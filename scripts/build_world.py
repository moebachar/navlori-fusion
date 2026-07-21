"""Build a Webots R2025a indoor-mall world from a converted ILN 2.0 floor in
ONE call. Self-contained: every lesson learned during the iterative build
(tile floor, shop shrink, custom-Solid wall decorations, proper windows,
post-emit wall dedup, path viz) is baked in. No follow-up cleanup script
needed.

Inputs (folder layout produced by scripts/convert_iln20_floor.py):
  <floor-dir>/
    meta/floor_info.json          - W, H in metres
    meta/geojson_map.json         - building polygons in EPSG:3857
    path_00/waypoints_raw.csv     - TIAGO++ spawn + first-target
    path_NN/ground_truth.csv      - dense 10 Hz GT for corridor detection
    path_NN/waypoints_raw.csv     - sparse landmark presses for path viz

Outputs:
  <out-dir>/<name>.wbt            - the Webots world
  <out-dir>/<name>_floor.png      - 5x5 m polished-concrete tile texture

Usage:
  .venv\\Scripts\\python.exe scripts\\build_world.py \\
      --floor-dir data/iln20_5d27099f_F2 \\
      --out-dir src/simulation/worlds \\
      --shrink 0.5 --paintings 20 --tvs 8 --windows 6

  # Defaults work too — every count knob defaults to a density-based "auto"
  # that scales with floor size:
  .venv\\Scripts\\python.exe scripts\\build_world.py \\
      --floor-dir data/iln20_5da138b7_B1 \\
      --out-dir src/simulation/worlds
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw


# ============================================================================
# Section 1 — defaults and styling (every value is configurable via CLI)
# ============================================================================
DEFAULTS = {
    "wall_thickness": 0.12,
    "wall_height": 3.20,
    "min_wall_length": 0.10,
    "min_wall_len_decoration": 1.5,
    "floor_tile_size": 5.0,
    "floor_margin": 200.0,
    "texture_px_per_m": 200,
    "seed": 1337,
    "shrink": 0.0,
    "window_min_wall_len": 4.5,
    "window_width": 1.4,
    "window_sill_z": 0.85,
    "window_lintel_z": 2.25,
    "window_glass_thick": 0.04,
    "ceiling_billboard_spacing": 28.0,
    "dedup_tol_perp": 0.20,
    "dedup_tol_yaw": 0.05,
    "dedup_overlap_frac": 0.50,
}

WALL_PALETTES = [
    # (basename, baseColor, roughness, metalness)
    ("plaster_offwhite",  (0.92, 0.90, 0.86), 0.85, 0.0),
    ("beige_warm",        (0.86, 0.80, 0.72), 0.80, 0.0),
    ("warm_cream",        (0.88, 0.84, 0.78), 0.85, 0.0),
    ("cool_white",        (0.95, 0.93, 0.90), 0.55, 0.0),
    ("muted_blue_grey",   (0.66, 0.71, 0.78), 0.70, 0.0),
    ("warm_terracotta",   (0.80, 0.55, 0.42), 0.85, 0.0),
    ("sage_green",        (0.60, 0.72, 0.62), 0.85, 0.0),
    ("light_marble",      (0.92, 0.90, 0.85), 0.30, 0.10),
    ("dark_marble",       (0.40, 0.42, 0.46), 0.40, 0.15),
    ("brushed_metal",     (0.55, 0.58, 0.62), 0.45, 0.30),
    ("accent_navy",       (0.20, 0.30, 0.55), 0.65, 0.0),
    ("accent_burgundy",   (0.45, 0.18, 0.22), 0.75, 0.0),
]

SIGN_COLORS = [
    (0.95, 0.18, 0.18),  # red
    (0.15, 0.45, 0.95),  # blue
    (0.95, 0.65, 0.10),  # orange
    (0.20, 0.78, 0.30),  # green
    (0.78, 0.20, 0.85),  # purple
    (0.95, 0.85, 0.10),  # yellow
    (0.92, 0.30, 0.55),  # pink
    (0.10, 0.30, 0.65),  # navy
    (0.55, 0.25, 0.10),  # brown
    (0.10, 0.55, 0.55),  # teal
    (0.92, 0.45, 0.10),  # amber
    (0.30, 0.30, 0.40),  # dark slate
]

PATH_COLORS = [
    (0.95, 0.45, 0.10),  # orange
    (0.20, 0.75, 0.35),  # green
    (0.40, 0.60, 0.95),  # blue
    (0.95, 0.85, 0.10),  # yellow
    (0.85, 0.20, 0.55),  # magenta
    (0.10, 0.75, 0.85),  # cyan
    (0.95, 0.20, 0.20),  # red
    (0.45, 0.25, 0.85),  # purple
]

BILLBOARD_TEXTS = [
    "ZARA", "H&M", "STARBUCKS", "APPLE", "NIKE", "ADIDAS", "UNIQLO",
    "SEPHORA", "MUJI", "GAP", "MANGO", "PRADA", "GUCCI", "SAMSUNG",
    "SONY", "KFC", "MCDONALD'S", "PIZZA HUT", "IKEA", "LEGO",
    "DECATHLON", "PUMA", "REEBOK", "MAC", "TIFFANY", "DIOR",
    "FOOD COURT", "RESTROOMS", "ELEVATOR", "INFORMATION", "CINEMA",
    "BOOKSTORE", "PHARMACY", "BANK", "CAFE", "BAKERY", "SPORTS",
]


# ============================================================================
# Section 2 — polygon math
# ============================================================================
def shoelace_area(ring) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def signed_area(ring) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += x1 * y2 - x2 * y1
    return a * 0.5


def polygon_centroid(ring):
    cx = cy = a2 = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        cross = x1 * y2 - x2 * y1
        a2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a2) < 1e-9:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (3 * a2), cy / (3 * a2)


def bbox(segs):
    xs = [x for s in segs for x in (s[0], s[2])]
    ys = [y for s in segs for y in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


def shrink_polygon_ring(ring, distance: float, min_area: float = 0.5):
    """Offset every edge of a closed polygon inward by `distance` (m); return
    the new closed ring, or None if the result is degenerate.

    Manual edge-offset: for each edge, translate both endpoints along the
    inward unit normal by `distance`, then intersect consecutive offset edges
    as infinite lines to find new vertices. Works for convex + gently concave
    shapes (mall shops are rectangular or L-shaped, well within range).
    """
    verts = list(ring[:-1]) if (ring and ring[0] == ring[-1]) else list(ring)
    n = len(verts)
    if n < 3 or distance <= 0:
        return None
    sa = signed_area(verts + [verts[0]])
    if abs(sa) < 1e-9:
        return None
    inward_sign = 1.0 if sa > 0 else -1.0

    offset_edges = []
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return None
        if inward_sign > 0:
            nx, ny = -dy / L, dx / L
        else:
            nx, ny = dy / L, -dx / L
        ox, oy = nx * distance, ny * distance
        offset_edges.append(((x1 + ox, y1 + oy), (x2 + ox, y2 + oy)))

    new_verts = []
    for i in range(n):
        a1, b1 = offset_edges[i]
        a2_, b2 = offset_edges[(i + 1) % n]
        x1, y1 = a1; x2, y2 = b1
        x3, y3 = a2_; x4, y4 = b2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            new_verts.append(b1)
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            new_verts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

    sa_new = signed_area(new_verts + [new_verts[0]])
    if sa_new * sa <= 0:
        return None
    if abs(sa_new) < min_area:
        return None
    new_verts.append(new_verts[0])
    return new_verts


def transform_xy(x, y, mx0, my0, sx, sy):
    return (x - mx0) * sx, (y - my0) * sy


# ============================================================================
# Section 3 — GeoJSON: split shop polygons from non-shop edges
# ============================================================================
def collect_segments_split(geo: dict):
    """Walk the GeoJSON once and return:
      shop_rings    : [{"ring": <closed Mercator ring>, "props": dict}, ...]
                      exterior rings of SHOP polygons (eligible for shrink)
      nonshop_segs  : [(x1,y1,x2,y2), ...] — floor outline polygon edges,
                      polygon holes, LineString features (no shrink applied)
    Together they form the full edge set; partitioned by "can this be shrunk?".
    """
    shop_rings = []
    nonshop_segs = []

    def emit_ring(ring):
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            nonshop_segs.append((x1, y1, x2, y2))

    def walk(geom, is_floor, props):
        gt = (geom or {}).get("type")
        coords = (geom or {}).get("coordinates")
        if gt == "LineString":
            emit_ring(coords); return
        polys = []
        if gt == "Polygon":
            polys = [coords]
        elif gt == "MultiPolygon":
            polys = coords or []
        elif gt == "GeometryCollection":
            for g in geom.get("geometries", []):
                walk(g, is_floor, props)
            return
        else:
            return
        for poly_rings in polys:
            if not poly_rings:
                continue
            ext = poly_rings[0]
            if len(ext) < 3:
                continue
            if is_floor:
                emit_ring(ext)
            else:
                shop_rings.append({"ring": ext, "props": props})
            for hole in poly_rings[1:]:
                if len(hole) >= 3:
                    emit_ring(hole)

    for f in geo.get("features", []):
        props = f.get("properties") or {}
        cat = props.get("category")
        is_floor = isinstance(cat, str) and cat.lower() == "floor"
        walk(f.get("geometry"), is_floor, props)
    return shop_rings, nonshop_segs


def segments_from_ring(ring):
    return [(x1, y1, x2, y2)
            for (x1, y1), (x2, y2) in zip(ring, ring[1:])]


# ============================================================================
# Section 4 — floor texture (generic 5x5m polished-concrete tile)
# ============================================================================
def render_tile_texture(out_png: str, tile_m: float, px_per_m: int,
                          seed: int):
    """Polished-concrete tile with subtle vein-noise + 1 m sub-grid + 5 m
    main-grid edge lines. Repeats across the whole floor via Floor.tileSize,
    so nothing in the texture can be misaligned with the walls."""
    side_px = int(tile_m * px_per_m)
    img = PILImage.new("RGB", (side_px, side_px), (218, 215, 210))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    for _ in range(80):
        cx = rng.randint(0, side_px - 1)
        cy = rng.randint(0, side_px - 1)
        r = rng.randint(6, 30)
        shade = rng.randint(208, 226)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=(shade, shade - 2, shade - 6))

    sub_px = int(1.0 * px_per_m)
    sub_color = (198, 195, 190)
    for x in range(sub_px, side_px, sub_px):
        draw.line([(x, 0), (x, side_px)], fill=sub_color, width=2)
    for y in range(sub_px, side_px, sub_px):
        draw.line([(0, y), (side_px, y)], fill=sub_color, width=2)

    edge_color = (170, 165, 158)
    draw.line([(0, 0), (side_px - 1, 0)],                  fill=edge_color, width=4)
    draw.line([(0, side_px - 1), (side_px - 1, side_px - 1)], fill=edge_color, width=4)
    draw.line([(0, 0), (0, side_px - 1)],                  fill=edge_color, width=4)
    draw.line([(side_px - 1, 0), (side_px - 1, side_px - 1)], fill=edge_color, width=4)

    img.save(out_png, "PNG", optimize=True)
    return side_px


# ============================================================================
# Section 5 — Webots emitters: walls
# ============================================================================
def fmt_wall(cfg, idx: int, x1: float, y1: float, x2: float, y2: float,
              palette_idx: int = 0) -> str:
    """Full-height wall Solid. Box thickness = cfg.wall_thickness, height =
    cfg.wall_height, length = distance between endpoints."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < cfg.min_wall_length:
        return ""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(dy, dx)
    _, (r, g, b), rough, met = WALL_PALETTES[palette_idx % len(WALL_PALETTES)]
    return (
        f"  DEF WALL_{idx:05d} Solid {{\n"
        f"    translation {mx:.4f} {my:.4f} {cfg.wall_height/2:.4f}\n"
        f"    rotation 0 0 1 {yaw:.5f}\n"
        f"    children [\n"
        f"      Shape {{\n"
        f"        appearance PBRAppearance {{\n"
        f"          baseColor {r} {g} {b}\n"
        f"          roughness {rough}\n"
        f"          metalness {met}\n"
        f"        }}\n"
        f"        geometry Box {{ size {length:.4f} {cfg.wall_thickness} "
        f"{cfg.wall_height} }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


def fmt_partial_wall(cfg, idx: int, mx: float, my: float, length: float,
                      yaw: float, z_center: float, height: float,
                      palette_idx: int = 0) -> str:
    """Wall piece with custom z-centre and height (sill / lintel around a
    window). Same thickness/material treatment as full walls."""
    _, (r, g, b), rough, met = WALL_PALETTES[palette_idx % len(WALL_PALETTES)]
    return (
        f"  DEF PWALL_{idx:05d} Solid {{\n"
        f"    translation {mx:.4f} {my:.4f} {z_center:.4f}\n"
        f"    rotation 0 0 1 {yaw:.5f}\n"
        f"    children [\n"
        f"      Shape {{\n"
        f"        appearance PBRAppearance {{\n"
        f"          baseColor {r} {g} {b}\n"
        f"          roughness {rough}\n"
        f"          metalness {met}\n"
        f"        }}\n"
        f"        geometry Box {{ size {length:.4f} {cfg.wall_thickness} "
        f"{height:.4f} }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


# ============================================================================
# Section 6 — windows (custom Solid: thin glass pane + 4 frame strips)
# ============================================================================
def fmt_window(cfg, idx: int, cx: float, cy: float, yaw: float) -> str:
    """Window built as: thin transparent glass + dark frame strips (top,
    bottom, left, right). Sits in the gap left by split_walls_for_windows()
    between two wall sub-segments and between a sill + lintel partial wall."""
    z_center = (cfg.window_sill_z + cfg.window_lintel_z) / 2
    glass_h = cfg.window_lintel_z - cfg.window_sill_z
    glass_w = cfg.window_width
    fw = 0.08  # frame strip thickness (vertical strips) / height (horizontal)
    return (
        f"DEF WIN_{idx:05d} Solid {{\n"
        f"  translation {cx:.4f} {cy:.4f} {z_center:.4f}\n"
        f"  rotation 0 0 1 {yaw:.5f}\n"
        f"  name \"window_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor 0.55 0.72 0.85\n"
        f"        emissiveColor 0.10 0.16 0.22\n"
        f"        roughness 0.05 metalness 0.20\n"
        f"        transparency 0.55\n"
        f"      }}\n"
        f"      geometry Box {{ size {glass_w:.3f} {cfg.window_glass_thick} "
        f"{glass_h:.3f} }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 { glass_h/2 - fw/2:.3f}\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {glass_w:.3f} {cfg.wall_thickness} "
        f"{fw:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 {-glass_h/2 + fw/2:.3f}\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {glass_w:.3f} {cfg.wall_thickness} "
        f"{fw:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"    Pose {{ translation {-glass_w/2 + fw/2:.3f} 0 0\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {fw:.3f} {cfg.wall_thickness} "
        f"{glass_h:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"    Pose {{ translation { glass_w/2 - fw/2:.3f} 0 0\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {fw:.3f} {cfg.wall_thickness} "
        f"{glass_h:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


# ============================================================================
# Section 7 — wall decorations
#
# ONE convention for every decoration:
#   Box size (W, T, H) where W = along the wall, T = thin (perpendicular),
#   H = vertical. Placed at (wall_mx, wall_my) + side * wall_normal * clearance
#   so the back face sits 5 mm off the wall. Rotation =
#     obj_yaw = wall_yaw       (for side = +1)
#     obj_yaw = wall_yaw + pi  (for side = -1)
#   so the Box's local +Y face points outward into the corridor.
#
# No PROTOs, no per-type rotation special cases.
# ============================================================================
def _wall_decor_pose(cfg, wall_mx, wall_my, wall_yaw, side, decor_thickness):
    nx = -math.sin(wall_yaw) * side
    ny =  math.cos(wall_yaw) * side
    gap = 0.005
    clearance = cfg.wall_thickness / 2 + decor_thickness / 2 + gap
    return (wall_mx + nx * clearance, wall_my + ny * clearance,
            wall_yaw if side == +1 else wall_yaw + math.pi)


def fmt_decor_painting_landscape(cfg, idx, mx, my, yaw, side, rng):
    T = 0.05
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 1.65
    r = rng.uniform(0.30, 0.85)
    g = rng.uniform(0.30, 0.85)
    b = rng.uniform(0.30, 0.85)
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_painting_landscape\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {r:.3f} {g:.3f} {b:.3f} "
        f"roughness 0.55 metalness 0 }}\n"
        f"      geometry Box {{ size 0.85 {T:.3f} 0.55 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0.001 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.13 0.08 "
        f"roughness 0.6 metalness 0 }}\n"
        f"      geometry Box {{ size 0.90 0.045 0.60 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decor_painting_portrait(cfg, idx, mx, my, yaw, side, rng):
    T = 0.05
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 1.65
    r = rng.uniform(0.30, 0.85)
    g = rng.uniform(0.30, 0.85)
    b = rng.uniform(0.30, 0.85)
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_painting_portrait\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {r:.3f} {g:.3f} {b:.3f} "
        f"roughness 0.55 metalness 0 }}\n"
        f"      geometry Box {{ size 0.50 {T:.3f} 0.85 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0.001 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.13 0.08 "
        f"roughness 0.6 metalness 0 }}\n"
        f"      geometry Box {{ size 0.55 0.045 0.90 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decor_tv(cfg, idx, mx, my, yaw, side, rng):
    T = 0.06
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 1.65
    sr = rng.uniform(0.05, 0.25)
    sg = rng.uniform(0.10, 0.40)
    sb = rng.uniform(0.30, 0.75)
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_tv\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.10 0.10 0.10 "
        f"roughness 0.45 metalness 0.20 }}\n"
        f"      geometry Box {{ size 1.20 {T:.3f} 0.70 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0.033 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {sr:.3f} {sg:.3f} {sb:.3f} "
        f"emissiveColor {sr*0.7:.3f} {sg*0.7:.3f} {sb*0.7:.3f} "
        f"roughness 0.2 metalness 0 }}\n"
        f"      geometry Box {{ size 1.05 0.01 0.58 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decor_mirror(cfg, idx, mx, my, yaw, side, rng):
    T = 0.04
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 1.50
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_mirror\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.15 0.15 0.18 "
        f"roughness 0.4 metalness 0.5 }}\n"
        f"      geometry Box {{ size 0.62 {T:.3f} 1.25 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0.022 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.92 0.94 0.96 "
        f"roughness 0.05 metalness 0.95 }}\n"
        f"      geometry Box {{ size 0.55 0.006 1.18 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decor_poster(cfg, idx, mx, my, yaw, side, rng):
    T = 0.03
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 1.55
    color_idx = abs(hash((idx, 7))) % len(SIGN_COLORS)
    r, g, b = SIGN_COLORS[color_idx]
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_poster\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {r:.3f} {g:.3f} {b:.3f} "
        f"emissiveColor {r*0.55:.3f} {g*0.55:.3f} {b*0.55:.3f} "
        f"roughness 0.3 metalness 0 }}\n"
        f"      geometry Box {{ size 0.90 {T:.3f} 1.10 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decor_sign(cfg, idx, mx, my, yaw, side, rng):
    T = 0.04
    px, py, oy = _wall_decor_pose(cfg, mx, my, yaw, side, T)
    z = 2.30
    color_idx = abs(hash((idx, 13))) % len(SIGN_COLORS)
    r, g, b = SIGN_COLORS[color_idx]
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_sign\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {r:.3f} {g:.3f} {b:.3f} "
        f"emissiveColor {r*0.65:.3f} {g*0.65:.3f} {b*0.65:.3f} "
        f"roughness 0.3 metalness 0 }}\n"
        f"      geometry Box {{ size 1.10 {T:.3f} 0.45 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 -0.025 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.18 0.20 "
        f"roughness 0.5 metalness 0.4 }}\n"
        f"      geometry Box {{ size 1.20 0.02 0.50 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


# Registry: type-name -> emitter callable
DECORATION_EMITTERS = {
    "painting_landscape": fmt_decor_painting_landscape,
    "painting_portrait":  fmt_decor_painting_portrait,
    "tv":                  fmt_decor_tv,
    "mirror":              fmt_decor_mirror,
    "poster":              fmt_decor_poster,
    "sign":                fmt_decor_sign,
}


# ============================================================================
# Section 8 — corner objects, floor objects, ceiling billboards
# ============================================================================
def fmt_corner_extinguisher(idx, x, y):
    return (
        f"DEF CORNER_{idx:05d} FireExtinguisher {{\n"
        f"  translation {x:.4f} {y:.4f} 0\n"
        f"  name \"corner_ext_{idx:05d}\"\n"
        f"}}\n"
    )


def fmt_corner_planter(idx, x, y):
    return (
        f"DEF CORNER_{idx:05d} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.35\n"
        f"  name \"corner_planter_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.30 0.22 0.15 "
        f"roughness 0.85 metalness 0 }}\n"
        f"      geometry Cylinder {{ radius 0.35 height 0.7 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 0.7\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.55 0.22 "
        f"roughness 0.95 metalness 0 }}\n"
        f"        geometry Sphere {{ radius 0.45 }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_floor_planter(idx, x, y):
    return (
        f"DEF FLOOROBJ_{idx:05d} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.45\n"
        f"  name \"planter_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.42 0.30 0.18 "
        f"roughness 0.85 metalness 0 }}\n"
        f"      geometry Cylinder {{ radius 0.45 height 0.9 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 0.85\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.18 0.55 0.22 "
        f"roughness 0.95 metalness 0 }}\n"
        f"        geometry Sphere {{ radius 0.55 }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_floor_bench(idx, x, y, yaw):
    return (
        f"DEF FLOOROBJ_{idx:05d} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.25\n"
        f"  rotation 0 0 1 {yaw:.5f}\n"
        f"  name \"bench_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.40 0.28 0.18 "
        f"roughness 0.7 metalness 0 }}\n"
        f"      geometry Box {{ size 1.8 0.5 0.45 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_floor_trashcan(idx, x, y):
    return (
        f"DEF FLOOROBJ_{idx:05d} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.4\n"
        f"  name \"trashcan_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.22 0.22 0.25 "
        f"roughness 0.5 metalness 0.4 }}\n"
        f"      geometry Cylinder {{ radius 0.27 height 0.85 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_floor_pillar(idx, x, y):
    return (
        f"DEF FLOOROBJ_{idx:05d} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 1.35\n"
        f"  name \"pillar_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.88 0.82 0.72 "
        f"roughness 0.55 metalness 0.10 }}\n"
        f"      geometry Cylinder {{ radius 0.30 height 2.7 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


FLOOR_OBJ_EMITTERS_NO_YAW = [fmt_floor_planter, fmt_floor_trashcan, fmt_floor_pillar]


def fmt_ceiling_billboard(idx, cx, cy, yaw, text, color_idx):
    r, g, b = SIGN_COLORS[color_idx % len(SIGN_COLORS)]
    panel_w = min(3.2, max(1.5, len(text) * 0.34))
    return (
        f"DEF BB_{idx:05d} Solid {{\n"
        f"  translation {cx:.4f} {cy:.4f} 2.55\n"
        f"  rotation 0 0 1 {yaw:.5f}\n"
        f"  name \"billboard_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor {r} {g} {b}\n"
        f"        emissiveColor {r*0.65:.3f} {g*0.65:.3f} {b*0.65:.3f}\n"
        f"        roughness 0.30 metalness 0.10\n"
        f"      }}\n"
        f"      geometry Box {{ size {panel_w:.2f} 0.06 0.50 }}\n"
        f"    }}\n"
        f"    Pose {{ translation 0 0 0.30\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.30 0.30 0.30 "
        f"roughness 0.6 metalness 0.4 }}\n"
        f"        geometry Box {{ size 0.05 0.05 0.20 }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_path_marker(idx, x, y, color, radius=0.13):
    r, g, b = color
    return (
        f"  DEF WP_{idx:05d} Solid {{\n"
        f"    translation {x:.4f} {y:.4f} 0.05\n"
        f"    name \"wp_{idx:05d}\"\n"
        f"    children [\n"
        f"      Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor {r} {g} {b} "
        f"emissiveColor {r*0.6:.2f} {g*0.6:.2f} {b*0.6:.2f} roughness 0.4 metalness 0 }}\n"
        f"        geometry Sphere {{ radius {radius:.3f} }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


def fmt_path_segment(idx, x1, y1, x2, y2, color):
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 0.05:
        return ""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(y2 - y1, x2 - x1)
    r, g, b = color
    return (
        f"  DEF PS_{idx:05d} Solid {{\n"
        f"    translation {mx:.4f} {my:.4f} 0.02\n"
        f"    rotation 0 0 1 {yaw:.5f}\n"
        f"    name \"pseg_{idx:05d}\"\n"
        f"    children [\n"
        f"      Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor {r} {g} {b} "
        f"emissiveColor {r*0.45:.2f} {g*0.45:.2f} {b*0.45:.2f} roughness 0.4 metalness 0 }}\n"
        f"        geometry Box {{ size {length:.4f} 0.06 0.025 }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


def fmt_marker(name, x, y, color_rgb):
    r, g, b = color_rgb
    return (
        f"DEF MARKER_{name} Solid {{\n"
        f"  translation {x:.4f} {y:.4f} 0.50\n"
        f"  name \"marker_{name.lower()}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {r} {g} {b} "
        f"emissiveColor {r*0.4:.2f} {g*0.4:.2f} {b*0.4:.2f} }}\n"
        f"      geometry Sphere {{ radius 0.35 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_ceiling(cx, cy, W, H, wall_height, overhang, thickness):
    """Single light-grey slab spanning the building footprint plus `overhang`
    metres on every side, suspended just above wall tops (bottom face at
    z = wall_height). One DEF CEILING Group with one Solid inside, so the
    user can hide it in Webots by right-clicking DEF CEILING in the scene
    tree -> Hide (R2025a), or by setting appearance.transparency to 1.0.
    """
    sx = W + 2 * overhang
    sy = H + 2 * overhang
    z = wall_height + thickness / 2
    return (
        f"# To hide the ceiling at runtime, right-click DEF CEILING in the\n"
        f"# Webots scene tree and choose Hide -- or edit appearance.transparency\n"
        f"# below to 1.0 for a permanent hide.\n"
        f"DEF CEILING Group {{\n"
        f"  children [\n"
        f"    Solid {{\n"
        f"      translation {cx:.4f} {cy:.4f} {z:.4f}\n"
        f"      name \"ceiling_slab\"\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{\n"
        f"            baseColor 0.92 0.92 0.90\n"
        f"            roughness 0.85\n"
        f"            metalness 0.0\n"
        f"            transparency 0.0\n"
        f"          }}\n"
        f"          geometry Box {{ size {sx:.4f} {sy:.4f} {thickness:.4f} }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


# ============================================================================
# Section 9 — geometry helpers (corridor side detection, distance)
# ============================================================================
def segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def min_dist_to_walls(px, py, walls):
    return min(segment_distance(px, py, *w) for w in walls) if walls else 1e9


def flat_traj_points(trajectories):
    return [pt for poly in trajectories for pt in poly]


def is_near_any_point(px, py, points, max_d):
    md2 = max_d * max_d
    for tx, ty in points:
        dx, dy = px - tx, py - ty
        if dx * dx + dy * dy <= md2:
            return True
    return False


def wall_corridor_side(seg, points, probe=2.5):
    """Probe +/- `probe` m along the wall normal; return +1 / -1 / 0
    (no clear corridor side)."""
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(yaw), math.cos(yaw)
    plus_near  = is_near_any_point(mx + nx * probe, my + ny * probe, points, probe)
    minus_near = is_near_any_point(mx - nx * probe, my - ny * probe, points, probe)
    if plus_near and not minus_near:
        return +1
    if minus_near and not plus_near:
        return -1
    return 0


# ============================================================================
# Section 10 — segment-level dedup (pre-build, cheap)
# ============================================================================
def dedupe_segments(segs, endpoint_tol=0.50, midpoint_tol=0.60, yaw_tol=0.10):
    """Drop duplicate wall segments using endpoint + midpoint+yaw heuristics.
    Catches the common case of adjacent shops sharing an edge."""
    seen_ep = set()
    seen_mid = set()
    out = []
    for x1, y1, x2, y2 in segs:
        a = (round(x1 / endpoint_tol), round(y1 / endpoint_tol))
        b = (round(x2 / endpoint_tol), round(y2 / endpoint_tol))
        key_ep = (a, b) if a <= b else (b, a)
        if key_ep in seen_ep:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        yaw = math.atan2(y2 - y1, x2 - x1) % math.pi
        key_mid = (round(mx / midpoint_tol), round(my / midpoint_tol),
                   round(yaw / yaw_tol))
        if key_mid in seen_mid:
            continue
        seen_ep.add(key_ep)
        seen_mid.add(key_mid)
        out.append((x1, y1, x2, y2))
    return out


# ============================================================================
# Section 11 — window placement (split walls)
# ============================================================================
def split_walls_for_windows(cfg, segs_local, trajectories, rng, count: int):
    """Pick `count` corridor-side walls (length >= window_min_wall_len) and
    cut each open for a window. Returns (full_segs, partial_walls, windows).

    full_segs    : the original list with the chosen wall replaced by its
                   two side-piece sub-segments
    partial_walls: [(mx,my,len,yaw,z_center,height), ...] for sills + lintels
    windows      : [(cx,cy,yaw), ...] for the glass placements
    """
    points = flat_traj_points(trajectories) if trajectories else []
    eligible_idx = []
    for i, s in enumerate(segs_local):
        x1, y1, x2, y2 = s
        length = math.hypot(x2 - x1, y2 - y1)
        if length < cfg.window_min_wall_len:
            continue
        if not points or wall_corridor_side(s, points) == 0:
            continue
        eligible_idx.append(i)
    if not eligible_idx:
        return list(segs_local), [], []

    count = max(0, min(count, len(eligible_idx)))
    chosen = set(rng.sample(eligible_idx, count))

    full_segs = []
    partials = []
    windows = []
    half_w = cfg.window_width / 2
    sill_h = cfg.window_sill_z
    sill_z_center = cfg.window_sill_z / 2
    lintel_h = cfg.wall_height - cfg.window_lintel_z
    lintel_z_center = (cfg.window_lintel_z + cfg.wall_height) / 2

    for i, s in enumerate(segs_local):
        if i not in chosen:
            full_segs.append(s); continue
        x1, y1, x2, y2 = s
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        yaw = math.atan2(y2 - y1, x2 - x1)
        ax = x1 + ux * (length / 2 - half_w)
        ay = y1 + uy * (length / 2 - half_w)
        bx = x1 + ux * (length / 2 + half_w)
        by = y1 + uy * (length / 2 + half_w)
        cx = x1 + ux * (length / 2)
        cy = y1 + uy * (length / 2)
        full_segs.append((x1, y1, ax, ay))
        full_segs.append((bx, by, x2, y2))
        partials.append((cx, cy, cfg.window_width, yaw,
                          sill_z_center, sill_h))
        partials.append((cx, cy, cfg.window_width, yaw,
                          lintel_z_center, lintel_h))
        windows.append((cx, cy, yaw))
    return full_segs, partials, windows


# ============================================================================
# Section 12 — corner detection
# ============================================================================
def find_corners(segs_local, tol=0.10):
    """Vertices shared by 2+ walls that meet at a non-collinear angle."""
    bucket = {}
    def key(x, y):
        return (round(x / tol), round(y / tol))
    for i, (x1, y1, x2, y2) in enumerate(segs_local):
        for px, py in ((x1, y1), (x2, y2)):
            bucket.setdefault(key(px, py), []).append((i, px, py, x1, y1, x2, y2))
    corners = []
    for _, items in bucket.items():
        if len(items) < 2:
            continue
        cx = sum(it[1] for it in items) / len(items)
        cy = sum(it[2] for it in items) / len(items)
        yaws = []
        for _, px, py, x1, y1, x2, y2 in items:
            if math.hypot(px - x1, py - y1) < 0.2:
                yaws.append(math.atan2(y2 - y1, x2 - x1))
            else:
                yaws.append(math.atan2(y1 - y2, x1 - x2))
        is_corner = False
        for i in range(len(yaws)):
            for j in range(i + 1, len(yaws)):
                d = abs(yaws[i] - yaws[j])
                while d > math.pi:
                    d -= 2 * math.pi
                d = abs(d)
                if 0.35 < d < math.pi - 0.35:
                    is_corner = True; break
            if is_corner: break
        if is_corner:
            corners.append((cx, cy))
    return corners


# ============================================================================
# Section 13 — floor object placement (against corridor-side walls)
# ============================================================================
def collect_floor_object_candidates(segs_local, trajectories, rng,
                                      offset=0.55, probe=2.5,
                                      min_wall_len=2.5, object_clearance=0.30):
    points = flat_traj_points(trajectories)
    out = []
    for s in segs_local:
        x1, y1, x2, y2 = s
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_wall_len:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        yaw = math.atan2(y2 - y1, x2 - x1)
        nx, ny = -math.sin(yaw), math.cos(yaw)
        plus_corr  = is_near_any_point(mx + nx * probe, my + ny * probe, points, probe)
        minus_corr = is_near_any_point(mx - nx * probe, my - ny * probe, points, probe)
        if plus_corr == minus_corr:
            continue
        side = +1 if plus_corr else -1
        px = mx + nx * offset * side
        py = my + ny * offset * side
        if is_near_any_point(px, py, points, object_clearance):
            continue
        out.append((px, py, yaw, side))
    rng.shuffle(out)
    return out


# ============================================================================
# Section 14 — ceiling billboard placement (along trajectories)
# ============================================================================
def collect_ceiling_billboard_candidates(trajectories, segs_local, rng,
                                            spacing=28.0, dedupe_radius=14.0,
                                            clearance_from_walls=1.2):
    placements = []
    for poly in trajectories:
        cum = 0.0
        last = poly[0]
        next_target = rng.uniform(spacing * 0.5, spacing * 1.5)
        for i in range(1, len(poly)):
            pt = poly[i]
            seg_len = math.hypot(pt[0] - last[0], pt[1] - last[1])
            while cum + seg_len >= next_target:
                f = (next_target - cum) / seg_len
                px = last[0] + f * (pt[0] - last[0])
                py = last[1] + f * (pt[1] - last[1])
                yaw = math.atan2(pt[1] - last[1], pt[0] - last[0])
                if min_dist_to_walls(px, py, segs_local) >= clearance_from_walls:
                    placements.append((px, py, yaw))
                next_target += rng.uniform(spacing * 0.6, spacing * 1.4)
            cum += seg_len
            last = pt
    kept = []
    for p in placements:
        if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < dedupe_radius
                    for q in kept):
            kept.append(p)
    rng.shuffle(kept)
    return kept


# ============================================================================
# Section 15 — post-emit wall dedup (final safety net)
#
# Runs on the assembled .wbt text right before we write it. Catches any
# overlapping WALL_/PWALL_ Solids that survived the segment-level dedup
# (typical cause: bucket-boundary near-miss).
# ============================================================================
WALL_HEAD_RE = re.compile(r"DEF (P?WALL_\d+) Solid \{")
TRANSLATION_RE = re.compile(
    r"translation\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
)
ROTATION_RE = re.compile(
    r"rotation\s+0\s+0\s+1\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
)
BOX_SIZE_RE = re.compile(
    r"geometry\s+Box\s*\{\s*size\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
)


def _find_block_end(text, brace_open_idx):
    depth = 0
    i = brace_open_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def _parse_walls_from_text(text):
    out = []
    for head in WALL_HEAD_RE.finditer(text):
        brace_idx = head.end() - 1
        end = _find_block_end(text, brace_idx)
        if end < len(text) and text[end] == "\n":
            end += 1
        block = text[head.start():end]
        mt = TRANSLATION_RE.search(block)
        mr = ROTATION_RE.search(block)
        mb = BOX_SIZE_RE.search(block)
        if not (mt and mr and mb):
            continue
        out.append({
            "name": head.group(1),
            "start": head.start(), "end": end,
            "cx": float(mt.group(1)), "cy": float(mt.group(2)),
            "cz": float(mt.group(3)),
            "yaw": float(mr.group(1)),
            "length": float(mb.group(1)),
            "thickness": float(mb.group(2)),
            "height": float(mb.group(3)),
        })
    return out


def _line_key(w, tol_yaw, tol_perp):
    yaw_mod = w["yaw"] % math.pi
    if yaw_mod > math.pi - tol_yaw:
        yaw_mod -= math.pi
    perp = -w["cx"] * math.sin(yaw_mod) + w["cy"] * math.cos(yaw_mod)
    return (round(yaw_mod / tol_yaw), round(perp / tol_perp))


def _axis_span(w):
    yaw_mod = w["yaw"] % math.pi
    s_mid = w["cx"] * math.cos(yaw_mod) + w["cy"] * math.sin(yaw_mod)
    return (s_mid - w["length"] / 2, s_mid + w["length"] / 2,
            w["cz"] - w["height"] / 2, w["cz"] + w["height"] / 2)


def _overlap_frac(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    return 0.0 if shorter <= 0 else inter / shorter


def dedup_wall_blocks_in_text(text, tol_yaw, tol_perp, overlap_frac_min):
    walls = _parse_walls_from_text(text)
    if not walls:
        return text, 0
    buckets = {}
    spans = []
    for i, w in enumerate(walls):
        spans.append(_axis_span(w))
        k = _line_key(w, tol_yaw, tol_perp)
        buckets.setdefault(k, []).append(i)

    order = sorted(range(len(walls)), key=lambda i: -walls[i]["length"])
    kept_idx = []
    dropped = set()

    for i in order:
        if i in dropped:
            continue
        w = walls[i]
        ki = _line_key(w, tol_yaw, tol_perp)
        s0, s1, z0, z1 = spans[i]
        cands = []
        for dy in (-1, 0, 1):
            for dp in (-1, 0, 1):
                neigh = buckets.get((ki[0] + dy, ki[1] + dp))
                if neigh:
                    cands.extend(neigh)
        is_dup = False
        for j in cands:
            if j == i or j in dropped or j not in kept_idx:
                continue
            wj = walls[j]
            d_yaw = abs((w["yaw"] - wj["yaw"]) % math.pi)
            d_yaw = min(d_yaw, math.pi - d_yaw)
            if d_yaw > tol_yaw:
                continue
            yaw_avg = (w["yaw"] + wj["yaw"]) / 2.0
            nx, ny = -math.sin(yaw_avg), math.cos(yaw_avg)
            d_perp = abs((w["cx"] - wj["cx"]) * nx + (w["cy"] - wj["cy"]) * ny)
            if d_perp > tol_perp:
                continue
            sj0, sj1, zj0, zj1 = spans[j]
            f_axis = _overlap_frac(s0, s1, sj0, sj1)
            f_z = _overlap_frac(z0, z1, zj0, zj1)
            if f_axis >= overlap_frac_min and f_z >= overlap_frac_min:
                dropped.add(i)
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)

    if not dropped:
        return text, 0
    pieces = []
    cursor = 0
    for w in walls:
        if w["name"] not in {walls[i]["name"] for i in kept_idx}:
            pieces.append(text[cursor:w["start"]])
            cursor = w["end"]
    pieces.append(text[cursor:])
    return "".join(pieces), len(dropped)


# ============================================================================
# Section 16 — assemble the .wbt
# ============================================================================
def build_wbt(cfg, name, W, H, segs_local, start_xy, second_xy,
                floor_texture_basename, trajectories,
                raw_waypoints_per_path, counts):
    """Assemble all the blocks into one .wbt text. `counts` is a dict with
    requested counts for every placement category (windows, paintings, tvs,
    mirrors, posters, signs, fire_extinguishers, corner_planters,
    floor_objects, ceiling_billboards). The corresponding placements never
    exceed available eligible slots."""
    cx, cy = W / 2, H / 2
    floor_w = W + 2 * cfg.floor_margin
    floor_h = H + 2 * cfg.floor_margin
    sx, sy = start_xy
    spawn_yaw = (math.atan2(second_xy[1] - sy, second_xy[0] - sx)
                  if second_xy else 0.0)

    rng = random.Random(cfg.seed)
    points = flat_traj_points(trajectories) if trajectories else []

    # ---- (a) segment-level dedup (cheap pre-pass)
    segs_local = dedupe_segments(segs_local)

    # ---- (b) windows: pick count corridor walls + cut them open
    win_rng = random.Random(cfg.seed + 13)
    segs_local, partial_walls, window_placements = split_walls_for_windows(
        cfg, segs_local, trajectories, win_rng, counts["windows"])

    # ---- (c) wall blocks (full + partial)
    wall_blocks = []
    for i, s in enumerate(segs_local):
        block = fmt_wall(cfg, i, *s, palette_idx=(i * 17 + 3) % len(WALL_PALETTES))
        if block:
            wall_blocks.append(block)
    for pi, (pm_x, pm_y, plen, pyaw, pzc, pht) in enumerate(partial_walls):
        wall_blocks.append(fmt_partial_wall(
            cfg, pi, pm_x, pm_y, plen, pyaw, pzc, pht,
            palette_idx=(pi * 13 + 5) % len(WALL_PALETTES)))

    # ---- (d) windows
    window_blocks = [fmt_window(cfg, i, *w)
                     for i, w in enumerate(window_placements)]

    # ---- (e) wall decorations: take eligible walls, allocate per type
    decor_eligible = []
    for s in segs_local:
        length = math.hypot(s[2] - s[0], s[3] - s[1])
        if length < cfg.min_wall_len_decoration:
            continue
        if points:
            side = wall_corridor_side(s, points)
            if side == 0:
                continue
        else:
            side = +1
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        wall_yaw = math.atan2(s[3] - s[1], s[2] - s[0])
        decor_eligible.append((mx, my, wall_yaw, side))

    decor_rng = random.Random(cfg.seed + 41)
    decor_rng.shuffle(decor_eligible)

    decor_blocks = []
    decor_idx = 0
    # Process types in stable order so the same seed produces the same world.
    for type_name in ("painting_landscape", "painting_portrait", "tv",
                       "mirror", "poster", "sign"):
        n = counts.get(type_name, 0)
        emitter = DECORATION_EMITTERS[type_name]
        take = min(n, len(decor_eligible))
        for _ in range(take):
            mx, my, wyaw, side = decor_eligible.pop()
            decor_blocks.append(emitter(cfg, decor_idx, mx, my, wyaw, side,
                                          decor_rng))
            decor_idx += 1

    # ---- (f) corner objects: pick from corridor-adjacent corners
    corner_blocks = []
    corner_rng = random.Random(cfg.seed + 23)
    corner_candidates = []
    if points:
        for cx_c, cy_c in find_corners(segs_local, tol=0.10):
            if is_near_any_point(cx_c, cy_c, points, 2.5):
                corner_candidates.append((cx_c, cy_c))
    corner_rng.shuffle(corner_candidates)
    n_ext = min(counts["fire_extinguishers"], len(corner_candidates))
    for ci in range(n_ext):
        cx_c, cy_c = corner_candidates.pop()
        corner_blocks.append(fmt_corner_extinguisher(ci, cx_c, cy_c))
    n_plant = min(counts["corner_planters"], len(corner_candidates))
    for pi_ in range(n_plant):
        cx_c, cy_c = corner_candidates.pop()
        corner_blocks.append(fmt_corner_planter(n_ext + pi_, cx_c, cy_c))

    # ---- (g) floor objects against corridor walls
    floor_obj_blocks = []
    if trajectories:
        floor_rng = random.Random(cfg.seed + 7)
        candidates = collect_floor_object_candidates(
            segs_local, trajectories, floor_rng)
        n_floor = min(counts["floor_objects"], len(candidates))
        for fi in range(n_floor):
            fx, fy, fyaw, _ = candidates.pop()
            # 25% benches (need yaw), 75% planter/trashcan/pillar
            if floor_rng.random() < 0.25:
                floor_obj_blocks.append(fmt_floor_bench(fi, fx, fy, fyaw))
            else:
                emitter = floor_rng.choice(FLOOR_OBJ_EMITTERS_NO_YAW)
                floor_obj_blocks.append(emitter(fi, fx, fy))

    # ---- (h) ceiling billboards
    bb_blocks = []
    if trajectories:
        bb_rng = random.Random(cfg.seed + 31)
        bb_cands = collect_ceiling_billboard_candidates(
            trajectories, segs_local, bb_rng,
            spacing=cfg.ceiling_billboard_spacing,
            dedupe_radius=14.0, clearance_from_walls=1.2)
        n_bb = min(counts["ceiling_billboards"], len(bb_cands))
        bb_names = BILLBOARD_TEXTS[:]
        bb_rng.shuffle(bb_names)
        for bi in range(n_bb):
            bx, by, byaw = bb_cands.pop()
            txt = bb_names[bi % len(bb_names)]
            bb_blocks.append(fmt_ceiling_billboard(bi, bx, by, byaw, txt, bi))

    # ---- (i) path visualisation
    path_blocks = []
    if raw_waypoints_per_path and counts["max_paths"] > 0:
        seq = list(raw_waypoints_per_path)
        if len(seq) > counts["max_paths"]:
            step = len(seq) / counts["max_paths"]
            seq = [seq[int(i * step)] for i in range(counts["max_paths"])]
        wpi = 0
        psi = 0
        for path_idx, wps in enumerate(seq):
            color = PATH_COLORS[path_idx % len(PATH_COLORS)]
            for x, y in wps:
                path_blocks.append(fmt_path_marker(wpi, x, y, color))
                wpi += 1
            for (x1, y1), (x2, y2) in zip(wps, wps[1:]):
                blk = fmt_path_segment(psi, x1, y1, x2, y2, color)
                if blk:
                    path_blocks.append(blk)
                    psi += 1

    # ---- assemble the final text
    walls_blob = "".join(wall_blocks)
    window_blob = "".join(window_blocks)
    decor_blob = "\n".join(decor_blocks)
    corner_blob = "\n".join(corner_blocks)
    floor_obj_blob = "\n".join(floor_obj_blocks)
    bb_blob = "\n".join(bb_blocks)
    path_blob = "".join(path_blocks)

    vp_h = max(20.0, max(W, H) * 0.35)

    extern_lines = [
        '"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
        'objects/backgrounds/protos/TexturedBackground.proto"',
        '"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
        'objects/backgrounds/protos/TexturedBackgroundLight.proto"',
        '"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
        'objects/floors/protos/Floor.proto"',
        '"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
        'robots/pal_robotics/tiagopp/protos/Tiago++.proto"',
    ]
    if any("FireExtinguisher" in b for b in corner_blocks):
        extern_lines.append(
            '"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
            'objects/factory/fire_extinguisher/protos/FireExtinguisher.proto"')
    extern_blob = "\n".join(f"EXTERNPROTO {x}" for x in extern_lines)

    return f"""#VRML_SIM R2025a utf8

{extern_blob}

WorldInfo {{
  basicTimeStep 32
  coordinateSystem "ENU"
}}
Viewpoint {{
  orientation -0.5773 0.5773 0.5773 2.0944
  position {cx:.2f} {cy - vp_h * 0.6:.2f} {vp_h:.2f}
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}

DirectionalLight {{
  direction 0.2 0.3 -1
  intensity 1.8
  ambientIntensity 0.7
  color 1 0.96 0.90
  castShadows FALSE
}}

Floor {{
  translation {cx:.4f} {cy:.4f} 0
  rotation 0 0 1 {cfg.floor_rotation:.5f}
  size {floor_w:.4f} {floor_h:.4f}
  tileSize {cfg.floor_tile_size:.2f} {cfg.floor_tile_size:.2f}
  appearance PBRAppearance {{
    baseColorMap ImageTexture {{
      url [ "{floor_texture_basename}" ]
    }}
    roughness 0.55
    metalness 0
  }}
}}

DEF WALLS Group {{
  children [
{walls_blob}  ]
}}

DEF WINDOWS Group {{
  children [
{window_blob}  ]
}}

DEF DECORATIONS Group {{
  children [
{decor_blob}
  ]
}}

DEF CORNER_OBJECTS Group {{
  children [
{corner_blob}
  ]
}}

DEF FLOOR_OBJECTS Group {{
  children [
{floor_obj_blob}
  ]
}}

DEF CEILING_BILLBOARDS Group {{
  children [
{bb_blob}
  ]
}}

DEF ROBOT_PATHS Group {{
  children [
{path_blob}  ]
}}

{fmt_marker("START_GREEN", sx, sy, (0.0, 0.9, 0.0))}
{fmt_marker("SECOND_RED",  second_xy[0], second_xy[1], (0.95, 0.05, 0.05))}

{fmt_ceiling(cx, cy, W, H, cfg.wall_height, cfg.ceiling_overhang, cfg.ceiling_thickness) if cfg.ceiling else ""}
DEF TIAGO Tiago++ {{
  translation {sx:.4f} {sy:.4f} 0
  rotation 0 0 1 {spawn_yaw:.5f}
  name "tiago"
  controller "<none>"
  supervisor TRUE
}}
"""


# ============================================================================
# Section 17 — CLI + driver
# ============================================================================
class Cfg:
    """Container for all geometry knobs, populated from CLI then passed through
    to every emitter so they don't need a global state."""
    pass


def _resolve_count(user_value, default_count):
    """User value: -1 = auto (use default_count), >=0 = exact requested."""
    return default_count if user_value < 0 else user_value


def main():
    ap = argparse.ArgumentParser(
        description="Build a Webots indoor-mall world from a converted "
                    "ILN 2.0 floor folder. One shot, no cleanup needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required IO
    ap.add_argument("--floor-dir", required=True,
                    help="path to data/iln20_<site>_<floor>/")
    ap.add_argument("--out-dir", required=True,
                    help="where to write <name>.wbt and <name>_floor.png")
    ap.add_argument("--name", default=None,
                    help="world basename (default: floor folder name)")

    # Geometry
    ap.add_argument("--shrink", type=float, default=DEFAULTS["shrink"],
                    help="shrink each shop polygon inward by N m before "
                         "emitting wall segments. Widens corridors by ~2N "
                         "between facing shops. Floor outline NOT shrunk.")
    ap.add_argument("--wall-thickness", type=float,
                    default=DEFAULTS["wall_thickness"])
    ap.add_argument("--wall-height", type=float,
                    default=DEFAULTS["wall_height"])
    ap.add_argument("--floor-tile-size", type=float,
                    default=DEFAULTS["floor_tile_size"],
                    help="floor texture repeats every N m (square tile)")
    ap.add_argument("--floor-margin", type=float,
                    default=DEFAULTS["floor_margin"],
                    help="extra ground around the building bbox so the floor "
                         "extends beyond every wall")
    ap.add_argument("--floor-rotation", type=float, default=0.0,
                    help="rotate Floor proto by this many radians around z")
    ap.add_argument("--texture-px-per-m", type=int,
                    default=DEFAULTS["texture_px_per_m"],
                    help="pixel resolution of the 5x5 m tile texture")

    # Window placement
    ap.add_argument("--windows", type=int, default=-1,
                    help="number of windows. -1 = auto (18 %% of long "
                         "corridor-side walls)")
    ap.add_argument("--window-min-wall-len", type=float,
                    default=DEFAULTS["window_min_wall_len"],
                    help="walls shorter than this can't host a window")

    # Wall decoration counts (per type)
    ap.add_argument("--paintings-landscape", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")
    ap.add_argument("--paintings-portrait", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")
    ap.add_argument("--tvs", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")
    ap.add_argument("--mirrors", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")
    ap.add_argument("--posters", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")
    ap.add_argument("--signs", type=int, default=-1,
                    help="-1 = auto (10 %% of eligible walls)")

    # Corner / floor / ceiling
    ap.add_argument("--fire-extinguishers", type=int, default=-1,
                    help="-1 = auto (~30 %% of corridor-adjacent corners)")
    ap.add_argument("--corner-planters", type=int, default=-1,
                    help="-1 = auto (~20 %% of remaining corners)")
    ap.add_argument("--floor-objects", type=int, default=-1,
                    help="bench / planter / trashcan / pillar against walls. "
                         "-1 = auto (~5 %% of corridor-side walls)")
    ap.add_argument("--ceiling-billboards", type=int, default=-1,
                    help="-1 = auto (one every ~28 m along trajectories)")
    ap.add_argument("--ceiling-billboard-spacing", type=float,
                    default=DEFAULTS["ceiling_billboard_spacing"],
                    help="only used in auto mode")

    # Path viz
    ap.add_argument("--max-paths", type=int, default=20,
                    help="cap visible robot paths so big mall doesn't get 200 "
                         "overlapping traces. 0 = disable path viz entirely.")

    # Ceiling (toggle-able in Webots via scene-tree Hide / Show)
    ap.add_argument("--no-ceiling", action="store_true",
                    help="skip the ceiling slab (default: ceiling included)")
    ap.add_argument("--ceiling-overhang", type=float, default=2.0,
                    help="ceiling extends this many metres past each wall "
                         "edge so its rim is not visible from inside the mall")
    ap.add_argument("--ceiling-thickness", type=float, default=0.10,
                    help="vertical thickness of the ceiling slab")

    # Wall dedup (post-emit safety net; on by default)
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip the post-emit wall dedup pass")
    ap.add_argument("--dedup-tol-perp", type=float,
                    default=DEFAULTS["dedup_tol_perp"])
    ap.add_argument("--dedup-tol-yaw", type=float,
                    default=DEFAULTS["dedup_tol_yaw"])
    ap.add_argument("--dedup-overlap-frac", type=float,
                    default=DEFAULTS["dedup_overlap_frac"])

    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = ap.parse_args()

    floor_dir = Path(args.floor_dir).resolve()
    if not floor_dir.is_dir():
        sys.exit(f"floor dir not found: {floor_dir}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or floor_dir.name

    # Pack the Cfg
    cfg = Cfg()
    cfg.wall_thickness = args.wall_thickness
    cfg.wall_height = args.wall_height
    cfg.min_wall_length = DEFAULTS["min_wall_length"]
    cfg.min_wall_len_decoration = DEFAULTS["min_wall_len_decoration"]
    cfg.floor_tile_size = args.floor_tile_size
    cfg.floor_margin = args.floor_margin
    cfg.floor_rotation = args.floor_rotation
    cfg.window_min_wall_len = args.window_min_wall_len
    cfg.window_width = DEFAULTS["window_width"]
    cfg.window_sill_z = DEFAULTS["window_sill_z"]
    cfg.window_lintel_z = DEFAULTS["window_lintel_z"]
    cfg.window_glass_thick = DEFAULTS["window_glass_thick"]
    cfg.ceiling_billboard_spacing = args.ceiling_billboard_spacing
    cfg.ceiling = not args.no_ceiling
    cfg.ceiling_overhang = args.ceiling_overhang
    cfg.ceiling_thickness = args.ceiling_thickness
    cfg.seed = args.seed

    # --- Load floor meta + GeoJSON
    meta = floor_dir / "meta"
    fi = json.load(open(meta / "floor_info.json"))
    mi = fi.get("map_info", fi)
    W, H = float(mi["width"]), float(mi["height"])
    print(f"[build] floor: {W:.2f} x {H:.2f} m  ({W*H:,.0f} m^2)")

    geo = json.load(open(meta / "geojson_map.json"))
    shop_rings_merc, nonshop_segs_merc = collect_segments_split(geo)
    n_shop_edges = sum(max(0, len(s["ring"]) - 1) for s in shop_rings_merc)
    print(f"[build] GeoJSON: {len(shop_rings_merc)} shop polygons "
          f"({n_shop_edges:,} edges), {len(nonshop_segs_merc):,} non-shop "
          f"segments")

    # --- bbox + transform (uses union of all segments for a stable frame)
    bbox_segs = list(nonshop_segs_merc)
    for s in shop_rings_merc:
        bbox_segs.extend(segments_from_ring(s["ring"]))
    xmin, ymin, xmax, ymax = bbox(bbox_segs)
    sx_scale = W / (xmax - xmin) if xmax > xmin else 1.0
    sy_scale = H / (ymax - ymin) if ymax > ymin else 1.0
    print(f"[build] mercator origin: ({xmin:.2f}, {ymin:.2f})  "
          f"scale: x={sx_scale:.6f} y={sy_scale:.6f}")

    # --- non-shop segs straight through
    nonshop_segs_local = [(
        (x1 - xmin) * sx_scale, (y1 - ymin) * sy_scale,
        (x2 - xmin) * sx_scale, (y2 - ymin) * sy_scale,
    ) for x1, y1, x2, y2 in nonshop_segs_merc]

    # --- shop rings: transform; optionally shrink
    shop_rings_local = []
    n_dropped = 0
    for s in shop_rings_merc:
        floor_ring = [transform_xy(x, y, xmin, ymin, sx_scale, sy_scale)
                      for x, y in s["ring"]]
        if args.shrink > 0:
            shrunk = shrink_polygon_ring(floor_ring, args.shrink)
            if shrunk is None:
                n_dropped += 1
                continue
            floor_ring = shrunk
        shop_rings_local.append({"ring": floor_ring, "props": s["props"]})
    if args.shrink > 0:
        print(f"[build] shrink: shop polygons -{args.shrink:.2f} m  "
              f"({len(shop_rings_local)} kept, {n_dropped} dropped as too small)")

    # --- build the full segment list
    shop_segs_local = []
    for s in shop_rings_local:
        shop_segs_local.extend(segments_from_ring(s["ring"]))
    segs_local = nonshop_segs_local + shop_segs_local

    # --- TIAGO++ spawn pose from path_00
    p00 = floor_dir / "path_00" / "waypoints_raw.csv"
    if not p00.is_file():
        sys.exit(f"missing {p00}")
    with open(p00) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 2:
        sys.exit("path_00 has < 2 raw waypoints")
    start = (float(rows[0]["gt_x"]), float(rows[0]["gt_y"]))
    second = (float(rows[1]["gt_x"]), float(rows[1]["gt_y"]))
    print(f"[build] TIAGO spawn ({start[0]:.2f}, {start[1]:.2f}) -> "
          f"facing ({second[0]:.2f}, {second[1]:.2f})")

    # --- trajectories (dense 10 Hz GT) for corridor detection
    trajectories = []
    for d in sorted(floor_dir.iterdir()):
        if not d.name.startswith("path_"):
            continue
        gt = d / "ground_truth.csv"
        if not gt.exists():
            continue
        with open(gt) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) >= 2:
            pts = [(float(r["gt_x"]), float(r["gt_y"])) for r in rows[::10]]
            if len(pts) >= 2:
                trajectories.append(pts)
    print(f"[build] trajectories: {len(trajectories)} traces "
          f"({sum(len(t) for t in trajectories):,} downsampled points)")

    # --- raw waypoints per path (for path viz markers + connecting boxes)
    raw_wps = []
    for d in sorted(floor_dir.iterdir()):
        if not d.name.startswith("path_"):
            continue
        wp = d / "waypoints_raw.csv"
        if not wp.exists():
            continue
        with open(wp) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) >= 2:
            raw_wps.append([(float(r["gt_x"]), float(r["gt_y"])) for r in rows])

    # --- count auto-resolution
    # Estimate eligible walls for decorations: long enough + corridor side
    pts_flat = flat_traj_points(trajectories)
    eligible_decor = 0
    eligible_long_corridor = 0   # for window auto
    for s in segs_local:
        length = math.hypot(s[2] - s[0], s[3] - s[1])
        if length < cfg.min_wall_len_decoration:
            continue
        if pts_flat and wall_corridor_side(s, pts_flat) == 0:
            continue
        eligible_decor += 1
        if length >= cfg.window_min_wall_len:
            eligible_long_corridor += 1

    # corners adjacent to corridors
    eligible_corners = 0
    if pts_flat:
        for cxc, cyc in find_corners(segs_local, tol=0.10):
            if is_near_any_point(cxc, cyc, pts_flat, 2.5):
                eligible_corners += 1

    counts = {
        "windows":             _resolve_count(args.windows, round(0.18 * eligible_long_corridor)),
        "painting_landscape":  _resolve_count(args.paintings_landscape, round(0.10 * eligible_decor)),
        "painting_portrait":   _resolve_count(args.paintings_portrait,  round(0.10 * eligible_decor)),
        "tv":                  _resolve_count(args.tvs,                  round(0.10 * eligible_decor)),
        "mirror":              _resolve_count(args.mirrors,              round(0.10 * eligible_decor)),
        "poster":              _resolve_count(args.posters,              round(0.10 * eligible_decor)),
        "sign":                _resolve_count(args.signs,                round(0.10 * eligible_decor)),
        "fire_extinguishers":  _resolve_count(args.fire_extinguishers,   round(0.30 * eligible_corners)),
        "corner_planters":     _resolve_count(args.corner_planters,      round(0.20 * eligible_corners)),
        "floor_objects":       _resolve_count(args.floor_objects,        round(0.30 * eligible_decor)),
        "ceiling_billboards":  _resolve_count(args.ceiling_billboards,   -1),
        "max_paths":           max(0, args.max_paths),
    }
    # ceiling billboards: auto handled by the placement function's density
    if counts["ceiling_billboards"] < 0:
        counts["ceiling_billboards"] = 10**6  # accept everything the density yields
    print(f"[build] eligible walls (decor) = {eligible_decor}  "
          f"long+corridor (windows) = {eligible_long_corridor}  "
          f"corridor-adjacent corners = {eligible_corners}")
    print(f"[build] requested counts: " +
          "  ".join(f"{k}={v}" for k, v in counts.items()))

    # --- render the floor texture
    floor_tex_basename = f"{name}_floor.png"
    floor_tex_path = out_dir / floor_tex_basename
    side_px = render_tile_texture(
        str(floor_tex_path), tile_m=cfg.floor_tile_size,
        px_per_m=args.texture_px_per_m, seed=cfg.seed)
    sz_mb = os.path.getsize(floor_tex_path) / 1024 / 1024
    print(f"[build] floor texture -> {floor_tex_path} "
          f"({side_px}x{side_px} px, {sz_mb:.2f} MB)")

    # --- build the .wbt text
    wbt = build_wbt(cfg, name, W, H, segs_local, start, second,
                      floor_tex_basename, trajectories, raw_wps, counts)

    # --- post-emit wall dedup safety net
    if not args.no_dedup:
        wbt, n_dedup = dedup_wall_blocks_in_text(
            wbt, args.dedup_tol_yaw, args.dedup_tol_perp,
            args.dedup_overlap_frac)
        print(f"[build] dedup: removed {n_dedup} duplicate wall block(s)")

    wbt_path = out_dir / f"{name}.wbt"
    wbt_path.write_text(wbt, encoding="utf-8")

    # --- final tally from the actually-written text. Use "_0" suffix on each
    # prefix so the count matches only indexed solids (WALL_00000 etc.) and
    # never the group-node headers (WALLS Group, CORNER_OBJECTS Group, ...).
    n_walls = wbt.count("DEF WALL_0") + wbt.count("DEF PWALL_0")
    n_win = wbt.count("DEF WIN_0")
    n_decor = wbt.count("DEF DECOR_0")
    n_corner = wbt.count("DEF CORNER_0")
    n_floor = wbt.count("DEF FLOOROBJ_0")
    n_bb = wbt.count("DEF BB_0")
    n_wp = wbt.count("DEF WP_0")
    n_ps = wbt.count("DEF PS_0")
    print(f"[build] world -> {wbt_path}  ({len(wbt):,} bytes)")
    print(f"[build]   walls={n_walls}  windows={n_win}  decorations={n_decor}  "
          f"corner_objects={n_corner}")
    has_ceiling = "DEF CEILING Group" in wbt
    print(f"[build]   floor_objects={n_floor}  ceiling_billboards={n_bb}  "
          f"path_waypoints={n_wp}  path_segments={n_ps}  "
          f"ceiling={'yes' if has_ceiling else 'no'}")


if __name__ == "__main__":
    main()
