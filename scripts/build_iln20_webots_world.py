"""Generate a Webots R2025a .wbt world from a converted ILN 2.0 floor.

Inputs (under data/iln20_<site8>_<floor>/):
  meta/floor_info.json   - floor extent in metres (width, height)
  meta/geojson_map.json  - wall polygons in Web Mercator (EPSG:3857) metres
  path_00/waypoints_raw.csv - used to place TIAGO++ + 2 scale-check spheres.

Outputs:
  src/simulation/worlds/<name>.wbt
  src/simulation/worlds/<name>_floor.png   (composite styled floor texture)

Modelling choices ("arena + visual richness" pass):
  * Floor: composite high-res texture generated with PIL — light concrete
    background, 5 m tile grid, each shop polygon filled with a soft tint and
    outlined, English shop name rendered at the polygon centroid, sized
    auto-fit to the polygon's bounding box.
  * Walls: every GeoJSON polygon edge -> visual-only Solid (Box 0.12 m thick,
    2.7 m tall, off-white PBR). No physics.
  * Decorations: ~25% of walls (long enough) get a wall-mounted PROTO from
    {Window, Mirror, LandscapePainting, PortraitPainting, Television,
     Blackboard, AdvertisingBoard, FireExtinguisher, Radiator}. Deterministic
    pick (seed). Object placed at the outward face of the wall, oriented to
    face outward, at a height appropriate to the object type.
  * Robot: TIAGO++ at path_00's first waypoint, facing the next waypoint.
    controller "<none>" — controller wired later.
  * Markers: green sphere @ path_00 wp0, red sphere @ path_00 wp1 (for the
    §4-step-4 visual scale check).

Mercator -> floor transform: bounding-box rescale (self-calibrates).

Coordinate frame: Webots ENU. Floor at z=0, spanning (0, 0) to (W, H).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WALL_THICKNESS = 0.12
WALL_HEIGHT = 3.20      # bumped from 2.7 — taller, more room for ceiling billboards
MIN_WALL_LENGTH = 0.10

DECORATE_PROB = 0.25
MIN_WALL_LEN_FOR_DECORATION = 1.5
DECOR_SEED = 1337
PX_PER_M_SMALL = 10
PX_PER_M_BIG = 6
TILE_M = 5.0

# A curated set of English mall brand names (used when GeoJSON names are
# Chinese or missing). Stable order; assigned to shops by polygon-area rank
# so the largest shops get the most "recognisable" names.
ENGLISH_SHOP_NAMES = [
    "ZARA", "H&M", "STARBUCKS", "APPLE", "NIKE", "ADIDAS", "UNIQLO",
    "SEPHORA", "MUJI", "GAP", "PULL&BEAR", "MANGO", "TOMMY HILFIGER",
    "CALVIN KLEIN", "GUESS", "PRADA", "GUCCI", "BOSE", "SAMSUNG", "LG",
    "XIAOMI", "HUAWEI", "SONY", "COSTA COFFEE", "KFC", "MCDONALD'S",
    "BURGER KING", "SUBWAY", "PIZZA HUT", "IKEA", "BEST BUY", "AMAZON",
    "GAMESTOP", "LEGO", "DECATHLON", "INTERSPORT", "TIMBERLAND",
    "CONVERSE", "NEW BALANCE", "PUMA", "REEBOK", "UNDER ARMOUR",
    "L'OREAL", "MAC", "BENEFIT", "ESTEE LAUDER", "NIVEA",
    "BATH & BODY WORKS", "PANDORA", "SWAROVSKI", "TIFFANY",
    "LOUIS VUITTON", "HERMES", "CHANEL", "DIOR", "BURBERRY", "ARMANI",
    "VERSACE", "FENDI", "BALENCIAGA", "OFF-WHITE", "SUPREME",
    "ROLEX", "OMEGA", "TAG HEUER", "CITIZEN", "CASIO",
    "FOOD COURT", "RESTROOMS", "ELEVATOR", "ESCALATOR", "INFORMATION",
    "CINEMA", "BOOKSTORE", "PHARMACY", "SUPERMARKET", "BANK",
    "CAFÉ", "BAKERY", "ICE CREAM", "JEWELLERY", "OPTICAL", "TOYS",
    "ELECTRONICS", "BOOKS", "MUSIC", "GAMES", "SPORTS", "GIFTS",
]

# Wall-mounted decorations - ALL built from primitive Boxes, no PROTOs.
# Each decoration uses one geometry convention:
#   Box size (W, T, H) where:
#     W = width along the wall (X axis)
#     T = thickness perpendicular to wall (Y axis, the thin dimension)
#     H = vertical height (Z axis)
#   The Box's outward face is +Y in local coords; with rotation
#     obj_yaw = wall_yaw           (for side = +1)
#     obj_yaw = wall_yaw + pi      (for side = -1)
#   the +Y face ends up pointing along the wall normal -> outward, into the
#   corridor. Same rotation rule for every decoration type, no per-PROTO
#   special-casing.
WALL_DECOR_EMITTERS = []  # populated by @_register_wall_decor below

# Corner objects (placed at wall-meeting points adjacent to corridors)
CORNER_OBJECTS = [
    ("FireExtinguisher",
     "objects/factory/fire_extinguisher/protos/FireExtinguisher.proto",
     0.50, ""),
    # planter is a custom Shape (see fmt_planter); also valid for corners
    ("_planter", None, None, None),
]

# Window proto for in-wall openings
WINDOW_PROTO_URL = "objects/apartment_structure/protos/Window.proto"

# Wall PBR palette — cycled per wall (deterministic hash) so the mall has
# visible material variety, not 2,646 identical off-white boxes.
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

# Shop sign color palette
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

DECORATE_PROB_NEW = 0.60   # bumped from 0.25 - the user wanted more density


# ---------------------------------------------------------------------------
# GeoJSON: walls (segments) + shops (polygon centroids + names)
# ---------------------------------------------------------------------------
def collect_segments(geo: dict) -> list[tuple[float, float, float, float]]:
    segs = []

    def add_ring(ring):
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            segs.append((x1, y1, x2, y2))

    def walk(geom):
        gt = geom.get("type")
        coords = geom.get("coordinates")
        if gt == "LineString":
            add_ring(coords)
        elif gt == "Polygon":
            for ring in coords:
                add_ring(ring)
        elif gt == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    add_ring(ring)
        elif gt == "GeometryCollection":
            for g in geom.get("geometries", []):
                walk(g)

    for f in geo.get("features", []):
        walk(f.get("geometry") or {})
    return segs


def collect_segments_split(geo: dict):
    """Walk the GeoJSON once and split features into:
      - shop_rings:    list of {"ring": <closed Mercator ring>, "props": ...}
                       — exterior rings of SHOP polygons (non-floor, polygonal).
                       These are the rings eligible for shrinking.
      - nonshop_segs:  list of (x1,y1,x2,y2) — every other edge: floor-outline
                       polygon, polygon holes, and LineString features. These
                       are emitted as walls verbatim (no shrink).

    Together they form exactly the same edge set as collect_segments(), just
    partitioned by "can this be shrunk?". Shop holes are kept as non-shrink
    walls so structural columns inside a shop don't get warped.
    """
    shop_rings = []
    nonshop_segs = []

    def emit_ring_edges(ring):
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            nonshop_segs.append((x1, y1, x2, y2))

    def walk(geom, is_floor):
        gt = (geom or {}).get("type")
        coords = (geom or {}).get("coordinates")
        if gt == "LineString":
            emit_ring_edges(coords)
            return
        polys = []
        if gt == "Polygon":
            polys = [coords]
        elif gt == "MultiPolygon":
            polys = coords or []
        elif gt == "GeometryCollection":
            for g in geom.get("geometries", []):
                walk(g, is_floor)
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
                emit_ring_edges(ext)
            else:
                shop_rings.append({"ring": ext, "props": props})
            # Holes (if any) are always non-shrink — treat as structural walls
            for hole in poly_rings[1:]:
                if len(hole) >= 3:
                    emit_ring_edges(hole)

    for f in geo.get("features", []):
        props = f.get("properties") or {}
        cat = props.get("category")
        is_floor = isinstance(cat, str) and cat.lower() == "floor"
        walk(f.get("geometry"), is_floor)

    return shop_rings, nonshop_segs


def collect_shops(geo: dict):
    """Return [{ring, props}, ...] for each Polygon / MultiPolygon feature
    that is a SHOP (not the floor outline and not a tiny structural blob).
    Coordinates left in Mercator.
    """
    shops = []
    for f in geo.get("features", []):
        props = f.get("properties") or {}
        # The floor-outline polygon has category == "floor" (plain string).
        # Shop polygons have category as a list of tag ObjectIds. Skip the
        # floor outline — otherwise it covers the whole area and hides
        # the corridors.
        cat = props.get("category")
        if isinstance(cat, str) and cat.lower() == "floor":
            continue
        g = f.get("geometry") or {}
        gt = g.get("type")
        coords = g.get("coordinates")
        rings_groups = []
        if gt == "Polygon":
            rings_groups = [coords]
        elif gt == "MultiPolygon":
            rings_groups = coords
        else:
            continue
        for rg in rings_groups:
            ext = rg[0]
            if len(ext) < 3:
                continue
            shops.append({"ring": ext, "props": props})
    return shops


def shoelace_area(ring) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def polygon_centroid(ring) -> tuple[float, float]:
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


def signed_area(ring) -> float:
    """Positive for CCW, negative for CW. Input may or may not be closed."""
    a = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        a += x1 * y2 - x2 * y1
    return a * 0.5


def shrink_polygon_ring(ring, distance: float, min_area: float = 0.5):
    """Offset every edge of a closed polygon inward by `distance` (metres) and
    return the resulting closed ring. Manual edge-offset algorithm: works for
    convex and gently concave polygons (typical mall shops are rectangular or
    L-shaped, well within reach).

    Returns None if the result is degenerate: winding flipped, area below
    `min_area` m2, or any edge collapsed.
    """
    # Strip closing point if present, take a clean vertex list
    verts = list(ring[:-1]) if (ring and ring[0] == ring[-1]) else list(ring)
    n = len(verts)
    if n < 3 or distance <= 0:
        return None

    sa = signed_area(verts + [verts[0]])
    if abs(sa) < 1e-9:
        return None
    # CCW (sa>0): inward normal = rotate edge dir +90deg = (-dy, dx)
    # CW  (sa<0): inward normal = rotate edge dir -90deg = ( dy,-dx)
    inward_sign = 1.0 if sa > 0 else -1.0

    # For each edge i (verts[i] -> verts[i+1]) compute its offset line
    # by translating both endpoints inward along the unit normal.
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

    # Intersect consecutive offset edges as infinite lines to get new vertices.
    new_verts = []
    for i in range(n):
        a1, b1 = offset_edges[i]
        a2_, b2 = offset_edges[(i + 1) % n]
        x1, y1 = a1; x2, y2 = b1
        x3, y3 = a2_; x4, y4 = b2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            # Parallel / collinear -> use the shared endpoint of edge i
            new_verts.append(b1)
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            new_verts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))

    sa_new = signed_area(new_verts + [new_verts[0]])
    if sa_new * sa <= 0:
        return None  # winding flipped -> shrunk past the medial axis
    if abs(sa_new) < min_area:
        return None  # too small to render usefully

    new_verts.append(new_verts[0])
    return new_verts


def bbox(segs):
    xs = [x for s in segs for x in (s[0], s[2])]
    ys = [y for s in segs for y in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


def transform_xy(x, y, mx0, my0, sx, sy):
    return (x - mx0) * sx, (y - my0) * sy


def dedupe_segments(segs, endpoint_tol: float = 0.50,
                      midpoint_tol: float = 0.60,
                      yaw_tol: float = 0.10):
    """Drop duplicate wall segments using two heuristics so adjacent shops
    that share an edge don't produce two visible overlapping walls.

    1. Endpoint match: two walls with the same endpoint pair (within
       `endpoint_tol` m, direction-agnostic) collapse to one.
    2. Midpoint + direction match: two walls whose midpoints match within
       `midpoint_tol` m AND whose yaws match within `yaw_tol` rad (mod π)
       collapse to one. Catches near-collinear overlapping walls of slightly
       different length whose endpoints don't quite line up.
    """
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
        yaw = math.atan2(y2 - y1, x2 - x1) % math.pi   # direction-agnostic
        key_mid = (round(mx / midpoint_tol), round(my / midpoint_tol),
                   round(yaw / yaw_tol))
        if key_mid in seen_mid:
            continue
        seen_ep.add(key_ep)
        seen_mid.add(key_mid)
        out.append((x1, y1, x2, y2))
    return out


def _wall_mount_yaw(wall_yaw: float, side: int) -> float:
    """Yaw for a Webots wall-mounted PROTO so its front face points outward
    along the wall normal.

    Webots PROTOs (LandscapePainting, PortraitPainting, Blackboard, Radiator,
    Television etc.) all use local +X as the 'face out from wall' direction.
    With that convention:
        side = +1 -> obj_yaw = wall_yaw + pi/2
        side = -1 -> obj_yaw = wall_yaw - pi/2
    """
    return wall_yaw + side * math.pi / 2


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------------------
# Composite floor texture
# ---------------------------------------------------------------------------
def find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_tile_texture(out_png: str, tile_m: float = 5.0,
                          px_per_m: int = 200, seed: int = DECOR_SEED):
    """Render a small SQUARE tile texture (default 5x5 m) that will be repeated
    across the whole floor via the Floor proto's tileSize parameter. Because
    the texture is generic - no shop polygons baked in - there is nothing
    that can be misaligned with the walls.

    Style: polished concrete with subtle veining + 1 m sub-grid + 5 m main grid
    at the tile edges. Looks like a real mall floor surface.
    """
    side_px = int(tile_m * px_per_m)
    img = PILImage.new("RGB", (side_px, side_px), (218, 215, 210))
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Subtle vein-like noise (random small blobs)
    for _ in range(80):
        cx = rng.randint(0, side_px - 1)
        cy = rng.randint(0, side_px - 1)
        r = rng.randint(6, 30)
        shade = rng.randint(208, 226)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=(shade, shade - 2, shade - 6))

    # 1 m sub-grid (light, in-tile)
    sub_px = int(1.0 * px_per_m)
    sub_color = (198, 195, 190)
    for x in range(sub_px, side_px, sub_px):
        draw.line([(x, 0), (x, side_px)], fill=sub_color, width=2)
    for y in range(sub_px, side_px, sub_px):
        draw.line([(0, y), (side_px, y)], fill=sub_color, width=2)

    # Tile edge lines (darker) at all four edges so the repeat is visible
    edge_color = (170, 165, 158)
    draw.line([(0, 0), (side_px - 1, 0)],            fill=edge_color, width=4)
    draw.line([(0, side_px - 1), (side_px - 1, side_px - 1)], fill=edge_color, width=4)
    draw.line([(0, 0), (0, side_px - 1)],            fill=edge_color, width=4)
    draw.line([(side_px - 1, 0), (side_px - 1, side_px - 1)], fill=edge_color, width=4)

    img.save(out_png, "PNG", optimize=True)
    return side_px, side_px, tile_m


def render_floor_texture(shops, W, H, mx0, my0, sx, sy,
                          px_per_m: int, out_png: str, seed: int = DECOR_SEED):
    """Build the composite floor texture and write it as RGB PNG."""
    img_w = int(W * px_per_m)
    img_h = int(H * px_per_m)

    # Background = light polished concrete with a slight warm tint
    img = PILImage.new("RGB", (img_w, img_h), (214, 210, 204))
    draw = ImageDraw.Draw(img)

    # Two-level grid: subtle 1 m sub-grid + stronger 5 m main grid
    sub_px = int(1.0 * px_per_m)
    sub_color = (200, 196, 190)
    if sub_px >= 4:   # only draw sub-grid when there's room for it
        for x in range(0, img_w + 1, sub_px):
            draw.line([(x, 0), (x, img_h)], fill=sub_color, width=1)
        for y in range(0, img_h + 1, sub_px):
            draw.line([(0, y), (img_w, y)], fill=sub_color, width=1)
    main_px = int(TILE_M * px_per_m)
    main_color = (170, 165, 158)
    for x in range(0, img_w + 1, main_px):
        draw.line([(x, 0), (x, img_h)], fill=main_color, width=2)
    for y in range(0, img_h + 1, main_px):
        draw.line([(0, y), (img_w, y)], fill=main_color, width=2)

    # Transform a single (mx, my) -> PIL pixel (img origin top-left, y down).
    def to_px(mx, my):
        fx, fy = transform_xy(mx, my, mx0, my0, sx, sy)
        return int(round(fx * px_per_m)), int(round(img_h - fy * px_per_m))

    # Compute area in floor metres for each shop; sort large -> small so big
    # shops get the most-recognisable brand names (in our curated list order).
    MIN_SHOP_AREA_M2 = 3.0   # filter structural blobs (storefront columns etc.)
    enriched = []
    for s in shops:
        ring_pix = [to_px(x, y) for x, y in s["ring"]]
        ring_floor = [(transform_xy(x, y, mx0, my0, sx, sy)) for x, y in s["ring"]]
        area_m2 = shoelace_area(ring_floor)
        if area_m2 < MIN_SHOP_AREA_M2:
            continue
        cx_m, cy_m = polygon_centroid(ring_floor)
        # bounding box of the polygon in METRES (for font auto-fit)
        xs = [p[0] for p in ring_floor]; ys = [p[1] for p in ring_floor]
        bbw_m = max(xs) - min(xs); bbh_m = max(ys) - min(ys)
        # display name: GeoJSON name if ASCII (decode HTML entities first),
        # else from curated list
        gname = html.unescape((s["props"].get("name") or "").strip())
        name = gname if (gname and is_ascii(gname)) else None
        enriched.append({"ring_pix": ring_pix, "area_m2": area_m2,
                         "cx_m": cx_m, "cy_m": cy_m,
                         "bbw_m": bbw_m, "bbh_m": bbh_m,
                         "name_geo": name, "_props": s["props"]})
    enriched.sort(key=lambda s: -s["area_m2"])

    rng = random.Random(seed)
    # Avoid duplicate names: track GeoJSON names already in use, and skip
    # those when iterating the curated fallback list.
    used = {s["name_geo"].upper() for s in enriched if s["name_geo"]}
    fallback_pool = [n for n in ENGLISH_SHOP_NAMES if n.upper() not in used]
    # cycle deterministically so a re-run picks the same names per polygon
    fallback_iter = iter(fallback_pool * (1 + len(enriched) // max(1, len(fallback_pool))))
    for s in enriched:
        s["name"] = s["name_geo"] or next(fallback_iter)

    # Shop tints - more saturation and variety so polygons read as distinct
    # surfaces (not all near-identical beige).
    SHOP_PALETTE = [
        (232, 218, 200),  # cream
        (218, 226, 218),  # mint
        (220, 222, 234),  # lilac
        (234, 222, 214),  # peach
        (218, 230, 232),  # pale teal
        (228, 224, 200),  # soft yellow
        (224, 214, 226),  # rose
        (210, 220, 224),  # blue-grey
        (234, 230, 220),  # warm white
        (216, 226, 210),  # sage
    ]
    for s in enriched:
        # Deterministic per-shop palette pick (same colour on re-runs)
        h = abs(hash(s["name"]))
        base = SHOP_PALETTE[h % len(SHOP_PALETTE)]
        # Per-shop micro-jitter so adjacent same-palette shops aren't identical
        jitter = (rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(-6, 6))
        fill = tuple(max(0, min(255, c + j)) for c, j in zip(base, jitter))
        # Narrow polygons get no outline (line would dominate); others get
        # a 2-px darker outline (stronger separation than before).
        narrow = (s["bbw_m"] < 1.5) or (s["bbh_m"] < 1.5)
        if narrow:
            draw.polygon(s["ring_pix"], fill=fill)
        else:
            draw.polygon(s["ring_pix"], fill=fill, outline=(60, 56, 52))
            # second pass for line thickness (only on non-narrow polygons,
            # where the fill has room to breathe under the heavier outline)
            for (a, b) in zip(s["ring_pix"], s["ring_pix"][1:] + [s["ring_pix"][0]]):
                draw.line([a, b], fill=(60, 56, 52), width=2)

        # Label: auto-fit font size to polygon bbox (with margin)
        margin = 0.78
        name = s["name"]
        # Polygon bbox in PIXELS, clamped to the image area so font auto-fit
        # never picks a size that would overflow the canvas (avoids the
        # "APPLE -> PPLE" edge-clip on shops at the floor border).
        cx_px_raw = int(round(s["cx_m"] * px_per_m))
        cy_px_raw = int(round(img_h - s["cy_m"] * px_per_m))
        bbw_avail = min(s["bbw_m"] * px_per_m,
                        2 * min(cx_px_raw, img_w - cx_px_raw))
        bbh_avail = min(s["bbh_m"] * px_per_m,
                        2 * min(cy_px_raw, img_h - cy_px_raw))
        bbw_px = max(8, int(bbw_avail * margin))
        bbh_px = max(8, int(bbh_avail * margin))
        # binary-search font size that fits
        lo, hi = 6, int(min(bbh_px, bbw_px * 1.6 / max(1, len(name))) * 2)
        hi = max(hi, lo + 1)
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            font = find_font(mid)
            try:
                bbox_ = draw.textbbox((0, 0), name, font=font)
                tw, th = bbox_[2] - bbox_[0], bbox_[3] - bbox_[1]
            except Exception:
                tw, th = mid * len(name) * 0.55, mid * 1.1
            if tw <= bbw_px and th <= bbh_px:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        font = find_font(best)
        try:
            bbox_ = draw.textbbox((0, 0), name, font=font)
            tw, th = bbox_[2] - bbox_[0], bbox_[3] - bbox_[1]
        except Exception:
            tw, th = best * len(name) * 0.55, best * 1.1
        cx_px = int(round(s["cx_m"] * px_per_m))
        cy_px = int(round(img_h - s["cy_m"] * px_per_m))
        tx = cx_px - tw // 2
        ty = cy_px - th // 2
        # Clamp text inside the image bounds so labels at the floor edge
        # don't get cropped off ("APPLE" -> "PPLE" bug).
        tx = max(2, min(tx, img_w - tw - 2))
        ty = max(2, min(ty, img_h - th - 2))
        # subtle shadow + main text
        draw.text((tx + 1, ty + 1), name, font=font, fill=(170, 165, 155))
        draw.text((tx, ty), name, font=font, fill=(35, 30, 25))

    # Note on Webots Floor proto texture orientation: the previous version
    # applied a FLIP_TOP_BOTTOM here on the hypothesis that Webots maps PIL
    # row 0 to floor -Y. User reported the flipped version was STILL
    # misaligned, so we revert: PIL row 0 (top of image) renders at floor
    # +Y (north). If alignment is still off we will try a 90° rotation next.
    img.save(out_png, "PNG", optimize=True)
    return img_w, img_h, len(enriched)


# ---------------------------------------------------------------------------
# Webots emitters: walls + decorations + markers + assembled .wbt
# ---------------------------------------------------------------------------
def fmt_wall(idx: int, x1: float, y1: float, x2: float, y2: float,
              palette_idx: int = 0) -> str:
    """Per-wall PBR cycled through WALL_PALETTES for visual variety."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < MIN_WALL_LENGTH:
        return ""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(dy, dx)
    pname, (r, g, b), rough, met = WALL_PALETTES[palette_idx % len(WALL_PALETTES)]
    return (
        f"  DEF WALL_{idx:05d} Solid {{\n"
        f"    translation {mx:.4f} {my:.4f} {WALL_HEIGHT/2:.4f}\n"
        f"    rotation 0 0 1 {yaw:.5f}\n"
        f"    children [\n"
        f"      Shape {{\n"
        f"        appearance PBRAppearance {{\n"
        f"          baseColor {r} {g} {b}\n"
        f"          roughness {rough}\n"
        f"          metalness {met}\n"
        f"        }}\n"
        f"        geometry Box {{ size {length:.4f} {WALL_THICKNESS} {WALL_HEIGHT} }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


# ----- floor objects (custom Shape Solids — no PROTO dependencies) -----
def fmt_planter(idx: int, x: float, y: float) -> str:
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
        f"    Pose {{\n"
        f"      translation 0 0 0.85\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 0.18 0.55 0.22 "
        f"roughness 0.95 metalness 0 }}\n"
        f"          geometry Sphere {{ radius 0.55 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_bench(idx: int, x: float, y: float, yaw: float) -> str:
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


def fmt_trashcan(idx: int, x: float, y: float) -> str:
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


def fmt_pillar(idx: int, x: float, y: float) -> str:
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


FLOOR_OBJ_FNS = [fmt_planter, fmt_bench, fmt_trashcan, fmt_pillar]


def fmt_partial_wall(idx: int, mx: float, my: float, length: float, yaw: float,
                       z_center: float, height: float,
                       palette_idx: int = 0) -> str:
    """Wall piece with custom z-centre and height (for sill / lintel)."""
    pname, (r, g, b), rough, met = WALL_PALETTES[palette_idx % len(WALL_PALETTES)]
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
        f"        geometry Box {{ size {length:.4f} {WALL_THICKNESS} {height:.4f} }}\n"
        f"      }}\n"
        f"    ]\n"
        f"  }}\n"
    )


def fmt_window(idx: int, cx: float, cy: float, yaw: float) -> str:
    """Custom window: thin transparent glass pane + 4 dark frame strips around
    the edges. Centred at (cx, cy, (sill+lintel)/2). Sits exactly in the gap
    left by split_walls_for_windows() between the wall sub-segments and
    between the sill and lintel.
    """
    z_center = (WIN_SILL_Z + WIN_LINTEL_Z) / 2          # ~1.55
    glass_h = WIN_LINTEL_Z - WIN_SILL_Z                  # 1.40
    glass_w = WIN_WIDTH                                  # 1.40
    fw = 0.08   # frame strip width (vertical strips) / height (horizontal strips)
    # Frame strips run the full window width / height; thickness matches wall.
    return (
        f"DEF WIN_{idx:05d} Solid {{\n"
        f"  translation {cx:.4f} {cy:.4f} {z_center:.4f}\n"
        f"  rotation 0 0 1 {yaw:.5f}\n"
        f"  name \"window_{idx:05d}\"\n"
        f"  children [\n"
        # Glass pane: thin in y (perpendicular to wall), wide x, tall z.
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor 0.55 0.72 0.85\n"
        f"        emissiveColor 0.10 0.16 0.22\n"
        f"        roughness 0.05 metalness 0.20\n"
        f"        transparency 0.55\n"
        f"      }}\n"
        f"      geometry Box {{ size {glass_w:.3f} {WIN_GLASS_THICK} {glass_h:.3f} }}\n"
        f"    }}\n"
        # Frame: top strip
        f"    Pose {{ translation 0 0 { glass_h/2 - fw/2:.3f}\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {glass_w:.3f} {WALL_THICKNESS} {fw:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        # Frame: bottom strip
        f"    Pose {{ translation 0 0 {-glass_h/2 + fw/2:.3f}\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {glass_w:.3f} {WALL_THICKNESS} {fw:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        # Frame: left strip
        f"    Pose {{ translation {-glass_w/2 + fw/2:.3f} 0 0\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {fw:.3f} {WALL_THICKNESS} {glass_h:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        # Frame: right strip
        f"    Pose {{ translation { glass_w/2 - fw/2:.3f} 0 0\n"
        f"      children [ Shape {{\n"
        f"        appearance PBRAppearance {{ baseColor 0.22 0.22 0.22 "
        f"roughness 0.45 metalness 0.30 }}\n"
        f"        geometry Box {{ size {fw:.3f} {WALL_THICKNESS} {glass_h:.3f} }}\n"
        f"      }} ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_corner_planter(idx: int, x: float, y: float) -> str:
    """Smaller planter variant for corners."""
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
        f"    Pose {{\n"
        f"      translation 0 0 0.7\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 0.22 0.55 0.22 "
        f"roughness 0.95 metalness 0 }}\n"
        f"          geometry Sphere {{ radius 0.45 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_corner_extinguisher(idx: int, x: float, y: float) -> str:
    """Fire extinguisher proto placed at corner."""
    return (
        f"DEF CORNER_{idx:05d} FireExtinguisher {{\n"
        f"  translation {x:.4f} {y:.4f} 0\n"
        f"  name \"corner_ext_{idx:05d}\"\n"
        f"}}\n"
    )


def fmt_ceiling_billboard(idx: int, cx: float, cy: float, yaw: float,
                            text: str, color_idx: int) -> str:
    """Custom hanging billboard - colored emissive panel suspended at z=2.6m
    with a thin support stem above. Visible from far in the corridor; never
    intersects walls because we place it over trajectory midlines."""
    r, g, b = SIGN_COLORS[color_idx % len(SIGN_COLORS)]
    panel_w = min(3.2, max(1.5, len(text) * 0.34))
    return (
        f"DEF BB_{idx:05d} Solid {{\n"
        f"  translation {cx:.4f} {cy:.4f} 2.55\n"
        f"  rotation 0 0 1 {yaw:.5f}\n"
        f"  name \"billboard_{idx:05d}\"\n"
        f"  children [\n"
        # main panel
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor {r} {g} {b}\n"
        f"        emissiveColor {r*0.65:.3f} {g*0.65:.3f} {b*0.65:.3f}\n"
        f"        roughness 0.30 metalness 0.10\n"
        f"      }}\n"
        f"      geometry Box {{ size {panel_w:.2f} 0.06 0.50 }}\n"
        f"    }}\n"
        # thin support stem going UP (toward ceiling)
        f"    Pose {{\n"
        f"      translation 0 0 0.30\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 0.30 0.30 0.30 "
        f"roughness 0.6 metalness 0.4 }}\n"
        f"          geometry Box {{ size 0.05 0.05 0.20 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


def place_ceiling_billboards(trajectories, segs_local, rng,
                                spacing: float = 30.0,
                                dedupe_radius: float = 12.0,
                                clearance_from_walls: float = 1.2):
    """Sample one billboard every `spacing` metres along trajectories; dedupe
    so they don't cluster. Each billboard is above a trajectory midpoint
    (always over walkable space, never inside a wall)."""
    placements = []  # (x, y, yaw)
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
                # don't place a billboard immediately over a wall
                if min_dist_to_walls(px, py, segs_local) >= clearance_from_walls:
                    placements.append((px, py, yaw))
                next_target += rng.uniform(spacing * 0.6, spacing * 1.4)
            cum += seg_len
            last = pt
    # dedupe spatially
    kept = []
    for p in placements:
        ok = True
        for q in kept:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < dedupe_radius:
                ok = False; break
        if ok:
            kept.append(p)
    return kept


def fmt_shop_sign(idx: int, cx: float, cy: float, name: str, color_idx: int) -> str:
    """Brightly-coloured glowing panel hovering at the shop centroid (z=2.6 m).
    The colour + emissive give the sign a 'lit storefront' look, providing
    strong visual landmarks for the synthetic camera."""
    r, g, b = SIGN_COLORS[color_idx % len(SIGN_COLORS)]
    sign_w = min(5.0, max(1.4, len(name) * 0.42))
    return (
        f"DEF SHOPSIGN_{idx:05d} Solid {{\n"
        f"  translation {cx:.4f} {cy:.4f} 2.60\n"
        f"  name \"shopsign_{idx:05d}\"\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{\n"
        f"        baseColor {r} {g} {b}\n"
        f"        emissiveColor {r*0.55:.3f} {g*0.55:.3f} {b*0.55:.3f}\n"
        f"        roughness 0.30 metalness 0\n"
        f"      }}\n"
        f"      geometry Box {{ size {sign_w:.2f} 0.06 0.55 }}\n"
        f"    }}\n"
        f"  ]\n"
        f"}}\n"
    )


# ----- floor-object placement: AGAINST walls on the corridor-facing side -----
def segment_distance(px: float, py: float, x1: float, y1: float,
                       x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def min_dist_to_walls(px: float, py: float, walls) -> float:
    return min(segment_distance(px, py, *w) for w in walls) if walls else 1e9


def _flat_traj_points(trajectories) -> list[tuple[float, float]]:
    return [pt for poly in trajectories for pt in poly]


def _is_near_any_point(px: float, py: float, points, max_d: float) -> bool:
    md2 = max_d * max_d
    for tx, ty in points:
        dx, dy = px - tx, py - ty
        if dx * dx + dy * dy <= md2:
            return True
    return False


def wall_corridor_side(s, points, probe: float = 2.5) -> int:
    """Return +1, -1, or 0 (no clear corridor side).
    Probes perpendicularly +/- probe metres and checks for nearby trajectory."""
    x1, y1, x2, y2 = s
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    yaw = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(yaw), math.cos(yaw)
    plus = (mx + nx * probe, my + ny * probe)
    minus = (mx - nx * probe, my - ny * probe)
    p_near = _is_near_any_point(plus[0], plus[1], points, probe)
    m_near = _is_near_any_point(minus[0], minus[1], points, probe)
    if p_near and not m_near:
        return +1
    if m_near and not p_near:
        return -1
    return 0


WIN_WIDTH = 1.4         # along the wall
WIN_HEIGHT = 1.4        # vertical
WIN_SILL_Z = 0.85       # bottom of glass = top of sill
WIN_LINTEL_Z = 2.25     # top of glass = bottom of lintel
WIN_GLASS_THICK = 0.04  # thin glass perpendicular to wall


def split_walls_for_windows(segs_local, trajectories, rng,
                              window_prob: float = 0.18,
                              min_len: float = 4.5):
    """For each chosen corridor-side wall, produce:
      * 2 full-height side wall segments (one before, one after the window)
      * 1 sill block (below the window, full wall thickness, z=0..WIN_SILL_Z)
      * 1 lintel block (above the window, z=WIN_LINTEL_Z..WALL_HEIGHT)
      * 1 window placement (glass + frame, drawn separately)
    The sill and lintel are emitted as partial-height walls (custom z, h).
    Returns (full_segs, partial_walls, windows).
       full_segs    : [(x1,y1,x2,y2), ...] - full-height wall pieces
       partial_walls: [(mx,my,length,yaw,z_center,height), ...] - sill/lintel
       windows      : [(cx,cy,yaw), ...]
    """
    points = _flat_traj_points(trajectories) if trajectories else []
    full_segs = []
    partials = []
    windows = []
    half_w = WIN_WIDTH / 2
    sill_h = WIN_SILL_Z                                  # 0..WIN_SILL_Z
    sill_z_center = WIN_SILL_Z / 2
    lintel_h = WALL_HEIGHT - WIN_LINTEL_Z                # WIN_LINTEL_Z..2.7
    lintel_z_center = (WIN_LINTEL_Z + WALL_HEIGHT) / 2

    for s in segs_local:
        x1, y1, x2, y2 = s
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            full_segs.append(s); continue
        if not points or wall_corridor_side(s, points) == 0:
            full_segs.append(s); continue
        if rng.random() > window_prob:
            full_segs.append(s); continue
        # split at midpoint, leave WIN_WIDTH gap for the window
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        yaw = math.atan2(y2 - y1, x2 - x1)
        ax_end_x, ay_end_y = x1 + ux * (length / 2 - half_w), y1 + uy * (length / 2 - half_w)
        bx_start_x, by_start_y = x1 + ux * (length / 2 + half_w), y1 + uy * (length / 2 + half_w)
        win_cx, win_cy = x1 + ux * (length / 2), y1 + uy * (length / 2)
        # side walls (full height)
        full_segs.append((x1, y1, ax_end_x, ay_end_y))
        full_segs.append((bx_start_x, by_start_y, x2, y2))
        # sill + lintel at the window region (partial height walls)
        partials.append((win_cx, win_cy, WIN_WIDTH, yaw,
                          sill_z_center, sill_h))
        partials.append((win_cx, win_cy, WIN_WIDTH, yaw,
                          lintel_z_center, lintel_h))
        windows.append((win_cx, win_cy, yaw))
    return full_segs, partials, windows


def find_corners(segs_local, tol: float = 0.10):
    """Find vertices shared by 2+ walls that meet at a non-collinear angle.
    Returns [(cx, cy), ...] in floor coords."""
    bucket = {}
    def key(x, y):
        return (round(x / tol), round(y / tol))
    for i, (x1, y1, x2, y2) in enumerate(segs_local):
        for px, py in ((x1, y1), (x2, y2)):
            k = key(px, py)
            bucket.setdefault(k, []).append((i, px, py, x1, y1, x2, y2))
    corners = []
    for k, items in bucket.items():
        if len(items) < 2:
            continue
        cx = sum(it[1] for it in items) / len(items)
        cy = sum(it[2] for it in items) / len(items)
        # compute outgoing yaws from this corner
        yaws = []
        for i, px, py, x1, y1, x2, y2 in items:
            # vector AWAY from corner toward the wall's other endpoint
            if math.hypot(px - x1, py - y1) < 0.2:
                yaws.append(math.atan2(y2 - y1, x2 - x1))
            else:
                yaws.append(math.atan2(y1 - y2, x1 - x2))
        # any pair non-collinear?
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


def place_floor_objects_along_walls(segs_local, trajectories, rng,
                                      obj_offset: float = 0.50,
                                      corridor_probe: float = 2.5,
                                      sample_prob: float = 0.10,
                                      min_wall_len: float = 2.5,
                                      object_clearance: float = 0.55):
    """For each wall, decide which side (if any) is the corridor by probing
    perpendicular `corridor_probe` metres each way and checking if a real
    trajectory point sits there. Place a floor object on the corridor side,
    against the wall, only for a small random subset of walls.

    Returns [(x, y, yaw, side), ...].
    """
    points = _flat_traj_points(trajectories)
    placements = []
    for s in segs_local:
        x1, y1, x2, y2 = s
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_wall_len:
            continue
        if rng.random() > sample_prob:
            continue
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        yaw = math.atan2(y2 - y1, x2 - x1)
        nx, ny = -math.sin(yaw), math.cos(yaw)
        plus  = (mx + nx * corridor_probe, my + ny * corridor_probe)
        minus = (mx - nx * corridor_probe, my - ny * corridor_probe)
        plus_corridor  = _is_near_any_point(plus[0],  plus[1],  points, corridor_probe)
        minus_corridor = _is_near_any_point(minus[0], minus[1], points, corridor_probe)
        if plus_corridor == minus_corridor:
            continue  # ambiguous (both or neither) — skip
        side = +1 if plus_corridor else -1
        px = mx + nx * obj_offset * side
        py = my + ny * obj_offset * side
        # don't drop an object on top of a trajectory point itself
        if _is_near_any_point(px, py, points, object_clearance):
            continue
        placements.append((px, py, yaw, side))
    return placements


def _wall_mount_yaw(wall_yaw: float, side: int) -> float:
    """Rotation about z for a wall-mounted object so that its default-forward
    axis points OUTWARD from the wall (away from the wall along the corridor
    normal). Webots wall PROTOs (Television, Painting, etc.) define their
    front face along local +Y, so:
      side = +1 -> object_yaw = wall_yaw      (front faces normal +1 direction)
      side = -1 -> object_yaw = wall_yaw + pi (front faces normal -1 direction)
    Previously this was `wall_yaw + side * pi/2`, which left the screen
    perpendicular to the wall plane (turned 90 degrees) - the user reported
    TVs sticking out orthogonally from walls.
    """
    return wall_yaw if side == +1 else wall_yaw + math.pi


def _wall_decor_pose(wall_mx: float, wall_my: float, wall_yaw: float,
                       side: int, decor_thickness: float):
    """Compute (translation_x, translation_y, rotation_yaw) for a wall
    decoration so its back face touches the wall surface with a tiny gap.

    Convention: every Box decoration has X = along wall (wide), Y = thin
    (perpendicular to wall), Z = vertical. After rotation by wall_yaw
    (side+1) / wall_yaw+pi (side-1), the Box's local +Y points outward
    along the wall normal -> the decoration's flat face is visible from
    the corridor. Same formula for every decoration type.
    """
    nx = -math.sin(wall_yaw) * side
    ny =  math.cos(wall_yaw) * side
    gap = 0.005
    clearance = WALL_THICKNESS / 2 + decor_thickness / 2 + gap
    px = wall_mx + nx * clearance
    py = wall_my + ny * clearance
    obj_yaw = wall_yaw if side == +1 else wall_yaw + math.pi
    return px, py, obj_yaw


def _register_wall_decor(fn):
    WALL_DECOR_EMITTERS.append(fn)
    return fn


@_register_wall_decor
def fmt_wall_painting_landscape(idx, mx, my, yaw, side, rng):
    """Landscape painting: 0.85 m wide x 0.55 m tall, random pastel color."""
    T = 0.05
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
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
        # Thin darker frame border (offset slightly outward, slightly larger)
        f"    Pose {{ translation 0 0.001 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.13 0.08 "
        f"roughness 0.6 metalness 0 }}\n"
        f"      geometry Box {{ size 0.90 0.045 0.60 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


@_register_wall_decor
def fmt_wall_painting_portrait(idx, mx, my, yaw, side, rng):
    """Portrait painting: 0.50 m wide x 0.85 m tall."""
    T = 0.05
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
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


@_register_wall_decor
def fmt_wall_tv(idx, mx, my, yaw, side, rng):
    """Flat-screen TV: dark bezel + bright emissive screen."""
    T = 0.06
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
    z = 1.65
    # Random screen tint (mostly blue/teal/dark with slight variety)
    sr = rng.uniform(0.05, 0.25)
    sg = rng.uniform(0.10, 0.40)
    sb = rng.uniform(0.30, 0.75)
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_tv\"\n"
        f"  children [\n"
        # dark bezel (outer)
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.10 0.10 0.10 "
        f"roughness 0.45 metalness 0.20 }}\n"
        f"      geometry Box {{ size 1.20 {T:.3f} 0.70 }}\n"
        f"    }}\n"
        # bright screen (slightly outward, smaller, emissive)
        f"    Pose {{ translation 0 0.033 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {sr:.3f} {sg:.3f} {sb:.3f} "
        f"emissiveColor {sr*0.7:.3f} {sg*0.7:.3f} {sb*0.7:.3f} "
        f"roughness 0.2 metalness 0 }}\n"
        f"      geometry Box {{ size 1.05 0.01 0.58 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


@_register_wall_decor
def fmt_wall_mirror(idx, mx, my, yaw, side, rng):
    """Wall mirror: highly reflective PBR rectangle."""
    T = 0.04
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
    z = 1.50
    return (
        f"DEF DECOR_{idx:05d} Solid {{\n"
        f"  translation {px:.4f} {py:.4f} {z:.4f}\n"
        f"  rotation 0 0 1 {oy:.5f}\n"
        f"  name \"decor_{idx:05d}_mirror\"\n"
        f"  children [\n"
        # dark frame
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.15 0.15 0.18 "
        f"roughness 0.4 metalness 0.5 }}\n"
        f"      geometry Box {{ size 0.62 {T:.3f} 1.25 }}\n"
        f"    }}\n"
        # mirror surface (highly reflective)
        f"    Pose {{ translation 0 0.022 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.92 0.94 0.96 "
        f"roughness 0.05 metalness 0.95 }}\n"
        f"      geometry Box {{ size 0.55 0.006 1.18 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


@_register_wall_decor
def fmt_wall_poster(idx, mx, my, yaw, side, rng):
    """Bright emissive poster - the most vibrant decoration. Replaces the
    custom_ad_panel; same role but built with the standard convention."""
    T = 0.03
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
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


@_register_wall_decor
def fmt_wall_sign(idx, mx, my, yaw, side, rng):
    """Wall sign - wide short emissive panel."""
    T = 0.04
    px, py, oy = _wall_decor_pose(mx, my, yaw, side, T)
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
        # dark backing strip
        f"    Pose {{ translation 0 -0.025 0 children [ Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor 0.18 0.18 0.20 "
        f"roughness 0.5 metalness 0.4 }}\n"
        f"      geometry Box {{ size 1.20 0.02 0.50 }}\n"
        f"    }} ] }}\n"
        f"  ]\n"
        f"}}\n"
    )


def fmt_decoration(idx: int, wall_mx: float, wall_my: float, wall_length: float,
                    wall_yaw: float, side: int, rng: random.Random) -> str:
    """Pick a random decoration emitter and call it. No PROTOs, no per-type
    rotation special cases - every emitter uses the standard convention from
    _wall_decor_pose()."""
    emitter = rng.choice(WALL_DECOR_EMITTERS)
    return emitter(idx, wall_mx, wall_my, wall_yaw, side, rng)


def fmt_path_marker(idx: int, x: float, y: float, color: tuple[float, float, float],
                      radius: float = 0.10) -> str:
    """Small sphere marker at a waypoint."""
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


def fmt_path_segment(idx: int, x1: float, y1: float, x2: float, y2: float,
                       color: tuple[float, float, float]) -> str:
    """Thin connecting box from (x1,y1) to (x2,y2) on the floor."""
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


def fmt_marker(name: str, x: float, y: float, color_rgb: tuple[float, float, float]) -> str:
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


def build_wbt(name: str, W: float, H: float, segs_local, start_xy, second_xy,
              floor_texture_basename: str, trajectories=None,
              shop_centroids=None, raw_waypoints_per_path=None,
              floor_rotation: float = 0.0,
              floor_tile_size: float = 5.0,
              floor_margin: float = 200.0,
              seed: int = DECOR_SEED) -> str:
    # Building frame is [0,W] x [0,H] in floor coords. The Floor proto is
    # centred at (W/2, H/2) but sized (W + 2*margin) x (H + 2*margin) so it
    # extends `margin` metres beyond the building outline in every direction.
    # Walls / decorations / paths stay at their original positions.
    cx, cy = W / 2, H / 2
    floor_w = W + 2 * floor_margin
    floor_h = H + 2 * floor_margin
    sx, sy = start_xy
    yaw = (math.atan2(second_xy[1] - sy, second_xy[0] - sx)
           if second_xy else 0.0)

    rng = random.Random(seed)
    points = _flat_traj_points(trajectories) if trajectories else []

    # ---- dedupe shared wall edges (adjacent shops share boundaries) ----
    segs_local = dedupe_segments(segs_local)

    # ---- cut some corridor-side walls open for windows ----
    win_rng = random.Random(seed + 13)
    segs_local, partial_walls, window_placements = split_walls_for_windows(
        segs_local, trajectories, win_rng,
        window_prob=0.18, min_len=4.5)

    # ---- walls: full-height + partial (sill/lintel) ----
    wall_blocks = []
    for i, s in enumerate(segs_local):
        wall_blocks.append(fmt_wall(i, *s, palette_idx=(i * 17 + 3) % len(WALL_PALETTES)))
    for pi, (mx, my, length, yaw, zc, ht) in enumerate(partial_walls):
        wall_blocks.append(fmt_partial_wall(pi, mx, my, length, yaw, zc, ht,
                                              palette_idx=(pi * 13 + 5) % len(WALL_PALETTES)))

    # ---- windows (custom Solid: glass + 4 frame strips) ----
    window_blocks = [fmt_window(i, *w) for i, w in enumerate(window_placements)]

    # ---- wall decorations: CORRIDOR SIDE ONLY, all custom-Solid ----
    decor_blocks = []
    didx = 0
    for i, s in enumerate(segs_local):
        dx, dy = s[2] - s[0], s[3] - s[1]
        length = math.hypot(dx, dy)
        if length < MIN_WALL_LEN_FOR_DECORATION:
            continue
        side = wall_corridor_side(s, points)
        if side == 0:
            continue
        if rng.random() < DECORATE_PROB_NEW:
            mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
            wall_yaw = math.atan2(dy, dx)
            decor_blocks.append(fmt_decoration(didx, mx, my, length, wall_yaw,
                                                side, rng))
            didx += 1

    # ---- corner objects (fire extinguisher / planter at corridor-adjacent corners) ----
    corner_blocks = []
    if points:
        corner_rng = random.Random(seed + 23)
        corners = find_corners(segs_local, tol=0.10)
        ci = 0
        for cx, cy in corners:
            # corridor-adjacent? trajectory within 2.5 m of corner
            if not _is_near_any_point(cx, cy, points, 2.5):
                continue
            # don't pile decorations next to a wall decoration (skip every other)
            if corner_rng.random() > 0.5:
                continue
            if corner_rng.random() < 0.6:
                corner_blocks.append(fmt_corner_extinguisher(ci, cx, cy))
            else:
                corner_blocks.append(fmt_corner_planter(ci, cx, cy))
            ci += 1

    # ---- floor objects against walls on corridor side ----
    floor_obj_blocks = []
    if trajectories:
        floor_rng = random.Random(seed + 7)
        placements = place_floor_objects_along_walls(
            segs_local, trajectories, floor_rng,
            obj_offset=0.55, corridor_probe=2.5,
            sample_prob=0.30, min_wall_len=2.5,
            object_clearance=0.30)
        for fi, (fx, fy, fyaw, _side) in enumerate(placements):
            choose = floor_rng.choice(FLOOR_OBJ_FNS)
            if choose is fmt_bench:
                floor_obj_blocks.append(choose(fi, fx, fy, fyaw))
            else:
                floor_obj_blocks.append(choose(fi, fx, fy))

    # ---- ceiling-hung billboards (custom, design from scratch) ----
    bb_blocks = []
    if trajectories:
        bb_rng = random.Random(seed + 31)
        bb_placements = place_ceiling_billboards(
            trajectories, segs_local, bb_rng,
            spacing=28.0, dedupe_radius=14.0, clearance_from_walls=1.2)
        bb_names = ENGLISH_SHOP_NAMES[:]
        bb_rng.shuffle(bb_names)
        for bi, (bx, by, byaw) in enumerate(bb_placements):
            txt = bb_names[bi % len(bb_names)]
            bb_blocks.append(fmt_ceiling_billboard(bi, bx, by, byaw, txt, color_idx=bi))

    sign_blocks = []   # floating shop signs stay removed

    # ---- path visualisation: small spheres at every raw waypoint + thin
    #      connecting boxes. One color per path index (cycled). Capped to
    #      20 paths for the big mall (otherwise ~4000 path objects).
    path_blocks = []
    if raw_waypoints_per_path:
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
        path_seq = list(raw_waypoints_per_path)
        if len(path_seq) > 20:
            # evenly sample 20 paths so corridor coverage stays representative
            step = len(path_seq) / 20
            path_seq = [path_seq[int(i * step)] for i in range(20)]
        wpi = 0
        psi = 0
        for path_idx, wps in enumerate(path_seq):
            color = PATH_COLORS[path_idx % len(PATH_COLORS)]
            for x, y in wps:
                path_blocks.append(fmt_path_marker(wpi, x, y, color, radius=0.13))
                wpi += 1
            for (x1, y1), (x2, y2) in zip(wps, wps[1:]):
                seg = fmt_path_segment(psi, x1, y1, x2, y2, color)
                if seg:
                    path_blocks.append(seg)
                    psi += 1

    walls_blob = "".join(wall_blocks)
    window_blob = "".join(window_blocks)
    decor_blob = "\n".join(decor_blocks)
    corner_blob = "\n".join(corner_blocks)
    floor_obj_blob = "\n".join(floor_obj_blocks)
    bb_blob = "\n".join(bb_blocks)
    path_blob = "".join(path_blocks)
    sign_blob = ""

    vp_h = max(20.0, max(W, H) * 0.35)

    # EXTERNPROTOs: only the unavoidable Webots PROTOs (background, floor,
    # robot). All wall decorations are custom Solids -> no PROTO needed.
    # FireExtinguisher PROTO is still used for corner objects.
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
            f'"https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/'
            f'objects/factory/fire_extinguisher/protos/FireExtinguisher.proto"')
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
  rotation 0 0 1 {floor_rotation:.5f}
  size {floor_w:.4f} {floor_h:.4f}
  tileSize {floor_tile_size:.2f} {floor_tile_size:.2f}
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

DEF TIAGO Tiago++ {{
  translation {sx:.4f} {sy:.4f} 0
  rotation 0 0 1 {yaw:.5f}
  name "tiago"
  controller "<none>"
  supervisor TRUE
}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--worlds-dir",
                    default=r"X:\navlori-fusion\src\simulation\worlds")
    ap.add_argument("--px-per-m", type=int, default=0,
                    help="floor texture resolution (default: auto by area)")
    ap.add_argument("--seed", type=int, default=DECOR_SEED)
    ap.add_argument("--floor-rotation", type=float, default=0.0,
                    help="rotate the Floor proto by this many radians around z "
                         "(try math.pi/2, math.pi, -math.pi/2 if texture is misaligned)")
    ap.add_argument("--shrink-shops", type=float, default=0.0,
                    help="shrink every shop polygon inward by this many metres "
                         "(in floor frame) before emitting wall segments. "
                         "Widens corridors by ~2x this value where shops face "
                         "each other. Floor outline + LineString features are "
                         "NOT shrunk so the building perimeter stays put. "
                         "Default 0.0 = no shrink.")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    if not ds.is_dir():
        sys.exit(f"dataset dir not found: {ds}")
    name = ds.name
    meta = ds / "meta"

    fi = json.load(open(meta / "floor_info.json"))
    mi = fi.get("map_info", fi)
    W, H = float(mi["width"]), float(mi["height"])
    print(f"[wbt] floor: {W:.2f} x {H:.2f} m  ({W*H:,.0f} m2)")

    geo = json.load(open(meta / "geojson_map.json"))
    shop_rings_mercator, nonshop_segs_mercator = collect_segments_split(geo)
    n_shop_edges = sum(max(0, len(s["ring"]) - 1) for s in shop_rings_mercator)
    print(f"[wbt] GeoJSON: {len(shop_rings_mercator)} shop polygons "
          f"({n_shop_edges:,} edges), {len(nonshop_segs_mercator):,} non-shop "
          f"segments (floor outline + LineStrings)")

    # bbox uses the union of every segment (shop + non-shop) so the frame is
    # identical to the pre-split build — shrink only affects what comes after.
    all_segs_for_bbox = list(nonshop_segs_mercator)
    for s in shop_rings_mercator:
        ring = s["ring"]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            all_segs_for_bbox.append((x1, y1, x2, y2))
    xmin, ymin, xmax, ymax = bbox(all_segs_for_bbox)
    sx = W / (xmax - xmin) if xmax > xmin else 1.0
    sy = H / (ymax - ymin) if ymax > ymin else 1.0
    print(f"[wbt] mercator origin: ({xmin:.2f}, {ymin:.2f})  scale: x={sx:.6f}  y={sy:.6f}")

    # Non-shop segments transform straight through (no shrink).
    nonshop_segs_local = [(
        (x1 - xmin) * sx, (y1 - ymin) * sy,
        (x2 - xmin) * sx, (y2 - ymin) * sy,
    ) for x1, y1, x2, y2 in nonshop_segs_mercator]

    # Shop rings: transform to floor frame; optionally shrink inward.
    shop_rings_local = []
    n_dropped = 0
    for s in shop_rings_mercator:
        floor_ring = [transform_xy(x, y, xmin, ymin, sx, sy) for x, y in s["ring"]]
        if args.shrink_shops > 0:
            shrunk = shrink_polygon_ring(floor_ring, args.shrink_shops)
            if shrunk is None:
                n_dropped += 1
                continue
            floor_ring = shrunk
        shop_rings_local.append({"ring": floor_ring, "props": s["props"]})
    if args.shrink_shops > 0:
        print(f"[wbt] shrink: shop polygons shrunk by {args.shrink_shops:.2f} m "
              f"inward  ({len(shop_rings_local)} kept, {n_dropped} dropped as "
              f"too small)")

    # Wall segments from the (possibly shrunken) shop rings.
    shop_segs_local = []
    for s in shop_rings_local:
        ring = s["ring"]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            shop_segs_local.append((x1, y1, x2, y2))

    segs_local = nonshop_segs_local + shop_segs_local

    # Determine px_per_m
    if args.px_per_m > 0:
        ppm = args.px_per_m
    else:
        # cap output PNG to ~12 MP
        ppm = PX_PER_M_SMALL if W * H < 20000 else PX_PER_M_BIG
    print(f"[wbt] floor texture resolution: {ppm} px/m -> {int(W*ppm)} x {int(H*ppm)} px")

    # path_00 waypoints for TIAGO + markers
    with open(ds / "path_00" / "waypoints_raw.csv") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < 2:
        sys.exit("path_00 has < 2 raw waypoints")
    start = (float(rows[0]["gt_x"]), float(rows[0]["gt_y"]))
    second = (float(rows[1]["gt_x"]), float(rows[1]["gt_y"]))
    print(f"[wbt] TIAGO start: ({start[0]:.2f}, {start[1]:.2f}) -> facing wp1 "
          f"({second[0]:.2f}, {second[1]:.2f})")

    worlds_dir = Path(args.worlds_dir)
    worlds_dir.mkdir(parents=True, exist_ok=True)
    floor_tex_basename = f"{name}_floor.png"
    floor_tex_path = worlds_dir / floor_tex_basename

    # REDO: drop the per-shop-polygon composite texture (was causing the
    # walls-vs-floor misalignment because Webots' UV mapping for the Floor
    # proto did not match my image-orientation assumption). Use a simple
    # 5 m square tile that repeats across the whole floor — generic surface,
    # nothing to align.
    print(f"[wbt] rendering tile floor texture (repeats every 5 m)...", flush=True)
    floor_tile_size = 5.0
    img_w, img_h, _ = render_tile_texture(
        str(floor_tex_path), tile_m=floor_tile_size, px_per_m=200,
        seed=args.seed,
    )
    sz_mb = os.path.getsize(floor_tex_path) / 1024 / 1024
    print(f"[wbt]   -> {floor_tex_path}  ({img_w}x{img_h} px, {sz_mb:.2f} MB, "
          f"tile size {floor_tile_size:.1f} m)")

    # Trajectories from DENSE ground_truth.csv (10 Hz interpolated), not the
    # sparse raw waypoints — needed so floor-object corridor detection finds
    # trajectory points close to every wall they pass by, not just at the
    # few landmark presses in each trace.
    trajectories = []
    for d in sorted(ds.iterdir()):
        if not d.name.startswith("path_"):
            continue
        gt_csv = d / "ground_truth.csv"
        if not gt_csv.exists():
            continue
        with open(gt_csv) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) >= 2:
            # downsample for speed — every 10th point (~1 Hz) is plenty for
            # corridor detection, no need for 10 Hz density
            pts = [(float(r["gt_x"]), float(r["gt_y"])) for r in rows[::10]]
            if len(pts) >= 2:
                trajectories.append(pts)

    # shop centroids + chosen names — derived from the SAME (possibly shrunken)
    # floor-frame rings used for walls, so signs stay inside the new perimeter.
    shop_centroids = []
    shop_meta = []
    for s in shop_rings_local:
        ring_floor = s["ring"]
        area = shoelace_area(ring_floor)
        if area < 3.0:
            continue
        cxs, cys = polygon_centroid(ring_floor)
        gname = html.unescape((s["props"].get("name") or "").strip())
        gname = gname if (gname and is_ascii(gname)) else None
        shop_meta.append({"cx": cxs, "cy": cys, "area": area, "geo_name": gname})
    shop_meta.sort(key=lambda x: -x["area"])
    used = {m["geo_name"].upper() for m in shop_meta if m["geo_name"]}
    fallback_pool = [n for n in ENGLISH_SHOP_NAMES if n.upper() not in used]
    fb_iter = iter(fallback_pool * (1 + len(shop_meta) // max(1, len(fallback_pool))))
    for m in shop_meta:
        m["name"] = m["geo_name"] or next(fb_iter)
        shop_centroids.append((m["cx"], m["cy"], m["name"]))

    # Raw waypoints per path (sparse landmark presses) — used for the visible
    # path markers + connecting lines. Sparser than `trajectories` (which is
    # dense 10 Hz GT used for corridor-side detection); much better for visual
    # display - one sphere per landmark, not 1000s of overlapping dots.
    raw_waypoints_per_path = []
    for d in sorted(ds.iterdir()):
        if not d.name.startswith("path_"):
            continue
        wp_csv = d / "waypoints_raw.csv"
        if not wp_csv.exists():
            continue
        with open(wp_csv) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) >= 2:
            raw_waypoints_per_path.append([
                (float(r["gt_x"]), float(r["gt_y"])) for r in rows
            ])

    wbt = build_wbt(name, W, H, segs_local, start, second, floor_tex_basename,
                     trajectories=trajectories,
                     shop_centroids=shop_centroids,
                     raw_waypoints_per_path=raw_waypoints_per_path,
                     floor_rotation=args.floor_rotation,
                     floor_tile_size=floor_tile_size,
                     seed=args.seed)
    wbt_path = worlds_dir / f"{name}.wbt"
    wbt_path.write_text(wbt, encoding="utf-8")
    n_decor = wbt.count("DEF DECOR_")
    n_floor = wbt.count("DEF FLOOROBJ_")
    n_win   = wbt.count("DEF WIN_")
    n_corner = wbt.count("DEF CORNER_")
    n_bb    = wbt.count("DEF BB_")
    n_wp    = wbt.count("DEF WP_")
    n_pseg  = wbt.count("DEF PS_")
    n_walls_emitted = wbt.count("DEF WALL_") + wbt.count("DEF PWALL_")
    print(f"[wbt] world  -> {wbt_path}  ({len(wbt):,} bytes, ~{wbt.count(chr(10)):,} lines)")
    print(f"[wbt]   walls={n_walls_emitted:,} (incl partial)  windows={n_win}  "
          f"decorations={n_decor}  corner_objects={n_corner}")
    print(f"[wbt]   floor_objects={n_floor}  ceiling_billboards={n_bb}  "
          f"path_waypoints={n_wp}  path_segments={n_pseg}")


if __name__ == "__main__":
    main()
