"""Inject a DEF CEILING Group into an existing Webots .wbt world.

The ceiling is a single light-grey slab spanning the building bbox (computed
from the actual WALL_/PWALL_ Solid translations + sizes in the file, so the
result fits any world built by scripts/build_world.py or the legacy
build_iln20_webots_world.py without assuming a known floor size).

Default sits at z = wall_top + thickness/2 so its bottom face touches the
top of the walls. Toggleable in Webots by right-clicking DEF CEILING in
the scene tree -> Hide.

Usage:
    .venv\\Scripts\\python.exe scripts\\add_ceiling_to_wbt.py \\
        --world src/simulation/worlds/iln20_5cd56b6a_F1.wbt
    .venv\\Scripts\\python.exe scripts\\add_ceiling_to_wbt.py \\
        --world <wbt> --out <other-wbt>     # write to a new file
    .venv\\Scripts\\python.exe scripts\\add_ceiling_to_wbt.py \\
        --world <wbt> --dry-run             # preview, don't modify
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path


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
TIAGO_HEAD_RE = re.compile(r"^DEF TIAGO\b", re.MULTILINE)
CEILING_HEAD_RE = re.compile(r"DEF CEILING\b")


def _block_end(text, brace_open_idx):
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


def parse_walls(text):
    out = []
    for head in WALL_HEAD_RE.finditer(text):
        brace = head.end() - 1
        end = _block_end(text, brace)
        block = text[head.start():end]
        mt = TRANSLATION_RE.search(block)
        mr = ROTATION_RE.search(block)
        mb = BOX_SIZE_RE.search(block)
        if not (mt and mr and mb):
            continue
        out.append({
            "cx": float(mt.group(1)), "cy": float(mt.group(2)),
            "cz": float(mt.group(3)),
            "yaw": float(mr.group(1)),
            "length": float(mb.group(1)),
            "thickness": float(mb.group(2)),
            "height": float(mb.group(3)),
            "is_partial": head.group(1).startswith("PWALL_"),
        })
    return out


def wall_endpoints(w):
    """Return ((x1,y1), (x2,y2)) for a wall: project length/2 along its axis
    from the midpoint (cx, cy)."""
    half = w["length"] / 2.0
    dx = math.cos(w["yaw"]) * half
    dy = math.sin(w["yaw"]) * half
    return (w["cx"] - dx, w["cy"] - dy), (w["cx"] + dx, w["cy"] + dy)


def fmt_ceiling(cx, cy, W, H, wall_height, overhang, thickness):
    sx = W + 2 * overhang
    sy = H + 2 * overhang
    z = wall_height + thickness / 2
    return (
        f"\n"
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
        f"}}\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", default=None,
                    help="output path (default: in-place with .bak backup)")
    ap.add_argument("--overhang", type=float, default=2.0,
                    help="extend ceiling N m past the building bbox on every "
                         "side (default 2.0)")
    ap.add_argument("--thickness", type=float, default=0.10,
                    help="slab vertical thickness (default 0.10)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing DEF CEILING block")
    args = ap.parse_args()

    world = Path(args.world).resolve()
    if not world.is_file():
        sys.exit(f"world not found: {world}")
    text = world.read_text(encoding="utf-8")

    existing = CEILING_HEAD_RE.search(text)
    if existing and not args.force:
        sys.exit(f"world already has a DEF CEILING -- pass --force to replace")

    walls = parse_walls(text)
    if not walls:
        sys.exit("no WALL_/PWALL_ solids found -- cannot derive bbox")

    # Building bbox from wall endpoints (excluding PWALL_ which are sill/lintel
    # partials around windows -- their XY footprint matches the parent wall
    # but we already have that from the full WALL_ endpoints).
    full_walls = [w for w in walls if not w["is_partial"]]
    pts = []
    for w in full_walls:
        (x1, y1), (x2, y2) = wall_endpoints(w)
        pts.append((x1, y1)); pts.append((x2, y2))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    W = xmax - xmin
    H = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # Wall height = max(Box.size_z) across full walls (PWALL partial heights
    # are smaller, so taking the max gives the actual room height).
    wall_height = max(w["height"] for w in full_walls)

    print(f"[ceiling] parsed {len(full_walls)} full walls + "
          f"{len(walls) - len(full_walls)} partials")
    print(f"[ceiling] building bbox: x in [{xmin:.2f}, {xmax:.2f}] "
          f"y in [{ymin:.2f}, {ymax:.2f}]")
    print(f"[ceiling] W x H = {W:.2f} x {H:.2f} m  center=({cx:.2f}, {cy:.2f})")
    print(f"[ceiling] wall_height={wall_height:.2f} m  "
          f"-> ceiling z_center={wall_height + args.thickness/2:.3f} m")

    block = fmt_ceiling(cx, cy, W, H, wall_height,
                          args.overhang, args.thickness)

    # If a ceiling already exists, locate & replace its full block.
    if existing:
        brace_idx = text.find("{", existing.start())
        if brace_idx == -1:
            sys.exit("malformed existing CEILING block")
        end = _block_end(text, brace_idx)
        # absorb trailing newline + any leading "# ..." comment lines just
        # above the DEF CEILING that we authored
        start = existing.start()
        line_start = text.rfind("\n", 0, start) + 1
        # walk up over consecutive comment lines
        cursor = line_start
        while cursor > 0:
            prev_line_end = cursor - 1
            prev_line_start = text.rfind("\n", 0, prev_line_end) + 1
            if text[prev_line_start:prev_line_end].lstrip().startswith("#"):
                cursor = prev_line_start
            else:
                break
        start = cursor
        if end < len(text) and text[end] == "\n":
            end += 1
        new_text = text[:start] + block + text[end:]
        action = "replaced existing"
    else:
        # Insert just before "DEF TIAGO" -- semantic end of structural blocks.
        m = TIAGO_HEAD_RE.search(text)
        if not m:
            sys.exit("no DEF TIAGO block found -- where do I insert?")
        ins = m.start()
        new_text = text[:ins] + block + text[ins:]
        action = "inserted new"

    if args.dry_run:
        print(f"[ceiling] dry-run: would have {action} CEILING block "
              f"({len(block):,} chars). No changes written.")
        return

    out_path = Path(args.out).resolve() if args.out else world
    if out_path == world:
        bak = world.with_suffix(world.suffix + ".bak-ceiling")
        shutil.copy2(world, bak)
        print(f"[ceiling] backup -> {bak}")
    out_path.write_text(new_text, encoding="utf-8")
    print(f"[ceiling] {action} -> {out_path}  "
          f"({len(text):,} -> {len(new_text):,} bytes, "
          f"{len(new_text) - len(text):+,d})")


if __name__ == "__main__":
    main()
