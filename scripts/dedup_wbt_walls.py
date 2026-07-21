"""Parse a Webots .wbt world file, detect duplicate / overlapping wall Solids
and drop the redundant ones.

Two wall Solids (DEF WALL_* or DEF PWALL_*) are considered "at the same place"
when ALL of these hold:
  1. yaws match (mod pi) within --tol-yaw rad
  2. perpendicular distance between their axis lines is below --tol-perp m
  3. their projected spans onto the shared axis overlap by at least
     --overlap-frac of the SHORTER wall's length
  4. their vertical (Z) spans also overlap by at least --overlap-frac of the
     shorter wall's height

Within a duplicate cluster the LONGER wall is kept; shorter / shorter-Z walls
are removed. The .wbt is then rewritten with the surviving blocks. By default
the change is in-place with a .bak file alongside; use --out to write to a
different path, or --dry-run to just report what would be dropped.

Usage:
    .venv\\Scripts\\python.exe scripts\\dedup_wbt_walls.py \\
        --world src/simulation/worlds/iln20_5d27099f_F2.wbt --dry-run
    .venv\\Scripts\\python.exe scripts\\dedup_wbt_walls.py \\
        --world src/simulation/worlds/iln20_5d27099f_F2.wbt
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path


# Matches the head of a WALL_/PWALL_ Solid block. Brace counting takes over
# from `{` to find the matching `}` — regex alone can't span balanced braces.
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


def find_block_end(text: str, brace_open_idx: int) -> int:
    """Given index of '{', return index right after its matching '}'."""
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
    raise ValueError(f"unbalanced braces starting at {brace_open_idx}")


def parse_walls(text: str):
    """Yield dicts describing every WALL_/PWALL_ Solid block found in `text`.

    Each dict has: name, start, end (text slice indices), cx, cy, cz, yaw,
    length, thickness, height. start/end span includes the trailing '\\n' if
    present.
    """
    out = []
    for head in WALL_HEAD_RE.finditer(text):
        name = head.group(1)
        # the '{' is the last char of the match
        brace_idx = head.end() - 1
        end = find_block_end(text, brace_idx)
        # absorb trailing newline so removal doesn't leave blank lines
        if end < len(text) and text[end] == "\n":
            end += 1
        block = text[head.start():end]

        mt = TRANSLATION_RE.search(block)
        mr = ROTATION_RE.search(block)
        mb = BOX_SIZE_RE.search(block)
        if not (mt and mr and mb):
            # malformed (or non-Box geometry) — skip silently, leave in file
            continue
        cx, cy, cz = float(mt.group(1)), float(mt.group(2)), float(mt.group(3))
        yaw = float(mr.group(1))
        length = float(mb.group(1))
        thickness = float(mb.group(2))
        height = float(mb.group(3))

        out.append({
            "name": name, "start": head.start(), "end": end,
            "cx": cx, "cy": cy, "cz": cz, "yaw": yaw,
            "length": length, "thickness": thickness, "height": height,
        })
    return out


def line_key(w, tol_yaw: float, tol_perp: float) -> tuple[int, int]:
    """Bucket key for the infinite line the wall lies on.

    Two walls share a key iff their yaws agree mod pi (within tol_yaw) and
    their perpendicular offsets from origin agree (within tol_perp).
    """
    yaw_mod = w["yaw"] % math.pi
    # Normalise so 0 and pi-eps don't sit in different buckets.
    if yaw_mod > math.pi - tol_yaw:
        yaw_mod -= math.pi
    # Perpendicular offset of the line through (cx, cy) with direction yaw
    # from the origin: signed distance = -cx*sin(yaw) + cy*cos(yaw).
    perp = -w["cx"] * math.sin(yaw_mod) + w["cy"] * math.cos(yaw_mod)
    return (round(yaw_mod / tol_yaw), round(perp / tol_perp))


def axis_span(w) -> tuple[float, float, float, float]:
    """Project the wall's midpoint onto its own axis, return (s0, s1, z0, z1)
    where s0<=s1 is the along-axis interval and z0<=z1 is the vertical one.
    """
    yaw_mod = w["yaw"] % math.pi
    s_mid = w["cx"] * math.cos(yaw_mod) + w["cy"] * math.sin(yaw_mod)
    half_L = w["length"] / 2.0
    half_H = w["height"] / 2.0
    return (s_mid - half_L, s_mid + half_L,
            w["cz"] - half_H, w["cz"] + half_H)


def overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of overlap divided by shorter interval length."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    if shorter <= 0:
        return 0.0
    return inter / shorter


def dedup(walls, tol_yaw: float, tol_perp: float, overlap_frac_min: float):
    """Return (kept_names, dropped_pairs) where dropped_pairs is a list of
    (dropped_name, kept_name, reason) tuples for the dry-run report.
    """
    # Group by line bucket. Also check the 8 neighbouring perp-buckets so a
    # wall that lands right on a tol boundary doesn't escape detection.
    buckets: dict[tuple[int, int], list[int]] = {}
    spans = []
    for i, w in enumerate(walls):
        spans.append(axis_span(w))
        key = line_key(w, tol_yaw, tol_perp)
        buckets.setdefault(key, []).append(i)

    # Sort each bucket by length DESCENDING — the longest wall is the "keeper"
    # candidate; shorter walls that overlap it get dropped.
    order = sorted(range(len(walls)), key=lambda i: -walls[i]["length"])

    kept_idx: list[int] = []
    dropped: list[tuple[str, str, str]] = []
    dropped_set: set[int] = set()

    for i in order:
        if i in dropped_set:
            continue
        w = walls[i]
        ki = line_key(w, tol_yaw, tol_perp)
        s0, s1, z0, z1 = spans[i]

        # Look in this bucket + the 8 neighbours (yaw +/-1, perp +/-1)
        candidates = []
        for dy in (-1, 0, 1):
            for dp in (-1, 0, 1):
                neigh = buckets.get((ki[0] + dy, ki[1] + dp))
                if neigh:
                    candidates.extend(neigh)

        for j in candidates:
            if j == i or j in dropped_set or j not in kept_idx:
                continue
            wj = walls[j]
            # Re-verify the line match precisely (bucket is fuzzy).
            d_yaw = abs((w["yaw"] - wj["yaw"]) % math.pi)
            d_yaw = min(d_yaw, math.pi - d_yaw)
            if d_yaw > tol_yaw:
                continue
            # Perpendicular distance between the two parallel lines:
            yaw_avg = (w["yaw"] + wj["yaw"]) / 2.0
            nx, ny = -math.sin(yaw_avg), math.cos(yaw_avg)
            d_perp = abs((w["cx"] - wj["cx"]) * nx +
                          (w["cy"] - wj["cy"]) * ny)
            if d_perp > tol_perp:
                continue

            sj0, sj1, zj0, zj1 = spans[j]
            f_axis = overlap_frac(s0, s1, sj0, sj1)
            f_z = overlap_frac(z0, z1, zj0, zj1)
            if f_axis >= overlap_frac_min and f_z >= overlap_frac_min:
                dropped_set.add(i)
                dropped.append((w["name"], wj["name"],
                                f"axis_overlap={f_axis:.2f} "
                                f"z_overlap={f_z:.2f} "
                                f"perp_d={d_perp:.3f} d_yaw={d_yaw:.3f}"))
                break
        if i not in dropped_set:
            kept_idx.append(i)

    kept_names = {walls[i]["name"] for i in kept_idx}
    return kept_names, dropped


def rewrite_without(text: str, walls, kept_names: set) -> str:
    """Return text with every wall block whose name is NOT in kept_names
    removed. Walls list must be in source order (which parse_walls guarantees,
    finditer is left-to-right)."""
    # Walk source order; emit kept blocks; skip dropped blocks.
    pieces = []
    cursor = 0
    for w in walls:
        if w["name"] in kept_names:
            continue  # keep -> leave intact in text, don't touch
        # drop -> emit [cursor, w["start"]) then skip the block
        pieces.append(text[cursor:w["start"]])
        cursor = w["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="path to .wbt to dedup")
    ap.add_argument("--out", default=None,
                    help="output path (default: in-place with .bak alongside)")
    ap.add_argument("--tol-yaw", type=float, default=0.05,
                    help="yaw tolerance in rad (default 0.05 ~ 3 deg)")
    ap.add_argument("--tol-perp", type=float, default=0.20,
                    help="perpendicular line tolerance in m (default 0.20). "
                         "Set above wall thickness so a pair of polygon-edge "
                         "walls offset by their own thickness still matches.")
    ap.add_argument("--overlap-frac", type=float, default=0.50,
                    help="minimum overlap fraction of the SHORTER wall, in "
                         "both axis and Z, to count as a duplicate "
                         "(default 0.50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report duplicates but don't modify the .wbt")
    ap.add_argument("--verbose", action="store_true",
                    help="print every dropped wall + kept counterpart")
    args = ap.parse_args()

    world = Path(args.world).resolve()
    if not world.is_file():
        sys.exit(f"world not found: {world}")

    text = world.read_text(encoding="utf-8")
    walls = parse_walls(text)
    print(f"[dedup] parsed {len(walls)} wall solids from {world.name}")
    if not walls:
        sys.exit(0)

    n_pwall = sum(1 for w in walls if w["name"].startswith("PWALL_"))
    n_wall = len(walls) - n_pwall
    print(f"[dedup]   WALL_={n_wall}  PWALL_={n_pwall}")

    kept_names, dropped = dedup(walls, args.tol_yaw, args.tol_perp,
                                  args.overlap_frac)
    print(f"[dedup] duplicates: {len(dropped)} walls would be removed "
          f"({len(kept_names)} kept)")

    if args.verbose:
        for d_name, k_name, why in dropped[:50]:
            print(f"  drop {d_name} (kept {k_name}: {why})")
        if len(dropped) > 50:
            print(f"  ... and {len(dropped) - 50} more")

    if args.dry_run:
        print("[dedup] --dry-run: no changes written")
        return

    if not dropped:
        print("[dedup] nothing to do")
        return

    new_text = rewrite_without(text, walls, kept_names)

    out_path = Path(args.out).resolve() if args.out else world
    if out_path == world:
        bak = world.with_suffix(world.suffix + ".bak")
        shutil.copy2(world, bak)
        print(f"[dedup] backup -> {bak}")
    out_path.write_text(new_text, encoding="utf-8")
    sz_before = len(text)
    sz_after = len(new_text)
    print(f"[dedup] wrote {out_path}  ({sz_before:,} -> {sz_after:,} bytes, "
          f"{sz_after - sz_before:+,d})")


if __name__ == "__main__":
    main()
