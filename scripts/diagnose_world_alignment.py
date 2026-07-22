"""Measure the geometric agreement between an iln20 world (.wbt) and the
dataset the replay controller drives from. No fixes — evidence only.

The controller anchors the robot at raw dataset coordinates, so replay is
correct iff the world's walls/markers sit in the SAME frame as the dataset's
ground truth. This script quantifies that:

  1. TIAGO start + green/red markers vs dataset path_00 waypoints 0/1
     (both were derived from the dataset at build time -> must match ~0).
  2. Wall bounding box vs the dataset floor extent [0,W]x[0,H].
  3. Coverage: fraction of ALL dataset waypoints inside the wall bbox.
  4. Transform identification: the world's WP_* path-marker spheres are
     verbatim copies of dataset waypoints. Fit them against the dataset
     under the 8 axis flips/rotations of the floor frame; report the
     residual of each. The winner (and its residual translation) IS the
     transform separating the two frames.

Usage:
    .venv\\Scripts\\python.exe scripts\\diagnose_world_alignment.py \\
        --world src/simulation/worlds/iln20_5d27099f_F2.wbt \\
        --dataset data/iln20_5d27099f_F2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


TRANSLATION_RE = r"translation\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"


def parse_named_translation(text: str, def_name: str):
    m = re.search(rf"DEF {def_name}\b[^\n]*\{{\s*\n\s*{TRANSLATION_RE}", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def parse_wp_markers(text: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b, _ in
            re.findall(rf"DEF WP_\d+ Solid \{{\s*\n\s*{TRANSLATION_RE}", text)]


def parse_wall_bbox(text: str):
    pts = []
    for block in re.finditer(
            rf"DEF WALL_\d+ Solid \{{\s*\n\s*{TRANSLATION_RE}\s*\n\s*"
            rf"rotation 0 0 1 (-?\d+\.?\d*)", text):
        cx, cy = float(block.group(1)), float(block.group(2))
        yaw = float(block.group(4))
        tail = text[block.end():block.end() + 600]
        mb = re.search(r"geometry Box \{ size (-?\d+\.?\d*)", tail)
        if not mb:
            continue
        half = float(mb.group(1)) / 2.0
        dx, dy = math.cos(yaw) * half, math.sin(yaw) * half
        pts.append((cx - dx, cy - dy))
        pts.append((cx + dx, cy + dy))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys), len(pts) // 2


def parse_floor(text: str):
    m = re.search(rf"Floor \{{\s*\n\s*{TRANSLATION_RE}.*?size\s+"
                  rf"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", text, re.DOTALL)
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)),
            float(m.group(4)), float(m.group(5)))


def load_waypoints(dataset: Path) -> dict[str, list[tuple[float, float]]]:
    """Raw waypoints per path; falls back to (subsampled) dense
    ground_truth.csv for datasets without waypoints_raw.csv (msiln_*)."""
    out = {}
    for d in sorted(dataset.iterdir()):
        if not d.name.startswith("path_"):
            continue
        f = d / "waypoints_raw.csv"
        stride = 1
        if not f.is_file():
            f = d / "ground_truth.csv"
            stride = 10  # 10 Hz dense -> ~1 Hz points
            if not f.is_file():
                continue
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        pts = [(float(r["gt_x"]), float(r["gt_y"])) for r in rows[::stride]]
        if pts:
            out[d.name] = pts
    return out


DIHEDRAL = [
    ("identity",        lambda x, y, W, H: (x, y)),
    ("flip-x",          lambda x, y, W, H: (W - x, y)),
    ("flip-y",          lambda x, y, W, H: (x, H - y)),
    ("rot180",          lambda x, y, W, H: (W - x, H - y)),
    ("swap-xy",         lambda x, y, W, H: (y, x)),
    ("swap+flip-x",     lambda x, y, W, H: (H - y, x)),
    ("swap+flip-y",     lambda x, y, W, H: (y, W - x)),
    ("swap+rot180",     lambda x, y, W, H: (H - y, W - x)),
]


def nearest_dist(p, cloud):
    return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in cloud)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 unless robot-start/floor/coverage checks pass "
                         "(and, when WP_ markers exist, the identity-transform "
                         "fit). Used by replay_site.py as an automated gate.")
    args = ap.parse_args()

    world = Path(args.world)
    dataset = Path(args.dataset)
    text = world.read_text(encoding="utf-8")

    fi = json.load(open(dataset / "meta" / "floor_info.json"))
    mi = fi.get("map_info", fi)
    W, H = float(mi["width"]), float(mi["height"])
    wps = load_waypoints(dataset)
    all_pts = [p for pts in wps.values() for p in pts]
    print(f"[dataset] {dataset.name}: floor {W:.1f} x {H:.1f} m, "
          f"{len(wps)} paths, {len(all_pts)} raw waypoints")

    gates: list[tuple[str, bool, str]] = []

    # ── 1. robot start (TIAGO or REPLAY_RIG) + markers vs path_00 ──
    robot_pt = parse_named_translation(text, "TIAGO")
    robot_label = "TIAGO"
    if robot_pt is None:
        robot_pt = parse_named_translation(text, "REPLAY_RIG")
        robot_label = "REPLAY_RIG"
    green = parse_named_translation(text, "MARKER_START_GREEN")
    red = parse_named_translation(text, "MARKER_SECOND_RED")
    p00 = wps.get("path_00") or (next(iter(wps.values())) if wps else [])
    rows = [(f"{robot_label} vs path wp0", robot_pt, p00[0] if p00 else None, True)]
    if green or red:  # debug markers are optional (absent in --clean worlds)
        rows += [("green  vs path wp0", green, p00[0] if p00 else None, False),
                 ("red    vs path wp1", red, p00[1] if len(p00) > 1 else None, False)]
    for label, world_pt, ds_pt, gated in rows:
        if world_pt and ds_pt:
            d = math.hypot(world_pt[0] - ds_pt[0], world_pt[1] - ds_pt[1])
            print(f"[markers] {label}: world=({world_pt[0]:.2f},{world_pt[1]:.2f}) "
                  f"dataset=({ds_pt[0]:.2f},{ds_pt[1]:.2f})  delta={d:.3f} m")
            if gated:
                gates.append(("robot-start", d < 0.05,
                              f"{label} delta {d:.3f} m (limit 0.05)"))
        else:
            print(f"[markers] {label}: MISSING ({'world' if not world_pt else 'dataset'})")
            if gated:
                gates.append(("robot-start", False, f"{label}: missing"))

    # ── 2. wall bbox vs floor extent ──
    wb = parse_wall_bbox(text)
    if wb:
        xmin, ymin, xmax, ymax, n_walls = wb
        print(f"[walls] {n_walls} walls, bbox x[{xmin:.2f},{xmax:.2f}] "
              f"y[{ymin:.2f},{ymax:.2f}]")
        print(f"[walls] dataset floor extent is x[0,{W:.2f}] y[0,{H:.2f}] "
              f"-> offsets: x0={xmin:.2f} y0={ymin:.2f} "
              f"x1={xmax - W:+.2f} y1={ymax - H:+.2f}")
    fl = parse_floor(text)
    if fl:
        print(f"[floor] proto at ({fl[0]:.2f},{fl[1]:.2f}) size {fl[2]:.1f} x {fl[3]:.1f} "
              f"(expected centre ({W/2:.2f},{H/2:.2f}))")
        d = math.hypot(fl[0] - W / 2, fl[1] - H / 2)
        gates.append(("floor-centre", d < 0.5,
                      f"floor proto centre off by {d:.2f} m (limit 0.5)"))
    else:
        gates.append(("floor-centre", False, "no Floor proto found"))

    # ── 3. waypoint coverage inside wall bbox ──
    if wb:
        margin = 1.0
        inside = sum(1 for x, y in all_pts
                     if xmin - margin <= x <= xmax + margin
                     and ymin - margin <= y <= ymax + margin)
        frac = inside / max(1, len(all_pts))
        print(f"[coverage] {inside}/{len(all_pts)} waypoints "
              f"({100 * frac:.1f}%) inside wall bbox (+{margin}m)")
        gates.append(("coverage", frac >= 0.995,
                      f"{100 * frac:.1f}% waypoints inside wall bbox "
                      f"(limit 99.5%)"))
    else:
        gates.append(("coverage", False, "no WALL_ solids parsed"))

    # ── 4. transform identification via WP_ markers (skipped in clean worlds:
    #     the builder emits none; robot-start + coverage carry the gate) ──
    markers = parse_wp_markers(text)
    print(f"[transform] {len(markers)} WP_ path markers in world"
          + (" (clean world -- fit skipped)" if not markers else ""))
    if markers and all_pts:
        sample = markers[:: max(1, len(markers) // 200)]
        print(f"[transform] fitting {len(sample)} markers against dataset "
              f"waypoints under 8 frame transforms:")
        results = []
        for name, T in DIHEDRAL:
            tpts = [T(x, y, W, H) for x, y in all_pts]
            ds = sorted(nearest_dist(m, tpts) for m in sample)
            med = ds[len(ds) // 2]
            results.append((med, name, ds[int(len(ds) * 0.9)]))
        results.sort()
        for med, name, p90 in results:
            tag = "  <-- best" if (med, name, p90) == results[0] else ""
            print(f"    {name:14s} median={med:7.3f} m   p90={p90:7.3f} m{tag}")
        best_med, best_name, _ = results[0]
        if best_name == "identity" and best_med < 0.05:
            print("[verdict] world and dataset are in the SAME frame "
                  "(markers coincide). Misalignment must come from elsewhere.")
        elif best_med < 0.5:
            print(f"[verdict] world frame = dataset frame under '{best_name}' "
                  f"-> the dataset (or world) is transformed by exactly that.")
        else:
            print("[verdict] no clean transform match -- world geometry was "
                  "edited independently of the dataset (manual shift?).")
        gates.append(("transform", best_name == "identity" and best_med < 0.05,
                      f"best='{best_name}' median={best_med:.3f} m"))

    # ── Gate summary ──
    if args.gate:
        print("\n[gate]")
        all_ok = True
        for name, ok, detail in gates:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            all_ok &= ok
        print(f"[gate] {'ALL PASS' if all_ok else 'FAILURE'}")
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
