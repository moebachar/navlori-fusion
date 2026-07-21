"""Render a static PNG preview of an ILN 2.0 Webots world before launching it.

Shows the floor texture + extruded walls (from the .wbt) + all GT trajectories
(red) + the start/wp1 marker spheres (green/red dots) overlaid in the same
metric frame. If the walls don't trace the floor-plan building outlines, or
the trajectories don't run between walls, the §6 frame-alignment trap has bit.

Usage:
    .venv\\Scripts\\python.exe scripts\\preview_iln20_webots_world.py \\
        --dataset-dir data/iln20_5d27099f_F2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


WALL_RE = re.compile(
    r"DEF WALL_\d+ Solid \{\s+"
    r"translation\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+"
    r"rotation\s+0\s+0\s+1\s+(-?\d+\.?\d*).*?"
    r"size\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)",
    re.DOTALL,
)


def parse_walls_from_wbt(wbt_text: str):
    """Yield (mx, my, length, thickness, yaw_rad) per wall Solid in the .wbt."""
    for m in WALL_RE.finditer(wbt_text):
        mx, my, _mz = float(m.group(1)), float(m.group(2)), float(m.group(3))
        yaw = float(m.group(4))
        length, thickness = float(m.group(5)), float(m.group(6))
        yield mx, my, length, thickness, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--worlds-dir",
                    default=r"X:\navlori-fusion\src\simulation\worlds")
    ap.add_argument("--out-dir",
                    default=None,
                    help="default: <dataset-dir>/preview/")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    name = ds.name
    wbt = Path(args.worlds_dir) / f"{name}.wbt"
    if not wbt.is_file():
        sys.exit(f"world not found: {wbt}\nrun build_iln20_webots_world.py first")
    out_dir = Path(args.out_dir) if args.out_dir else ds / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    fi = json.load(open(ds / "meta" / "floor_info.json"))
    mi = fi.get("map_info", fi)
    W, H = float(mi["width"]), float(mi["height"])

    img = Image.open(ds / "meta" / "floor_image.png")
    wbt_text = wbt.read_text(encoding="utf-8")
    walls = list(parse_walls_from_wbt(wbt_text))
    print(f"[preview] parsed {len(walls):,} walls from {wbt.name}")

    # gather all trace polylines (use raw waypoints — same as scoring/overlay)
    polys = []
    for d in sorted(ds.iterdir()):
        if not d.name.startswith("path_"):
            continue
        wp_csv = d / "waypoints_raw.csv"
        if not wp_csv.exists():
            continue
        with open(wp_csv) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) >= 2:
            polys.append(np.asarray(
                [(float(r["gt_x"]), float(r["gt_y"])) for r in rows]))
    print(f"[preview] {len(polys)} traces with >=2 raw waypoints")

    # path_00 markers for sanity (same as in the world)
    with open(ds / "path_00" / "waypoints_raw.csv") as fh:
        rows = list(csv.DictReader(fh))
    start = (float(rows[0]["gt_x"]), float(rows[0]["gt_y"]))
    second = (float(rows[1]["gt_x"]), float(rows[1]["gt_y"]))

    fig, ax = plt.subplots(figsize=(12, 12 * H / W))
    # 1. floor image (same convention as the inspection overlay)
    ax.imshow(img, extent=[0, W, 0, H], origin="upper",
              interpolation="bilinear", aspect="equal", alpha=0.55)

    # 2. walls — render each Box as a thin oriented rectangle in 2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D
    for mx, my, length, thickness, yaw in walls:
        rect = Rectangle((-length/2, -thickness/2), length, thickness,
                         facecolor="#222222", edgecolor="none", alpha=0.85)
        trans = Affine2D().rotate(yaw).translate(mx, my) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)

    # 3. traces (red)
    for wp in polys:
        ax.plot(wp[:, 0], wp[:, 1], linewidth=0.5, alpha=0.5, color="#ff2a55")

    # 4. start/wp1 markers (green/red dots)
    ax.scatter(*start, s=120, c="#00cc44", edgecolor="black", linewidth=1, zorder=10,
               label="path_00 start (green sphere in Webots)")
    ax.scatter(*second, s=120, c="#e6194b", marker="D", edgecolor="black",
               linewidth=1, zorder=10, label="path_00 wp1 (red sphere in Webots)")

    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{name}  preview  ({W:.1f}x{H:.1f} m, {len(walls):,} walls, {len(polys)} traces)\n"
                 f"Webots-frame check: dark walls should trace floor-plan rooms, "
                 f"red trajectories should walk between walls",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    out = out_dir / f"{name}_preview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[preview] -> {out}")


if __name__ == "__main__":
    main()
