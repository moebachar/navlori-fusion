"""Render trajectory overlays on the 20 staged ILN 2.0 candidate floor plans.

For each floor_image.png in the inspection dir, finds the matching
site/floor in the dataset, reads every trace's TYPE_WAYPOINT polyline,
and draws them on top of the floor raster — so you can see at a glance
whether trajectories actually cover the corridors / rooms.

Coordinate alignment:
  - floor_info.json gives the floor extent in METRES (width, height)
    of the rectangle bounded by the floor image.
  - waypoints are in the SAME metric frame, origin at the floor-plan
    corner (per the handoff §6 — but watch the y-axis convention).
  - image pixels: (0,0) = top-left, +y goes DOWN. World metres: (0,0)
    typically = bottom-left, +y goes UP. So waypoint y is FLIPPED to
    match image y.
"""
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INSPECT = r"X:\navlori-fusion\data\iln20\inspection"
DATA = r"X:\navlori-fusion\data\iln20\data"

# parse `..._<site8>_<floor>.png` from the staged filenames
_RE = re.compile(r"_([0-9a-f]{8})_([^.]+)\.png$")


def find_site_floor(site8: str, floor: str) -> str | None:
    """Match the 8-char prefix back to the full site id."""
    for s in os.listdir(DATA):
        if s.startswith(site8):
            fd = os.path.join(DATA, s, floor)
            if os.path.isdir(fd):
                return fd
    return None


def load_waypoints(trace_path: str) -> np.ndarray:
    """Return (N, 2) array of (x, y) waypoints in metres, sorted by time."""
    pts = []
    with open(trace_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[1] == "TYPE_WAYPOINT":
                try:
                    pts.append((int(c[0]), float(c[2]), float(c[3])))
                except ValueError:
                    pass
    pts.sort()
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([(x, y) for _, x, y in pts], dtype=np.float32)


def render_one(png_name: str) -> str:
    m = _RE.search(png_name)
    if not m:
        return f"skip {png_name}: no site8/floor in name"
    site8, floor = m.group(1), m.group(2)
    fd = find_site_floor(site8, floor)
    if fd is None:
        return f"skip {png_name}: no site dir for {site8}/{floor}"

    info_p = os.path.join(fd, "floor_info.json")
    img_p = os.path.join(fd, "floor_image.png")
    pdir = os.path.join(fd, "path_data_files")
    if not (os.path.isfile(info_p) and os.path.isfile(img_p) and os.path.isdir(pdir)):
        return f"skip {png_name}: missing assets in {fd}"

    info = json.load(open(info_p))
    mi = info.get("map_info", info)
    width_m = float(mi["width"])
    height_m = float(mi["height"])

    img = Image.open(img_p)
    w_px, h_px = img.size

    # gather every trace's polyline
    polylines = []
    n_skipped = 0
    for fn in os.listdir(pdir):
        if not fn.endswith(".txt"):
            continue
        wp = load_waypoints(os.path.join(pdir, fn))
        if len(wp) >= 2:
            polylines.append(wp)
        else:
            n_skipped += 1

    if not polylines:
        return f"skip {png_name}: no usable traces"

    # plot
    fig, ax = plt.subplots(figsize=(10, 10 * h_px / w_px))
    # imshow with extent = METRIC frame, image y is flipped via origin="upper"
    ax.imshow(img, extent=[0, width_m, 0, height_m], origin="upper",
              interpolation="bilinear", aspect="equal")
    # waypoints: y is in the SAME metric frame; image is drawn top-down,
    # so to match the visual orientation we plot y as-is and rely on imshow's
    # origin="upper" + invert. Test both, pick the one that aligns: convention
    # in ILN 2.0 is waypoint y measured from BOTTOM-LEFT, so we flip y for
    # the imshow which puts pixel (0,0) at TOP-LEFT. Equivalent: invert_yaxis.
    for wp in polylines:
        # forward-flip: world y_bottom_left -> image y_top_left
        x = wp[:, 0]
        y = height_m - wp[:, 1]
        ax.plot(x, y, linewidth=0.6, alpha=0.35, color="#ff2a55")
        ax.scatter(x[0], y[0], s=8, c="#00cc44", zorder=4, alpha=0.6)
        ax.scatter(x[-1], y[-1], s=8, c="#e6194b", marker="s",
                    zorder=4, alpha=0.6)

    ax.set_xlim(0, width_m)
    ax.set_ylim(height_m, 0)  # keep image orientation (top-left origin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m, image-down)")
    ax.set_title(
        f"{site8}/{floor}  —  {len(polylines)} traces, "
        f"{width_m:.0f}×{height_m:.0f} m  ({width_m*height_m:,.0f} m²)",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)

    out = os.path.join(INSPECT, os.path.splitext(png_name)[0] + "_overlay.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"ok  {png_name}: {len(polylines)} traces (skipped {n_skipped})"


def main():
    names = sorted(
        f for f in os.listdir(INSPECT)
        if f.endswith(".png") and "_overlay" not in f
    )
    print(f"[overlay] rendering {len(names)} candidates", flush=True)
    for i, name in enumerate(names, 1):
        msg = render_one(name)
        print(f"[overlay] [{i:>2}/{len(names)}] {msg}", flush=True)
    print("[overlay] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
