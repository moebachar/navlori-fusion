"""Generate a 4-panel orientation diagnostic for one floor — picks the
correct (image flip × waypoint y-flip) combination. Open the output PNG
and visually pick which panel shows the trajectories tracing the
corridors of the floor plan.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Use the most readable / smallest floor for the diagnostic
SITE = "5cd56b9be2acfd2d33b5f99e"   # div-straight, 6,635 m², 19 traces
FLOOR = "F1"
OUT = r"X:\navlori-fusion\data\iln20\inspection\_ORIENTATION_DIAGNOSTIC.png"
DATA = r"X:\navlori-fusion\data\iln20\data"


def load_waypoints(p):
    pts = []
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4 and c[1] == "TYPE_WAYPOINT":
                try:
                    pts.append((int(c[0]), float(c[2]), float(c[3])))
                except ValueError:
                    pass
    pts.sort()
    return np.asarray([(x, y) for _, x, y in pts], dtype=np.float32) if pts else None


def main():
    fd = os.path.join(DATA, SITE, FLOOR)
    info = json.load(open(os.path.join(fd, "floor_info.json")))
    mi = info.get("map_info", info)
    W, H = float(mi["width"]), float(mi["height"])
    img = Image.open(os.path.join(fd, "floor_image.png"))

    polylines = []
    pdir = os.path.join(fd, "path_data_files")
    for fn in sorted(os.listdir(pdir)):
        if fn.endswith(".txt"):
            wp = load_waypoints(os.path.join(pdir, fn))
            if wp is not None and len(wp) >= 2:
                polylines.append(wp)

    # 4 panels = 2 (image origin) × 2 (waypoint y-flip)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    cfgs = [
        # (image_origin, waypoint y-flip)  →  panel title
        ("upper", False, "A:  imshow origin='upper', y NOT flipped\n(image upright, waypoints as-is)"),
        ("upper", True,  "B:  imshow origin='upper', y FLIPPED (H - y)\n(if waypoint origin is top-left)"),
        ("lower", False, "C:  imshow origin='lower', y NOT flipped\n(image upside-down vs file, waypoints as-is)"),
        ("lower", True,  "D:  imshow origin='lower', y FLIPPED (H - y)"),
    ]
    for ax, (orig, flip, title) in zip(axes.flat, cfgs):
        ax.imshow(img, extent=[0, W, 0, H], origin=orig,
                  interpolation="bilinear", aspect="equal")
        for wp in polylines:
            x = wp[:, 0]
            y = (H - wp[:, 1]) if flip else wp[:, 1]
            ax.plot(x, y, linewidth=0.6, alpha=0.5, color="#ff2a55")
            ax.scatter(x[0], y[0], s=15, c="#00cc44", zorder=4)
            ax.scatter(x[-1], y[-1], s=15, c="#e6194b", marker="s", zorder=4)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)   # default y-up axes
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Orientation diagnostic — {SITE[:12]}…/{FLOOR}  "
                 f"({W:.0f}×{H:.0f} m, {len(polylines)} traces)\n"
                 f"Pick the panel where red trajectories TRACE THE CORRIDORS of the building.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
