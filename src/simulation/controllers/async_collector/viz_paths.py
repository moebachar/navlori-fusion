"""
NavLoRI — Visualize all 30 paths from paths.json overlaid on the wall map.
Generates:
  - all_paths_overview.png  (all 30 paths on one figure)
  - paths_grid.png          (6x5 grid, one path per cell)
"""

import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_FILE = os.path.join(HERE, "paths.json")
OUT_DIR = os.path.join(HERE, "viz")

# ── Wall geometry (from Tiago++'s world.wbt) ────────────────────────────────
ROBOT_RADIUS = 0.30
SAFETY_COEFF = 1.5
SAFE_CLEARANCE = ROBOT_RADIUS * SAFETY_COEFF

RAW_WALLS = [
    (1.48, -1.47, 2.2327, 0.2, 2.7),
    (1.27, -3.35, -2.5454, 0.2, 2.707),
    (-0.61, -6.52, 2.2301, 0.2, 6.9709),
    (-10.33, -5.40, 2.15, 0.2, 7.5),
    (-2.16, -10.77, -2.6475, 0.2, 5.0),
    (1.10, -11.53, -1.0, 0.2, 5.1671),
    (3.99, -11.34, -2.6168, 0.2, 3.0),
    (-1.06, -16.10, 2.1194, 0.2, 13.8),
    (-8.01, -18.25, 0.6777, 0.2, 1.9),
    (-6.55, -16.13, -1.0151, 0.2, 5.1559),
    (-5.62, -12.74, 0.5114, 0.2, 4.6),
    (-8.43, -11.76, 2.1214, 0.2, 3.8606),
    (-12.77, -8.61, 0.5831, 0.2, 2.9),
    (-10.79, -11.60, 0.5831, 0.2, 2.7),
    (-6.30, -4.47, -2.482, 0.2, 3.1004),
    (-4.05, -4.58, -0.8897, 0.2, 3.6),
    (-3.72, -1.98, 0.5912, 0.2, 3.4447),
    (-2.75, 1.41, -0.8125, 0.2, 6.0),
    (0.84, 1.52, -2.5178, 0.2, 5.3),
]


def wall_polygons(inflate=0.0):
    """Return list of (4,2) arrays representing wall rectangles."""
    polys = []
    for tx, ty, angle, sx, sy in RAW_WALLS:
        hx = sx / 2 + inflate
        hy = sy / 2 + inflate
        corners_local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        ca, sa = math.cos(angle), math.sin(angle)
        corners = [(tx + ca * cx - sa * cy, ty + sa * cx + ca * cy)
                    for cx, cy in corners_local]
        polys.append(np.array(corners))
    return polys


def draw_walls(ax, inflate=0.0, color="#888888", alpha=0.5, label=None):
    """Draw wall rectangles on an axes."""
    for i, poly in enumerate(wall_polygons(inflate)):
        p = plt.Polygon(poly, closed=True, facecolor=color, edgecolor="black",
                        linewidth=0.5, alpha=alpha,
                        label=label if i == 0 else None)
        ax.add_patch(p)


def draw_safety_zone(ax):
    """Draw inflated walls (safety margin) as translucent overlay."""
    for poly in wall_polygons(SAFE_CLEARANCE):
        p = plt.Polygon(poly, closed=True, facecolor="#ff0000",
                        edgecolor="none", alpha=0.08)
        ax.add_patch(p)


def load_paths():
    with open(PATHS_FILE) as f:
        data = json.load(f)
    paths = []
    for p in data["paths"]:
        wps = [(w["x"], w["y"]) for w in p["waypoints"]]
        paths.append({"id": p["id"], "waypoints": wps,
                       "length": p.get("length_m", 0)})
    return paths, data["nodes"]


def set_map_limits(ax):
    ax.set_xlim(-14.5, 5.5)
    ax.set_ylim(-20, 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f0")
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ── Generate colormap for 30 paths ──────────────────────────────────────────
def path_colors(n=30):
    cmap = plt.cm.get_cmap("tab20", 20)
    cmap2 = plt.cm.get_cmap("tab20b", 20)
    colors = []
    for i in range(n):
        if i < 20:
            colors.append(cmap(i))
        else:
            colors.append(cmap2(i - 20))
    return colors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths, nodes = load_paths()
    colors = path_colors(len(paths))

    # ════════════════════════════════════════════════════════════════════
    # 1. Overview: all 30 paths on one figure
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 18))
    draw_safety_zone(ax)
    draw_walls(ax, inflate=0.0, color="#666666", alpha=0.7, label="Walls")

    # Draw all nodes
    nx = [n["x"] for n in nodes]
    ny = [n["y"] for n in nodes]
    ax.scatter(nx, ny, s=12, c="black", zorder=5, alpha=0.4, label=f"Nodes ({len(nodes)})")

    for p, c in zip(paths, colors):
        wps = p["waypoints"]
        xs = [w[0] for w in wps]
        ys = [w[1] for w in wps]
        ax.plot(xs, ys, color=c, linewidth=1.2, alpha=0.7)
        ax.scatter(xs[0], ys[0], color=c, s=40, marker="o", zorder=6,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(xs[-1], ys[-1], color=c, s=40, marker="s", zorder=6,
                   edgecolors="black", linewidths=0.5)

    set_map_limits(ax)
    ax.set_title("NavLoRI — All 30 Paths Overview", fontsize=16, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    overview_path = os.path.join(OUT_DIR, "all_paths_overview.png")
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {overview_path}")

    # ════════════════════════════════════════════════════════════════════
    # 2. Grid: 6 rows x 5 cols, one path per cell
    # ════════════════════════════════════════════════════════════════════
    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 28))
    fig.suptitle("NavLoRI — Individual Paths (30)", fontsize=18, fontweight="bold", y=0.995)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if idx < len(paths):
            p = paths[idx]
            wps = p["waypoints"]
            xs = [w[0] for w in wps]
            ys = [w[1] for w in wps]

            draw_walls(ax, inflate=0.0, color="#999999", alpha=0.5)

            ax.plot(xs, ys, color=colors[idx], linewidth=2, alpha=0.9, zorder=4)

            # Waypoint markers
            ax.scatter(xs, ys, c="white", s=20, zorder=5,
                       edgecolors=colors[idx], linewidths=1)
            # Start (green) and end (red)
            ax.scatter(xs[0], ys[0], c="#00cc44", s=60, marker="o", zorder=6,
                       edgecolors="black", linewidths=0.8, label="Start")
            ax.scatter(xs[-1], ys[-1], c="#ff3333", s=60, marker="s", zorder=6,
                       edgecolors="black", linewidths=0.8, label="End")

            # Direction arrows at midpoints
            for i in range(len(xs) - 1):
                mx = (xs[i] + xs[i + 1]) / 2
                my = (ys[i] + ys[i + 1]) / 2
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                norm = math.hypot(dx, dy)
                if norm > 0.5:
                    ax.annotate("", xy=(mx + dx * 0.15 / norm, my + dy * 0.15 / norm),
                                xytext=(mx, my),
                                arrowprops=dict(arrowstyle="->", color=colors[idx],
                                                lw=1.5))

            set_map_limits(ax)
            length = p["length"]
            ax.set_title(f"Path {p['id']}  ({len(wps)} wp, {length:.1f}m)",
                         fontsize=9, fontweight="bold")
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    grid_path = os.path.join(OUT_DIR, "paths_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {grid_path}")

    # ════════════════════════════════════════════════════════════════════
    # 3. Stats summary
    # ════════════════════════════════════════════════════════════════════
    lengths = [p["length"] for p in paths]
    wpcounts = [len(p["waypoints"]) for p in paths]
    print(f"\n{'='*50}")
    print(f"  30 Paths Summary")
    print(f"{'='*50}")
    print(f"  Total distance:  {sum(lengths):.1f} m")
    print(f"  Avg path length: {np.mean(lengths):.1f} m  (min={min(lengths):.1f}, max={max(lengths):.1f})")
    print(f"  Avg waypoints:   {np.mean(wpcounts):.1f}  (min={min(wpcounts)}, max={max(wpcounts)})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
