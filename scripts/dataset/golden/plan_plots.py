#!/usr/bin/env python
# Golden-run paths drawn on the CAD floor plan (Plan_RDC_walls-only_cropped.jpg): one per run + all together.
import json, re, shutil, numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
ROOT = "/mnt/x/side_navlori"; DATA = ROOT + "/data"; FPD = DATA + "/golden_floorplan"; F = 4
tf = json.load(open(f"{FPD}/plan_transform.json")); S, R_, TX, TY = tf["px_per_m"], tf["rotation_rad"], tf["tx_px"], tf["ty_px"]
plan = Image.open(f"{ROOT}/{tf['image']}").convert("RGB"); W0, H0 = plan.size
plan = np.array(plan.resize((W0 // F, H0 // F), Image.LANCZOS))
def w2p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); c, s = np.cos(R_), np.sin(R_)
    return (TX + S * (c * x - s * y)) / F, (TY - S * (s * x + c * y)) / F
CAD = {int(k): (v["x"], v["y"]) for k, v in json.load(open(f"{DATA}/tags_ground_truth.json")).items()}
RUNS = [f"golden_run_{i}" for i in range(1, 13)]
INFO = {r: json.load(open(f"{DATA}/{r}/ground_truth/gt_info.json")) for r in RUNS}

def base(figsize=(24, 8.2)):
    fig, ax = plt.subplots(figsize=figsize); ax.imshow(plan, interpolation="antialiased"); ax.axis("off")
    x0, y0 = plan.shape[1] * 0.015, plan.shape[0] * 0.97; L = 5 * S / F               # 5 m scale bar
    ax.plot([x0, x0 + L], [y0, y0], "-", color="k", lw=4); ax.text(x0 + L / 2, y0 - 12, "5 m", ha="center", va="bottom", fontsize=12)
    return fig, ax
def tags(ax, seen=()):
    for t, (x, y) in CAD.items():
        u, v = w2p(x, y); s_ = t in seen
        ax.plot(u, v, "s", ms=10 if s_ else 5, mfc="gold" if s_ else "white", mec="k", mew=1.0 if s_ else 0.6, zorder=6)
        ax.annotate(str(t), (u, v), xytext=(4, 4), textcoords="offset points", fontsize=11 if s_ else 7,
                    fontweight="bold" if s_ else "normal", color="k" if s_ else "0.4", zorder=7)
def acc(i):
    if i["status"] == "ok":
        h = i.get("heldout_median_cm")
        return f"held-out tag error {h:.0f} cm median" if h == h and h is not None else "2 tags only (fit 1 cm)"
    return f"map-matched: {i['wall_within_10cm_pct']:.0f}% walls within 10 cm"

for run in RUNS:
    g = pd.read_csv(f"{DATA}/{run}/ground_truth/gt_pose.csv"); i = INFO[run]
    t = (g.t_ns.values - g.t_ns.values[0]) / 1e9; u, v = w2p(g.x.values, g.y.values)
    fig, ax = base(); tags(ax, i["used_tags"])
    sc = ax.scatter(u, v, c=t, cmap="viridis", s=5, zorder=4)
    ax.plot(u[0], v[0], "o", ms=14, mfc="lime", mec="k", zorder=8, label="start")
    ax.plot(u[-1], v[-1], "s", ms=13, mfc="red", mec="k", zorder=8, label="end")
    ax.legend(loc="upper right", fontsize=11); fig.colorbar(sc, ax=ax, fraction=0.015, pad=0.01, label="time since start (s)")
    ax.set_title(f"{run}  |  {t[-1]:.0f} s, {i['path_m']:.1f} m  |  tags {i['used_tags']}  |  {acc(i)}", fontsize=14)
    plt.tight_layout(); fig.savefig(f"{DATA}/{run}/ground_truth/gt_on_plan.png", dpi=110); plt.close(fig)
    # README: show the plan view first; method.md: list the new file
    rd = open(f"{DATA}/{run}/README.md").read()
    if "gt_on_plan.png" not in rd:
        rd = rd.replace("![path](ground_truth/gt_overview.png)",
                        "![path on the floor plan](ground_truth/gt_on_plan.png)\n\n(on the lidar-built map: `ground_truth/gt_overview.png`)")
        open(f"{DATA}/{run}/README.md", "w").write(rd)
    md = open(f"{DATA}/{run}/ground_truth/method.md").read()
    if "gt_on_plan.png" not in md:
        open(f"{DATA}/{run}/ground_truth/method.md", "w").write(md.replace("`gt_overview.png`", "`gt_on_plan.png` (CAD plan) · `gt_overview.png`"))
    print(run, "ok", flush=True)

fig, ax = base((26, 9))
COLS = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#9a6324","#000075","#808000","#800000","#469990"]
for k, run in enumerate(RUNS):
    g = pd.read_csv(f"{DATA}/{run}/ground_truth/gt_pose.csv"); u, v = w2p(g.x.values, g.y.values); col = COLS[k]
    ax.plot(u, v, "-", color=col, lw=2.6, zorder=4, label=f"{run} ({INFO[run]['path_m']:.0f} m)")
    ax.plot(u[0], v[0], "o", ms=9, mfc=col, mec="k", zorder=5)
tags(ax, seen={t for r in RUNS for t in INFO[r]["used_tags"]})
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6, fontsize=12, frameon=False)
ax.set_title(f"All 12 golden runs on the floor plan  |  {sum(INFO[r]['path_m'] for r in RUNS):.0f} m, "
             f"{sum(INFO[r]['duration_s'] for r in RUNS)/60:.1f} min  |  dot = start, gold = tags seen", fontsize=15)
fig.savefig(f"{FPD}/all_paths_on_plan.png", dpi=110, bbox_inches="tight"); plt.close(fig)
shutil.copy2(f"{FPD}/all_paths_on_plan.png", f"{ROOT}/gt_review/all_paths_on_plan.png"); print("PLAN_PLOTS_DONE")
