"""Generate simple, presentation-ready plots for DPVO eval run.

Outputs to <run>/presentation/:
  01_ate_per_path.png   -- per-path ATE RMSE, colored by split
  02_trajectories.png   -- pred vs GT (top-down) for all paths
  03_val_vs_test.png    -- median / mean / p95 XY error, val vs test
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("runs/dpvo_20260427_055514")
OUT = RUN / "presentation"
OUT.mkdir(exist_ok=True)

with open(RUN / "dpvo_eval.json") as f:
    ev = json.load(f)
frames = pd.read_csv(RUN / "dpvo_eval_frames.csv")

VAL = "#2E86AB"
TEST = "#E63946"

# ---------- 01: ATE per path ----------
paths = sorted(ev["per_path"].keys(), key=int)
ate = [ev["per_path"][p]["ate_rmse_3d"] for p in paths]
splits = [ev["per_path"][p]["split"] for p in paths]
colors = [VAL if s == "val" else TEST for s in splits]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([f"path {p}" for p in paths], ate, color=colors, edgecolor="black")
for b, v in zip(bars, ate):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("ATE RMSE (m)")
ax.set_title("DPVO — ATE RMSE per path")
ax.grid(axis="y", alpha=0.3)
val_patch = plt.Rectangle((0, 0), 1, 1, color=VAL, label="val")
test_patch = plt.Rectangle((0, 0), 1, 1, color=TEST, label="test")
ax.legend(handles=[val_patch, test_patch], loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "01_ate_per_path.png", dpi=150)
plt.close()

# ---------- 02: trajectories ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, p in zip(axes.flat, paths):
    sub = frames[frames["path_id"] == int(p)]
    ax.plot(sub["gt_x"], sub["gt_y"], color="black", lw=2, label="GT")
    ax.plot(sub["pred_x_aligned"], sub["pred_y_aligned"],
            color=VAL if ev["per_path"][p]["split"] == "val" else TEST,
            lw=1.5, ls="--", label="DPVO")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(
        f"path {p} ({ev['per_path'][p]['split']})  "
        f"ATE={ev['per_path'][p]['ate_rmse_3d']:.2f} m"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="best")
fig.suptitle("DPVO predicted trajectory vs ground truth", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "02_trajectories.png", dpi=150)
plt.close()

# ---------- 03: val vs test overall ----------
ov = ev["overall"]
metrics = ["median", "mean", "p95"]
val_vals = [ov[f"val_{m}_xy"] for m in metrics]
test_vals = [ov[f"test_{m}_xy"] for m in metrics]

x = np.arange(len(metrics))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
b1 = ax.bar(x - w / 2, val_vals, w, color=VAL, label=f"val (n={ov['val_n']})", edgecolor="black")
b2 = ax.bar(x + w / 2, test_vals, w, color=TEST, label=f"test (n={ov['test_n']})", edgecolor="black")
for bars, vals in [(b1, val_vals), (b2, test_vals)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics])
ax.set_ylabel("XY error (m)")
ax.set_title("DPVO — val vs test (per-frame XY error)")
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_val_vs_test.png", dpi=150)
plt.close()

print(f"Wrote 3 plots to {OUT}")
