"""Debug script: iterate on per-path overlay visualizations.

Two known issues from the user:
1. IMUWiFine paths have x/y scales that differ wildly (corridor floor) → equal
   aspect compresses the trajectory into a horizontal line.
2. MSILN: baseline predictions land ~50 m from GT on hard paths, blowing up
   the axis so the tight GT+Ours cluster is unreadable.

This script renders MULTIPLE variants of the figures so we can pick the best
approach before editing the notebook builder. PNGs go to _debug_pngs/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("_debug_pngs")
OUT.mkdir(exist_ok=True)

from src.pipeline.training import load_trained
from src.pipeline.baselines import (
    load_position_regressor, load_preprocessor,
    predict_imuwifine_msiln, load_imuwifine_msiln,
)
from src.pipeline.baselines._msiln_loader import (
    load_ap_vocab, TRAIN_PATHS as MSILN_TRAIN, TEST_PATHS as MSILN_TEST,
)
from src.pipeline.baselines.imuwifine import _MsilnWindowDataset
from src.pipeline.baselines._msiln_loader import load_msiln_paths_for_imuwifine

print("Loading Ours-MSILN + IMUWiFine baseline ckpts...", flush=True)
trainer_msiln = load_trained("runs/main_table/msiln_site1_b1/transformer",
                             arch="transformer", dataset="msiln_site1_b1")
iwf_model, _, _ = load_imuwifine_msiln("runs/main_table/msiln_site1_b1/imuwifine/model.pt")
trainer_iwfine = load_trained("runs/main_table/imuwifine/transformer",
                              arch="transformer", dataset="imuwifine")

# Per-path slicing for Ours-MSILN
gt_rows = trainer_msiln.dm.test_ds._gt_rows
path_ids_msiln = np.array([r["path_id"] for r in gt_rows])
pred_all, gt_all = trainer_msiln.predict(split="test")
pred_all = pred_all.cpu().numpy(); gt_all = gt_all.cpu().numpy()

ours_per_path_msiln = {}
for pid in sorted(set(path_ids_msiln.tolist())):
    mask = path_ids_msiln == pid
    ours_per_path_msiln[pid] = {"pred": pred_all[mask], "gt": gt_all[mask]}

# wlanloc per-path
print("Running wlanloc per-path on MSILN...", flush=True)
PositionRegressor = load_position_regressor()
DataPreprocessor  = load_preprocessor()
apv = load_ap_vocab()
rssi_cols = [f"wifi_rssi_{m}" for m in apv.keys()]
MSILN_ROOT = Path("data/msiln_site1_b1")

def load_path_wifi(pid):
    pdir = MSILN_ROOT / f"path_{pid:02d}"
    wifi = pd.read_csv(pdir / "wifi.csv"); gt = pd.read_csv(pdir / "ground_truth.csv")
    for c in rssi_cols:
        if c not in wifi.columns: wifi[c] = np.nan
    X = wifi[rssi_cols].values.astype(np.float64)
    X = np.where(np.isnan(X), 100.0, X)
    t_w = wifi["sim_time"].values.astype(np.float64)
    t_g = gt["sim_time"].values.astype(np.float64)
    xy = np.stack([np.interp(t_w, t_g, gt["gt_x"].values),
                   np.interp(t_w, t_g, gt["gt_y"].values)], axis=1)
    return X, xy.astype(np.float32)

Xtr = np.vstack([load_path_wifi(p)[0] for p in MSILN_TRAIN])
Ytr = np.vstack([load_path_wifi(p)[1] for p in MSILN_TRAIN])
pre = DataPreprocessor(); Xtr_pp = pre.fit_transform(Xtr)
reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
reg.fit_location(0, 0, Xtr_pp, Ytr)

wlanloc_per_path = {}
for pid in MSILN_TEST:
    X, gt = load_path_wifi(pid)
    pred = reg.models[(0, 0)].predict(pre.transform(X)).astype(np.float32)
    wlanloc_per_path[pid] = {"pred": pred, "gt": gt}

# IMUWiFine per-path
print("Running IMUWiFine baseline per-path on MSILN...", flush=True)
iwf_per_path_msiln = {}
for pid in MSILN_TEST:
    paths_one, _ = load_msiln_paths_for_imuwifine([pid], target_hz=10.0)
    if not paths_one: continue
    ds_one = _MsilnWindowDataset(paths_one, window=30, stride=30)
    preds, gts = [], []
    iwf_model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    iwf_model.to(dev)
    with torch.no_grad():
        for i in range(len(ds_one)):
            x, y = ds_one[i]
            x = x.unsqueeze(0).to(dev)
            preds.append(iwf_model(x).cpu().squeeze(0).numpy()); gts.append(y.numpy())
    iwf_per_path_msiln[pid] = {"pred": np.concatenate(preds, axis=0),
                                "gt": np.concatenate(gts, axis=0)}


def msiln_path_overlay(pid, variant: str, out_path: Path):
    """Try a specific visualization strategy."""
    ours = ours_per_path_msiln[pid]; wlan = wlanloc_per_path[pid]; iwf = iwf_per_path_msiln[pid]
    fig, ax = plt.subplots(figsize=(9, 7))

    if variant == "v1_full_scatter":
        # Original: scatter everything, auto axis. Baselines blow up the view.
        ax.scatter(ours["gt"][:, 0],   ours["gt"][:, 1],   s=14, alpha=0.55, label="GT",        color="#1f77b4")
        ax.scatter(wlan["pred"][:, 0], wlan["pred"][:, 1], s=14, alpha=0.55, label="wlanloc",   color="#2ca02c")
        ax.scatter(iwf["pred"][:, 0],  iwf["pred"][:, 1],  s=14, alpha=0.55, label="IMUWiFine", color="#9467bd")
        ax.scatter(ours["pred"][:, 0], ours["pred"][:, 1], s=14, alpha=0.55, label="Ours",      color="#d62728")
        ax.set_aspect("equal")

    elif variant == "v2_lines_full":
        # GT + Ours as connected lines (time-ordered); baselines as scatter.
        ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.2, alpha=0.9, label="GT",   color="#1f77b4")
        ax.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.6, alpha=0.85, label="Ours", color="#d62728")
        ax.scatter(wlan["pred"][:, 0], wlan["pred"][:, 1], s=36, alpha=0.7, marker="x",
                   label="wlanloc",   color="#2ca02c")
        ax.scatter(iwf["pred"][:, 0],  iwf["pred"][:, 1],  s=36, alpha=0.7, marker="^",
                   label="IMUWiFine", color="#9467bd")
        ax.set_aspect("equal")

    elif variant == "v3_lines_tight_axis":
        # Same as v2 but axis limits = GT bbox + 5 m margin (baselines outside get clipped).
        ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.2, alpha=0.9, label="GT",   color="#1f77b4")
        ax.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.6, alpha=0.85, label="Ours", color="#d62728")
        ax.scatter(wlan["pred"][:, 0], wlan["pred"][:, 1], s=36, alpha=0.7, marker="x",
                   label="wlanloc",   color="#2ca02c")
        ax.scatter(iwf["pred"][:, 0],  iwf["pred"][:, 1],  s=36, alpha=0.7, marker="^",
                   label="IMUWiFine", color="#9467bd")
        gtx, gty = ours["gt"][:, 0], ours["gt"][:, 1]
        margin = 5.0
        ax.set_xlim(gtx.min() - margin, gtx.max() + margin)
        ax.set_ylim(gty.min() - margin, gty.max() + margin)
        ax.set_aspect("equal")

    elif variant == "v4_two_panel":
        # Two-panel: left = tight view (GT+Ours only); right = full view with baselines.
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        a = axes[0]
        a.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.2, alpha=0.9, label="GT",   color="#1f77b4")
        a.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.6, alpha=0.85, label="Ours", color="#d62728")
        a.set_aspect("equal"); a.legend(loc="best"); a.set_title(f"path_{pid} — GT vs Ours (tight)")
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")

        b = axes[1]
        b.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.2, alpha=0.9, label="GT",   color="#1f77b4")
        b.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.6, alpha=0.85, label="Ours", color="#d62728")
        b.scatter(wlan["pred"][:, 0], wlan["pred"][:, 1], s=36, alpha=0.7, marker="x",
                  label="wlanloc",   color="#2ca02c")
        b.scatter(iwf["pred"][:, 0],  iwf["pred"][:, 1],  s=36, alpha=0.7, marker="^",
                  label="IMUWiFine", color="#9467bd")
        b.set_aspect("equal"); b.legend(loc="best"); b.set_title(f"path_{pid} — full view")
        b.set_xlabel("x (m)"); b.set_ylabel("y (m)")
        plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
        return

    ax.legend(loc="best"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"MSILN path_{pid} — {variant}")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


# IMUWiFine per-path (Ours only)
gt_rows_iwf = trainer_iwfine.dm.test_ds._gt_rows
path_ids_iwf = np.array([r["path_id"] for r in gt_rows_iwf])
pred_iwf_all, gt_iwf_all = trainer_iwfine.predict(split="test")
pred_iwf_all = pred_iwf_all.cpu().numpy(); gt_iwf_all = gt_iwf_all.cpu().numpy()

ours_per_path_iwf = {}
for pid in sorted(set(path_ids_iwf.tolist())):
    mask = path_ids_iwf == pid
    ours_per_path_iwf[pid] = {"pred": pred_iwf_all[mask], "gt": gt_iwf_all[mask]}


def iwfine_path_overlay(pid, variant: str, out_path: Path):
    ours = ours_per_path_iwf[pid]
    fig, ax = plt.subplots(figsize=(9, 5))
    if variant == "v1_equal":
        ax.scatter(ours["gt"][:, 0],   ours["gt"][:, 1],   s=14, alpha=0.55, label="GT",   color="#1f77b4")
        ax.scatter(ours["pred"][:, 0], ours["pred"][:, 1], s=14, alpha=0.55, label="Ours", color="#d62728")
        ax.set_aspect("equal")
    elif variant == "v2_auto":
        ax.scatter(ours["gt"][:, 0],   ours["gt"][:, 1],   s=14, alpha=0.55, label="GT",   color="#1f77b4")
        ax.scatter(ours["pred"][:, 0], ours["pred"][:, 1], s=14, alpha=0.55, label="Ours", color="#d62728")
        ax.set_aspect("auto")
    elif variant == "v3_lines_auto":
        ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.0, alpha=0.9, label="GT",   color="#1f77b4")
        ax.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.5, alpha=0.85, label="Ours", color="#d62728")
        ax.set_aspect("auto")
    elif variant == "v4_lines_equal_datalim":
        # "datalim" — keep aspect 1:1 by stretching axes, not data
        ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.0, alpha=0.9, label="GT",   color="#1f77b4")
        ax.plot(ours["pred"][:, 0], ours["pred"][:, 1], lw=1.5, alpha=0.85, label="Ours", color="#d62728")
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    gtx, gty = ours["gt"][:, 0], ours["gt"][:, 1]
    xr, yr = float(gtx.max() - gtx.min()), float(gty.max() - gty.min())
    ax.set_title(f"IMUWiFine path_{pid} — {variant}  (x_range={xr:.1f} m, y_range={yr:.1f} m, ratio={xr/max(yr,1e-3):.1f})")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def msiln_path_overlay_v5(pid, out_path: Path):
    """v5 (proposed final): GT as line; Ours+baselines as scatter; axis clipped to GT bbox + 5m."""
    ours = ours_per_path_msiln[pid]; wlan = wlanloc_per_path[pid]; iwf = iwf_per_path_msiln[pid]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.5, alpha=0.9, label="GT", color="#1f77b4")
    ax.scatter(wlan["pred"][:, 0], wlan["pred"][:, 1], s=50, alpha=0.7, marker="x",
               label="wlanloc",   color="#2ca02c")
    ax.scatter(iwf["pred"][:, 0],  iwf["pred"][:, 1],  s=50, alpha=0.7, marker="^",
               label="IMUWiFine", color="#9467bd")
    ax.scatter(ours["pred"][:, 0], ours["pred"][:, 1], s=12, alpha=0.5, label="Ours", color="#d62728")
    gtx, gty = ours["gt"][:, 0], ours["gt"][:, 1]
    margin = 5.0
    ax.set_xlim(gtx.min() - margin, gtx.max() + margin)
    ax.set_ylim(gty.min() - margin, gty.max() + margin)
    ax.set_aspect("equal")
    ax.legend(loc="best"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    mae_ours = float(np.linalg.norm(ours["pred"] - ours["gt"], axis=1).mean())
    ax.set_title(f"MSILN path_{pid} (Ours MAE {mae_ours:.2f} m)")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


def iwfine_path_overlay_v5(pid, out_path: Path):
    """v5: scatter Ours + GT as connected line; aspect=auto; y axis padded to min 1m height."""
    ours = ours_per_path_iwf[pid]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ours["gt"][:, 0],   ours["gt"][:, 1],   lw=2.2, alpha=0.9, label="GT", color="#1f77b4")
    ax.scatter(ours["pred"][:, 0], ours["pred"][:, 1], s=14, alpha=0.55, label="Ours", color="#d62728")
    gtx, gty = ours["gt"][:, 0], ours["gt"][:, 1]
    xm = 1.0; ax.set_xlim(gtx.min() - xm, gtx.max() + xm)
    yc = (gty.min() + gty.max()) / 2; yh = max(gty.max() - gty.min(), 1.0) * 1.5
    ax.set_ylim(yc - yh / 2, yc + yh / 2)
    ax.set_aspect("auto")
    ax.legend(loc="best"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    mae_ours = float(np.linalg.norm(ours["pred"] - ours["gt"], axis=1).mean())
    ax.set_title(f"IMUWiFine path_{pid} (Ours MAE {mae_ours:.2f} m)")
    plt.tight_layout(); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()


# Try 3 MSILN paths × 4 variants
MSILN_PATHS_TO_TRY = [128, 131, 132]
MSILN_VARIANTS = ["v1_full_scatter", "v2_lines_full", "v3_lines_tight_axis", "v4_two_panel"]
for pid in MSILN_PATHS_TO_TRY:
    for v in MSILN_VARIANTS:
        out = OUT / f"msiln_path{pid}_{v}.png"
        msiln_path_overlay(pid, v, out)
        print(f"  wrote {out}", flush=True)

# Try 3 IMUWiFine paths × 4 variants
# Sort paths by Ours MAE; pick top-2 (tightest) + 1 middling
mae_per_path_iwf = {pid: float(np.linalg.norm(d["pred"] - d["gt"], axis=1).mean())
                     for pid, d in ours_per_path_iwf.items()}
sorted_pids = sorted(mae_per_path_iwf, key=mae_per_path_iwf.get)
IWF_PATHS_TO_TRY = [sorted_pids[0], sorted_pids[1], sorted_pids[len(sorted_pids) // 2]]
print(f"\nIMUWiFine: trying paths {IWF_PATHS_TO_TRY} (MAEs: "
       f"{[mae_per_path_iwf[p] for p in IWF_PATHS_TO_TRY]})", flush=True)
IWF_VARIANTS = ["v1_equal", "v2_auto", "v3_lines_auto", "v4_lines_equal_datalim"]
for pid in IWF_PATHS_TO_TRY:
    for v in IWF_VARIANTS:
        out = OUT / f"iwfine_path{pid}_{v}.png"
        iwfine_path_overlay(pid, v, out)
        print(f"  wrote {out}", flush=True)

# v5 final candidates
print("\nRendering v5 candidates...", flush=True)
for pid in MSILN_PATHS_TO_TRY:
    out = OUT / f"msiln_path{pid}_v5_FINAL.png"
    msiln_path_overlay_v5(pid, out)
    print(f"  wrote {out}", flush=True)
for pid in IWF_PATHS_TO_TRY:
    out = OUT / f"iwfine_path{pid}_v5_FINAL.png"
    iwfine_path_overlay_v5(pid, out)
    print(f"  wrote {out}", flush=True)

# Helpful debug info — print path bboxes
print("\nMSILN per-path GT bboxes + baseline span:")
for pid in MSILN_TEST:
    g = ours_per_path_msiln[pid]["gt"]
    w = wlanloc_per_path[pid]["pred"]; i = iwf_per_path_msiln[pid]["pred"]
    print(f"  path_{pid}: GT x=[{g[:,0].min():6.1f}, {g[:,0].max():6.1f}], y=[{g[:,1].min():6.1f}, {g[:,1].max():6.1f}]  "
          f"wlanloc x=[{w[:,0].min():6.1f}, {w[:,0].max():6.1f}]  "
          f"IMUWiFine x=[{i[:,0].min():6.1f}, {i[:,0].max():6.1f}]")

print("\nIMUWiFine per-path GT bboxes:")
for pid in IWF_PATHS_TO_TRY:
    g = ours_per_path_iwf[pid]["gt"]
    xr = float(g[:, 0].max() - g[:, 0].min()); yr = float(g[:, 1].max() - g[:, 1].min())
    print(f"  path_{pid}: x_range={xr:.1f} m, y_range={yr:.1f} m, ratio={xr/max(yr,1e-3):.2f}")
