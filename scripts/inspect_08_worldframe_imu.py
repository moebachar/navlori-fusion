"""M4 hypothesis test — does world-frame IMU carry more displacement signal?

The motion leg is weak (Probe 5: 12% skill on IPIN) — suspected cause is
body-frame accel: the same physical motion looks different depending on
device heading. Test cheaply (no training): build IMU windows, rotate the
horizontal accel into world frame via yaw, and compare kNN(IMU->1s disp)
skill for:
  RAW   — features as-is (body frame), what the pipeline uses
  WORLD — + yaw-rotated (ax,ay) world-frame accel channels
  WORLD-ONLY — only world-frame accel + gyro (drop raw body accel)

If WORLD >> RAW, a heading-aware encoder (M4) is worth building.

Run: .venv/Scripts/python.exe scripts/inspect_08_worldframe_imu.py [dataset]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

from src.pipeline.fusion.builder import load_config  # noqa: E402

IMU_COLS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
            "roll_deg", "pitch_deg", "yaw_deg"]


def build_windows(root, pids, win=32):
    """Return (windows list of (win, feat_raw), disp targets, valid)."""
    Xr, Y = [], []
    for p in pids:
        pdir = root / f"path_{p:02d}"
        gf, imf = pdir / "ground_truth.csv", pdir / "imu.csv"
        if not (gf.exists() and imf.exists()):
            continue
        g = pd.read_csv(gf)
        im = pd.read_csv(imf)
        if len(im) < win + 1 or len(g) < 2:
            continue
        it = im["sim_time"].values
        feats = im[IMU_COLS].values.astype(np.float64)
        gt_t = g["sim_time"].values
        gt_xy = g[["gt_x", "gt_y"]].values
        # one sample per GT row: IMU window ending at gt time, disp over 1s
        for k in range(len(gt_t)):
            t = gt_t[k]
            j = int(np.searchsorted(it, t, side="right") - 1)
            if j < win:
                continue
            # displacement over last 1s
            t0 = t - 1.0
            k0 = int(np.searchsorted(gt_t, t0, side="right") - 1)
            if k0 < 0:
                continue
            Xr.append(feats[j - win + 1:j + 1])
            Y.append(gt_xy[k] - gt_xy[k0])
    return Xr, np.array(Y)


def to_world(win):
    """Add yaw-rotated world-frame horizontal accel channels."""
    ax, ay = win[:, 0], win[:, 1]
    yaw = np.deg2rad(win[:, 8])
    axw = np.cos(yaw) * ax - np.sin(yaw) * ay
    ayw = np.sin(yaw) * ax + np.cos(yaw) * ay
    return axw, ayw


def feat_raw(wins):
    return np.array([w.reshape(-1) for w in wins])


def feat_world(wins):
    out = []
    for w in wins:
        axw, ayw = to_world(w)
        out.append(np.concatenate([w.reshape(-1), axw, ayw]))
    return np.array(out)


def feat_world_only(wins):
    out = []
    for w in wins:
        axw, ayw = to_world(w)
        # world accel + gyro (cols 3,4,5), drop body accel + raw orientation
        out.append(np.concatenate([axw, ayw, w[:, 3:6].reshape(-1)]))
    return np.array(out)


def skill(ftr, ytr, fva, yva):
    # z-score features (per-dim) so kNN isn't dominated by scale
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-8
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance").fit((ftr - mu) / sd, ytr)
    pred = knn.predict((fva - mu) / sd)
    err = np.linalg.norm(pred - yva, axis=1).mean()
    zero = np.linalg.norm(yva, axis=1).mean()
    return err, 1 - err / zero


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ipin2024_floor-2"
    cfg = load_config(name)
    d = cfg.dataset
    root = ROOT / str(d.root) / d.collection_dir
    print(f"=== {name} — world-frame IMU hypothesis ===")
    wtr, ytr = build_windows(root, list(d.split.train_paths))
    wva, yva = build_windows(root, list(d.split.val_paths))
    print(f"  train={len(wtr)} val={len(wva)} windows; mean|disp|={np.linalg.norm(yva,axis=1).mean():.3f}m")
    for label, fn in [("RAW (body)", feat_raw), ("WORLD (+yaw accel)", feat_world),
                      ("WORLD-ONLY", feat_world_only)]:
        err, sk = skill(fn(wtr), ytr, fn(wva), yva)
        print(f"  {label:20s}  disp MAE={err:.3f}m  skill={sk*100:.0f}%")


if __name__ == "__main__":
    main()
