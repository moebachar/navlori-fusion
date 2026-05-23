"""PROBE 2 — the WiFi-staleness floor.

For each GT sample at time t, find the most recent real WiFi scan time
t_w <= t and measure:
  * staleness = t - t_w  (seconds the carried-forward scan is old)
  * lag_disp  = ||gt(t) - gt(t_w)||  (meters moved since that scan)

``lag_disp`` is the IRREDUCIBLE error of any WiFi-only predictor with the
current carry-forward scheme: even a perfect fingerprint that nails the
position at scan time is off by lag_disp at query time.

Also computes the "centroid floor": assign every GT sample the centroid of
all GT positions sharing its WiFi scan, then the MAE of that assignment —
the best a WiFi-only model can do given one scan maps to many positions.

Pure data, no model. Run:
  .venv/Scripts/python.exe scripts/inspect_02_wifi_staleness.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import load_config  # noqa: E402

DATASETS = ["simulation", "ipin2024_floor-2", "ronin_a000"]


def analyze_path(pdir: Path):
    g = pd.read_csv(pdir / "ground_truth.csv")
    w = pd.read_csv(pdir / "wifi.csv") if (pdir / "wifi.csv").exists() else None
    if w is None or len(w) == 0 or len(g) < 2:
        return None
    gt_t = g["sim_time"].values
    gt_xy = g[["gt_x", "gt_y"]].values
    w_t = np.sort(w["sim_time"].values)

    # For each GT time, index of most recent wifi scan <= t.
    idx = np.searchsorted(w_t, gt_t, side="right") - 1
    valid = idx >= 0
    if valid.sum() == 0:
        return None
    gt_t, gt_xy, idx = gt_t[valid], gt_xy[valid], idx[valid]
    scan_t = w_t[idx]

    staleness = gt_t - scan_t                       # seconds

    # lag_disp: distance from current pos to pos at scan time. Need GT at
    # scan time -> nearest GT row to scan_t.
    gt_all_t = g["sim_time"].values
    gt_all_xy = g[["gt_x", "gt_y"]].values
    j = np.searchsorted(gt_all_t, scan_t, side="right") - 1
    j = j.clip(0, len(gt_all_t) - 1)
    pos_at_scan = gt_all_xy[j]
    lag_disp = np.linalg.norm(gt_xy - pos_at_scan, axis=1)

    # centroid floor: group GT samples by their wifi scan idx, assign centroid.
    centroid_pred = np.zeros_like(gt_xy)
    for u in np.unique(idx):
        m = idx == u
        centroid_pred[m] = gt_xy[m].mean(axis=0)
    centroid_err = np.linalg.norm(gt_xy - centroid_pred, axis=1)

    # samples per scan
    _, counts = np.unique(idx, return_counts=True)
    return {
        "staleness": staleness, "lag_disp": lag_disp,
        "centroid_err": centroid_err, "samples_per_scan": counts,
    }


def analyze_dataset(name):
    print(f"\n{'='*80}\n{name}\n{'='*80}")
    cfg = load_config(name)
    d = cfg.dataset
    root = ROOT / str(d.root) / d.collection_dir
    for split in ("train", "val"):
        pids = list(getattr(d.split, f"{split}_paths"))
        if not pids:
            continue
        agg = {"staleness": [], "lag_disp": [], "centroid_err": [], "samples_per_scan": []}
        for p in pids:
            r = analyze_path(root / f"path_{p:02d}")
            if r is None:
                continue
            for k in agg:
                agg[k].append(r[k])
        if not agg["staleness"]:
            continue
        stale = np.concatenate(agg["staleness"])
        lag = np.concatenate(agg["lag_disp"])
        cent = np.concatenate(agg["centroid_err"])
        spc = np.concatenate(agg["samples_per_scan"])
        print(f"\n  [{split}]  {len(stale)} GT samples")
        print(f"    staleness (s):       median={np.median(stale):.2f}  "
              f"p90={np.percentile(stale,90):.2f}  max={stale.max():.2f}")
        print(f"    GT samples / scan:   median={np.median(spc):.0f}  "
              f"max={spc.max():.0f}")
        print(f"    lag_disp (m) = move since last scan:")
        print(f"        median={np.median(lag):.2f}  mean={lag.mean():.2f}  "
              f"p90={np.percentile(lag,90):.2f}  max={lag.max():.2f}")
        print(f"    >>> WiFi-only IRREDUCIBLE error floor (mean lag_disp) = "
              f"{lag.mean():.2f} m")
        print(f"    >>> centroid floor (best WiFi-only MAE w/ carry-forward) = "
              f"{cent.mean():.2f} m")


def main():
    for ds in (sys.argv[1:] or DATASETS):
        try:
            analyze_dataset(ds)
        except Exception as e:
            print(f"  ERROR {ds}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
