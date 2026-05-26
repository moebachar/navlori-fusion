"""PLAN_19 Step 1a — wlan_localization on IMUWiFine floor 4 (NEW measurement).

Clone of `_eval_wlanloc_msiln.py` (RESULT_15 template), retargeted at
`data/imuwifine_floor4/` with the train/val/test split from
`configs/data/imuwifine.yaml`:
  train = paths 0-39, val = paths 40-59, test = paths 60-79.

Uses the vendored PositionRegressor + DataPreprocessor from
``C:\\Users\\FabLab\\AppData\\Local\\Temp\\wlan_localization\\src`` via
`importlib` (Demand #3 honoured; no vendored edits).

Run: ``.venv/Scripts/python.exe scripts/_eval_wlanloc_imuwifine.py``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "imuwifine_floor4"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_19"

from src.pipeline.baselines import load_position_regressor, load_preprocessor  # noqa: E402

TRAIN_PATHS = list(range(0, 40))
VAL_PATHS = list(range(40, 60))
TEST_PATHS = list(range(60, 80))


def load_split_rssi(path_ids: list[int], master_cols=None):
    all_X, all_xy = [], []
    rssi_cols_master = master_cols
    for pid in path_ids:
        pdir = DATA / f"path_{pid:02d}"
        if not pdir.exists():
            continue
        wifi = pd.read_csv(pdir / "wifi.csv")
        gt = pd.read_csv(pdir / "ground_truth.csv")
        rssi_cols = [c for c in wifi.columns if c.startswith("wifi_rssi_")]
        if rssi_cols_master is None:
            rssi_cols_master = rssi_cols
        else:
            for c in rssi_cols_master:
                if c not in wifi.columns:
                    wifi[c] = np.nan
        X = wifi[rssi_cols_master].values.astype(np.float64)
        X = np.where(np.isnan(X), -100.0, X)
        gt_t = gt["sim_time"].values
        gt_xy = gt[["gt_x", "gt_y"]].values
        wifi_t = wifi["sim_time"].values
        xy = np.zeros((len(wifi_t), 2), dtype=np.float64)
        for i, t in enumerate(wifi_t):
            xy[i, 0] = np.interp(t, gt_t, gt_xy[:, 0])
            xy[i, 1] = np.interp(t, gt_t, gt_xy[:, 1])
        all_X.append(X)
        all_xy.append(xy)
    if not all_X:
        return np.zeros((0, 0)), np.zeros((0, 2)), rssi_cols_master
    X = np.concatenate(all_X, axis=0)
    xy = np.concatenate(all_xy, axis=0)
    return X, xy, rssi_cols_master


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PositionRegressor = load_position_regressor()
    DataPreprocessor = load_preprocessor()
    print("Loading IMUWiFine floor 4...", flush=True)
    Xtr_raw, Ytr, master = load_split_rssi(TRAIN_PATHS)
    Xva_raw, Yva, _ = load_split_rssi(VAL_PATHS, master)
    Xte_raw, Yte, _ = load_split_rssi(TEST_PATHS, master)
    print(f"  train {len(Xtr_raw)}  val {len(Xva_raw)}  test {len(Xte_raw)}  "
          f"APs {Xtr_raw.shape[1]}", flush=True)

    not_detected_in = -100.0
    not_detected_wlanloc = 100.0
    Xtr_raw = np.where(Xtr_raw == not_detected_in, not_detected_wlanloc, Xtr_raw)
    Xva_raw = np.where(Xva_raw == not_detected_in, not_detected_wlanloc, Xva_raw)
    Xte_raw = np.where(Xte_raw == not_detected_in, not_detected_wlanloc, Xte_raw)

    pre = DataPreprocessor()
    print("Fitting wlan_localization preprocessor (Box-Cox + PCA)...", flush=True)
    t0 = time.time()
    try:
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)
        Xte = pre.transform(Xte_raw)
    except Exception as e:
        print(f"Preprocessor failed: {type(e).__name__}: {e}", flush=True)
        print("Falling back to plain RSSI affine (no preprocessor).", flush=True)
        for X in [Xtr_raw, Xva_raw, Xte_raw]:
            X[X == 100.0] = -100.0
        Xtr = (Xtr_raw + 100.0) / 100.0
        Xva = (Xva_raw + 100.0) / 100.0
        Xte = (Xte_raw + 100.0) / 100.0
    print(f"  preprocessor in dim {Xtr_raw.shape[1]} -> out dim {Xtr.shape[1]}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    print("Training PositionRegressor (k=3, manhattan, distance)...", flush=True)
    reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
    t0 = time.time()
    reg.fit_location(0, 0, Xtr, Ytr)
    print(f"  fitted in {time.time()-t0:.1f}s", flush=True)

    out = {}
    for split, X, Y in [("val", Xva, Yva), ("test", Xte, Yte)]:
        preds = reg.models[(0, 0)].predict(X)
        errs = np.sqrt(((preds - Y) ** 2).sum(1))
        mean = float(errs.mean())
        out[split] = {
            "n": int(len(errs)),
            "mean": mean,
            "median": float(np.median(errs)),
            "p25": float(np.percentile(errs, 25)),
            "p75": float(np.percentile(errs, 75)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()),
        }
        print(f"{split}: mean Euclidean = {mean:.3f} m  "
              f"median={out[split]['median']:.3f}  p90={out[split]['p90']:.3f}  "
              f"max={out[split]['max']:.3f}  (n={out[split]['n']})", flush=True)

    out["method"] = "wlan_localization PositionRegressor (k=3, manhattan, distance-weighted, IMUWiFine floor 4)"
    out["preprocessor"] = "DataPreprocessor (vendored, fit on train)"
    with open(OUT_DIR / "wlanloc_imuwifine.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'wlanloc_imuwifine.json'}", flush=True)


if __name__ == "__main__":
    main()
