"""PLAN_15 Step 2 — wlan_localization on MSILN site1/B1 cross-session.

Mirrors `scripts/eval_wlanloc_uji.py` (restored RESULT_01) but
targeting MSILN cross-session train/val/test split per
`configs/data/msiln_site1_b1.yaml`. Uses the vendored
`PositionRegressor` from
`C:\\Users\\FabLab\\AppData\\Local\\Temp\\wlan_localization\\src`
loaded via the same `importlib` shim — Demand #3 honoured.

MSILN is single-site/single-floor so the cascade-oracle isn't
applicable; we use the global regression mode (apples-to-apples with
our learned fusion encoder).

Run: ``.venv/Scripts/python.exe scripts/_eval_wlanloc_msiln.py``
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "msiln_site1_b1"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_15"
WLANLOC_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src")

# Train = paths 0-93 (Nov 24 surveyor), val = 94-127 (Nov 25),
# test = 128-132 (Dec 5+6).
TRAIN_PATHS = list(range(0, 94))
VAL_PATHS = list(range(94, 128))
TEST_PATHS = list(range(128, 133))


def _stub_logger():
    base = types.ModuleType("wlan_localization")
    utils = types.ModuleType("wlan_localization.utils")
    logmod = types.ModuleType("wlan_localization.utils.logger")
    import logging
    logmod.get_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("wlan_localization", base)
    sys.modules.setdefault("wlan_localization.utils", utils)
    sys.modules.setdefault("wlan_localization.utils.logger", logmod)


def _load_pure(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, WLANLOC_SRC / "wlan_localization" / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_position_regressor():
    _stub_logger()
    return _load_pure("models/position_regressor.py", "wlan_pos_reg").PositionRegressor


def _load_preprocessor():
    _stub_logger()
    return _load_pure("data/preprocessor.py", "wlan_preproc").DataPreprocessor


def load_split_rssi(path_ids: list[int]):
    """Aggregate WiFi RSSI + GT across paths.

    For each path, join wifi.csv to ground_truth.csv on nearest-time
    (each WiFi scan gets the closest-in-time GT pose).
    """
    all_X, all_xy = [], []
    rssi_cols_master = None
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
            # Reindex this path's wifi onto the master AP set so all
            # samples share the same column order.
            for c in rssi_cols_master:
                if c not in wifi.columns:
                    wifi[c] = np.nan
        X = wifi[rssi_cols_master].values.astype(np.float64)
        # MSILN convention: NaN = AP not seen → fill -100 (wlan_localization's
        # not-detected sentinel is 100 = "out of range" but it accepts
        # any NaN-filling strategy).
        X = np.where(np.isnan(X), -100.0, X)
        # Match samples to nearest GT in time.
        gt_t = gt["sim_time"].values
        gt_xy = gt[["gt_x", "gt_y"]].values
        wifi_t = wifi["sim_time"].values
        idx = np.clip(np.searchsorted(gt_t, wifi_t), 0, len(gt_t) - 1)
        # Linear interp (between idx-1 and idx where applicable).
        xy = np.zeros((len(wifi_t), 2), dtype=np.float64)
        for i, t in enumerate(wifi_t):
            xy[i, 0] = np.interp(t, gt_t, gt_xy[:, 0])
            xy[i, 1] = np.interp(t, gt_t, gt_xy[:, 1])
        all_X.append(X)
        all_xy.append(xy)
    if not all_X:
        return np.zeros((0, 0)), np.zeros((0, 2))
    X = np.concatenate(all_X, axis=0)
    xy = np.concatenate(all_xy, axis=0)
    return X, xy


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PositionRegressor = _load_position_regressor()
    DataPreprocessor = _load_preprocessor()
    print("Loading MSILN site1/B1...", flush=True)
    Xtr_raw, Ytr = load_split_rssi(TRAIN_PATHS)
    Xva_raw, Yva = load_split_rssi(VAL_PATHS)
    Xte_raw, Yte = load_split_rssi(TEST_PATHS)
    print(f"  train {len(Xtr_raw)}  val {len(Xva_raw)}  test {len(Xte_raw)}  "
          f"APs {Xtr_raw.shape[1]}", flush=True)

    # wlan_localization's DataPreprocessor expects "not-detected = 100";
    # MSILN convention is NaN → -100. Re-translate so the preprocessor
    # behaves on its expected sentinel.
    not_detected_msiln = -100.0
    not_detected_wlanloc = 100.0
    Xtr_raw = np.where(Xtr_raw == not_detected_msiln, not_detected_wlanloc, Xtr_raw)
    Xva_raw = np.where(Xva_raw == not_detected_msiln, not_detected_wlanloc, Xva_raw)
    Xte_raw = np.where(Xte_raw == not_detected_msiln, not_detected_wlanloc, Xte_raw)

    pre = DataPreprocessor()
    print("Fitting wlan_localization preprocessor (Box-Cox + PCA)...", flush=True)
    t0 = time.time()
    try:
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)
        Xte = pre.transform(Xte_raw)
    except Exception as e:
        print(f"Preprocessor failed: {type(e).__name__}: {e}", flush=True)
        # Fall back to a minimal RSSI normalisation (no Box-Cox / PCA):
        # convert 100 (not-detected) to -100, then (x+100)/100.
        print("Falling back to plain RSSI affine (no preprocessor).", flush=True)
        for X in [Xtr_raw, Xva_raw, Xte_raw]:
            X[X == 100.0] = -100.0
        Xtr = (Xtr_raw + 100.0) / 100.0
        Xva = (Xva_raw + 100.0) / 100.0
        Xte = (Xte_raw + 100.0) / 100.0
    print(f"  preprocessor in dim {Xtr_raw.shape[1]} -> out dim {Xtr.shape[1]}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # MSILN is single-site → use global PositionRegressor (apples-to-apples
    # with our learned fusion).
    print("Training PositionRegressor (k=3, manhattan, distance)...", flush=True)
    reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
    t0 = time.time()
    reg.fit_location(0, 0, Xtr, Ytr)
    print(f"  fitted in {time.time()-t0:.1f}s", flush=True)

    # Evaluate on val + test.
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

    out["method"] = "wlan_localization PositionRegressor (k=3, manhattan, distance-weighted, MSILN cross-session)"
    out["preprocessor"] = "DataPreprocessor (vendored, fit on train)"
    with open(OUT_DIR / "wlanloc_msiln.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'wlanloc_msiln.json'}", flush=True)


if __name__ == "__main__":
    main()
