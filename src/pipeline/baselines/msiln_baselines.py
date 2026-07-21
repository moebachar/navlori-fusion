"""Two non-learned MSILN baselines for paper Table 2 (PLAN_40).

(1) WiFi-kNN cross-session: k-nearest-neighbor in WiFi RSSI feature space.
    Cross-session train -> val/test under the locked MSILN site1/B1 splits.
    Matches the WiFi-kNN reference cited in scope.md §7.3 (RESULT_15
    anchor 9.47 m test).

(2) PDR-from-first-waypoint: pedestrian dead reckoning using the MIT-
    licensed indoor_location_competition_20's step-detection algorithm,
    anchored only at the trajectory start. Demonstrates how pure inertial
    dead-reckoning drifts on MSILN cross-session test — a real
    upper-bound baseline that uses NO WiFi at test time.

The competition repo's ``compute_step_positions(acce, ahrs, posi_datas)``
expects per-step waypoint anchors throughout the trajectory — that is a
calibration / offline-mapping routine, NOT a test-time localization. We
use only ``compute_steps`` + ``compute_stride_length`` + ``compute_headings``
+ ``compute_rel_positions`` and anchor at the first waypoint, giving an
honest PDR-only baseline.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from ._msiln_loader import (
    MSILN_ROOT, TRAIN_PATHS, VAL_PATHS, TEST_PATHS,
    load_ap_vocab, load_msiln_paths_for_competition,
)
from ._paths import EXTERNAL_METHODS


# ---------------------------------------------------------------------------
# (1) WiFi-kNN cross-session baseline
# ---------------------------------------------------------------------------

def _load_msiln_wifi_features(path_ids, ap_vocab):
    """For each path, return per-scan (X_rssi, Y_xy) by joining wifi.csv
    to nearest GT timestamp."""
    rssi_cols = [f"wifi_rssi_{mac}" for mac in ap_vocab.keys()]
    Xs, Ys = [], []
    for pid in path_ids:
        pdir = MSILN_ROOT / f"path_{pid:02d}"
        if not pdir.is_dir():
            continue
        wifi = pd.read_csv(pdir / "wifi.csv")
        gt   = pd.read_csv(pdir / "ground_truth.csv")
        if wifi.empty:
            continue
        for c in rssi_cols:
            if c not in wifi.columns:
                wifi[c] = np.nan
        X = wifi[rssi_cols].values.astype(np.float32)
        X = np.where(np.isnan(X), -100.0, X)
        gt_t = gt["sim_time"].values.astype(np.float64)
        wifi_t = wifi["sim_time"].values.astype(np.float64)
        xy = np.stack([np.interp(wifi_t, gt_t, gt["gt_x"].values),
                        np.interp(wifi_t, gt_t, gt["gt_y"].values)], axis=1)
        Xs.append(X); Ys.append(xy.astype(np.float32))
    if not Xs:
        return None, None
    return np.vstack(Xs), np.vstack(Ys)


def run_wifi_knn_msiln(verbose: bool = True, return_predictions: bool = False) -> dict:
    """Cross-session WiFi-kNN on MSILN site1/B1.

    Uses ``sklearn.neighbors.KNeighborsRegressor`` with default settings:
    ``n_neighbors=5``, Euclidean distance, uniform weights. This is the
    generic "WiFi-kNN" baseline as commonly reported in the indoor-
    localization literature (no bespoke per-paper tuning).

    With ``return_predictions=True`` the result dict gains ``val_pred``,
    ``val_gt``, ``test_pred``, ``test_gt`` ndarrays for plotting.
    """
    from sklearn.neighbors import KNeighborsRegressor

    ap_vocab = load_ap_vocab()
    if verbose:
        print("[WiFi-kNN] Loading MSILN train/val/test WiFi scans...", flush=True)
    Xtr, Ytr = _load_msiln_wifi_features(TRAIN_PATHS, ap_vocab)
    Xva, Yva = _load_msiln_wifi_features(VAL_PATHS, ap_vocab)
    Xte, Yte = _load_msiln_wifi_features(TEST_PATHS, ap_vocab)

    if verbose:
        print(f"[WiFi-kNN] train={len(Xtr)} val={len(Xva)} test={len(Xte)} scans  "
              f"({Xtr.shape[1]} APs)", flush=True)

    knn = KNeighborsRegressor()  # defaults: k=5, Euclidean, uniform weights
    knn.fit(Xtr, Ytr)

    out = {"k": knn.n_neighbors, "metric": str(knn.metric),
            "weights": knn.weights, "n_train_scans": int(len(Xtr))}
    for split, Xq, Yq in [("val", Xva, Yva), ("test", Xte, Yte)]:
        pred = knn.predict(Xq).astype(np.float32)
        errs = np.linalg.norm(pred - Yq, axis=1)
        out[f"{split}_mae"]    = float(errs.mean())
        out[f"{split}_median"] = float(np.median(errs))
        out[f"{split}_n"]      = int(len(errs))
        if return_predictions:
            out[f"{split}_pred"] = pred
            out[f"{split}_gt"]   = Yq.astype(np.float32)
        if verbose:
            print(f"[WiFi-kNN] {split:>4s}: MAE {out[f'{split}_mae']:.3f}m  "
                  f"median {out[f'{split}_median']:.3f}m  n={out[f'{split}_n']}",
                   flush=True)
    return out


def run_wlanloc_msiln(verbose: bool = True) -> dict:
    """wlan_localization on MSILN site1/B1 cross-session — predictions + MAE.

    Runs live (no caching). Used to obtain test-set predictions for the
    GT-vs-baselines overlay figure. Reuses the official preprocessor +
    global k=3 manhattan distance-weighted PositionRegressor.
    """
    from src.pipeline.baselines import load_position_regressor, load_preprocessor

    PositionRegressor = load_position_regressor()
    DataPreprocessor  = load_preprocessor()

    ap_vocab = load_ap_vocab()
    if verbose:
        print("[wlanloc] Loading MSILN train/val/test WiFi scans...", flush=True)
    Xtr, Ytr = _load_msiln_wifi_features(TRAIN_PATHS, ap_vocab)
    Xva, Yva = _load_msiln_wifi_features(VAL_PATHS, ap_vocab)
    Xte, Yte = _load_msiln_wifi_features(TEST_PATHS, ap_vocab)

    # Convert MSILN -100 sentinel to wlanloc's 100 sentinel
    Xtr = np.where(Xtr == -100.0, 100.0, Xtr).astype(np.float64)
    Xva = np.where(Xva == -100.0, 100.0, Xva).astype(np.float64)
    Xte = np.where(Xte == -100.0, 100.0, Xte).astype(np.float64)

    pre = DataPreprocessor()
    Xtr_pp = pre.fit_transform(Xtr)
    Xva_pp = pre.transform(Xva)
    Xte_pp = pre.transform(Xte)

    reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
    reg.fit_location(0, 0, Xtr_pp, Ytr.astype(np.float64))

    out = {}
    for split, Xq, Yq in [("val", Xva_pp, Yva), ("test", Xte_pp, Yte)]:
        pred = reg.models[(0, 0)].predict(Xq).astype(np.float32)
        errs = np.linalg.norm(pred - Yq, axis=1)
        out[f"{split}_mae"] = float(errs.mean())
        out[f"{split}_n"]   = int(len(errs))
        out[f"{split}_pred"] = pred
        out[f"{split}_gt"]   = Yq.astype(np.float32)
        if verbose:
            print(f"[wlanloc] {split:>4s}: MAE {out[f'{split}_mae']:.3f}m  n={out[f'{split}_n']}",
                   flush=True)
    return out


# ---------------------------------------------------------------------------
# (2) PDR-from-first-waypoint baseline (MIT shim, no edits to vendored source)
# ---------------------------------------------------------------------------

def _load_competition_compute_f():
    """Load the MIT-licensed competition compute_f module via importlib shim."""
    repo = EXTERNAL_METHODS / "indoor_location_competition_20"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(
        "_msiln_compute_f", repo / "compute_f.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pdr_from_start_msiln(verbose: bool = True) -> dict:
    """PDR-only baseline: anchor at first GT waypoint, dead-reckon to end.

    Uses the competition's step-detection (compute_steps), stride-length
    (Weinberg), and heading (rotation-vector) functions. No WiFi at test
    time. Drift accumulates over the trajectory length. This is the "what
    if we only had IMU + a known start" upper bound.
    """
    cf = _load_competition_compute_f()
    paths = load_msiln_paths_for_competition(TEST_PATHS + VAL_PATHS)
    out_split = {"val": [], "test": []}

    if verbose:
        print(f"[PDR-from-start] Loaded {len(paths)} paths (val+test)", flush=True)

    for p in paths:
        pid = p["path_id"]
        acce = p["acce"]
        ahrs = p["ahrs"]
        waypoints = p["waypoints"]
        if len(waypoints) < 1:
            continue

        try:
            step_ts, step_idx, step_acce_mm = cf.compute_steps(acce)
            if len(step_ts) < 2:
                continue
            headings    = cf.compute_headings(ahrs)
            stride_lens = cf.compute_stride_length(step_acce_mm)
            step_head   = cf.compute_step_heading(step_ts, headings)
            rel_pos     = cf.compute_rel_positions(stride_lens, step_head)
        except Exception as e:
            if verbose:
                print(f"[PDR-from-start] path_{pid:02d} skipped: {type(e).__name__}: {e}",
                       flush=True)
            continue

        # Integrate rel_pos from first waypoint (anchor only at start)
        start_xy = waypoints[0, 1:3]
        # rel_pos columns: [timestamp, dx, dy]
        traj = np.cumsum(rel_pos[:, 1:3], axis=0) + start_xy
        traj_t = rel_pos[:, 0]

        # Match each step prediction to the nearest GT waypoint in time → error
        gt_t = waypoints[:, 0]
        gt_xy = waypoints[:, 1:3]
        errs = []
        for t, pred_xy in zip(traj_t, traj):
            gt_idx = np.argmin(np.abs(gt_t - t))
            errs.append(np.linalg.norm(pred_xy - gt_xy[gt_idx]))
        if not errs:
            continue
        split = "test" if pid in TEST_PATHS else "val"
        out_split[split].append({
            "path_id": pid,
            "n_steps": len(errs),
            "mae": float(np.mean(errs)),
            "final_drift": float(errs[-1]),
        })

    summary = {}
    for split, rows in out_split.items():
        if not rows:
            summary[f"{split}_mae"] = float("nan")
            summary[f"{split}_n_paths"] = 0
            continue
        all_errs = []
        for r in rows:
            all_errs.append(r["mae"])
        summary[f"{split}_mae"]      = float(np.mean(all_errs))
        summary[f"{split}_n_paths"]  = len(rows)
        if verbose:
            print(f"[PDR-from-start] {split:>4s}: mean per-path MAE {summary[f'{split}_mae']:.3f}m "
                  f"over {summary[f'{split}_n_paths']} paths", flush=True)
    summary["per_path"] = out_split
    return summary


__all__ = [
    "run_wifi_knn_msiln",
    "run_pdr_from_start_msiln",
]
