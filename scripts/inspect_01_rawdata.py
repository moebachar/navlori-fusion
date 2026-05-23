"""PROBE 1 — raw data integrity. No model, no assumptions.

For each dataset + split, walk the actual path CSVs and report:
  * GT: (x,y) extent, sample count, timestamp monotonicity, rate, NaN.
  * WiFi: scan count, rate, RSSI value range, missing encoding, APs-visible
    per scan (sparsity), fraction of all-missing scans.
  * IMU: rate, per-feature value range, NaN/inf.
  * Cross-modality timestamp span alignment.

Goal: catch garbage-in before blaming anything downstream.

Run: .venv/Scripts/python.exe scripts/inspect_01_rawdata.py [dataset ...]
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

DATASETS = ["simulation", "ipin2024_floor-2", "imuwifine", "ronin_a000"]


def _wifi_rssi_cols(df):
    return [c for c in df.columns if c.startswith("wifi_rssi_")]


def inspect_path(pdir: Path) -> dict:
    out = {"path": pdir.name}
    gt = pdir / "ground_truth.csv"
    if not gt.exists():
        return {**out, "error": "no ground_truth.csv"}
    g = pd.read_csv(gt)
    if len(g) == 0:
        return {**out, "error": "empty GT"}
    t = g["sim_time"].values
    out["gt_n"] = len(g)
    out["gt_dur_s"] = round(float(t[-1] - t[0]), 1)
    out["gt_rate_hz"] = round(len(g) / max(t[-1] - t[0], 1e-9), 1)
    out["gt_monotonic"] = bool(np.all(np.diff(t) >= 0))
    out["gt_x_range"] = [round(float(g["gt_x"].min()), 2), round(float(g["gt_x"].max()), 2)]
    out["gt_y_range"] = [round(float(g["gt_y"].min()), 2), round(float(g["gt_y"].max()), 2)]
    out["gt_nan"] = int(g[["gt_x", "gt_y"]].isna().sum().sum())
    # GT step size (per-tick displacement) — sanity on motion realism.
    d = np.linalg.norm(np.diff(g[["gt_x", "gt_y"]].values, axis=0), axis=1)
    out["gt_step_cm"] = [round(float(np.median(d) * 100), 1), round(float(d.max() * 100), 1)]

    wifi = pdir / "wifi.csv"
    if wifi.exists():
        w = pd.read_csv(wifi)
        cols = _wifi_rssi_cols(w)
        out["wifi_n"] = len(w)
        if len(w) > 0 and cols:
            wt = w["sim_time"].values
            out["wifi_rate_hz"] = round(len(w) / max(t[-1] - t[0], 1e-9), 2)
            vals = w[cols].values.astype(np.float64)
            finite = vals[np.isfinite(vals)]
            out["wifi_rssi_range"] = [round(float(finite.min()), 1), round(float(finite.max()), 1)] if finite.size else None
            out["wifi_nan_frac"] = round(float(np.isnan(vals).mean()), 3)
            # APs visible per scan = non-NaN entries per row
            vis = np.isfinite(vals).sum(axis=1)
            out["wifi_aps_per_scan"] = [int(vis.min()), int(np.median(vis)), int(vis.max())]
            out["wifi_n_aps_total"] = len(cols)
            out["wifi_empty_scans"] = int((vis == 0).sum())

    imu = pdir / "imu.csv"
    if imu.exists():
        im = pd.read_csv(imu)
        out["imu_n"] = len(im)
        if len(im) > 1:
            it = im["sim_time"].values
            out["imu_rate_hz"] = round(len(im) / max(it[-1] - it[0], 1e-9), 1)
            feat = [c for c in im.columns if c != "sim_time"]
            arr = im[feat].values.astype(np.float64)
            out["imu_nan"] = int(np.isnan(arr).sum())
            out["imu_inf"] = int(np.isinf(arr).sum())
            out["imu_feat_absmax"] = round(float(np.nanmax(np.abs(arr))), 1)
    return out


def inspect_dataset(name: str):
    print(f"\n{'='*90}\nDATASET: {name}\n{'='*90}")
    cfg = load_config(name)
    d = cfg.dataset
    root = ROOT / str(d.root) / d.collection_dir
    splits = {"train": list(d.split.train_paths),
              "val": list(d.split.val_paths),
              "test": list(d.split.test_paths)}
    for split, pids in splits.items():
        if not pids:
            continue
        recs = [inspect_path(root / f"path_{p:02d}") for p in pids]
        recs = [r for r in recs if "error" not in r]
        if not recs:
            print(f"  [{split}] no usable paths")
            continue
        # Aggregate
        gt_n = sum(r["gt_n"] for r in recs)
        xr = [min(r["gt_x_range"][0] for r in recs), max(r["gt_x_range"][1] for r in recs)]
        yr = [min(r["gt_y_range"][0] for r in recs), max(r["gt_y_range"][1] for r in recs)]
        gt_nan = sum(r["gt_nan"] for r in recs)
        mono = all(r["gt_monotonic"] for r in recs)
        rates_gt = np.mean([r["gt_rate_hz"] for r in recs])
        step_med = np.median([r["gt_step_cm"][0] for r in recs])
        step_max = max(r["gt_step_cm"][1] for r in recs)
        print(f"\n  [{split}]  {len(recs)} paths  {gt_n} GT samples")
        print(f"    GT  x={xr} y={yr}  extent={round(xr[1]-xr[0],1)}x{round(yr[1]-yr[0],1)}m"
              f"  rate~{rates_gt:.1f}Hz  mono={mono}  nan={gt_nan}")
        print(f"    GT step (cm/tick): median~{step_med:.1f}  max={step_max:.1f}")
        if "wifi_n" in recs[0]:
            wn = sum(r.get("wifi_n", 0) for r in recs)
            wrates = np.mean([r.get("wifi_rate_hz", 0) for r in recs])
            naps = recs[0].get("wifi_n_aps_total")
            nanfrac = np.mean([r.get("wifi_nan_frac", 0) for r in recs])
            vis_med = np.median([r.get("wifi_aps_per_scan", [0, 0, 0])[1] for r in recs])
            empties = sum(r.get("wifi_empty_scans", 0) for r in recs)
            rng = recs[0].get("wifi_rssi_range")
            print(f"    WiFi {wn} scans  rate~{wrates:.2f}Hz  {naps} APs  "
                  f"rssi_range={rng}  nan_frac={nanfrac:.2f}  "
                  f"APs_visible/scan(med)={vis_med:.0f}  empty_scans={empties}")
        if "imu_n" in recs[0]:
            inn = sum(r.get("imu_n", 0) for r in recs)
            irate = np.mean([r.get("imu_rate_hz", 0) for r in recs])
            inan = sum(r.get("imu_nan", 0) for r in recs)
            iinf = sum(r.get("imu_inf", 0) for r in recs)
            iabs = max(r.get("imu_feat_absmax", 0) for r in recs)
            print(f"    IMU  {inn} samples  rate~{irate:.1f}Hz  nan={inan}  inf={iinf}  absmax={iabs}")


def main():
    targets = sys.argv[1:] or DATASETS
    for ds in targets:
        try:
            inspect_dataset(ds)
        except Exception as e:
            print(f"  ERROR on {ds}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
