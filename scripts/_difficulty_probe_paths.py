"""Step 0a of PLAN_05 — difficulty-matched probe for Webots paths.

Computes per-path difficulty features (length, mean speed, mean
curvature, n_pairs at stride=5) for the val + test paths used in
RESULT_03's Camera audit. Then computes difficulty-normalised MAE
(MAE / path_length) and reports whether the test-beats-val finding
survives normalisation.

Run: ``.venv/Scripts/python.exe scripts/_difficulty_probe_paths.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "async_collection"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_05"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]


def difficulty_features(pid: int) -> dict:
    gt = pd.read_csv(DATA / f"path_{pid:02d}" / "ground_truth.csv")
    t = gt["sim_time"].values.astype(np.float64)
    x = gt["gt_x"].values.astype(np.float64)
    y = gt["gt_y"].values.astype(np.float64)
    head = gt["gt_heading_rad"].values.astype(np.float64)
    # Path length.
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    length = float(seg.sum())
    duration = float(t[-1] - t[0])
    mean_speed = length / max(duration, 1e-6)
    # Heading angular velocity (curvature proxy).
    dh = np.unwrap(head)
    dh_dt = np.diff(dh) / np.diff(t).clip(1e-6)
    mean_abs_omega = float(np.abs(dh_dt).mean())
    return {
        "path_id": pid,
        "length_m": length,
        "duration_s": duration,
        "mean_speed_m_s": mean_speed,
        "mean_abs_omega_rad_s": mean_abs_omega,
        "n_gt_samples": int(len(t)),
    }


def main():
    with open(ROOT / "runs" / "overnight" / "run2_iter_03" / "webots_dpvo.json") as f:
        rj = json.load(f)
    pa = rj["runs"]["P-A"]
    val_mae = pa["val_dist"]["per_path"]   # keys are stringified ints
    test_mae = pa["test_dist"]["per_path"]

    rows = []
    for split, paths, mae_d in [
        ("val", VAL_PATHS, val_mae),
        ("test", TEST_PATHS, test_mae),
    ]:
        for pid in paths:
            feat = difficulty_features(pid)
            mae = mae_d[str(pid)]["mean"]
            n_pairs = mae_d[str(pid)]["n_frames"]
            feat["split"] = split
            feat["mae_m"] = float(mae)
            feat["n_pairs_stride5"] = int(n_pairs)
            feat["mae_per_m"] = float(mae / max(feat["length_m"], 1e-6))
            feat["mae_per_curv"] = float(mae / max(feat["mean_abs_omega_rad_s"], 1e-6))
            rows.append(feat)

    out = {"per_path": rows}

    # Aggregate by split.
    for split in ("val", "test"):
        subset = [r for r in rows if r["split"] == split]
        out[f"{split}_aggregate"] = {
            "mean_mae": float(np.mean([r["mae_m"] for r in subset])),
            "mean_length": float(np.mean([r["length_m"] for r in subset])),
            "mean_speed": float(np.mean([r["mean_speed_m_s"] for r in subset])),
            "mean_omega": float(np.mean([r["mean_abs_omega_rad_s"] for r in subset])),
            "mean_mae_per_m": float(np.mean([r["mae_per_m"] for r in subset])),
        }

    gap_raw = (out["test_aggregate"]["mean_mae"]
                - out["val_aggregate"]["mean_mae"]) / max(out["val_aggregate"]["mean_mae"], 1e-6) * 100.0
    gap_norm = (out["test_aggregate"]["mean_mae_per_m"]
                 - out["val_aggregate"]["mean_mae_per_m"]) / max(out["val_aggregate"]["mean_mae_per_m"], 1e-6) * 100.0
    out["raw_gap_pct"] = float(gap_raw)
    out["difficulty_normalised_gap_pct"] = float(gap_norm)

    with open(OUT_DIR / "camera_difficulty_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    # Pretty-print.
    print(f"{'path':<6}{'split':<6}{'len(m)':>8}{'speed':>8}{'|omega|':>10}{'n_pairs':>8}"
          f"{'MAE':>8}{'MAE/m':>10}")
    for r in rows:
        print(f"{r['path_id']:<6}{r['split']:<6}"
              f"{r['length_m']:>8.2f}{r['mean_speed_m_s']:>8.3f}"
              f"{r['mean_abs_omega_rad_s']:>10.3f}{r['n_pairs_stride5']:>8}"
              f"{r['mae_m']:>8.3f}{r['mae_per_m']:>10.4f}")
    print()
    print(f"val   aggregate: MAE={out['val_aggregate']['mean_mae']:.3f}  "
          f"length={out['val_aggregate']['mean_length']:.2f}  "
          f"MAE/m={out['val_aggregate']['mean_mae_per_m']:.4f}")
    print(f"test  aggregate: MAE={out['test_aggregate']['mean_mae']:.3f}  "
          f"length={out['test_aggregate']['mean_length']:.2f}  "
          f"MAE/m={out['test_aggregate']['mean_mae_per_m']:.4f}")
    print(f"raw test-val gap         = {gap_raw:+.1f}%")
    print(f"difficulty-normalised gap = {gap_norm:+.1f}%")
    if gap_norm > 20.0:
        print(">> verdict: keep label fails difficulty-normalised gate (gap > 20 %)")
    elif gap_norm > 0.0:
        print(">> verdict: test harder per-m than val (gap > 0); raw test < val likely reflects easier total length, not transfer")
    else:
        print(">> verdict: even per-m, test ≤ val; original 'no overfit' transfer claim holds")


if __name__ == "__main__":
    main()
