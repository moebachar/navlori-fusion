"""Evaluate DPVO predicted trajectories against ground truth.

For each path with a TUM-format trajectory (`saved_trajectories/path_XX.txt`):
  1. Read predicted (frame_index, x, y, z, qx, qy, qz, qw).
  2. Read GT (sim_time, gt_x, gt_y, gt_z, ...) from data/async_collection/path_XX/ground_truth.csv.
  3. Match every predicted frame to its closest GT row by per-path frame index
     (DPVO writes frame numbers as its timestamps; we map them back to camera
     frame indices, then to sim_time, then to nearest GT row).
  4. Solve Sim(3) Umeyama alignment (predicted -> GT, with scale).
  5. Apply alignment, compute:
       - ATE: RMSE of euclidean (x, y, z) error after alignment
       - per-frame (x, y) euclidean error    <- comparable to ACE+PnP
       - RPE: relative pose error (translation, rotation) over a fixed delta

Output: <run_dir>/dpvo_eval.json  + <run_dir>/dpvo_eval_frames.csv

Usage (from repo root):
    python scripts/eval_dpvo.py                        # newest dpvo_* run
    python scripts/eval_dpvo.py --run runs/dpvo_...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "async_collection"

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sim(3) Umeyama alignment
# ---------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve Sim(3) such that  dst ≈ s · R · src + t  (least squares).

    src, dst : (N, 3) float arrays of corresponding 3D points.
    Returns (s, R(3x3), t(3,)).

    Reference: Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE PAMI 1991.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n  # (3, 3)

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / sigma_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_tum_trajectory(path: Path) -> pd.DataFrame:
    """TUM format: time tx ty tz qx qy qz qw  (whitespace separated)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 8:
            continue
        rows.append([float(x) for x in toks[:8]])
    arr = np.array(rows)
    return pd.DataFrame(arr, columns=["t", "tx", "ty", "tz",
                                       "qx", "qy", "qz", "qw"])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) -> (3, 3). Hamilton convention, normalised."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z),  2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),      1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),      2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Per-path evaluation
# ---------------------------------------------------------------------------

def evaluate_path(traj_file: Path, pid: int) -> dict:
    pdir = DATA / f"path_{pid:02d}"
    cam_csv = pdir / "camera.csv"
    gt_csv = pdir / "ground_truth.csv"
    if not (cam_csv.exists() and gt_csv.exists()):
        return {"path_id": pid, "status": "missing", "n": 0}

    pred = read_tum_trajectory(traj_file)
    cam_df = pd.read_csv(cam_csv)
    gt_df = pd.read_csv(gt_csv).sort_values("sim_time").reset_index(drop=True)

    # DPVO writes per-image-index timestamps (0, 1, 2, ...). Map them to the
    # corresponding camera-frame sim_time via cam_df, then nearest GT row.
    cam_df = cam_df.sort_values("sim_time").reset_index(drop=True)
    if len(pred) > len(cam_df):
        # DPVO might output for every input frame even when stride>1 omitted —
        # we still want to clip safely.
        pred = pred.iloc[:len(cam_df)].reset_index(drop=True)

    # Match each predicted frame to its sim_time
    pred_sim_t = cam_df["sim_time"].iloc[:len(pred)].to_numpy()

    # For each pred frame, nearest GT row (search-sorted on sim_time)
    gt_t = gt_df["sim_time"].to_numpy()
    idx = np.searchsorted(gt_t, pred_sim_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = idx - 1
    right = idx
    pick = np.where(np.abs(gt_t[left] - pred_sim_t) < np.abs(gt_t[right] - pred_sim_t),
                    left, right)
    gt_xyz = gt_df.iloc[pick][["gt_x", "gt_y", "gt_z"]].to_numpy()

    pred_xyz = pred[["tx", "ty", "tz"]].to_numpy()

    # Sim(3) align predicted -> GT
    s, R, t = umeyama(pred_xyz, gt_xyz)
    pred_aligned = (s * (R @ pred_xyz.T)).T + t

    # Errors
    diff = pred_aligned - gt_xyz                # (N, 3)
    euclid_3d = np.linalg.norm(diff, axis=1)    # (N,)
    euclid_xy = np.linalg.norm(diff[:, :2], axis=1)  # (N,)
    ate_rmse = float(np.sqrt((euclid_3d ** 2).mean()))

    # RPE: pairwise translation error every k frames
    k = max(1, len(pred_aligned) // 30)         # ~30 RPE samples
    di = np.arange(k, len(pred_aligned))
    rel_pred = pred_aligned[di] - pred_aligned[di - k]
    rel_gt = gt_xyz[di] - gt_xyz[di - k]
    rpe_t = float(np.linalg.norm(rel_pred - rel_gt, axis=1).mean())

    return {
        "path_id": pid,
        "status": "ok",
        "n": int(len(pred)),
        "scale": s,
        "ate_rmse_3d": ate_rmse,
        "median_xy": float(np.median(euclid_xy)),
        "mean_xy":   float(np.mean(euclid_xy)),
        "p95_xy":    float(np.percentile(euclid_xy, 95)),
        "median_3d": float(np.median(euclid_3d)),
        "p95_3d":    float(np.percentile(euclid_3d, 95)),
        "rpe_t":     rpe_t,
        "_per_frame": [
            {"sim_time": float(pred_sim_t[i]),
             "pred_x_aligned": float(pred_aligned[i, 0]),
             "pred_y_aligned": float(pred_aligned[i, 1]),
             "gt_x": float(gt_xyz[i, 0]), "gt_y": float(gt_xyz[i, 1]),
             "euclid_xy": float(euclid_xy[i]),
             "euclid_3d": float(euclid_3d[i])}
            for i in range(len(pred_aligned))
        ],
    }


def _pick_run(override: str | None) -> Path:
    if override:
        return Path(override)
    runs = sorted((ROOT / "runs").glob("dpvo_*"))
    runs = [r for r in runs if (r / "saved_trajectories").exists()]
    if not runs:
        sys.exit("No dpvo_* run with saved_trajectories/ found. "
                 "Run scripts/run_dpvo_paths.py first.")
    return runs[-1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = _pick_run(args.run)
    print(f"run dir: {run_dir}")

    traj_dir = run_dir / "saved_trajectories"
    summary = {"per_path": {}, "overall": {}}
    csv_rows: list[dict] = []

    val_paths, test_paths = {2, 13, 14}, {15, 16, 17}
    for f in sorted(traj_dir.glob("path_*.txt")):
        pid = int(f.stem.split("_")[1])
        res = evaluate_path(f, pid)
        if res["status"] != "ok":
            print(f"  path {pid}: skipped ({res['status']})")
            continue
        per_frame = res.pop("_per_frame")
        split = ("val" if pid in val_paths
                 else "test" if pid in test_paths
                 else "other")
        res["split"] = split
        summary["per_path"][str(pid)] = res
        for r in per_frame:
            r["path_id"] = pid
            r["split"] = split
            csv_rows.append(r)
        print(f"  path {pid:>2} ({split}): n={res['n']:>4}  "
              f"scale={res['scale']:.3f}  "
              f"ATE={res['ate_rmse_3d']:.3f}m  "
              f"median(xy)={res['median_xy']:.3f}m  "
              f"p95(xy)={res['p95_xy']:.3f}m  "
              f"RPE_t={res['rpe_t']:.3f}m")

    # Overall pooled (xy and 3d) over all per-frame errors
    if csv_rows:
        xy = np.array([r["euclid_xy"] for r in csv_rows])
        d3 = np.array([r["euclid_3d"] for r in csv_rows])
        summary["overall"] = {
            "n": int(len(xy)),
            "median_xy": float(np.median(xy)),
            "mean_xy":   float(np.mean(xy)),
            "p95_xy":    float(np.percentile(xy, 95)),
            "median_3d": float(np.median(d3)),
            "p95_3d":    float(np.percentile(d3, 95)),
        }
        for split in ("val", "test"):
            sub = [r for r in csv_rows if r["split"] == split]
            if sub:
                xs = np.array([r["euclid_xy"] for r in sub])
                summary["overall"][f"{split}_n"] = int(len(xs))
                summary["overall"][f"{split}_median_xy"] = float(np.median(xs))
                summary["overall"][f"{split}_mean_xy"]   = float(np.mean(xs))
                summary["overall"][f"{split}_p95_xy"]    = float(np.percentile(xs, 95))

    out_json = run_dir / "dpvo_eval.json"
    out_csv  = run_dir / "dpvo_eval_frames.csv"
    out_json.write_text(json.dumps(summary, indent=2))
    if csv_rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print()
    print("=" * 64)
    if "median_xy" in summary["overall"]:
        o = summary["overall"]
        print(f"OVERALL (all paths):  median(xy)={o['median_xy']:.3f}m  "
              f"p95(xy)={o['p95_xy']:.3f}m  n={o['n']}")
        for split in ("val", "test"):
            if f"{split}_n" in o:
                print(f"   {split:5s}:  median(xy)={o[f'{split}_median_xy']:.3f}m  "
                      f"p95(xy)={o[f'{split}_p95_xy']:.3f}m  "
                      f"n={o[f'{split}_n']}")
    print(f"saved: {out_json}")
    if csv_rows:
        print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
