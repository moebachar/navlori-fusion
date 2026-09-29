"""Classical dead reckoning on side_golden (12 real TurtleBot3 runs), val + test paths.

Same protocol as the RoNIN eval (start from the first GT pose, integrate, never re-anchor):
  * wheel odometry  — the TurtleBot3's own /odom pose (diff-drive wheel encoders + its IMU yaw),
                      rigidly moved so its first pose equals the first GT pose (x, y, yaw).
  * wheel + gyro    — speed from odom_linear_vel, heading = GT yaw(0) + integral of gyro_z,
                      gyro bias taken from the first 3 s (the robot stands still at every run start).
Error = Euclidean distance to GT at every GT timestamp (~10 Hz).

Run: ``.venv/Scripts/python.exe scripts/_eval_odom_dr_side_golden.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.pipeline.fusion.builder import load_config  # noqa: E402

DATASET = "side_golden"
OUT_DIR = ROOT / "runs" / "side_golden" / "odom_dr"
STILL_S = 3.0


def wheel_odom(od: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    th_o = np.radians(od.odom_theta_deg.values)
    dth = np.radians(gt.gt_yaw_deg.values[0]) - th_o[0]
    c, s = np.cos(dth), np.sin(dth)
    dx, dy = od.odom_x.values - od.odom_x.values[0], od.odom_y.values - od.odom_y.values[0]
    x = gt.gt_x.values[0] + c * dx - s * dy
    y = gt.gt_y.values[0] + s * dx + c * dy
    t = gt.sim_time.values
    return np.stack([np.interp(t, od.sim_time.values, x), np.interp(t, od.sim_time.values, y)], 1)


def wheel_gyro(od: pd.DataFrame, imu: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    ti, wz = imu.sim_time.values, imu.gyro_z.values
    bias = wz[ti < STILL_S].mean()
    th = np.radians(gt.gt_yaw_deg.values[0]) + np.concatenate([[0], np.cumsum((wz[1:] - bias) * np.diff(ti))])
    to = od.sim_time.values
    th_o = np.interp(to, ti, th)
    v = od.odom_linear_vel.values
    dt = np.diff(to, prepend=to[0])
    x = gt.gt_x.values[0] + np.cumsum(v * np.cos(th_o) * dt)
    y = gt.gt_y.values[0] + np.cumsum(v * np.sin(th_o) * dt)
    t = gt.sim_time.values
    return np.stack([np.interp(t, to, x), np.interp(t, to, y)], 1)


def stats(e: np.ndarray) -> dict:
    return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)),
                p90=float(np.percentile(e, 90)), max=float(e.max()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(DATASET)
    root = ROOT / str(cfg.dataset.root) / cfg.dataset.collection_dir
    out = {"method": {"wheel_odom": "TurtleBot3 /odom pose, aligned to GT pose at t=0",
                      "wheel_gyro": "odom speed + gyro-integrated heading (bias from first 3 s), from GT pose at t=0"}}
    traj = {}
    for split in ["val", "test"]:
        pids = list(getattr(cfg.dataset.split, f"{split}_paths"))
        errs = {"wheel_odom": [], "wheel_gyro": []}
        per_path = {"wheel_odom": {}, "wheel_gyro": {}}
        for pid in pids:
            pdir = root / f"path_{pid:02d}"
            gt = pd.read_csv(pdir / "ground_truth.csv")
            od = pd.read_csv(pdir / "odometry.csv")
            imu = pd.read_csv(pdir / "imu.csv")
            gxy = gt[["gt_x", "gt_y"]].values
            for name, pred in [("wheel_odom", wheel_odom(od, gt)), ("wheel_gyro", wheel_gyro(od, imu, gt))]:
                e = np.linalg.norm(pred - gxy, axis=1)
                errs[name].append(e)
                per_path[name][pid] = float(e.mean())
                traj[f"{name}_path_{pid:02d}"] = pred
            traj[f"gt_path_{pid:02d}"] = gxy
            traj[f"t_path_{pid:02d}"] = gt.sim_time.values
        out[split] = {}
        for name in errs:
            s = stats(np.concatenate(errs[name]))
            s["per_path"] = per_path[name]
            out[split][name] = s
            print(f"{split:4s} {name:10s}: mean {s['mean']:.3f} m  median {s['median']:.3f}  p90 {s['p90']:.3f}  "
                  f"max {s['max']:.3f}  per-path {', '.join(f'{p}:{v:.2f}' for p, v in per_path[name].items())}",
                  flush=True)
    (OUT_DIR / "odom_dr_side_golden.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(OUT_DIR / "trajectories.npz", **traj)
    print(f"\nwrote {OUT_DIR / 'odom_dr_side_golden.json'}", flush=True)


if __name__ == "__main__":
    main()
