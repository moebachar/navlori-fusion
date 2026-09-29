"""Convert the side_navlori golden runs (real TurtleBot3, 2026-09-24) to navlori-fusion format.

Source: ``<side_root>/data/golden_run_N/`` (N = 1..12), packaged by the side_navlori
ground-truth pipeline (lidar SLAM + gyro heading, placed by an AprilTag network).
Output: ``data/side_golden/path_{N-1:02d}/`` with the standard CSVs:

- ground_truth.csv : sim_time, gt_x, gt_y (+ gt_yaw_deg, extra column), ~10 Hz (one pose per lidar scan)
- imu.csv          : 50 Hz, accel (m/s^2), gyro (rad/s), roll/pitch/yaw (deg) from the IMU quaternion
- odometry.csv     : 15 Hz, TurtleBot /odom pose + twist, wheel speeds in m/s (joint_states velocity is
                     already linear wheel speed despite the source column name ``*_vel_radps``)
- wifi.csv         : one row per scan (scan end time), APs of the train-run vocabulary, NaN = not seen
- camera.csv       : 5 Hz (every 6th 30 Hz frame, copied to camera/rgb_NNNNNN.jpg)

``sim_time`` is seconds from the first ground-truth pose of the run; every sensor is cropped to the GT span.

Run: ``.venv/Scripts/python.exe scripts/convert_side_golden.py``
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMU_HZ = 50
ODOM_HZ = 15
CAM_EVERY = 6                 # 30 Hz -> 5 Hz
WIFI_MAX_AGE_MS = 4000        # drop cached BSSIDs older than this within a scan
MIN_SCANS_IN_TRAIN = 3        # AP vocabulary: BSSIDs seen in >= this many train scans
RUNS = list(range(1, 13))
SPLIT = {"train": [1, 4, 5, 7, 9, 10, 11], "val": [3, 8], "test": [2, 6, 12]}   # golden run numbers


def quat_to_rpy_deg(qx, qy, qz, qw):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def resample(t_src, cols, t_dst, angle_cols=()):
    """Linear interpolation of each column onto t_dst; angle columns (rad) are unwrapped first."""
    t_src, idx = np.unique(t_src, return_index=True)
    out = {}
    for name, v in cols.items():
        v = np.asarray(v, float)[idx]
        if name in angle_cols:
            out[name] = np.angle(np.exp(1j * np.interp(t_dst, t_src, np.unwrap(v))))
        else:
            out[name] = np.interp(t_dst, t_src, v)
    return out


def load_wifi(run_dir: Path) -> pd.DataFrame:
    w = pd.read_csv(run_dir / "wifi" / "wifi.csv")
    return w[w.last_seen_ms <= WIFI_MAX_AGE_MS]


def convert_run(run: int, src: Path, dst: Path, vocab: list[str], camera: bool) -> dict:
    rd = src / f"golden_run_{run}"
    dst.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(rd / "ground_truth" / "gt_pose.csv")
    t0, t1 = int(gt.t_ns.iloc[0]), int(gt.t_ns.iloc[-1])
    sec = lambda t_ns: (np.asarray(t_ns, np.int64) - t0) / 1e9
    T_end = (t1 - t0) / 1e9
    pd.DataFrame({"sim_time": np.round(sec(gt.t_ns), 4), "gt_x": np.round(gt.x, 4), "gt_y": np.round(gt.y, 4),
                  "gt_yaw_deg": np.round(np.degrees(gt.yaw), 3)}).to_csv(dst / "ground_truth.csv", index=False)

    imu = pd.read_csv(rd / "imu" / "imu.csv")
    ti = np.arange(0.0, T_end + 1e-9, 1.0 / IMU_HZ)
    r, p, y = quat_to_rpy_deg(imu.qx.values, imu.qy.values, imu.qz.values, imu.qw.values)
    rs = resample(sec(imu.t_ns), dict(accel_x=imu.ax, accel_y=imu.ay, accel_z=imu.az, gyro_x=imu.wx, gyro_y=imu.wy,
                                      gyro_z=imu.wz, roll=np.radians(r), pitch=np.radians(p), yaw=np.radians(y)),
                  ti, angle_cols=("roll", "pitch", "yaw"))
    imu_df = pd.DataFrame({"sim_time": np.round(ti, 4), **{k: rs[k] for k in
                           ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]},
                           "roll_deg": np.degrees(rs["roll"]), "pitch_deg": np.degrees(rs["pitch"]),
                           "yaw_deg": np.degrees(rs["yaw"])})
    imu_df.to_csv(dst / "imu.csv", index=False, float_format="%.6f")

    od = pd.read_csv(rd / "wheel_odom" / "odom.csv")
    js = pd.read_csv(rd / "wheel_odom" / "joint_states.csv")
    to = np.arange(0.0, T_end + 1e-9, 1.0 / ODOM_HZ)
    o = resample(sec(od.t_ns), dict(x=od.x, y=od.y, yaw=od.yaw, v=od.v_lin, w=od.w_ang), to, angle_cols=("yaw",))
    j = resample(sec(js.t_ns), dict(l=js.left_vel_radps, r=js.right_vel_radps), to)
    odom_df = pd.DataFrame({"sim_time": np.round(to, 4), "odom_x": o["x"], "odom_y": o["y"],
                            "odom_theta_deg": np.degrees(o["yaw"]), "odom_linear_vel": o["v"], "odom_angular_vel": o["w"],
                            "wheel_left_vel": j["l"], "wheel_right_vel": j["r"]})
    odom_df.to_csv(dst / "odometry.csv", index=False, float_format="%.6f")

    w = load_wifi(rd)
    col = {b: i for i, b in enumerate(vocab)}
    rows = []
    for _, s in w.groupby("scan_idx"):
        ts = sec(s.t_end_ns.iloc[0])
        if ts < 0 or ts > T_end:
            continue
        rssi = np.full(len(vocab), np.nan)
        for b, v in zip(s.bssid, s.rssi_dbm):
            if b in col and not (rssi[col[b]] >= v):
                rssi[col[b]] = v
        seen = ~np.isnan(rssi)
        if not seen.any():
            continue
        k = int(np.nanargmax(rssi))
        rows.append([round(float(ts), 3), int(seen.sum()), float(rssi[k]), vocab[k].replace(":", ""), *rssi])
    wifi_cols = ["sim_time", "wifi_visible_count", "wifi_strongest_rssi", "wifi_strongest_mac"] + \
                [f"wifi_rssi_{b.replace(':', '')}" for b in vocab]
    pd.DataFrame(rows, columns=wifi_cols).to_csv(dst / "wifi.csv", index=False)

    cam = pd.read_csv(rd / "camera" / "camera.csv")
    cam = cam[(cam.t_ns >= t0) & (cam.t_ns <= t1)].iloc[::CAM_EVERY].reset_index(drop=True)
    cdir = dst / "camera"
    if camera:
        cdir.mkdir(exist_ok=True)
    crow = []
    for i, c in cam.iterrows():
        name = f"rgb_{i:06d}.jpg"
        if camera and not (cdir / name).exists():
            shutil.copy2(rd / "camera" / "images" / c.filename, cdir / name)
        crow.append(dict(sim_time=round(float(sec(c.t_ns)), 4), frame_id=f"{i:06d}", rgb_path=f"camera/{name}",
                         depth_path="", cam_x=np.nan, cam_y=np.nan, cam_z=np.nan))
    pd.DataFrame(crow).to_csv(dst / "camera.csv", index=False)

    info = json.load(open(rd / "ground_truth" / "gt_info.json"))
    meta = dict(dataset="side_golden", source=f"side_navlori/data/golden_run_{run}", golden_run=run,
                path_id=run - 1, split=next(k for k, v in SPLIT.items() if run in v), duration_s=round(T_end, 2),
                path_m=round(float(info["path_m"]), 2), gt_status=info["status"], n_gt=len(gt), n_imu=len(imu_df),
                n_odom=len(odom_df), n_wifi_scans=len(rows), n_camera=len(crow), imu_hz=IMU_HZ, odom_hz=ODOM_HZ,
                camera_hz=30 / CAM_EVERY, wifi_max_age_ms=WIFI_MAX_AGE_MS)
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gt.x, gt.y, "-", lw=1.5, label="ground truth")
    ax.plot(gt.x.iloc[0], gt.y.iloc[0], "go", ms=8, label="start")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title(f"path_{run-1:02d} (golden_run_{run}) | {meta['split']} | {T_end:.0f} s, {meta['path_m']:.0f} m | "
                 f"WiFi {len(rows)} scans")
    fig.tight_layout(); fig.savefig(dst / "trajectory.png", dpi=90); plt.close(fig)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(r"X:\side_navlori\data"))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "side_golden")
    ap.add_argument("--no-camera", action="store_true", help="write camera.csv but do not copy images")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for run in SPLIT["train"]:
        w = load_wifi(args.src / f"golden_run_{run}")
        for b, n in w.groupby("bssid").scan_idx.nunique().items():
            counts[b] = counts.get(b, 0) + int(n)
    vocab = sorted(b for b, n in counts.items() if n >= MIN_SCANS_IN_TRAIN)
    print(f"AP vocabulary: {len(vocab)} BSSIDs (seen in >= {MIN_SCANS_IN_TRAIN} train scans)", flush=True)

    metas = []
    for run in RUNS:
        m = convert_run(run, args.src, args.out / f"path_{run-1:02d}", vocab, camera=not args.no_camera)
        metas.append(m)
        print(f"  path_{run-1:02d} <- golden_run_{run:<2d} [{m['split']:5s}] {m['duration_s']:6.1f} s  {m['path_m']:5.1f} m  "
              f"gt {m['n_gt']}  imu {m['n_imu']}  odom {m['n_odom']}  wifi {m['n_wifi_scans']}  cam {m['n_camera']}", flush=True)

    split = {k: [r - 1 for r in v] for k, v in SPLIT.items()}
    (args.out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (args.out / "ap_vocab.json").write_text(json.dumps({b: i for i, b in enumerate(vocab)}, indent=2), encoding="utf-8")
    (args.out / "metadata.json").write_text(json.dumps(dict(
        dataset="side_golden", n_paths=len(metas), n_aps=len(vocab), splits=split,
        total_duration_s=round(sum(m["duration_s"] for m in metas), 1), total_path_m=round(sum(m["path_m"] for m in metas), 1),
        robot="TurtleBot3 (real)", recorded="2026-09-24",
        ground_truth="lidar SLAM translation + gyro heading, placed in the building frame by an AprilTag network"),
        indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), args.out / "convert_side_golden.py")
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
