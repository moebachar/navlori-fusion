#!/usr/bin/env python
# Tag-coordinate consistency: for tag sightings close in time (drift-free), compare the distance between
# the tags as measured by the robot (gyro+SLAM track + camera) with the distance in tags_ground_truth.json.
# Also: SLAM vs wheel path length per run (SLAM translation sanity).
import json, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
TAGS = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
TAG_SIZE = 0.150
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)
def inverse(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]), -(-s * a[..., 0] + c * a[..., 1]), -a[..., 2]], -1)
def between(a, b): return compose(inverse(a), b)
def interp_pose(T, P, t): return np.stack([np.interp(t, T, P[:, i]) for i in range(3)], -1)
rows, lens = [], []
for i in range(1, 13):
    run = f"golden_run_{i}"
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv"); T = gt.t_ns.values.astype(np.int64); N = len(T)
    S = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]; ts = lambda t: (t - T[0]) / 1e9
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns"); od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    ti = ts(imu.t_ns.values); bias = imu.wz.values[(ti > 0.3) & (ti < 4.0)].mean()
    Gs = np.interp(ts(T), ti, np.cumsum((imu.wz.values - bias) * np.diff(ti, prepend=ti[0])))
    O = interp_pose(od.t_ns.values, np.c_[od.x.values, od.y.values, np.unwrap(od.yaw.values)], T)
    dS, dO = between(S[:-1], S[1:]), between(O[:-1], O[1:])
    jump = np.hypot(*(dS[:, :2] - dO[:, :2]).T) > 0.03 + 0.3 * np.hypot(*dO[:, :2].T)
    step = np.c_[np.where(jump[:, None], dO[:, :2], dS[:, :2]), np.diff(Gs)]
    F = np.zeros((N, 3)); F[0] = S[0]
    for k in range(N - 1): F[k + 1] = compose(F[k], step[k])
    ls, lw = np.hypot(*dS[:, :2].T).sum(), np.hypot(*dO[:, :2].T).sum()
    lens.append(dict(run=run, slam_path_m=ls, wheel_path_m=lw, slam_over_wheel=ls / lw))
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px / det.side_px < 0.015) & (det.side_px > 18) & det.tag_id.isin(list(TAGS)) & (det.t_ns >= T[0]) & (det.t_ns <= T[-1])].copy()
    b = (t_bc[:, None] + R_bc @ (TAG_SIZE * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    p = compose(interp_pose(T, F, det.t_ns.values), np.c_[b, np.zeros(len(b))])[:, :2]
    det["px"], det["py"] = p[:, 0], p[:, 1]
    det = det.sort_values(["tag_id", "t_ns"]); det["cl"] = ((det.tag_id != det.tag_id.shift()) | (det.t_ns.diff() > 2e9)).cumsum()
    C = det.groupby("cl").agg(tag=("tag_id", "first"), t=("t_ns", "median"), px=("px", "median"), py=("py", "median"), n=("t_ns", "size"))
    C = C.reset_index(drop=True)
    for a in range(len(C)):
        for b_ in range(a + 1, len(C)):
            if C.tag[a] == C.tag[b_]: continue
            dm = np.hypot(C.px[a] - C.px[b_], C.py[a] - C.py[b_]); dj = np.linalg.norm(TAGS[C.tag[a]] - TAGS[C.tag[b_]])
            rows.append(dict(run=run, pair=f"{min(C.tag[a],C.tag[b_])}-{max(C.tag[a],C.tag[b_])}", dt_s=abs(C.t[b_] - C.t[a]) / 1e9,
                             robot_m=dm, cad_m=dj, diff_cm=(dm - dj) * 100))
P = pd.DataFrame(rows)
print("SLAM vs wheel path length:"); print(pd.DataFrame(lens).round(3).to_string(index=False))
print("\nTag pairs seen within 30 s of each other (drift-free): robot-measured vs CAD distance")
print(P[P.dt_s < 30].sort_values(["pair", "run"]).round(2).to_string(index=False))
print("\nAll pairs, grouped (median over sightings):")
print(P.groupby("pair").agg(n=("diff_cm", "size"), med_diff_cm=("diff_cm", "median"), min_dt=("dt_s", "min"), cad_m=("cad_m", "first")).round(1).to_string())
