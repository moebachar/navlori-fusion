#!/usr/bin/env python
# Left building area: runs 1,3,4,5,6 each placed by ITS OWN tags (rigid), maps + tag sightings overlaid.
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
CAD = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
exec(open("/root/navlori/scripts_local/tag_ba.py").read().split("obs = pd.DataFrame(OBS)")[0].split("# ---------------- per run")[0])
RUNS_L = ["golden_run_1", "golden_run_3", "golden_run_4", "golden_run_5", "golden_run_6"]
fig, axs = plt.subplots(2, 3, figsize=(20, 13)); axs = axs.ravel()
cols = plt.get_cmap("tab10")
for i, run in enumerate(RUNS_L):
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv"); T = gt.t_ns.values.astype(np.int64); N = len(T)
    S = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]; ts = lambda t: (t - T[0]) / 1e9
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns"); od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    ti = ts(imu.t_ns.values); bias = imu.wz.values[(ti > 0.3) & (ti < 4.0)].mean()
    Gs = np.interp(ts(T), ti, np.cumsum((imu.wz.values - bias) * np.diff(ti, prepend=ti[0])))
    O = interp_pose(od.t_ns.values, np.c_[od.x.values, od.y.values, np.unwrap(od.yaw.values)], T)
    dS, dO = between(S[:-1], S[1:]), between(O[:-1], O[1:])
    jump = np.hypot(*(dS[:, :2] - dO[:, :2]).T) > 0.03 + 0.3 * np.hypot(*dO[:, :2].T)
    F = np.zeros((N, 3)); F[0] = S[0]; step = np.c_[np.where(jump[:, None], dO[:, :2], dS[:, :2]), np.diff(Gs)]
    for k in range(N - 1): F[k + 1] = compose(F[k], step[k])
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json")); t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px / det.side_px < 0.015) & (det.side_px > 18) & (det.t_ns >= T[0]) & (det.t_ns <= T[-1])]
    b = (t_bc[:, None] + R_bc @ (0.15 * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    p = compose(interp_pose(T, F, det.t_ns.values), np.c_[b, np.zeros(len(b))])[:, :2]
    z = np.load(f"{STG}/{run}/lidar/scans.npz"); off = z["offsets"]; pts = []
    for k in range(0, N, 3):
        a, r = z["angles"][off[k]:off[k + 1]:2], z["ranges"][off[k]:off[k + 1]:2]; ok = np.isfinite(r) & (r > 0.1) & (r < 8)
        xb, yb = -0.064 + r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]); c, s = np.cos(F[k, 2]), np.sin(F[k, 2])
        pts.append(np.c_[F[k, 0] + c * xb - s * yb, F[k, 1] + s * xb + c * yb])
    M = np.concatenate(pts)
    ax = axs[i]; ax.scatter(M[:, 0], M[:, 1], s=0.2, c="0.7", rasterized=True)
    sc = ax.scatter(F[:, 0], F[:, 1], c=ts(T), s=2, cmap="viridis")
    for j, tid in enumerate(sorted(det.tag_id.unique())):
        m = det.tag_id.values == tid; pm = np.median(p[m], 0)
        ax.scatter(p[m, 0], p[m, 1], s=8, color=cols(j)); ax.annotate(f"tag {tid}\n t={ts(det.t_ns.values[m]).min():.0f}-{ts(det.t_ns.values[m]).max():.0f}s", pm, fontsize=10, color=cols(j), fontweight="bold")
    ax.set_aspect("equal"); ax.set_title(f"{run} (own frame, gyro+SLAM)"); ax.grid(alpha=.3)
    fig.colorbar(sc, ax=ax, fraction=0.03, label="t (s)")
ax = axs[5]
for t in [0, 1, 2, 3, 4, 5, 6]:
    ax.plot(*CAD[t], "ks", mfc="yellow", ms=10); ax.annotate(str(t), CAD[t], xytext=(6, 6), textcoords="offset points", fontsize=12)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title("CAD tag positions (left area)")
plt.tight_layout(); fig.savefig("/mnt/x/side_navlori/gt_review/left_area_runs_own_frame.png", dpi=90)
print("ok")
