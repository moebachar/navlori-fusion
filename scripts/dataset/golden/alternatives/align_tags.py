#!/usr/bin/env python
# AprilTag -> world alignment of lidar-SLAM trajectories (golden runs).
#  A) estimate tag physical size from lidar wall range vs unit-size PnP
#  B) observed tag positions in each run's SLAM frame
#  C) robust rigid SLAM->world fit on the known tag coordinates; residuals per tag
import json, sys, numpy as np, pandas as pd
from itertools import combinations
from scipy.spatial.transform import Rotation as Rot

RUNS = sys.argv[1:] or ["golden_run_7", "golden_run_10", "golden_run_11"]
DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
TAGS = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
LIDAR = np.array([-0.064, 0.0])               # base_scan origin in base_footprint (yaw 0)

def load(run):
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv")
    z = np.load(f"{STG}/{run}/lidar/scans.npz")
    return det, t_bc, R_bc, gt, z

def good(det):  # usable detections: sharp fit, not tiny
    return det[(det.reproj_px < 1.0) & (det.side_px > 18)].copy()

def lidar_range(z, t_ns, bearing):
    ts, off = z["t_ns"], z["offsets"]
    k = int(np.clip(np.searchsorted(ts, t_ns), 1, len(ts) - 1))
    k = k if abs(ts[k] - t_ns) < abs(ts[k - 1] - t_ns) else k - 1
    a, r = z["angles"][off[k]:off[k + 1]], z["ranges"][off[k]:off[k + 1]]
    d = np.abs(np.angle(np.exp(1j * (a - bearing))))
    rr = r[(d < np.radians(1.5)) & np.isfinite(r)]
    return float(np.median(rr)) if len(rr) else np.nan

def size_from_lidar(row, t_bc, R_bc, z):
    v = (R_bc @ np.array([row.tx_u, row.ty_u, row.tz_u]))[:2]   # unit-size tag offset (base frame)
    a = t_bc[:2] - LIDAR; s = 0.15
    for _ in range(3):
        p = t_bc[:2] + s * v - LIDAR
        r = lidar_range(z, row.t_ns, np.arctan2(p[1], p[0]))
        if not np.isfinite(r): return np.nan
        A, B, C = v @ v, 2 * a @ v, a @ a - r * r
        disc = B * B - 4 * A * C
        if disc < 0: return np.nan
        s = (-B + np.sqrt(disc)) / (2 * A)
    return s

def slam_pose(gt, t):
    yaw = np.unwrap(gt.yaw.values)
    return (np.interp(t, gt.t_ns.values, gt.x.values), np.interp(t, gt.t_ns.values, gt.y.values),
            np.interp(t, gt.t_ns.values, yaw))

def rigid_fit(P, W):  # least-squares 2D rotation+translation mapping P -> W
    mp, mw = P.mean(0), W.mean(0)
    H = (P - mp).T @ (W - mw); U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0: Vt[1] *= -1; Rm = Vt.T @ U.T
    return Rm, mw - Rm @ mp

def similarity_scale(P, W):
    mp, mw = P.mean(0), W.mean(0)
    return np.sqrt(((W - mw) ** 2).sum() / ((P - mp) ** 2).sum())

data = {r: load(r) for r in RUNS}

# ---- A) tag size ----
S = []
for run, (det, t_bc, R_bc, gt, z) in data.items():
    g = good(det)
    g = g[g.side_px > 30]                                   # near, well-resolved tags only
    for row in g.itertuples():
        s = size_from_lidar(row, t_bc, R_bc, z)
        if np.isfinite(s) and 0.03 < s < 0.6: S.append((run, row.tag_id, s))
S = pd.DataFrame(S, columns=["run", "tag_id", "s"])
per_tag = S.groupby("tag_id").s.agg(["median", "count", lambda x: x.quantile(.75) - x.quantile(.25)])
per_tag.columns = ["median_m", "n", "iqr_m"]
print("A) tag size from lidar (per tag):"); print((per_tag * [100, 1, 100]).round(2).rename(columns={"median_m": "size_cm", "iqr_m": "iqr_cm"}).to_string())
SIZE = float(S.s.median())
if len(sys.argv) > 1 and False: pass
print(f"   -> global tag size estimate: {SIZE*100:.2f} cm  (from {len(S)} detections)\n")

# ---- B+C) per run ----
summary = {}
for run, (det, t_bc, R_bc, gt, z) in data.items():
    g = good(det)
    x, y, yaw = slam_pose(gt, g.t_ns.values)
    pb = (t_bc[:, None] + (R_bc @ (SIZE * g[["tx_u", "ty_u", "tz_u"]].values.T)))[:2]  # tag in base frame
    c, s_ = np.cos(yaw), np.sin(yaw)
    g["px"] = x + c * pb[0] - s_ * pb[1]; g["py"] = y + s_ * pb[0] + c * pb[1]
    g["rng"] = np.hypot(pb[0], pb[1])
    tags = sorted(t for t in g.tag_id.unique() if t in TAGS)
    med = {t: np.median(g.loc[g.tag_id == t, ["px", "py"]].values, 0) for t in tags}
    spread = {t: float(np.median(np.hypot(*(g.loc[g.tag_id == t, ["px", "py"]].values - med[t]).T))) for t in tags}
    # RANSAC over tag pairs, then refit on inliers
    best = None
    for a_, b_ in combinations(tags, 2):
        Rm, T = rigid_fit(np.array([med[a_], med[b_]]), np.array([TAGS[a_], TAGS[b_]]))
        res = {t: np.linalg.norm(Rm @ med[t] + T - TAGS[t]) for t in tags}
        inl = [t for t in tags if res[t] < 0.5]
        score = (len(inl), -sum(res[t] for t in inl))
        if best is None or score > best[0]: best = (score, inl)
    inl = best[1]
    Rm, T = rigid_fit(np.array([med[t] for t in inl]), np.array([TAGS[t] for t in inl]))
    scale = similarity_scale(np.array([med[t] for t in inl]), np.array([TAGS[t] for t in inl])) if len(inl) >= 3 else np.nan
    th = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
    print(f"===== {run}: {len(g)} good detections, tags {tags}; inliers {inl}")
    print(f"   SLAM->world: rotation {th:.2f} deg, translation ({T[0]:.3f}, {T[1]:.3f}) m;  best-fit scale (CAD/lidar) {scale:.4f}")
    for t in tags:
        tt = g.tag_id == t
        e = Rm @ med[t] + T - TAGS[t]
        tm = (g.loc[tt, "t_ns"].values - gt.t_ns.iloc[0]) / 1e9
        print(f"   tag {t:2d}: n={tt.sum():3d}  seen t={tm.min():6.1f}-{tm.max():6.1f}s  range {g.loc[tt,'rng'].median():.2f}m  "
              f"self-spread {spread[t]*100:5.1f}cm  RESIDUAL {np.linalg.norm(e)*100:6.1f} cm  ({e[0]*100:+.1f},{e[1]*100:+.1f}){'' if t in inl else '  <-- OUTLIER'}")
    summary[run] = dict(R=Rm.tolist(), T=T.tolist(), theta_deg=th, inliers=inl, size=SIZE)
    g.to_csv(f"{CAM}/{run}/tag_obs_slam.csv", index=False)
json.dump(summary, open(f"{CAM}/rigid_alignment.json", "w"), indent=1)
print("ALIGN_DONE")
