#!/usr/bin/env python
# Joint alignment of all golden runs + AprilTag positions ("tag bundle adjustment").
#  - each run = its fused gyro+SLAM trajectory, treated as a rigid body (validated: stiff model)
#  - each tag = unknown 2D position with a soft, robust prior at its CAD coordinate
#  - each tag sighting cluster = observation of that tag in the run frame
# Runs sharing tags tie the network together, so a CAD coordinate that several runs contradict is corrected
# (and reported), instead of bending the trajectories. Leave-one-sighting-out gives held-out errors.
import json, os, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
RUNS = [f"golden_run_{i}" for i in range(1, 13)]
CAD = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
TAG_SIZE = 0.150; REL_REPROJ = 0.015; MIN_SIDE = 18; SIG_CAD = 0.10; LIDAR_X = -0.064

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)
def inverse(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]), -(-s * a[..., 0] + c * a[..., 1]), -a[..., 2]], -1)
def between(a, b): return compose(inverse(a), b)
def interp_pose(T, P, t): return np.stack([np.interp(t, T, P[:, i]) for i in range(3)], -1)
def rot(th): c, s = np.cos(th), np.sin(th); return np.array([[c, -s], [s, c]])
def procrustes(P, Q):
    mp, mq = P.mean(0), Q.mean(0); U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq)); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, mq - R @ mp

# ---------------- per run: fused trajectory + tag sighting clusters ----------------
RUN = {}; OBS = []
for run in RUNS:
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv"); T = gt.t_ns.values.astype(np.int64); N = len(T)
    S = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]; ts = lambda t, T0=T[0]: (t - T0) / 1e9
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns"); od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    ti = ts(imu.t_ns.values); bias = imu.wz.values[(ti > 0.3) & (ti < 4.0)].mean()
    Gs = np.interp(ts(T), ti, np.cumsum((imu.wz.values - bias) * np.diff(ti, prepend=ti[0])))
    O = interp_pose(od.t_ns.values, np.c_[od.x.values, od.y.values, np.unwrap(od.yaw.values)], T)
    dS, dO = between(S[:-1], S[1:]), between(O[:-1], O[1:])
    jump = np.hypot(*(dS[:, :2] - dO[:, :2]).T) > 0.03 + 0.3 * np.hypot(*dO[:, :2].T)
    step = np.c_[np.where(jump[:, None], dO[:, :2], dS[:, :2]), np.diff(Gs)]
    F = np.zeros((N, 3)); F[0] = S[0]
    for k in range(N - 1): F[k + 1] = compose(F[k], step[k])
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px / det.side_px < REL_REPROJ) & (det.side_px > MIN_SIDE) & det.tag_id.isin(list(CAD)) &
              (det.t_ns >= T[0]) & (det.t_ns <= T[-1])].copy()
    b = (t_bc[:, None] + R_bc @ (TAG_SIZE * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    det["rng"] = np.hypot(b[:, 0], b[:, 1])
    det["obl"] = [np.degrees(np.arccos(abs(Rot.from_rotvec([r.rx, r.ry, r.rz]).as_matrix()[2, 2]))) for r in det.itertuples()]
    pf = compose(interp_pose(T, F, det.t_ns.values), np.c_[b, np.zeros(len(b))])[:, :2]      # tag in run (fused) frame
    ps = compose(interp_pose(T, S, det.t_ns.values), np.c_[b, np.zeros(len(b))])[:, :2]      # same, raw SLAM frame
    det["fx"], det["fy"], det["sx"], det["sy"] = pf[:, 0], pf[:, 1], ps[:, 0], ps[:, 1]
    det = det.sort_values(["tag_id", "t_ns"]); det["cl"] = ((det.tag_id != det.tag_id.shift()) | (det.t_ns.diff() > 2e9)).cumsum()
    for cl, g in det.groupby("cl"):
        rng = float(g.rng.median()); obl = float(g.obl.median())
        sig = (0.03 + 0.03 * rng) * (2.0 if obl > 60 else 1.0)
        OBS.append(dict(run=run, tag=int(g.tag_id.iloc[0]), p=np.array([g.fx.median(), g.fy.median()]),
                        ps=np.array([g.sx.median(), g.sy.median()]), sig=sig, n=len(g), rng=rng, obl=obl,
                        t0=ts(g.t_ns.min()), t1=ts(g.t_ns.max())))
    RUN[run] = dict(T=T, F=F, S=S, Gs=Gs, jump=jump, gt=gt, dS=dS)
    print(f"{run}: tags {sorted(set(o['tag'] for o in OBS if o['run']==run))}  ({sum(o['run']==run for o in OBS)} sightings)", flush=True)

obs = pd.DataFrame(OBS)
ba_runs = [r for r in RUNS if obs[obs.run == r].tag.nunique() >= 2]
single = [r for r in RUNS if r not in ba_runs]
O_ = obs[obs.run.isin(ba_runs)].reset_index(drop=True)
tags = sorted(O_.tag.unique()); ti_ = {t: i for i, t in enumerate(tags)}; ri_ = {r: i for i, r in enumerate(ba_runs)}
print(f"\nBA runs: {ba_runs}\nsingle-tag runs (-> map matching): {single}\ntags in BA: {tags}", flush=True)

def unpack(v):
    return v[:3 * len(ba_runs)].reshape(-1, 3), v[3 * len(ba_runs):].reshape(-1, 2)
def solve(mask, v0):
    O = O_[mask]; ridx = O.run.map(ri_).values; tidx = O.tag.map(ti_).values
    P = np.stack(O.p.values); sg = O.sig.values; cad = np.array([CAD[t] for t in tags])
    def fun(v):
        X, Tg = unpack(v)
        c, s = np.cos(X[ridx, 2]), np.sin(X[ridx, 2])
        pred = np.c_[X[ridx, 0] + c * P[:, 0] - s * P[:, 1], X[ridx, 1] + s * P[:, 0] + c * P[:, 1]]
        return np.r_[((pred - Tg[tidx]) / sg[:, None]).ravel(), ((Tg - cad) / SIG_CAD).ravel()]
    r = least_squares(fun, v0, loss="cauchy", f_scale=2.0, x_scale="jac", max_nfev=2000)
    print("DEBUG status", r.status, r.message, "nfev", r.nfev, "cost0", 0.5*np.sum(fun(v0)**2), "cost", r.cost, "|dx|", np.abs(r.x-v0).max(), flush=True)
    return r.x

# init: runs = Procrustes of their sighting medians onto CAD, tags = CAD
X0 = []
for r in ba_runs:
    o = O_[O_.run == r].groupby("tag").p.apply(lambda s: np.median(np.stack(s.values), 0))
    R, t = procrustes(np.stack(o.values), np.array([CAD[t] for t in o.index])); X0.append([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])
v0 = np.r_[np.array(X0).ravel(), np.array([CAD[t] for t in tags]).ravel()]
full = np.ones(len(O_), bool); v = solve(full, v0); X, Tg = unpack(v)

import sys; sys.exit(0)
print("\nTag positions (estimated from all runs) vs CAD:")
tag_rows = []
for t in tags:
    d = Tg[ti_[t]] - CAD[t]; n = int((O_.tag == t).sum()); rs = sorted(set(O_[O_.tag == t].run.str.split("_").str[-1].astype(int)))
    tag_rows.append(dict(tag=t, cad_x=CAD[t][0], cad_y=CAD[t][1], est_x=Tg[ti_[t]][0], est_y=Tg[ti_[t]][1],
                         offset_cm=100 * np.linalg.norm(d), dx_cm=100 * d[0], dy_cm=100 * d[1], sightings=n, runs=",".join(map(str, rs)),
                         verdict=("CHECK: CAD looks off" if np.linalg.norm(d) > 0.25 and n >= 2 else
                                  "only 1 sighting: weak" if n < 2 else "ok")))
TR = pd.DataFrame(tag_rows); print(TR.round(2).to_string(index=False))

# ---- per sighting: residual now + held out (sighting removed, network re-solved) ----
res_rows = []
for k in range(len(O_)):
    o = O_.iloc[k]; r = ri_[o.run]
    left = O_[(O_.run == o.run) & (np.arange(len(O_)) != k)].tag.nunique()
    def pred(X_): c, s = np.cos(X_[r, 2]), np.sin(X_[r, 2]); return np.array([X_[r, 0] + c * o.p[0] - s * o.p[1], X_[r, 1] + s * o.p[0] + c * o.p[1]])
    now = np.linalg.norm(pred(X) - Tg[ti_[o.tag]])
    if left >= 2:
        m = full.copy(); m[k] = False; Xh, Tgh = unpack(solve(m, v)); ho = pred(Xh) - Tgh[ti_[o.tag]]
    else:
        ho = np.array([np.nan, np.nan])
    res_rows.append(dict(run=o.run, tag=o.tag, t_start=o.t0, t_end=o.t1, n=o.n, range_m=o.rng, oblique_deg=o.obl,
                         corrected_cm=100 * now, heldout_cm=100 * np.linalg.norm(ho), heldout_dx_cm=100 * ho[0], heldout_dy_cm=100 * ho[1],
                         vs_cad_cm=100 * np.linalg.norm(pred(X) - CAD[o.tag])))
RES = pd.DataFrame(res_rows)
print("\nPer run: held-out error at tags (network re-solved without that sighting)")
print(RES.groupby("run").agg(sightings=("tag", "size"), heldout_med_cm=("heldout_cm", "median"), heldout_max_cm=("heldout_cm", "max"),
                             fit_med_cm=("corrected_cm", "median")).reindex(ba_runs).round(1).to_string())
os.makedirs(f"{STG}/_network", exist_ok=True)
TR.to_csv(f"{STG}/_network/tag_positions_estimated.csv", index=False); RES.to_csv(f"{STG}/_network/sighting_residuals.csv", index=False)
json.dump({str(t): dict(x=float(Tg[ti_[t]][0]), y=float(Tg[ti_[t]][1])) for t in tags}, open(f"{STG}/_network/tags_estimated.json", "w"), indent=1)

# ---- write per-run ground truth for BA runs ----
for r in ba_runs:
    d = RUN[r]; G = X[ri_[r]]; T, F, S, gt = d["T"], d["F"], d["S"], d["gt"]
    Pw = compose(np.broadcast_to(G, F.shape), F)
    o = O_[O_.run == r]                                                            # raw SLAM rigidly fitted to the same tag estimates
    Rs, ts_ = procrustes(np.stack(o.ps.values), np.array([Tg[ti_[t]] for t in o.tag])) if o.tag.nunique() >= 2 else (np.eye(2), np.zeros(2))
    Srig = compose(np.broadcast_to(np.array([ts_[0], ts_[1], np.arctan2(Rs[1, 0], Rs[0, 0])]), S.shape), S)
    out = gt.copy(); out["x"], out["y"], out["yaw"] = Pw[:, 0], Pw[:, 1], wrap(Pw[:, 2])
    out["slam_jump_step"] = np.r_[0, d["jump"].astype(int)]
    out.to_csv(f"{STG}/{r}/ground_truth/gt_pose_world.csv", index=False)
    pd.DataFrame(dict(t_ns=T, x=Srig[:, 0], y=Srig[:, 1], yaw=wrap(Srig[:, 2]),
                      heading_slam_minus_gyro_deg=np.degrees(wrap(S[:, 2] - (d["Gs"] - d["Gs"][0] + S[0, 2]))),
                      slam_jump_step=np.r_[0, d["jump"].astype(int)])).to_csv(f"{STG}/{r}/ground_truth/gt_slam_aligned.csv", index=False)
    rr = RES[RES.run == r]; rr.to_csv(f"{STG}/{r}/ground_truth/tag_residuals.csv", index=False)
    z = np.load(f"{STG}/{r}/lidar/scans.npz"); off = z["offsets"]; pts = []
    for i in range(0, len(T), 3):
        a, rg = z["angles"][off[i]:off[i + 1]:2], z["ranges"][off[i]:off[i + 1]:2]
        ok = np.isfinite(rg) & (rg > 0.1) & (rg < 8); a, rg = a[ok], rg[ok]
        xb, yb = LIDAR_X + rg * np.cos(a), rg * np.sin(a); c, s = np.cos(Pw[i, 2]), np.sin(Pw[i, 2])
        pts.append(np.c_[Pw[i, 0] + c * xb - s * yb, Pw[i, 1] + s * xb + c * yb])
    np.savez_compressed(f"{STG}/{r}/ground_truth/map_points_world.npz", xy=np.concatenate(pts))
    tt = lambda x: (x - T[0]) / 1e9
    json.dump(dict(run=r, status="ok", method="tag network: fused gyro+SLAM trajectory placed rigidly by joint tag adjustment",
                   tag_size_m=TAG_SIZE, used_tags=sorted(int(t) for t in o.tag.unique()), n_sightings=int(len(o)),
                   heldout_median_cm=float(rr.heldout_cm.median()), heldout_max_cm=float(rr.heldout_cm.max()),
                   fit_median_cm=float(rr.corrected_cm.median()), wheel_fallback_steps=int(d["jump"].sum()),
                   slam_gyro_jump_steps=int((np.degrees(np.abs(wrap(d["dS"][:, 2] - np.diff(d["Gs"])))) > 2).sum()),
                   first_tag_s=float(o.t0.min()), last_tag_end_s=float(o.t1.max()), duration_s=float(tt(T[-1])),
                   path_m=float(np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum()), transform=[float(g) for g in G]),
              open(f"{STG}/{r}/ground_truth/gt_info.json", "w"), indent=1)
for r in single:
    json.dump(dict(run=r, status="needs_map_matching", seen_tags=sorted(int(t) for t in obs[obs.run == r].tag.unique())),
              open(f"{STG}/{r}/ground_truth/gt_info.json", "w"), indent=1)
print("\nTAG_BA_DONE", flush=True)
