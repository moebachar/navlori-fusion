#!/usr/bin/env python
# Joint alignment of the golden runs on the AprilTag network (runs rigid, tags with soft CAD priors).
import json, os, sys, numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_common import *

RUNS = [f"golden_run_{i}" for i in range(1, 13)]; SIG_CAD = 0.10
RUN, OBS = {}, []
for run in RUNS:
    f = fused(run); det = sightings(run, f["T"], dict(F=f["F"], S=f["S"])); t0 = f["T"][0]
    for cl, g in det.groupby("cl"):
        rng, obl = float(g.rng.median()), float(g.obl.median())
        OBS.append(dict(run=run, tag=int(g.tag_id.iloc[0]), p=np.array([g.F_x.median(), g.F_y.median()]),
                        ps=np.array([g.S_x.median(), g.S_y.median()]), sig=(0.03 + 0.03 * rng) * (2.0 if obl > 60 else 1.0),
                        n=len(g), rng=rng, obl=obl, t0=(g.t_ns.min() - t0) / 1e9, t1=(g.t_ns.max() - t0) / 1e9))
    RUN[run] = f
    print(f"{run}: gyro offset {f['bias_dps']:+.3f} deg/s via {f['bias_method']}; tags {sorted(det.tag_id.unique().tolist())}", flush=True)
obs = pd.DataFrame(OBS)
ba = [r for r in RUNS if obs[obs.run == r].tag.nunique() >= 2]; single = [r for r in RUNS if r not in ba]
O_ = obs[obs.run.isin(ba)].reset_index(drop=True)
tags = sorted(O_.tag.unique()); ti_ = {t: i for i, t in enumerate(tags)}; ri_ = {r: i for i, r in enumerate(ba)}
cad = np.array([CAD[t] for t in tags])
def unpack(v): return v[:3 * len(ba)].reshape(-1, 3), v[3 * len(ba):].reshape(-1, 2)
def solve(mask, v0):
    O = O_[mask]; ri = O.run.map(ri_).values; tj = O.tag.map(ti_).values; P = np.stack(O.p.values); sg = O.sig.values
    def fun(v):
        X, Tg = unpack(v); c, s = np.cos(X[ri, 2]), np.sin(X[ri, 2])
        pr = np.c_[X[ri, 0] + c * P[:, 0] - s * P[:, 1], X[ri, 1] + s * P[:, 0] + c * P[:, 1]]
        return np.r_[((pr - Tg[tj]) / sg[:, None]).ravel(), ((Tg - cad) / SIG_CAD).ravel()]
    return least_squares(fun, v0, loss="cauchy", f_scale=2.0, x_scale="jac", xtol=1e-12, ftol=1e-12, max_nfev=5000).x
X0 = []
for r in ba:
    o = O_[O_.run == r].groupby("tag").p.apply(lambda s: np.median(np.stack(s.values), 0))
    R, t = procrustes(np.stack(o.values), np.array([CAD[k] for k in o.index])); X0.append([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])
v0 = np.r_[np.array(X0).ravel(), cad.ravel()]; full = np.ones(len(O_), bool); v = solve(full, v0); X, Tg = unpack(v)

TR = pd.DataFrame([dict(tag=t, cad_x=CAD[t][0], cad_y=CAD[t][1], est_x=Tg[ti_[t]][0], est_y=Tg[ti_[t]][1],
                        offset_cm=100 * np.linalg.norm(Tg[ti_[t]] - CAD[t]), sightings=int((O_.tag == t).sum()),
                        runs=",".join(sorted({r.split("_")[-1] for r in O_[O_.tag == t].run}, key=int))) for t in tags])
print("\nTag network (estimated vs CAD):"); print(TR.round(2).to_string(index=False))
rows = []
for k in range(len(O_)):
    o = O_.iloc[k]; r = ri_[o.run]
    pred = lambda X_: np.array([X_[r, 0] + np.cos(X_[r, 2]) * o.p[0] - np.sin(X_[r, 2]) * o.p[1], X_[r, 1] + np.sin(X_[r, 2]) * o.p[0] + np.cos(X_[r, 2]) * o.p[1]])
    left = O_[(O_.run == o.run) & (np.arange(len(O_)) != k)].tag.nunique()
    if left >= 2:
        m = full.copy(); m[k] = False; Xh, Tgh = unpack(solve(m, v)); ho = pred(Xh) - Tgh[ti_[o.tag]]
    else: ho = np.array([np.nan, np.nan])
    rows.append(dict(run=o.run, tag=o.tag, t_start=o.t0, t_end=o.t1, n=o.n, range_m=o.rng, oblique_deg=o.obl,
                     fit_cm=100 * np.linalg.norm(pred(X) - Tg[ti_[o.tag]]), heldout_cm=100 * np.linalg.norm(ho),
                     heldout_dx_cm=100 * ho[0], heldout_dy_cm=100 * ho[1], vs_cad_cm=100 * np.linalg.norm(pred(X) - CAD[o.tag])))
RES = pd.DataFrame(rows)
print("\nPer run (held out = sighting removed, whole network re-solved):")
print(RES.groupby("run").agg(sightings=("tag", "size"), fit_med_cm=("fit_cm", "median"), heldout_med_cm=("heldout_cm", "median"),
                             heldout_max_cm=("heldout_cm", "max")).reindex(ba).round(1).to_string())
NET = f"{STG}/_network"; os.makedirs(NET, exist_ok=True)
TR.to_csv(f"{NET}/tag_positions_estimated.csv", index=False); RES.to_csv(f"{NET}/sighting_residuals.csv", index=False)
json.dump({str(t): dict(x=float(Tg[ti_[t]][0]), y=float(Tg[ti_[t]][1])) for t in tags}, open(f"{NET}/tags_estimated.json", "w"), indent=1)

for r in ba:
    f = RUN[r]; G = X[ri_[r]]; T, F, S, gt = f["T"], f["F"], f["S"], f["gt"]
    Pw = compose(np.broadcast_to(G, F.shape), F)
    o = O_[O_.run == r]; Rs, ts_ = procrustes(np.stack(o.ps.values), np.array([Tg[ti_[k]] for k in o.tag]))
    Srig = compose(np.broadcast_to(np.array([ts_[0], ts_[1], np.arctan2(Rs[1, 0], Rs[0, 0])]), S.shape), S)
    out = gt.copy(); out["x"], out["y"], out["yaw"] = Pw[:, 0], Pw[:, 1], wrap(Pw[:, 2]); out["slam_jump_step"] = np.r_[0, f["jump"].astype(int)]
    out.to_csv(f"{STG}/{r}/ground_truth/gt_pose_world.csv", index=False)
    pd.DataFrame(dict(t_ns=T, x=Srig[:, 0], y=Srig[:, 1], yaw=wrap(Srig[:, 2]), heading_slam_minus_gyro_deg=f["slam_gyro_heading_dev_deg"],
                      slam_jump_step=np.r_[0, f["jump"].astype(int)])).to_csv(f"{STG}/{r}/ground_truth/gt_slam_aligned.csv", index=False)
    rr = RES[RES.run == r]; rr.to_csv(f"{STG}/{r}/ground_truth/tag_residuals.csv", index=False)
    np.savez_compressed(f"{STG}/{r}/ground_truth/map_points_world.npz", xy=scan_points(r, T, Pw))
    json.dump(dict(run=r, status="ok", method="tag network: gyro+SLAM trajectory placed rigidly by joint tag adjustment over all runs",
                   tag_size_m=TAG_SIZE, gyro_offset_dps=f["bias_dps"], gyro_offset_method=f["bias_method"],
                   used_tags=sorted(int(k) for k in o.tag.unique()), n_sightings=int(len(o)),
                   fit_median_cm=float(rr.fit_cm.median()), heldout_median_cm=float(rr.heldout_cm.median()), heldout_max_cm=float(rr.heldout_cm.max()),
                   wheel_fallback_steps=int(f["jump"].sum()), slam_heading_jumps=int((np.abs(np.diff(f["slam_gyro_heading_dev_deg"])) > 2).sum()),
                   first_tag_s=float(o.t0.min()), last_tag_end_s=float(o.t1.max()), duration_s=float(f["t"][-1]),
                   path_m=float(np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum()), transform=[float(g) for g in G]),
              open(f"{STG}/{r}/ground_truth/gt_info.json", "w"), indent=1)
for r in single:
    json.dump(dict(run=r, status="needs_map_matching", seen_tags=sorted(int(k) for k in obs[obs.run == r].tag.unique())),
              open(f"{STG}/{r}/ground_truth/gt_info.json", "w"), indent=1)
print("\nsingle-tag runs -> map matching:", single); print("TAG_BA_DONE", flush=True)
