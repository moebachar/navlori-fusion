#!/usr/bin/env python
# Golden runs that see a single tag: place the gyro+SLAM trajectory by matching its lidar map onto the
# building map of the tag-network runs (multi-start 2D ICP), the one tag (network-estimated position) as anchor.
import json, os, sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_common import *

ALL = [f"golden_run_{i}" for i in range(1, 13)]; VOX = 0.05
NET = json.load(open(f"{STG}/_network/tags_estimated.json")); TAGW = {int(k): np.array([v["x"], v["y"]]) for k, v in NET.items()}
def voxel(p, v=VOX):
    k = np.floor(p / v).astype(np.int64); _, i = np.unique(k, axis=0, return_index=True); return p[np.sort(i)]
def rot(th): c, s = np.cos(th), np.sin(th); return np.array([[c, -s], [s, c]])
single = [r for r in ALL if json.load(open(f"{STG}/{r}/ground_truth/gt_info.json"))["status"] == "needs_map_matching"]
ref_runs = [r for r in ALL if json.load(open(f"{STG}/{r}/ground_truth/gt_info.json"))["status"] == "ok"]
REF = voxel(np.concatenate([np.load(f"{STG}/{r}/ground_truth/map_points_world.npz")["xy"] for r in ref_runs])); tree = cKDTree(REF)
print(f"reference: {len(REF)} voxels from {len(ref_runs)} runs; to match: {single}", flush=True)
for run in single:
    f = fused(run); T, F, S, gt = f["T"], f["F"], f["S"], f["gt"]
    det = sightings(run, T, dict(F=F, S=S)); tid = int(det.tag_id.mode()[0]); d = det[det.tag_id == tid]
    ptag = np.array([d.F_x.median(), d.F_y.median()]); W = TAGW.get(tid, CAD[tid])
    own_full = scan_points(run, T, F); own = voxel(own_full)
    results = []
    for th0 in np.radians(np.arange(0, 360, 5)):
        R = rot(th0); t = W - R @ ptag
        for dmax in [1.0, 0.7, 0.5, 0.35, 0.25, 0.2, 0.15, 0.15, 0.12, 0.12, 0.1, 0.1]:
            q = own @ R.T + t; dist, j = tree.query(q, distance_upper_bound=dmax); m = np.isfinite(dist)
            if m.sum() < 50: break
            R, t = procrustes(np.vstack([own[m], ptag[None]]), np.vstack([REF[j[m]], W[None]]), np.r_[np.ones(m.sum()), 0.1 * m.sum()])
        dist = tree.query(own @ R.T + t)[0]
        results.append((float((dist < 0.10).mean()), float(np.median(dist[dist < 0.10])) if (dist < 0.10).any() else 9,
                        np.degrees(np.arctan2(R[1, 0], R[0, 0])), R, t, float(np.linalg.norm(R @ ptag + t - W))))
    results.sort(key=lambda x: -x[0]); best = results[0]
    alt = next((x for x in results if abs(wrap(np.radians(x[2] - best[2]))) > np.radians(10)), None)
    inl, gap, th, R, t, terr = best
    print(f"{run}: tag {tid} | best rotation {th:.2f} deg, walls within 10 cm {inl*100:.1f}%, median gap {gap*100:.1f} cm, tag error {terr*100:.1f} cm"
          f" | runner-up (other heading) {alt[0]*100:.1f}% at {alt[2]:.0f} deg" if alt else "", flush=True)
    G = np.array([t[0], t[1], np.radians(th)]); Pw = compose(np.broadcast_to(G, F.shape), F)
    Rs, ts_ = procrustes(S[:, :2], Pw[:, :2]); Srig = compose(np.broadcast_to(np.array([ts_[0], ts_[1], np.arctan2(Rs[1, 0], Rs[0, 0])]), S.shape), S)
    out = gt.copy(); out["x"], out["y"], out["yaw"] = Pw[:, 0], Pw[:, 1], wrap(Pw[:, 2]); out["slam_jump_step"] = np.r_[0, f["jump"].astype(int)]
    out.to_csv(f"{STG}/{run}/ground_truth/gt_pose_world.csv", index=False)
    pd.DataFrame(dict(t_ns=T, x=Srig[:, 0], y=Srig[:, 1], yaw=wrap(Srig[:, 2]), heading_slam_minus_gyro_deg=f["slam_gyro_heading_dev_deg"],
                      slam_jump_step=np.r_[0, f["jump"].astype(int)])).to_csv(f"{STG}/{run}/ground_truth/gt_slam_aligned.csv", index=False)
    np.savez_compressed(f"{STG}/{run}/ground_truth/map_points_world.npz", xy=own_full @ R.T + t)
    pd.DataFrame([dict(run=run, tag=tid, t_start=(d.t_ns.min() - T[0]) / 1e9, t_end=(d.t_ns.max() - T[0]) / 1e9, n=len(d), range_m=float(d.rng.median()),
                       oblique_deg=float(d.obl.median()), fit_cm=terr * 100, heldout_cm=np.nan, heldout_dx_cm=np.nan, heldout_dy_cm=np.nan,
                       vs_cad_cm=100 * float(np.linalg.norm(R @ ptag + t - CAD[tid])))]).to_csv(f"{STG}/{run}/ground_truth/tag_residuals.csv", index=False)
    json.dump(dict(run=run, status="map_matched", method="gyro+SLAM trajectory placed by 2D ICP of its lidar map onto the tag-network runs' map; single tag as anchor",
                   tag_size_m=TAG_SIZE, gyro_offset_dps=f["bias_dps"], gyro_offset_method=f["bias_method"], used_tags=[tid], n_sightings=1,
                   fit_median_cm=terr * 100, wall_within_10cm_pct=inl * 100, wall_gap_median_cm=gap * 100,
                   runner_up_heading_deg=(alt[2] if alt else None), runner_up_walls_pct=(alt[0] * 100 if alt else None),
                   heldout_median_cm=None, heldout_max_cm=None, wheel_fallback_steps=int(f["jump"].sum()),
                   slam_heading_jumps=int((np.abs(np.diff(f["slam_gyro_heading_dev_deg"])) > 2).sum()),
                   first_tag_s=float((d.t_ns.min() - T[0]) / 1e9), last_tag_end_s=float((d.t_ns.max() - T[0]) / 1e9),
                   duration_s=float(f["t"][-1]), path_m=float(np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum()),
                   transform=[float(g) for g in G], reference_runs=ref_runs), open(f"{STG}/{run}/ground_truth/gt_info.json", "w"), indent=1)
print("GT_ICP_DONE", flush=True)
