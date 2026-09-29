#!/usr/bin/env python
# Tag-corrected ground truth: lidar-SLAM trajectory + AprilTag landmarks (known CAD coords)
# in a 2D pose graph. SLAM supplies the local shape (relative-pose factors between keyframes);
# tag sightings anchor it to the building frame. Leave-one-sighting-out gives held-out accuracy.
import json, os, sys, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = sys.argv[1:] or ["golden_run_7", "golden_run_10", "golden_run_11"]
DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
REV = "/mnt/x/side_navlori/gt_review"; os.makedirs(REV, exist_ok=True)
TAGS = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
TAG_SIZE = 0.1515          # m, estimated from lidar (all tags agree within a few mm)
EXCLUDE = {12}             # coordinate inconsistent in 2 independent runs -> held out, reported
KF = 5                     # keyframe every 5 scans (~0.5 s)
LIDAR_X = -0.064
CMAP = plt.get_cmap("tab10")

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)
def inverse(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]), -(-s * a[..., 0] + c * a[..., 1]), -a[..., 2]], -1)
def between(a, b): return compose(inverse(a), b)
def interp_pose(T, P, t): return np.stack([np.interp(t, T, P[:, i]) for i in range(3)], -1)
def rigid_fit(P, W):
    mp, mw = P.mean(0), W.mean(0); U, _, Vt = np.linalg.svd((P - mp).T @ (W - mw)); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, mw - R @ mp

all_maps = {}
for run in RUNS:
    print(f"\n===== {run} =====", flush=True)
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv")
    T = gt.t_ns.values.astype(np.int64); P = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]
    ts = lambda t: (t - T[0]) / 1e9
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px < 1.0) & (det.side_px > 18) & det.tag_id.isin(list(TAGS)) &
              (det.t_ns >= T[0]) & (det.t_ns <= T[-1])].copy()
    b = (t_bc[:, None] + R_bc @ (TAG_SIZE * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    det["bx"], det["by"] = b[:, 0], b[:, 1]; det["rng"] = np.hypot(b[:, 0], b[:, 1])
    det = det.sort_values(["tag_id", "t_ns"])
    newc = (det.tag_id != det.tag_id.shift()) | (det.t_ns.diff() > 2e9)
    det["cluster"] = newc.cumsum()
    # keyframes + nearest keyframe per detection
    kidx = np.unique(np.r_[np.arange(0, len(T), KF), len(T) - 1]); Tk, Pk = T[kidx], P[kidx]; n = len(kidx)
    j = np.clip(np.searchsorted(Tk, det.t_ns.values), 1, n - 1)
    j = np.where(np.abs(Tk[j - 1] - det.t_ns.values) < np.abs(Tk[j] - det.t_ns.values), j - 1, j)
    det["kf"] = j
    det = det.sort_values("side_px", ascending=False).drop_duplicates(["tag_id", "kf"]).sort_values("t_ns").reset_index(drop=True)
    dl = between(Pk[det.kf.values], interp_pose(T, P, det.t_ns.values))
    q = compose(dl, np.c_[det.bx, det.by, np.zeros(len(det))])[:, :2]           # tag in keyframe frame
    W = np.array([TAGS[t] for t in det.tag_id]); sig = 0.03 + 0.03 * det.rng.values
    kfj = det.kf.values
    use = ~det.tag_id.isin(EXCLUDE).values
    # rigid initialisation on per-tag medians
    tag_slam = compose(Pk[kfj], np.c_[q, np.zeros(len(q))])[:, :2]
    used_tags = sorted(det.tag_id[use].unique())
    med = np.array([np.median(tag_slam[use & (det.tag_id.values == t)], 0) for t in used_tags])
    Rm, Tr = rigid_fit(med, np.array([TAGS[t] for t in used_tags]))
    G = np.array([Tr[0], Tr[1], np.arctan2(Rm[1, 0], Rm[0, 0])])
    X0 = compose(np.broadcast_to(G, Pk.shape), Pk)
    D = between(Pk[:-1], Pk[1:]); dist = np.hypot(D[:, 0], D[:, 1])
    sxy = 0.01 + 0.03 * dist; sth = np.radians(0.3) + 0.03 * np.abs(D[:, 2])

    def tag_pred(X, m):
        k = kfj[m]; c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
        return np.c_[X[k, 0] + c * q[m, 0] - s * q[m, 1], X[k, 1] + s * q[m, 0] + c * q[m, 1]]

    def solve(mask, x0):
        k, qm, Wm, sm, mm = kfj[mask], q[mask], W[mask], sig[mask], int(mask.sum())
        def fun(v):
            X = v.reshape(-1, 3)
            E = between(X[:-1], X[1:]) - D; E[:, 2] = wrap(E[:, 2])
            c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
            pr = np.c_[X[k, 0] + c * qm[:, 0] - s * qm[:, 1], X[k, 1] + s * qm[:, 0] + c * qm[:, 1]]
            return np.r_[np.c_[E[:, 0] / sxy, E[:, 1] / sxy, E[:, 2] / sth].ravel(), ((pr - Wm) / sm[:, None]).ravel()]
        J = lil_matrix((3 * (n - 1) + 2 * mm, 3 * n), dtype=np.int8)
        for i in range(n - 1): J[3 * i:3 * i + 3, 3 * i:3 * i + 6] = 1
        for i, kk in enumerate(k): J[3 * (n - 1) + 2 * i:3 * (n - 1) + 2 * i + 2, 3 * kk:3 * kk + 3] = 1
        r = least_squares(fun, x0.ravel(), jac_sparsity=J, loss="huber", f_scale=3.0, x_scale="jac", max_nfev=300)
        return r.x.reshape(-1, 3)

    X = solve(use, X0)
    # per-sighting residuals: rigid / corrected / held-out (leave that sighting out)
    rows = []
    for cid, grp in det.groupby("cluster"):
        m = (det.cluster == cid).values; tid = int(grp.tag_id.iloc[0])
        e_rig = np.linalg.norm(tag_pred(X0, m) - W[m], axis=1)
        e_cor = np.linalg.norm(tag_pred(X, m) - W[m], axis=1)
        Xh = X if tid in EXCLUDE else solve(use & ~m, X)
        e_ho = np.linalg.norm(tag_pred(Xh, m) - W[m], axis=1)
        dv = np.median(tag_pred(Xh, m) - W[m], 0)
        rows.append(dict(tag=tid, cluster=cid, t_start=ts(grp.t_ns.min()), t_end=ts(grp.t_ns.max()), n=int(m.sum()),
                         range_m=float(grp.rng.median()), used=tid not in EXCLUDE,
                         rigid_cm=100 * np.median(e_rig), corrected_cm=100 * np.median(e_cor),
                         heldout_cm=100 * np.median(e_ho), heldout_dx_cm=100 * dv[0], heldout_dy_cm=100 * dv[1]))
    R = pd.DataFrame(rows).sort_values("t_start")
    print(R.round(1).to_string(index=False), flush=True)
    # densify: interpolate keyframe corrections (world <- slam) onto every SLAM pose
    C = compose(X, inverse(Pk)); C[:, 2] = np.unwrap(C[:, 2])
    Ct = np.stack([np.interp(T, Tk, C[:, i]) for i in range(3)], -1)
    Pw = compose(Ct, P); Prig = compose(np.broadcast_to(G, P.shape), P)
    t_first, t_last = det[use].t_ns.min(), det[use].t_ns.max()
    anchored = (T >= t_first) & (T <= t_last)
    out = gt.copy(); out["x"], out["y"], out["yaw"] = Pw[:, 0], Pw[:, 1], wrap(Pw[:, 2])
    out["corr_m"] = np.hypot(*(Pw[:, :2] - Prig[:, :2]).T); out["anchored"] = anchored.astype(int)
    out.to_csv(f"{STG}/{run}/ground_truth/gt_pose_world.csv", index=False)
    R.to_csv(f"{STG}/{run}/ground_truth/tag_residuals.csv", index=False)
    print(f"  unanchored: first {ts(t_first):.1f}s at start, last {ts(T[-1])-ts(t_last):.1f}s at end;  "
          f"max correction vs rigid {out.corr_m.max():.2f} m;  median held-out error (used tags) "
          f"{R[R.used].heldout_cm.median():.1f} cm", flush=True)

    # corrected lidar map in the world frame (every 3rd scan, every 2nd beam)
    z = np.load(f"{STG}/{run}/lidar/scans.npz"); off = z["offsets"]; pts = []
    for i in range(0, len(T), 3):
        a, r = z["angles"][off[i]:off[i + 1]:2], z["ranges"][off[i]:off[i + 1]:2]
        ok = np.isfinite(r) & (r > 0.1) & (r < 8); a, r = a[ok], r[ok]
        xb, yb = LIDAR_X + r * np.cos(a), r * np.sin(a); c, s = np.cos(Pw[i, 2]), np.sin(Pw[i, 2])
        pts.append(np.c_[Pw[i, 0] + c * xb - s * yb, Pw[i, 1] + s * xb + c * yb])
    mapw = np.concatenate(pts); all_maps[run] = (mapw, Pw); np.savez_compressed(f"{STG}/{run}/ground_truth/map_points_world.npz", xy=mapw)

    # ---------------- figure 1: ground truth on the building frame ----------------
    seen = sorted(det.tag_id.unique()); col = {t: CMAP(i % 10) for i, t in enumerate(seen)}
    tsec = ts(T)
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.scatter(mapw[:, 0], mapw[:, 1], s=0.25, c="#b9c6d6", zorder=0, rasterized=True)
    ax.plot(Prig[:, 0], Prig[:, 1], "--", color="0.45", lw=1.1, zorder=1, label="SLAM, single rigid fit (before correction)")
    sc = ax.scatter(Pw[:, 0], Pw[:, 1], c=tsec, cmap="viridis", s=3, zorder=2, label="ground truth (tag-corrected)")
    ax.plot(*Pw[0, :2], "o", ms=12, mfc="lime", mec="k", zorder=5, label="start")
    ax.plot(*Pw[-1, :2], "s", ms=12, mfc="red", mec="k", zorder=5, label="end")
    for tid, (x, y) in TAGS.items():
        s_ = tid in seen
        ax.plot(x, y, "s", ms=11 if s_ else 6, mfc=col[tid] if s_ else "white", mec="k", zorder=4)
        ax.annotate(str(tid), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=11 if s_ else 7,
                    fontweight="bold" if s_ else "normal", color="k" if s_ else "0.5", zorder=6)
    for cid, grp in det.groupby("cluster"):
        m = (det.cluster == cid).values; tid = int(grp.tag_id.iloc[0]); pr = tag_pred(X, m)
        ax.scatter(pr[:, 0], pr[:, 1], s=6, color=col[tid], zorder=3)
        k = kfj[m][len(grp) // 2]; ax.plot([X[k, 0], TAGS[tid][0]], [X[k, 1], TAGS[tid][1]], "-", color=col[tid], lw=0.8, alpha=0.7, zorder=3)
    for tid in EXCLUDE & set(seen):
        rr = R[R.tag == tid].iloc[0]
        ax.annotate(f"tag {tid}: CAD coord looks off\n(seen {rr.heldout_cm:.0f} cm away)", TAGS[tid], xytext=(15, -35),
                    textcoords="offset points", fontsize=9, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"))
    xs = np.r_[Pw[:, 0], [TAGS[t][0] for t in seen]]; ys = np.r_[Pw[:, 1], [TAGS[t][1] for t in seen]]
    ax.set_xlim(xs.min() - 3, xs.max() + 3); ax.set_ylim(ys.min() - 3, ys.max() + 3)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x (m, building/CAD frame)"); ax.set_ylabel("y (m)")
    fig.colorbar(sc, ax=ax, fraction=0.025, label="time since start (s)")
    ax.legend(loc="best", fontsize=9)
    L = np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum()
    ax.set_title(f"{run}: ground truth on the building frame  |  {tsec[-1]:.0f} s, {L:.1f} m  |  tags used {[t for t in used_tags]}  |  "
                 f"median held-out tag error {R[R.used].heldout_cm.median():.0f} cm", fontsize=12)
    plt.tight_layout(); fig.savefig(f"{REV}/{run}_gt_world.png", dpi=110); plt.close(fig)

    # ---------------- figure 2: diagnostics ----------------
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw=dict(height_ratios=[1, 0.8, 1.1]))
    axs[0].plot(tsec, out.corr_m, color="tab:purple"); axs[0].set_ylabel("correction vs rigid (m)")
    for _, rr in R.iterrows():
        for a_ in axs[:2]: a_.axvspan(rr.t_start, rr.t_end + 0.3, color=col[rr.tag], alpha=0.25)
        axs[0].text((rr.t_start + rr.t_end) / 2, axs[0].get_ylim()[1] * 0.9, f"#{int(rr.tag)}", ha="center", fontsize=9)
    axs[0].set_title("How much the tags moved the SLAM trajectory (shaded = tag in view)"); axs[0].grid(alpha=0.3)
    axs[1].plot(tsec, gt.rmse_m * 1000, color="0.3", lw=0.8); axs[1].set_ylabel("SLAM scan fit (mm)"); axs[1].grid(alpha=0.3)
    axs[1].set_xlabel("time since start (s)")
    lab = [f"#{int(r_.tag)}\n{r_.t_start:.0f}s" for _, r_ in R.iterrows()]; xx = np.arange(len(R)); w = 0.27
    axs[2].bar(xx - w, R.rigid_cm, w, label="single rigid fit", color="0.6")
    axs[2].bar(xx, R.corrected_cm, w, label="tag-corrected (tag used)", color="tab:green")
    axs[2].bar(xx + w, R.heldout_cm, w, label="held out (tag hidden from the fit)", color="tab:orange")
    for i_, r_ in enumerate(R.itertuples()):
        if not r_.used: axs[2].text(i_, max(r_.rigid_cm, r_.heldout_cm) + 3, "not used\n(coord?)", ha="center", fontsize=8, color="crimson")
    axs[2].set_xticks(xx); axs[2].set_xticklabels(lab); axs[2].set_ylabel("tag position error (cm)")
    axs[2].set_title("Per tag sighting: error of the ground truth at that tag"); axs[2].legend(fontsize=9); axs[2].grid(alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(f"{REV}/{run}_gt_diagnostics.png", dpi=105); plt.close(fig)
    print(f"  figures -> {REV}/{run}_gt_world.png, _gt_diagnostics.png", flush=True)

# ---------------- all runs together: do their maps agree? ----------------
fig, ax = plt.subplots(figsize=(18, 8))
for i, (run, (mapw, Pw)) in enumerate(all_maps.items()):
    ax.scatter(mapw[::2, 0], mapw[::2, 1], s=0.2, color=CMAP(i), alpha=0.35, rasterized=True)
    ax.plot(Pw[:, 0], Pw[:, 1], "-", color=CMAP(i), lw=1.6, label=run)
for tid, (x, y) in TAGS.items():
    ax.plot(x, y, "ks", ms=6, mfc="yellow"); ax.annotate(str(tid), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(markerscale=6); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("All corrected runs + their lidar maps in one building frame (walls from different runs should coincide)")
plt.tight_layout(); fig.savefig(f"{REV}/all_runs_world.png", dpi=100); plt.close(fig)
print("GT_POSEGRAPH_DONE", flush=True)
