#!/usr/bin/env python
# Tag-corrected ground truth v2 — fused odometry + AprilTag pose graph.
#  motion model per scan step: rotation from the GYRO (never jumps), translation from lidar SLAM
#  (cm-accurate) with a wheel-odometry fallback on steps where SLAM jumps; then a 2D pose graph
#  pins that trajectory to the tags' CAD coordinates. Leave-one-sighting-out = held-out accuracy.
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
TAG_SIZE = 0.150; EXCLUDE = {12}; KF = 5; LIDAR_X = -0.064   # 15.0 cm black square (print_tags.html); lidar cross-check 15.15
NOFIG = os.environ.get("NOFIG") == "1"                                                     # compute + save, no review figures
# Stiff motion model (chosen by sweep 2026-09-27): held-out tag error was no worse than looser settings and the
# correction stays smooth (<1 cm), i.e. the gyro+SLAM path is right and the ~20 cm tag residuals are tag/CAD noise.
SXY0, SXY1 = float(os.environ.get("SXY0", 0.001)), float(os.environ.get("SXY1", 0.002))     # translation sigma per keyframe: a + b*dist
STH0, STH1 = np.radians(float(os.environ.get("STH0_DEG", 0.02))), float(os.environ.get("STH1", 0.002))
QUICK = os.environ.get("QUICK") == "1"                                                    # numbers only, no figures
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

all_maps = {}; summary = []
for run in RUNS:
    print(f"\n===== {run} =====", flush=True)
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv")
    T = gt.t_ns.values.astype(np.int64); N = len(T)
    S = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]                     # lidar SLAM
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns")
    od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    ts = lambda t: (t - T[0]) / 1e9
    ti = ts(imu.t_ns.values); bias = imu.wz.values[(ti > 0.3) & (ti < 4.0)].mean()
    gy = np.cumsum((imu.wz.values - bias) * np.diff(ti, prepend=ti[0]))                 # gyro heading
    Gs = np.interp(ts(T), ti, gy)
    O = interp_pose(od.t_ns.values, np.c_[od.x.values, od.y.values, np.unwrap(od.yaw.values)], T)  # wheel odom
    # ---- fused per-scan-step motion ----
    dS, dO = between(S[:-1], S[1:]), between(O[:-1], O[1:])
    jump = np.hypot(*(dS[:, :2] - dO[:, :2]).T) > 0.03 + 0.3 * np.hypot(*dO[:, :2].T)
    step = np.c_[np.where(jump[:, None], dO[:, :2], dS[:, :2]), np.diff(Gs)]
    F = np.zeros((N, 3)); F[0] = S[0]
    for i in range(N - 1): F[i + 1] = compose(F[i], step[i])
    rot_disagree = np.degrees(np.abs(wrap(dS[:, 2] - np.diff(Gs))))
    print(f"  steps using wheel fallback: {jump.sum()} / {N-1};  SLAM-vs-gyro step rotation > 2 deg: {(rot_disagree > 2).sum()} steps "
          f"(largest {rot_disagree.max():.1f} deg at t={ts(T[1:][rot_disagree.argmax()]):.0f}s)", flush=True)
    # ---- tag observations ----
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px < 1.0) & (det.side_px > 18) & det.tag_id.isin(list(TAGS)) & (det.t_ns >= T[0]) & (det.t_ns <= T[-1])].copy()
    b = (t_bc[:, None] + R_bc @ (TAG_SIZE * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    det["bx"], det["by"] = b[:, 0], b[:, 1]; det["rng"] = np.hypot(b[:, 0], b[:, 1])
    det = det.sort_values(["tag_id", "t_ns"])
    det["cluster"] = ((det.tag_id != det.tag_id.shift()) | (det.t_ns.diff() > 2e9)).cumsum()
    kidx = np.unique(np.r_[np.arange(0, N, KF), N - 1]); Tk, Fk = T[kidx], F[kidx]; n = len(kidx)
    j = np.clip(np.searchsorted(Tk, det.t_ns.values), 1, n - 1)
    det["kf"] = np.where(np.abs(Tk[j - 1] - det.t_ns.values) < np.abs(Tk[j] - det.t_ns.values), j - 1, j)
    det = det.sort_values("side_px", ascending=False).drop_duplicates(["tag_id", "kf"]).sort_values("t_ns").reset_index(drop=True)
    kfj = det.kf.values
    q = compose(between(Fk[kfj], interp_pose(T, F, det.t_ns.values)), np.c_[det.bx, det.by, np.zeros(len(det))])[:, :2]
    W = np.array([TAGS[t] for t in det.tag_id]); sig = 0.03 + 0.03 * det.rng.values
    use = ~det.tag_id.isin(EXCLUDE).values
    used_tags = sorted(int(t) for t in det.tag_id[use].unique())
    if len(used_tags) < 2:
        print(f"  !!! only {len(used_tags)} distinct usable tag(s) {used_tags}: rotation not observable from tag positions -> SKIPPED", flush=True)
        json.dump(dict(run=run, status="skipped_lt2_tags", used_tags=used_tags), open(f"{STG}/{run}/ground_truth/gt_info.json", "w"))
        continue
    tag_F = compose(Fk[kfj], np.c_[q, np.zeros(len(q))])[:, :2]
    med = np.array([np.median(tag_F[use & (det.tag_id.values == t)], 0) for t in used_tags])
    Rm, Tr = rigid_fit(med, np.array([TAGS[t] for t in used_tags]))
    G = np.array([Tr[0], Tr[1], np.arctan2(Rm[1, 0], Rm[0, 0])])
    X0 = compose(np.broadcast_to(G, Fk.shape), Fk)
    D = between(Fk[:-1], Fk[1:]); dist = np.hypot(D[:, 0], D[:, 1])
    sxy = SXY0 + SXY1 * dist                                    # translation: SLAM/wheels
    sth = STH0 + STH1 * np.abs(D[:, 2])                         # rotation: gyro, tight

    def tag_pred(X, m):
        k = kfj[m]; c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
        return np.c_[X[k, 0] + c * q[m, 0] - s * q[m, 1], X[k, 1] + s * q[m, 0] + c * q[m, 1]]
    def solve(mask, x0):
        k, qm, Wm, sm, mm = kfj[mask], q[mask], W[mask], sig[mask], int(mask.sum())
        def fun(v):
            X = v.reshape(-1, 3); E = between(X[:-1], X[1:]) - D; E[:, 2] = wrap(E[:, 2])
            c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
            pr = np.c_[X[k, 0] + c * qm[:, 0] - s * qm[:, 1], X[k, 1] + s * qm[:, 0] + c * qm[:, 1]]
            return np.r_[np.c_[E[:, 0] / sxy, E[:, 1] / sxy, E[:, 2] / sth].ravel(), ((pr - Wm) / sm[:, None]).ravel()]
        J = lil_matrix((3 * (n - 1) + 2 * mm, 3 * n), dtype=np.int8)
        for i in range(n - 1): J[3 * i:3 * i + 3, 3 * i:3 * i + 6] = 1
        for i, kk in enumerate(k): J[3 * (n - 1) + 2 * i:3 * (n - 1) + 2 * i + 2, 3 * kk:3 * kk + 3] = 1
        return least_squares(fun, x0.ravel(), jac_sparsity=J, loss="huber", f_scale=3.0, x_scale="jac", max_nfev=300).x.reshape(-1, 3)

    X = solve(use, X0)
    rows = []
    for cid, grp in det.groupby("cluster"):
        m = (det.cluster == cid).values; tid = int(grp.tag_id.iloc[0])
        left = set(det.tag_id[use & ~m].unique())
        if tid in EXCLUDE: Xh = X
        elif len(left) < 2: Xh = None          # hiding it leaves < 2 tags: rotation free -> no honest held-out value
        else: Xh = solve(use & ~m, X)
        dv = np.median(tag_pred(Xh, m) - W[m], 0) if Xh is not None else np.array([np.nan, np.nan])
        rows.append(dict(tag=tid, t_start=ts(grp.t_ns.min()), t_end=ts(grp.t_ns.max()), n=int(m.sum()), range_m=float(grp.rng.median()),
                         used=tid not in EXCLUDE, rigid_cm=100 * np.median(np.linalg.norm(tag_pred(X0, m) - W[m], axis=1)),
                         corrected_cm=100 * np.median(np.linalg.norm(tag_pred(X, m) - W[m], axis=1)),
                         heldout_cm=100 * np.median(np.linalg.norm(tag_pred(Xh, m) - W[m], axis=1)) if Xh is not None else np.nan,
                         heldout_dx_cm=100 * dv[0], heldout_dy_cm=100 * dv[1]))
    R = pd.DataFrame(rows).sort_values("t_start"); print(R.round(1).to_string(index=False), flush=True)
    C = compose(X, inverse(Fk)); C[:, 2] = np.unwrap(C[:, 2])
    Pw = compose(np.stack([np.interp(T, Tk, C[:, i]) for i in range(3)], -1), F)
    # reference: plain lidar SLAM rigidly fitted to the same tags (what we had before)
    tag_S = compose(interp_pose(T, S, det.t_ns.values), np.c_[det.bx, det.by, np.zeros(len(det))])[:, :2]
    medS = np.array([np.median(tag_S[use & (det.tag_id.values == t)], 0) for t in used_tags])
    RmS, TrS = rigid_fit(medS, np.array([TAGS[t] for t in used_tags]))
    Srig = compose(np.broadcast_to(np.array([TrS[0], TrS[1], np.arctan2(RmS[1, 0], RmS[0, 0])]), S.shape), S)
    t_first, t_last = det[use].t_ns.min(), det[use].t_ns.max()
    out = gt.copy(); out["x"], out["y"], out["yaw"] = Pw[:, 0], Pw[:, 1], wrap(Pw[:, 2])
    out["slam_jump_step"] = np.r_[0, jump.astype(int)]; out["anchored"] = ((T >= t_first) & (T <= t_last)).astype(int)
    out.to_csv(f"{STG}/{run}/ground_truth/gt_pose_world.csv", index=False)
    R.to_csv(f"{STG}/{run}/ground_truth/tag_residuals.csv", index=False)
    ho = R[R.used].heldout_cm
    summary.append(dict(run=run, dur_s=ts(T[-1]), path_m=np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum(), tags=used_tags,
                        heldout_med_cm=ho.median(), heldout_max_cm=ho.max(), unanchored_start_s=ts(t_first), unanchored_end_s=ts(T[-1]) - ts(t_last)))
    Frig_ = compose(np.broadcast_to(G, F.shape), F); corr = np.hypot(*(Pw[:, :2] - Frig_[:, :2]).T)
    rough = np.abs(np.diff(corr, 2)).max() * 1e4 if len(corr) > 2 else 0
    print(f"  held-out tag error: median {ho.median():.1f} cm, max {ho.max():.1f} cm | correction max {corr.max()*100:.1f} cm, "
          f"roughness {rough:.2f} | unanchored: {ts(t_first):.0f}s start, {ts(T[-1])-ts(t_last):.0f}s end", flush=True)
    summary[-1].update(corr_max_cm=corr.max() * 100, roughness=rough)
    if QUICK: continue
    z = np.load(f"{STG}/{run}/lidar/scans.npz"); off = z["offsets"]; pts = []
    for i in range(0, N, 3):
        a, r = z["angles"][off[i]:off[i + 1]:2], z["ranges"][off[i]:off[i + 1]:2]
        ok = np.isfinite(r) & (r > 0.1) & (r < 8); a, r = a[ok], r[ok]
        xb, yb = LIDAR_X + r * np.cos(a), r * np.sin(a); c, s = np.cos(Pw[i, 2]), np.sin(Pw[i, 2])
        pts.append(np.c_[Pw[i, 0] + c * xb - s * yb, Pw[i, 1] + s * xb + c * yb])
    mapw = np.concatenate(pts); all_maps[run] = (mapw, Pw)
    np.savez_compressed(f"{STG}/{run}/ground_truth/map_points_world.npz", xy=mapw)
    # raw lidar SLAM, rigidly placed on the same tags (for the "raw vs corrected" preview)
    pd.DataFrame(dict(t_ns=T, x=Srig[:, 0], y=Srig[:, 1], yaw=wrap(Srig[:, 2]),
                      heading_slam_minus_gyro_deg=np.degrees(wrap(S[:, 2] - (Gs - Gs[0] + S[0, 2]))),
                      slam_jump_step=np.r_[0, jump.astype(int)])).to_csv(f"{STG}/{run}/ground_truth/gt_slam_aligned.csv", index=False)
    json.dump(dict(run=run, status="ok", tag_size_m=TAG_SIZE, used_tags=used_tags, seen_tags=sorted(int(t) for t in det.tag_id.unique()),
                   excluded_tags=sorted(EXCLUDE & set(int(t) for t in det.tag_id.unique())), n_sightings=int(len(R)),
                   heldout_median_cm=float(ho.median()), heldout_max_cm=float(ho.max()), wheel_fallback_steps=int(jump.sum()),
                   slam_gyro_jump_steps=int((rot_disagree > 2).sum()), unanchored_start_s=float(ts(t_first)),
                   unanchored_end_s=float(ts(T[-1]) - ts(t_last)), duration_s=float(ts(T[-1])),
                   path_m=float(np.hypot(*np.diff(Pw[:, :2], axis=0).T).sum()), sigmas=dict(sxy=[SXY0, SXY1], sth_deg=[float(np.degrees(STH0)), STH1])),
              open(f"{STG}/{run}/ground_truth/gt_info.json", "w"), indent=1)
    if NOFIG: continue

    # ---------------- figure 1: GT on the building frame ----------------
    seen = sorted(int(t) for t in det.tag_id.unique()); col = {t: CMAP(i % 10) for i, t in enumerate(seen)}
    tsec = ts(T)
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.scatter(mapw[:, 0], mapw[:, 1], s=0.25, c="#b9c6d6", zorder=0, rasterized=True)
    ax.plot(Srig[:, 0], Srig[:, 1], "--", color="0.45", lw=1.1, zorder=1, label="lidar SLAM alone (fitted to tags) — before fix")
    sc = ax.scatter(Pw[:, 0], Pw[:, 1], c=tsec, cmap="viridis", s=3, zorder=2, label="ground truth (gyro heading + SLAM + tags)")
    ax.plot(*Pw[0, :2], "o", ms=12, mfc="lime", mec="k", zorder=5, label="start")
    ax.plot(*Pw[-1, :2], "s", ms=12, mfc="red", mec="k", zorder=5, label="end")
    for tid, (x, y) in TAGS.items():
        s_ = tid in seen
        ax.plot(x, y, "s", ms=11 if s_ else 6, mfc=col[tid] if s_ else "white", mec="k", zorder=4)
        ax.annotate(str(tid), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=11 if s_ else 7,
                    fontweight="bold" if s_ else "normal", color="k" if s_ else "0.5", zorder=6)
    for cid, grp in det.groupby("cluster"):
        m = (det.cluster == cid).values; tid = int(grp.tag_id.iloc[0]); pr = tag_pred(X, m)
        ax.scatter(pr[:, 0], pr[:, 1], s=7, color=col[tid], zorder=3)
        k = kfj[m][len(grp) // 2]; ax.plot([X[k, 0], TAGS[tid][0]], [X[k, 1], TAGS[tid][1]], "-", color=col[tid], lw=0.8, alpha=0.7, zorder=3)
    for tid in EXCLUDE & set(seen):
        rr = R[R.tag == tid].iloc[0]
        ax.annotate(f"tag {tid}: CAD coordinate looks off\n(robot sees it {rr.heldout_cm:.0f} cm away from it)", TAGS[tid], xytext=(-250, -60),
                    textcoords="offset points", fontsize=9, color="crimson", arrowprops=dict(arrowstyle="->", color="crimson"),
                    bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))
    xs = np.r_[Pw[:, 0], [TAGS[t][0] for t in seen]]; ys = np.r_[Pw[:, 1], [TAGS[t][1] for t in seen]]
    ax.set_xlim(xs.min() - 3, xs.max() + 3); ax.set_ylim(ys.min() - 3, ys.max() + 3)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x (m, building / CAD frame)"); ax.set_ylabel("y (m)")
    fig.colorbar(sc, ax=ax, fraction=0.025, label="time since start (s)"); ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{run}: {tsec[-1]:.0f} s, {summary[-1]['path_m']:.1f} m | tags used {used_tags} | held-out tag error: "
                 f"median {ho.median():.0f} cm, max {ho.max():.0f} cm", fontsize=12)
    plt.tight_layout(); fig.savefig(f"{REV}/{run}_gt_world.png", dpi=110); plt.close(fig)

    # ---------------- figure 2: diagnostics ----------------
    fig, axs = plt.subplots(3, 1, figsize=(14, 10.5), gridspec_kw=dict(height_ratios=[1, 0.9, 1.1]))
    dd = np.degrees(wrap(S[:, 2] - (Gs - Gs[0] + S[0, 2])))
    axs[0].plot(tsec, dd, color="tab:red", lw=1); axs[0].axhline(0, color="k", lw=0.5)
    jt = tsec[1:][jump]; axs[0].plot(jt, np.zeros_like(jt), "|", color="k", ms=10, label="step where SLAM jumped (wheels used)")
    axs[0].set_ylabel("SLAM heading - gyro (deg)"); axs[0].legend(fontsize=9); axs[0].grid(alpha=0.3)
    axs[0].set_title("Lidar SLAM heading vs gyro (the GT takes heading from the gyro)")
    Frig = compose(np.broadcast_to(G, F.shape), F)
    err = np.hypot(*(Pw[:, :2] - Frig[:, :2]).T)
    axs[1].plot(tsec, err, color="tab:purple"); axs[1].set_ylabel("correction by tags (m)"); axs[1].grid(alpha=0.3)
    axs[1].set_ylim(0, max(0.3, err.max() * 1.2))
    axs[1].set_title("How far the tags moved the gyro+SLAM trajectory (small = the motion model was already right)")
    for _, rr in R.iterrows():
        for a_ in axs[:2]: a_.axvspan(rr.t_start, rr.t_end + 0.3, color=col[int(rr.tag)], alpha=0.25)
        axs[1].text((rr.t_start + rr.t_end) / 2, axs[1].get_ylim()[1] * 0.88, f"#{int(rr.tag)}", ha="center", fontsize=9)
    axs[1].set_xlabel("time since start (s)  —  shaded = tag in view")
    lab = [f"#{int(r_.tag)}\n{r_.t_start:.0f}s" for _, r_ in R.iterrows()]; xx = np.arange(len(R)); w = 0.27
    axs[2].bar(xx - w, R.rigid_cm, w, label="fused odometry, single rigid fit", color="0.6")
    axs[2].bar(xx, R.corrected_cm, w, label="final GT (tag used)", color="tab:green")
    axs[2].bar(xx + w, R.heldout_cm, w, label="final GT with this tag HIDDEN = honest accuracy", color="tab:orange")
    for i_, r_ in enumerate(R.itertuples()):
        if not r_.used: axs[2].text(i_, max(r_.rigid_cm, r_.heldout_cm) + 2, "not used\n(coordinate?)", ha="center", fontsize=8, color="crimson")
    axs[2].set_ylim(0, max(R.rigid_cm.max(), R.heldout_cm.max()) * 1.35)
    axs[2].set_xticks(xx); axs[2].set_xticklabels(lab); axs[2].set_ylabel("error at the tag (cm)")
    axs[2].set_title("Per tag sighting: ground-truth error measured at that tag"); axs[2].legend(fontsize=9); axs[2].grid(alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(f"{REV}/{run}_gt_diagnostics.png", dpi=105); plt.close(fig)

if QUICK or NOFIG:
    print("\n" + pd.DataFrame(summary).round(1).to_string(index=False)); print("GT_FUSED_DONE", flush=True); sys.exit(0)
fig, ax = plt.subplots(figsize=(18, 8))
for i, (run, (mapw, Pw)) in enumerate(all_maps.items()):
    ax.scatter(mapw[::2, 0], mapw[::2, 1], s=0.2, color=CMAP(i), alpha=0.35, rasterized=True)
    ax.plot(Pw[:, 0], Pw[:, 1], "-", color=CMAP(i), lw=1.6, label=run)
for tid, (x, y) in TAGS.items():
    ax.plot(x, y, "ks", ms=6, mfc="yellow"); ax.annotate(str(tid), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(markerscale=6); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("All 3 corrected runs + their lidar maps in one building frame — walls from different runs should coincide")
plt.tight_layout(); fig.savefig(f"{REV}/all_runs_world.png", dpi=100); plt.close(fig)
# independent check (no tags involved): how far apart are the walls seen by different runs where they overlap?
from scipy.spatial import cKDTree
names = list(all_maps)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        A, B = all_maps[names[a]][0][::3], all_maps[names[b]][0][::3]
        d = cKDTree(B).query(A, distance_upper_bound=0.5)[0]; d = d[np.isfinite(d)]
        if len(d) > 500:
            print(f"wall agreement {names[a]} vs {names[b]}: {len(d)} overlapping points, median gap {np.median(d)*100:.1f} cm, "
                  f"p90 {np.percentile(d, 90)*100:.1f} cm", flush=True)
print("\n" + pd.DataFrame(summary).round(1).to_string(index=False)); print("GT_FUSED_DONE", flush=True)
