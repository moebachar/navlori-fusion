#!/usr/bin/env python
# Package the 12 golden runs into the conventional side_navlori run layout (like data/run2) + floor plan + plots.
import os, re, sys, json, shutil, subprocess, tempfile, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_common import DATA, STG, CAM, CAD, TAG_SIZE, wrap

RUNS = [f"golden_run_{i}" for i in range(1, 13)]
BAGS = DATA + "/bags"; FPD = DATA + "/golden_floorplan"; REV = "/mnt/x/side_navlori/gt_review"
KALIBR = "/mnt/x/side_navlori/calib/kalibr_640x480/calib_cam-camchain.yaml"
NET = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(f"{STG}/_network/tags_estimated.json")).items()}
RES = 0.05
os.makedirs(BAGS, exist_ok=True); os.makedirs(FPD, exist_ok=True)
INFO = {r: json.load(open(f"{STG}/{r}/ground_truth/gt_info.json")) for r in RUNS}
log = lambda *a: print(*a, flush=True)

# ---------------- floor plan: union of all tag-anchored lidar maps ----------------
allpts = np.concatenate([np.load(f"{STG}/{r}/ground_truth/map_points_world.npz")["xy"] for r in RUNS])
x0, y0 = np.floor(allpts.min(0) - 1.0); x1, y1 = np.ceil(allpts.max(0) + 1.0)
nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
H = np.histogram2d(allpts[:, 0], allpts[:, 1], bins=[nx, ny], range=[[x0, x1], [y0, y1]])[0].T
Wl = np.log1p(H); Wl[H < 2] = 0
FP = 1 - 0.88 * np.clip(Wl / np.percentile(Wl[Wl > 0], 97), 0, 1)                  # 1 = white, walls dark
EXT = [x0, x1, y0, y1]
Image.fromarray((FP[::-1] * 255).astype(np.uint8)).save(f"{FPD}/floorplan.png")
open(f"{FPD}/floorplan.yaml", "w").write(
    f"# Floor plan built from the lidar of all 12 golden runs, after tag-network alignment.\n"
    f"# Pixel (0,0) is the TOP-left; world = origin + (col, rows-1-row) * resolution.\nimage: floorplan.png\n"
    f"resolution: {RES}\norigin: [{x0}, {y0}]\nframe: building / CAD frame of tags_ground_truth.json (m)\nsize_px: [{nx}, {ny}]\n")
log(f"floor plan {nx}x{ny} px at {RES} m, x {x0}..{x1}, y {y0}..{y1}")

def floor(ax, alpha=1.0):
    ax.imshow(FP, extent=EXT, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=alpha, zorder=0)
def tags(ax, seen=(), fs=8):
    for t in sorted(CAD):
        x, y = NET.get(t, CAD[t]); s = t in seen
        ax.plot(x, y, "s", ms=9 if s else 5, mfc="gold" if s else "white", mec="k", mew=1.0 if s else 0.6, zorder=6)
        ax.annotate(str(t), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=fs + (2 if s else 0),
                    fontweight="bold" if s else "normal", color="k" if s else "0.45", zorder=7)

def acc_text(i):
    if i["status"] == "ok":
        h = i.get("heldout_median_cm")
        return (f"held-out tag error {h:.0f} cm (median), {i['heldout_max_cm']:.0f} cm max" if h == h and h is not None
                else f"only 2 tags -> fit {i['fit_median_cm']:.0f} cm (no held-out)")
    return f"map-matched: {i['wall_within_10cm_pct']:.0f}% walls within 10 cm, median gap {i['wall_gap_median_cm']:.1f} cm"

# ---------------- per run ----------------
summary = []
for run in RUNS:
    i = INFO[run]; out = f"{DATA}/{run}"; stg = f"{STG}/{run}"
    # 1) move the raw bag out of the way (data/bags is gitignored)
    if os.path.exists(f"{out}/metadata.yaml") and not os.path.exists(f"{BAGS}/{run}"):
        try: os.rename(out, f"{BAGS}/{run}")
        except OSError: shutil.move(out, f"{BAGS}/{run}")
        log(f"{run}: raw bag -> data/bags/{run}")
    for d in ["camera", "imu", "wheel_odom", "mag", "wifi", "lidar", "ground_truth", "calib"]: os.makedirs(f"{out}/{d}", exist_ok=True)
    # 2) sensor data (exported with data/scripts/export_*.py)
    for rel in ["imu/imu.csv", "mag/mag.csv", "wheel_odom/odom.csv", "wheel_odom/joint_states.csv", "wifi/wifi.csv",
                "wifi/wifi_raw.jsonl", "lidar/scans.npz", "lidar/scans_meta.csv"]:
        shutil.copy2(f"{stg}/{rel}", f"{out}/{rel}")
    shutil.copy2(f"{CAM}/{run}/camera/camera.csv", f"{out}/camera/camera.csv")
    if not os.path.isdir(f"{out}/camera/images"):
        subprocess.run(["cp", "-r", f"{CAM}/{run}/camera/images", f"{out}/camera/"], check=True)
    # 3) ground truth
    g = pd.read_csv(f"{stg}/ground_truth/gt_pose_world.csv"); g.to_csv(f"{out}/ground_truth/gt_pose.csv", index=False)
    shutil.copy2(f"{stg}/ground_truth/gt_pose.csv", f"{out}/ground_truth/gt_pose_slam_raw.csv")
    sa = pd.read_csv(f"{stg}/ground_truth/gt_slam_aligned.csv"); sa.to_csv(f"{out}/ground_truth/gt_pose_slam_aligned.csv", index=False)
    shutil.copy2(f"{stg}/ground_truth/tag_residuals.csv", f"{out}/ground_truth/tag_residuals.csv")
    shutil.copy2(f"{stg}/ground_truth/gt_info.json", f"{out}/ground_truth/gt_info.json")
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    for a in "xyz": det[f"t{a}_m"] = det[f"t{a}_u"] * TAG_SIZE
    det.to_csv(f"{out}/ground_truth/tag_detections.csv", index=False)
    xy = np.load(f"{stg}/ground_truth/map_points_world.npz")["xy"]
    kk = cKDTree(xy).query(xy, k=8)[1]; nb = xy[kk] - xy[kk].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", nb, nb); w, V = np.linalg.eigh(C); nrm = V[:, :, 0]
    np.savez_compressed(f"{out}/ground_truth/map_points.npz", xy=xy, normals=nrm)
    # occupancy-style map (points only, free space not ray-cast)
    mx0, my0 = np.floor(xy.min(0) - 1); mx1, my1 = np.ceil(xy.max(0) + 1); mnx, mny = int((mx1 - mx0) / RES), int((my1 - my0) / RES)
    Hm = np.histogram2d(xy[:, 0], xy[:, 1], bins=[mnx, mny], range=[[mx0, mx1], [my0, my1]])[0].T
    Image.fromarray(np.where(Hm >= 2, 0, 254).astype(np.uint8)[::-1]).save(f"{out}/ground_truth/map.png")
    open(f"{out}/ground_truth/map.yaml", "w").write(f"image: map.png\nresolution: {RES}\norigin: [{mx0}, {my0}, 0.0]\nnegate: 0\n"
        f"occupied_thresh: 0.65\nfree_thresh: 0.196\nframe: building / CAD frame (m)\n# occupied = lidar hits of this run; free space NOT ray-cast\n")
    t = (g.t_ns.values - g.t_ns.values[0]) / 1e9; seen = i["used_tags"]; L = i["path_m"]
    # --- plot A: this path alone on the floor plan (gt_overview.png) ---
    fig, ax = plt.subplots(figsize=(18, 8.2)); floor(ax); tags(ax, seen)
    sc = ax.scatter(g.x, g.y, c=t, cmap="viridis", s=4, zorder=4)
    ax.plot(g.x.iloc[0], g.y.iloc[0], "o", ms=13, mfc="lime", mec="k", zorder=8, label="start")
    ax.plot(g.x.iloc[-1], g.y.iloc[-1], "s", ms=12, mfc="red", mec="k", zorder=8, label="end")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01, label="time since start (s)"); ax.legend(loc="upper left")
    ax.set_title(f"{run}  |  {t[-1]:.0f} s, {L:.1f} m  |  tags {seen}  |  {acc_text(i)}", fontsize=13)
    plt.tight_layout(); fig.savefig(f"{out}/ground_truth/gt_overview.png", dpi=100); plt.close(fig)
    # --- plot B: raw lidar SLAM vs corrected (slam_vs_corrected.png) ---
    pos_err = np.hypot(g.x - sa.x, g.y - sa.y); hd = np.degrees(wrap(np.radians(sa.heading_slam_minus_gyro_deg)))
    fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1, 1])
    ax = fig.add_subplot(gs[0]); floor(ax, 0.9); tags(ax, seen)
    ax.plot(sa.x, sa.y, "--", color="crimson", lw=1.6, zorder=4, label="raw lidar SLAM (best rigid fit to the tags)")
    ax.plot(g.x, g.y, "-", color="tab:blue", lw=2.2, zorder=5, label="ground truth (gyro heading + SLAM distances + tag network)")
    jm = sa.slam_jump_step.values.astype(bool)
    if jm.any(): ax.plot(sa.x[jm], sa.y[jm], "x", color="k", ms=10, mew=2, zorder=6, label="SLAM jump (step replaced by wheels)")
    ax.plot(g.x.iloc[0], g.y.iloc[0], "o", ms=12, mfc="lime", mec="k", zorder=8); ax.plot(g.x.iloc[-1], g.y.iloc[-1], "s", ms=11, mfc="red", mec="k", zorder=8)
    pad = 2.5; ax.set_xlim(min(g.x.min(), sa.x.min()) - pad, max(g.x.max(), sa.x.max()) + pad)
    ax.set_ylim(min(g.y.min(), sa.y.min()) - pad, max(g.y.max(), sa.y.max()) + pad); ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=10); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{run}: raw lidar SLAM vs corrected ground truth  |  gyro offset removed {i['gyro_offset_dps']:+.2f} deg/s", fontsize=13)
    a2 = fig.add_subplot(gs[1]); a2.plot(t, pos_err, color="tab:purple"); a2.set_ylabel("position gap (m)"); a2.grid(alpha=.3)
    a2.set_title(f"distance between raw SLAM and ground truth: median {np.median(pos_err):.2f} m, max {pos_err.max():.2f} m")
    a3 = fig.add_subplot(gs[2]); a3.plot(t, hd, color="crimson"); a3.axhline(0, color="k", lw=.5); a3.set_ylabel("SLAM heading - GT (deg)")
    a3.set_xlabel("time since start (s)"); a3.grid(alpha=.3)
    plt.tight_layout(); fig.savefig(f"{out}/ground_truth/slam_vs_corrected.png", dpi=95); plt.close(fig)
    # --- map preview: this run's own lidar map + path ---
    fig, ax = plt.subplots(figsize=(12, 9)); ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="0.55", rasterized=True)
    ax.plot(g.x, g.y, "-", color="tab:blue", lw=1.5); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title(f"{run}: lidar map of this run (building frame)")
    plt.tight_layout(); fig.savefig(f"{out}/ground_truth/map_preview.png", dpi=90); plt.close(fig)
    # 4) calibration
    with tempfile.TemporaryDirectory() as td:
        src = open(f"{DATA}/scripts/export_calib.py").read()
        src = re.sub(r'BAG = r".*?"', f'BAG = r"{BAGS}/{run}"', src, count=1); src = re.sub(r'OUT = r".*?"', f'OUT = r"{td}"', src, count=1)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, "export_calib.py", "exec"), {"__name__": "__main__"})
        shutil.copy2(f"{td}/extrinsics.yaml", f"{out}/calib/extrinsics.yaml")
    kal = open(KALIBR).read()
    fx, fy, cx, cy = [float(v) for v in re.search(r"intrinsics: \[(.*?)\]", kal).group(1).split(",")]
    k1, k2, p1, p2 = [float(v) for v in re.search(r"distortion_coeffs: \[(.*?)\]", kal).group(1).split(",")]
    open(f"{out}/calib/camera_intrinsics.yaml", "w").write(
        f"# MEASURED camera intrinsics for /camera/image_raw at 640x480 (imx219, cropped+binned sensor mode, ~28 deg HFOV).\n"
        f"# Kalibr (ethz-asl, Dockerfile_ros1_20_04), aprilgrid 8x6 30 mm, 42/266 frames used, reprojection error 0.35/0.39 px (1 sigma).\n"
        f"# Recorded 2026-09-25 on the same camera and mode as this run. Source: calib/kalibr_640x480/.\n"
        f"image_width: 640\nimage_height: 480\ncamera_name: imx219_640x480\ncamera_matrix:\n  rows: 3\n  cols: 3\n"
        f"  data: [{fx:.6f}, 0, {cx:.6f}, 0, {fy:.6f}, {cy:.6f}, 0, 0, 1]\ndistortion_model: plumb_bob\ndistortion_coefficients:\n  rows: 1\n  cols: 5\n"
        f"  data: [{k1:.8f}, {k2:.8f}, {p1:.8f}, {p2:.8f}, 0.0]\nrectification_matrix:\n  rows: 3\n  cols: 3\n  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
        f"projection_matrix:\n  rows: 3\n  cols: 4\n  data: [{fx:.6f}, 0, {cx:.6f}, 0, 0, {fy:.6f}, {cy:.6f}, 0, 0, 0, 1, 0]\n")
    cam = pd.read_csv(f"{out}/camera/camera.csv"); imu = pd.read_csv(f"{out}/imu/imu.csv"); od = pd.read_csv(f"{out}/wheel_odom/odom.csv")
    mag = pd.read_csv(f"{out}/mag/mag.csv"); wifi = pd.read_csv(f"{out}/wifi/wifi.csv", keep_default_na=False)
    z = np.load(f"{out}/lidar/scans.npz"); dur = t[-1]
    rate = lambda n: n / dur
    mag_dead = bool((mag.select_dtypes("number").drop(columns=[c for c in mag.columns if c.startswith("t_")], errors="ignore").abs().sum().sum()) == 0)
    open(f"{out}/calib/robot.yaml", "w").write(
        f"# Platform + measured rates for {run} (recorded 2026-09-24)\nplatform: TurtleBot3 Waffle Pi (ROBOTIS), ROS 2 Humble\n"
        f"camera: {{model: Raspberry Pi Camera v2 (imx219), mode: 640x480 RGB888 -> JPEG, rate_hz: {rate(len(cam)):.1f}, intrinsics: calib/camera_intrinsics.yaml}}\n"
        f"imu: {{source: OpenCR, rate_hz: {rate(len(imu)):.1f}, gyro_offset_removed_dps: {i['gyro_offset_dps']:.3f}, gyro_offset_method: \"{i['gyro_offset_method']}\"}}\n"
        f"wheel_odom: {{topics: [/odom, /joint_states], rate_hz: {rate(len(od)):.1f}}}\nlidar: {{model: LDS-02, rate_hz: {rate(len(z['t_ns'])):.2f}}}\n"
        f"wifi: {{scans: {wifi.scan_idx.nunique()}, mean_interval_s: {dur / max(1, wifi.scan_idx.nunique()):.1f}, access_points: {wifi.bssid.nunique()}}}\n"
        f"magnetometer: {{status: {'DEAD (all zeros)' if mag_dead else 'values present'}}}\n"
        f"apriltags: {{family: tag36h11, black_square_m: {TAG_SIZE}, positions: data/tags_ground_truth.json (CAD), estimated: calib/apriltags.yaml}}\n")
    open(f"{out}/calib/apriltags.yaml", "w").write(
        "# AprilTag network (tag36h11, 15.0 cm black square). cad = data/tags_ground_truth.json;\n"
        "# estimated = joint adjustment over all 12 golden runs (use to spot CAD coordinates that are off).\nfamily: tag36h11\nsize_m: 0.150\ntags:\n" +
        "".join(f"  {k}: {{cad: [{CAD[k][0]:.4f}, {CAD[k][1]:.4f}]" + (f", estimated: [{NET[k][0]:.4f}, {NET[k][1]:.4f}], offset_cm: {100*np.linalg.norm(NET[k]-CAD[k]):.1f}" if k in NET else ", estimated: null") + "}\n"
                for k in sorted(CAD)))
    # 5) docs
    res = pd.read_csv(f"{out}/ground_truth/tag_residuals.csv")
    method = "joint AprilTag network over all golden runs" if i["status"] == "ok" else "lidar map matching onto the tag-network runs (single tag seen)"
    open(f"{out}/ground_truth/method.md", "w").write(f"""# Ground truth — {run}

**Frame:** building / CAD frame of `data/tags_ground_truth.json` (metres). **Rows:** one pose per lidar scan (`t_ns` = scan stamp), `base_footprint`.

## How it is built
1. **Heading = gyro.** OpenCR `/imu` z-rate integrated; its constant offset ({i['gyro_offset_dps']:+.3f} deg/s here) is removed using {i['gyro_offset_method']}.
   Lidar SLAM heading is *not* used: it can jump (golden_run_7 had a 13 deg jump for 33 s).
2. **Distance = lidar SLAM.** Per-scan translation from the scan-to-map lidar SLAM (`data/scripts/build_ground_truth.py`),
   replaced by wheel odometry on steps where SLAM jumps ({i['wheel_fallback_steps']} steps here).
3. **Placement in the building = {method}.** The trajectory from 1+2 is rigid (tested: letting the tags bend it did not improve held-out accuracy,
   so the ~15-25 cm residuals at tags are tag/CAD noise, not path error).
   AprilTags: tag36h11, 15.0 cm, camera intrinsics measured with Kalibr (`calib/camera_intrinsics.yaml`).

## Accuracy
- {acc_text(i)}.
- Held-out = that tag sighting removed and the whole network re-solved, then the error at that tag measured (`tag_residuals.csv`).
- Walls from different golden runs coincide to ~6-7 cm (median) where they overlap.
- Local (relative) accuracy is much better than the absolute numbers: cm-level from lidar SLAM + gyro.

## Files
`gt_pose.csv` (this GT) · `gt_pose_slam_raw.csv` (raw lidar SLAM, its own frame) · `gt_pose_slam_aligned.csv` (raw SLAM, best rigid fit to the tags) ·
`tag_detections.csv` (all tag detections, PnP in camera frame: `t*_m` metric) · `tag_residuals.csv` · `gt_info.json` ·
`map_points.npz` (lidar map of this run, building frame) · `map.png/.yaml` · `gt_overview.png` · `slam_vs_corrected.png` · `map_preview.png`
""")
    wifi_n = wifi.scan_idx.nunique()
    open(f"{out}/README.md", "w").write(f"""# {run}

TurtleBot3 Waffle Pi, CESI building, recorded **2026-09-24** (one of 12 "golden runs": fresh bringup per run,
5 s still at start/end, start/end facing a mapped AprilTag). **{t[-1]:.0f} s, {L:.1f} m**, tags seen: {seen}.

![path](ground_truth/gt_overview.png)

## Contents (same layout as `run1`/`run2`, loads with `data/scripts/load_dataset.py`)
| folder | what | rate |
|---|---|---|
| `camera/` | `images/<t_ns>.jpg` 640x480 + `camera.csv` | {rate(len(cam)):.1f} Hz |
| `imu/` | `imu.csv` (OpenCR gyro/accel/orientation) | {rate(len(imu)):.0f} Hz |
| `wheel_odom/` | `odom.csv`, `joint_states.csv` | {rate(len(od)):.0f} Hz |
| `mag/` | `mag.csv` | {'dead (all zeros)' if mag_dead else 'present'} |
| `lidar/` | `scans.npz` (+ `scans_meta.csv`) LDS-02 | {rate(len(z['t_ns'])):.1f} Hz |
| `wifi/` | `wifi.csv`, `wifi_raw.jsonl` | {wifi_n} scans (~1 per {dur/max(1,wifi_n):.1f} s) |
| `ground_truth/` | `gt_pose.csv` + method, residuals, maps, plots | per lidar scan |
| `calib/` | measured camera intrinsics, extrinsics (from `/tf_static`), robot, AprilTags | |

## Ground truth
Building frame (m), same as `data/tags_ground_truth.json`. Gyro heading + lidar-SLAM distances, placed by the
{method}. **{acc_text(i)}.** Details: [`ground_truth/method.md`](ground_truth/method.md);
raw SLAM vs corrected: `ground_truth/slam_vs_corrected.png`.

## Differences from run1/run2
- Camera **640x480** with **measured** intrinsics (Kalibr), not the nominal 820x616 ones.
- Ground truth in the **building frame** (AprilTags), not the SLAM start frame; raw SLAM kept as `gt_pose_slam_raw.csv`.
- IMU/odom at ~{rate(len(imu)):.0f} Hz (was 20 Hz). `/cmd_vel` was not recorded (controller published elsewhere).
""")
    summary.append(dict(run=run, dur_s=round(t[-1]), path_m=round(L, 1), tags=",".join(map(str, seen)), gt=i["status"], accuracy=acc_text(i),
                        camera=len(cam), imu_hz=round(rate(len(imu))), wifi_scans=wifi_n))
    log(f"{run}: packaged ({len(cam)} frames, {t[-1]:.0f} s, {L:.1f} m) | {acc_text(i)}")

# ---------------- all paths on one floor plan ----------------
fig, ax = plt.subplots(figsize=(22, 10)); floor(ax)
cm = plt.get_cmap("tab20")
for k, run in enumerate(RUNS):
    g = pd.read_csv(f"{DATA}/{run}/ground_truth/gt_pose.csv")
    ax.plot(g.x, g.y, "-", color=cm(k * 20 // len(RUNS) % 20), lw=2.2, zorder=4, label=f"{run} ({INFO[run]['path_m']:.0f} m)")
    ax.plot(g.x.iloc[0], g.y.iloc[0], "o", ms=7, mfc=cm(k * 20 // len(RUNS) % 20), mec="k", zorder=5)
tags(ax, seen=set(NET), fs=9)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.legend(loc="upper left", fontsize=10, ncol=2, framealpha=0.9)
ax.set_title(f"All 12 golden runs on the floor plan (building frame)  |  {sum(INFO[r]['path_m'] for r in RUNS):.0f} m, "
             f"{sum(INFO[r]['duration_s'] for r in RUNS)/60:.1f} min in total  |  dot = start", fontsize=14)
plt.tight_layout(); fig.savefig(f"{FPD}/all_paths.png", dpi=100); plt.close(fig)
S = pd.DataFrame(summary); S.to_csv(f"{FPD}/golden_runs_summary.csv", index=False)
for f_ in ["all_paths.png"]: shutil.copy2(f"{FPD}/{f_}", f"{REV}/{f_}")
log(S.to_string(index=False)); log("PACKAGE_DONE")
