"""Shared pieces of the golden-run ground-truth pipeline (fused motion model, tag sightings, maps)."""
import json, numpy as np, pandas as pd
from scipy.stats import theilslopes
from scipy.spatial.transform import Rotation as Rot

DATA = "/mnt/x/side_navlori/data"; STG = DATA + "/staging_golden"; CAM = "/root/navlori/golden_cam"
CAD = {int(k): np.array([v["x"], v["y"]]) for k, v in json.load(open(DATA + "/tags_ground_truth.json")).items()}
TAG_SIZE = 0.150            # m, black square (print_tags.html: 15 cm image, no white margin; lidar check 15.1 cm)
LIDAR_X = -0.064            # base_scan origin in base_footprint
REL_REPROJ, MIN_SIDE = 0.015, 18
GYRO_STILL = json.load(open(STG + "/_network/gyro_bias.json"))   # lidar-confirmed stillness estimates

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def compose(a, b):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], a[..., 2] + b[..., 2]], -1)
def inverse(a):
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-(c * a[..., 0] + s * a[..., 1]), -(-s * a[..., 0] + c * a[..., 1]), -a[..., 2]], -1)
def between(a, b): return compose(inverse(a), b)
def interp_pose(T, P, t): return np.stack([np.interp(t, T, P[:, i]) for i in range(3)], -1)
def procrustes(P, Q, w=None):
    w = np.ones(len(P)) if w is None else w; w = w / w.sum(); mp, mq = w @ P, w @ Q
    U, _, Vt = np.linalg.svd((P - mp).T @ ((Q - mq) * w[:, None])); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, mq - R @ mp

def fused(run):
    """Motion model: heading = gyro (bias removed), translation = lidar SLAM steps (wheel steps where SLAM jumps)."""
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv"); T = gt.t_ns.values.astype(np.int64); N = len(T)
    S = np.c_[gt.x.values, gt.y.values, np.unwrap(gt.yaw.values)]; t = (T - T[0]) / 1e9
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns"); od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    ti = (imu.t_ns.values - T[0]) / 1e9
    graw = np.interp(t, ti, np.cumsum(imu.wz.values * np.diff(ti, prepend=ti[0])))
    d = graw - S[:, 2]; k = np.arange(0, N, max(1, N // 400))
    slope, icpt = theilslopes(d[k], t[k])[:2]; spread = np.ptp(np.percentile(d - (icpt + slope * t), [5, 95]))
    if spread < np.radians(20):
        bias, how = slope, "gyro-vs-SLAM heading trend (Theil-Sen)"
    else:
        bias, how = np.radians(GYRO_STILL[run]["bias_still_dps"]), "lidar-confirmed stillness (SLAM heading unusable)"
    G = graw - bias * t
    O = interp_pose(od.t_ns.values, np.c_[od.x.values, od.y.values, np.unwrap(od.yaw.values)], T)
    dS, dO = between(S[:-1], S[1:]), between(O[:-1], O[1:])
    jump = np.hypot(*(dS[:, :2] - dO[:, :2]).T) > 0.03 + 0.3 * np.hypot(*dO[:, :2].T)
    step = np.c_[np.where(jump[:, None], dO[:, :2], dS[:, :2]), np.diff(G)]
    F = np.zeros((N, 3)); F[0] = S[0]
    for i in range(N - 1): F[i + 1] = compose(F[i], step[i])
    return dict(gt=gt, T=T, t=t, S=S, F=F, G=G, jump=jump, dS=dS, bias_dps=float(np.degrees(bias)), bias_method=how,
                slam_gyro_heading_dev_deg=np.degrees(wrap(S[:, 2] - (G - G[0] + S[0, 2]))))

def sightings(run, T, frames):
    """Tag detections -> positions in each given trajectory frame (dict name->(N,3) poses) + clusters."""
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv")
    det = det[(det.reproj_px / det.side_px < REL_REPROJ) & (det.side_px > MIN_SIDE) & det.tag_id.isin(list(CAD)) &
              (det.t_ns >= T[0]) & (det.t_ns <= T[-1])].copy()
    b = (t_bc[:, None] + R_bc @ (TAG_SIZE * det[["tx_u", "ty_u", "tz_u"]].values.T))[:2].T
    det["bx"], det["by"] = b[:, 0], b[:, 1]; det["rng"] = np.hypot(b[:, 0], b[:, 1])
    det["obl"] = [np.degrees(np.arccos(abs(Rot.from_rotvec([r.rx, r.ry, r.rz]).as_matrix()[2, 2]))) for r in det.itertuples()]
    for name, P in frames.items():
        p = compose(interp_pose(T, P, det.t_ns.values), np.c_[b, np.zeros(len(b))])[:, :2]
        det[f"{name}_x"], det[f"{name}_y"] = p[:, 0], p[:, 1]
    det = det.sort_values(["tag_id", "t_ns"]); det["cl"] = ((det.tag_id != det.tag_id.shift()) | (det.t_ns.diff() > 2e9)).cumsum()
    return det.reset_index(drop=True)

def scan_points(run, T, P, every=3, beam=2, rmax=8.0):
    z = np.load(f"{STG}/{run}/lidar/scans.npz"); off = z["offsets"]; pts = []
    for i in range(0, len(T), every):
        a, r = z["angles"][off[i]:off[i + 1]:beam], z["ranges"][off[i]:off[i + 1]:beam]
        ok = np.isfinite(r) & (r > 0.1) & (r < rmax); a, r = a[ok], r[ok]
        xb, yb = LIDAR_X + r * np.cos(a), r * np.sin(a); c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
        pts.append(np.c_[P[i, 0] + c * xb - s * yb, P[i, 1] + s * xb + c * yb])
    return np.concatenate(pts)
