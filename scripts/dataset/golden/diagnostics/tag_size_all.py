#!/usr/bin/env python
# Per-tag printed size from lidar (wall range at the tag's bearing vs unit-size PnP), all golden runs.
import json, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
STG = "/mnt/x/side_navlori/data/staging_golden"; CAM = "/root/navlori/golden_cam"; LIDAR = np.array([-0.064, 0.0])
def lidar_range(z, t_ns, bearing):
    ts, off = z["t_ns"], z["offsets"]; k = int(np.clip(np.searchsorted(ts, t_ns), 1, len(ts) - 1))
    k = k if abs(ts[k] - t_ns) < abs(ts[k - 1] - t_ns) else k - 1
    a, r = z["angles"][off[k]:off[k + 1]], z["ranges"][off[k]:off[k + 1]]
    d = np.abs(np.angle(np.exp(1j * (a - bearing)))); rr = r[(d < np.radians(1.5)) & np.isfinite(r)]
    return float(np.median(rr)) if len(rr) else np.nan
rows = []
for i in range(1, 13):
    run = f"golden_run_{i}"
    det = pd.read_csv(f"{CAM}/{run}/tag_detections.csv"); z = np.load(f"{STG}/{run}/lidar/scans.npz")
    ex = json.load(open(f"{CAM}/{run}/extrinsics_cam.json"))
    t_bc = np.array(ex["t_base_cam"]); R_bc = Rot.from_quat(ex["q_base_cam_xyzw"]).as_matrix()
    det = det[(det.side_px > 30) & (det.reproj_px / det.side_px < 0.015)]
    for row in det.itertuples():
        v = (R_bc @ np.array([row.tx_u, row.ty_u, row.tz_u]))[:2]; a = t_bc[:2] - LIDAR; s = 0.15
        for _ in range(3):
            p = t_bc[:2] + s * v - LIDAR; r = lidar_range(z, row.t_ns, np.arctan2(p[1], p[0]))
            if not np.isfinite(r): s = np.nan; break
            A, B, C = v @ v, 2 * a @ v, a @ a - r * r; disc = B * B - 4 * A * C
            if disc < 0: s = np.nan; break
            s = (-B + np.sqrt(disc)) / (2 * A)
        # tag plane obliqueness: angle between camera axis and tag normal
        Rt = Rot.from_rotvec([row.rx, row.ry, row.rz]).as_matrix(); obl = np.degrees(np.arccos(abs(Rt[2, 2])))
        rows.append(dict(run=run, tag=row.tag_id, size_cm=s * 100, rel_reproj_pct=100 * row.reproj_px / row.side_px, oblique_deg=obl))
D = pd.DataFrame(rows).dropna()
D = D[(D.size_cm > 3) & (D.size_cm < 60)]
g = D.groupby("tag").agg(n=("size_cm", "size"), size_cm=("size_cm", "median"),
                         iqr_cm=("size_cm", lambda x: x.quantile(.75) - x.quantile(.25)),
                         rel_reproj_pct=("rel_reproj_pct", "median"), oblique_deg=("oblique_deg", "median"),
                         runs=("run", lambda x: ",".join(sorted({r.split("_")[-1] for r in x}, key=int))))
print(g.round(2).to_string())
