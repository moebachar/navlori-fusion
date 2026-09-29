#!/usr/bin/env python
# Gyro bias per run from lidar-confirmed stillness: consecutive scans (resampled on a common angle grid) that
# are identical within ~1 cm => robot neither moving nor turning => mean gyro there = bias.
import numpy as np, pandas as pd, json
from scipy.stats import theilslopes
STG = "/mnt/x/side_navlori/data/staging_golden"
GRID = np.linspace(-np.pi, np.pi, 360, endpoint=False)
def resample(a, r):
    ok = np.isfinite(r) & (r > 0.1) & (r < 8)
    if ok.sum() < 50: return np.full(len(GRID), np.nan)
    a, r = np.mod(a[ok] + np.pi, 2 * np.pi) - np.pi, r[ok]; o = np.argsort(a); a, r = a[o], r[o]
    idx = np.clip(np.searchsorted(a, GRID), 1, len(a) - 1)
    near = np.where(np.abs(a[idx - 1] - GRID) < np.abs(a[idx] - GRID), idx - 1, idx)
    out = r[near]; out[np.abs(a[near] - GRID) > np.radians(1.5)] = np.nan
    return out
res = {}
for i in range(1, 13):
    run = f"golden_run_{i}"
    z = np.load(f"{STG}/{run}/lidar/scans.npz"); off = z["offsets"]; T = z["t_ns"]; n = len(T)
    R = np.stack([resample(z["angles"][off[k]:off[k + 1]], z["ranges"][off[k]:off[k + 1]]) for k in range(n)])
    d = np.r_[np.nan, np.nanmedian(np.abs(np.diff(R, axis=0)), axis=1)]
    still = d < 0.010
    # require 1 s of continuous stillness (10 scans) and drop 0.3 s at the edges of each still block
    blk = np.zeros(n, bool); k = 0
    while k < n:
        if still[k]:
            j = k
            while j < n and still[j]: j += 1
            if j - k >= 10: blk[k + 3:j - 3] = True
            k = j
        else: k += 1
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns")
    ks = np.clip(np.searchsorted(T, imu.t_ns.values), 0, n - 1)
    m = blk[ks] & (imu.t_ns.values >= T[0]) & (imu.t_ns.values <= T[-1])
    b = imu.wz.values[m].mean() if m.sum() > 50 else np.nan
    sd = imu.wz.values[m].std() if m.sum() > 50 else np.nan
    t = (T - T[0]) / 1e9; still_t = t[blk]
    blocks = []
    if blk.any():
        e = np.flatnonzero(np.diff(np.r_[0, blk.astype(int), 0]))
        blocks = [f"{t[a_]:.0f}-{t[min(b_-1,n-1)]:.0f}s" for a_, b_ in zip(e[::2], e[1::2])]
    # SLAM-trend estimate for comparison
    g = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv"); tg = (g.t_ns.values - T[0]) / 1e9; ti = (imu.t_ns.values - T[0]) / 1e9
    dd = np.interp(tg, ti, np.cumsum(imu.wz.values * np.diff(ti, prepend=ti[0]))) - np.unwrap(g.yaw.values)
    kk = np.arange(0, len(tg), max(1, len(tg) // 400)); sl = theilslopes(dd[kk], tg[kk])[0]
    res[run] = dict(bias_still_dps=float(np.degrees(b)), still_imu_samples=int(m.sum()), still_blocks=blocks,
                    gyro_noise_dps=float(np.degrees(sd)), bias_slam_trend_dps=float(np.degrees(sl)))
    print(f"{run:14s} lidar-still bias {np.degrees(b):+8.3f} deg/s (noise {np.degrees(sd):.2f}, {m.sum()} samples, still {blocks}) | SLAM-trend {np.degrees(sl):+8.3f}", flush=True)
json.dump(res, open(f"{STG}/_network/gyro_bias.json", "w"), indent=1)
