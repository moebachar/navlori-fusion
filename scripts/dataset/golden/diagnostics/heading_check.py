#!/usr/bin/env python
# Where does lidar-SLAM heading disagree with the gyro? (gyro bias from the static start freeze)
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RUNS = sys.argv[1:] or ["golden_run_7", "golden_run_10", "golden_run_11"]
STG = "/mnt/x/side_navlori/data/staging_golden"; REV = "/mnt/x/side_navlori/gt_review"
for run in RUNS:
    gt = pd.read_csv(f"{STG}/{run}/ground_truth/gt_pose.csv")
    imu = pd.read_csv(f"{STG}/{run}/imu/imu.csv").sort_values("t_ns")
    od = pd.read_csv(f"{STG}/{run}/wheel_odom/odom.csv").sort_values("t_ns")
    T = gt.t_ns.values; t0 = T[0]
    ti = (imu.t_ns.values - t0) / 1e9; wz = imu.wz.values
    static = (ti > 0.3) & (ti < 4.0); bias = wz[static].mean()
    dt = np.diff(ti, prepend=ti[0]); gyro = np.cumsum((wz - bias) * dt)
    t = (T - t0) / 1e9
    g = np.interp(t, ti, gyro); g -= g[0]
    s = np.unwrap(gt.yaw.values); s -= s[0]
    o = np.interp(t, (od.t_ns.values - t0) / 1e9, np.unwrap(od.yaw.values)); o -= o[0]
    d_sg = np.degrees(s - g); d_og = np.degrees(o - g)
    # biggest heading disagreement changes over a 10 s window
    win = 10.0; j = np.searchsorted(t, t + win); ok = j < len(t)
    jump = np.full(len(t), np.nan); jump[ok] = d_sg[j[ok]] - d_sg[ok]
    k = np.nanargmax(np.abs(jump))
    print(f"{run}: gyro bias {np.degrees(bias):.3f} deg/s | end disagreement SLAM-gyro {d_sg[-1]:+.1f} deg, odom-gyro {d_og[-1]:+.1f} deg "
          f"| biggest SLAM-gyro change in 10 s: {jump[k]:+.1f} deg starting t={t[k]:.0f}s | total turned {np.degrees(np.abs(np.diff(g)).sum()):.0f} deg")
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(t, np.degrees(g), label="gyro (integrated)", lw=1.2); ax[0].plot(t, np.degrees(s), label="lidar SLAM", lw=1)
    ax[0].plot(t, np.degrees(o), label="wheel odometry", lw=1, alpha=0.8); ax[0].set_ylabel("heading (deg)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(t, d_sg, color="tab:red", label="SLAM - gyro"); ax[1].plot(t, d_og, color="tab:green", label="odometry - gyro")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("heading disagreement (deg)"); ax[1].set_xlabel("time since start (s)")
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[0].set_title(f"{run}: where do the heading sources disagree?")
    plt.tight_layout(); fig.savefig(f"{REV}/{run}_heading.png", dpi=100); plt.close(fig)
