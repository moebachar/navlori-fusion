"""Smoke test: load a real dataset path and verify the spline pins every
original waypoint exactly + reports realistic kinematic peaks."""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from path_loader import load_path, summarise
from trajectory import HermiteTrajectory, feasibility


def main():
    if len(sys.argv) < 2:
        print("usage: _smoke_trajectory.py <path_dir>")
        sys.exit(1)
    rp = load_path(sys.argv[1], 0)
    print(summarise(rp))

    ts = [w.t for w in rp.waypoints]
    xs = [w.x for w in rp.waypoints]
    ys = [w.y for w in rp.waypoints]
    traj = HermiteTrajectory(ts, xs, ys)

    # 1. Exact-pin check at every original waypoint
    max_err = 0.0
    for w in rp.waypoints:
        x, y = traj.evaluate_position(w.t)
        e = math.hypot(x - w.x, y - w.y)
        max_err = max(max_err, e)
    print(f"[exact-pin] max position error at original WPs: {max_err:.2e} m")
    assert max_err < 1e-8, "endpoint pinning broke"
    print("[OK] endpoint pinning < 1e-8 m at every waypoint")

    # 2. Kinematic profile
    report = feasibility(traj, max_speed=1.0, max_omega=3.0, n_samples=5000)
    print(f"[kinematics] peak_speed = {report['peak_speed_m_s']:.3f} m/s "
          f"({'OK' if report['peak_speed_m_s'] <= 1.0 else 'OVER'} the 1.0 m/s test limit)")
    print(f"[kinematics] peak_yaw_rate = {report['peak_yaw_rate_rad_s']:.3f} rad/s "
          f"(={math.degrees(report['peak_yaw_rate_rad_s']):.1f} deg/s, "
          f"window={report['yaw_window_s']*1000:.0f}ms)")
    print(f"[kinematics] peak_accel = {report['peak_accel_m_s2']:.3f} m/s^2")
    print(f"[kinematics] feasible (at limits 1.0 m/s, 3.0 rad/s): "
          f"{report['feasible']}")

    # 3. Densify the trajectory at 30 Hz, count samples
    dense_hz = 30.0
    n_dense = int((rp.t_end - rp.t_start) * dense_hz) + 1
    print(f"[densify] at {dense_hz:.0f} Hz: {n_dense} samples "
          f"(was {rp.n_waypoints} original)")


if __name__ == "__main__":
    main()
