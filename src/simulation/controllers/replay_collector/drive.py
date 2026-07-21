"""Differential-drive tracker that follows a HermiteTrajectory.

Inputs each sim step:
    - target trajectory state at sim_time (spline.evaluate(t))
    - current robot pose (x, y, yaw) from the Supervisor

Outputs:
    - (v_lin, omega) command suitable for set_wheels(lm, rm, v, omega)

Strategy
--------
Feed-forward from the spline:
    - v_ff      = ||(vx_des, vy_des)||                 (spline tangent magnitude)
    - omega_ff  = (yaw_des_next - yaw_curr) / dt       (computed by sampling
                  the spline at t + dt_lookahead)

Feedback (small, additive):
    - v_corr     = K_v * (look_ahead_dist_signed)      where look_ahead_dist_signed
                   is the component of (target_pos - robot_pos) along the heading
    - omega_corr = K_yaw * yaw_error  +  K_lat * lateral_error

This keeps the robot pinned to the spline even when the feed-forward is
slightly off (e.g. wheel slip, physics noise, spline noise at low speed).

Wheel-speed limits are clamped at the wheel level (set_wheels). When the
required (v, omega) exceeds the differential-drive capability the controller
just runs at saturation; the smoke harness + feasibility pre-pass should
have already flagged the path as infeasible at default TIAGO++ settings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


@dataclass
class DriveConfig:
    # Lookahead window for the heading target. The robot aims at
    # spline(t + dt_lookahead) rather than spline(t) -- gives smoother
    # tracking on curves than chasing the instantaneous tangent.
    dt_lookahead: float = 0.20

    # Feedback gains
    k_v: float = 1.0          # forward gain on along-track error (m/s per m)
    k_yaw: float = 2.5        # angular gain on heading error (rad/s per rad)
    k_lat: float = 0.8        # angular gain on lateral error (rad/s per m)

    # When the desired speed drops below this, freeze the heading target
    # to whatever was last commanded -- avoids spinning chasing noise.
    near_stop_speed: float = 0.03


@dataclass
class DriveCommand:
    v: float
    omega: float
    # Diagnostics for logging
    ff_v: float
    ff_omega: float
    pose_err_along: float
    pose_err_lat: float
    yaw_err: float


class DifferentialDriveTracker:
    def __init__(self, traj, cfg: DriveConfig | None = None):
        self.traj = traj
        self.cfg = cfg or DriveConfig()
        self._last_heading_target: float | None = None

    def step(self, sim_time: float, robot_x: float, robot_y: float,
             robot_yaw: float) -> DriveCommand:
        cfg = self.cfg
        st = self.traj.evaluate(sim_time)

        # Heading target = direction from current spline pos to a small
        # look-ahead point on the spline.
        la_t = sim_time + cfg.dt_lookahead
        lx, ly = self.traj.evaluate_position(la_t)
        dx_la = lx - st.x
        dy_la = ly - st.y
        dist_la = math.hypot(dx_la, dy_la)
        if dist_la > 1e-4:
            heading_target = math.atan2(dy_la, dx_la)
            self._last_heading_target = heading_target
        else:
            heading_target = (self._last_heading_target
                              if self._last_heading_target is not None
                              else robot_yaw)

        # Pose error in the robot's local frame.
        ex_world = st.x - robot_x
        ey_world = st.y - robot_y
        cos_h = math.cos(robot_yaw)
        sin_h = math.sin(robot_yaw)
        err_along = cos_h * ex_world + sin_h * ey_world  # forward error
        err_lat = -sin_h * ex_world + cos_h * ey_world   # left-positive error

        # Yaw error (target heading - current yaw, wrapped).
        yaw_err = _wrap_pi(heading_target - robot_yaw)

        # Feed-forward velocity from spline tangent.
        v_ff = st.speed
        # Feed-forward yaw rate from finite difference of heading_target
        # over dt_lookahead.
        omega_ff = yaw_err / max(cfg.dt_lookahead, 1e-3)

        # Feedback.
        v_cmd = v_ff + cfg.k_v * err_along
        if v_ff < cfg.near_stop_speed and abs(yaw_err) < math.radians(10):
            # near-stop: zero forward, just settle heading
            v_cmd = max(0.0, v_cmd * 0.5)

        omega_cmd = omega_ff + cfg.k_yaw * yaw_err + cfg.k_lat * err_lat

        return DriveCommand(
            v=v_cmd, omega=omega_cmd,
            ff_v=v_ff, ff_omega=omega_ff,
            pose_err_along=err_along, pose_err_lat=err_lat,
            yaw_err=yaw_err,
        )


def wheel_velocities_for(v: float, omega: float, axle: float, wheel_r: float,
                           max_wheel: float) -> tuple[float, float]:
    """Convert (v, omega) -> (left, right) wheel angular velocities, clamped
    to +/- max_wheel. If the request is infeasible, scale BOTH wheels by the
    same factor so the (v/omega) ratio is preserved (better than clipping
    one wheel and skewing the motion)."""
    vl = (v - omega * axle / 2.0) / wheel_r
    vr = (v + omega * axle / 2.0) / wheel_r
    scale = 1.0
    peak = max(abs(vl), abs(vr))
    if peak > max_wheel:
        scale = max_wheel / peak
        vl *= scale
        vr *= scale
    return vl, vr


# ---------------------------------------------------------------------------
# Synthetic odometry
# ---------------------------------------------------------------------------
# We pose-anchor the robot to the spline each step (constraint #5: exact
# timing) which means physical wheel encoders won't reflect realistic motion
# (they'd be saturated against the wheel-velocity cap when the spline
# demands > ~0.62 m/s). Instead we synthesize odometry from the spline state,
# applying realistic wheel-encoder-like noise so downstream pipelines see
# data with the right statistics.

class OdometrySynthesizer:
    """Integrate spline velocities into a (noisy) odometry pose track.

    The synthesizer maintains its own (x, y, theta) estimate that drifts
    from ground truth in the same way physical wheel-encoder odometry would
    -- via integration noise. This gives downstream training realistic
    odom-vs-GT divergence without needing to physically drive the wheels at
    the (potentially infeasible) spline rate.
    """

    def __init__(self,
                 axle: float = 0.4044,
                 wheel_r: float = 0.0985,
                 # Per-step gaussian noise on incremental wheel rotation,
                 # tuned so cumulative odom drift over 10 m is ~10-30 cm
                 # (typical for cheap encoders on indoor mobile bases).
                 sigma_left: float = 0.004,
                 sigma_right: float = 0.004,
                 seed: int = 12345):
        import random
        self.rng = random.Random(seed)
        self.axle = axle
        self.wheel_r = wheel_r
        self.sigma_l = sigma_left
        self.sigma_r = sigma_right
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # Accumulated wheel positions (rad) for the simulated encoders.
        self.left_pos = 0.0
        self.right_pos = 0.0
        self._initialised = False
        # Last commanded wheel velocities, for `wheel_left_vel/wheel_right_vel`
        self._last_vl = 0.0
        self._last_vr = 0.0

    def reset(self, x0: float, y0: float, theta0: float):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.left_pos = 0.0
        self.right_pos = 0.0
        self._initialised = True

    def step(self, v: float, omega: float, dt: float) -> dict:
        """Advance one step using commanded (v, omega), produce an odom row.

        Returns a dict matching the project's `odometry.csv` column family:
            odom_x, odom_y, odom_theta_deg,
            odom_linear_vel, odom_angular_vel,
            wheel_left_vel, wheel_right_vel
        """
        if not self._initialised:
            self.reset(0.0, 0.0, 0.0)

        # Inverse kinematics: linear -> (left, right) angular velocities.
        vl = (v - omega * self.axle / 2.0) / self.wheel_r
        vr = (v + omega * self.axle / 2.0) / self.wheel_r
        self._last_vl = vl
        self._last_vr = vr

        # Incremental wheel rotations with gaussian noise.
        dl_rot = vl * dt + self.rng.gauss(0.0, self.sigma_l)
        dr_rot = vr * dt + self.rng.gauss(0.0, self.sigma_r)
        self.left_pos += dl_rot
        self.right_pos += dr_rot

        # Forward kinematics back to (dx, dy, dtheta) in robot frame.
        dl = dl_rot * self.wheel_r
        dr = dr_rot * self.wheel_r
        lin = (dl + dr) / 2.0
        ang = (dr - dl) / self.axle
        # midpoint integration of heading -> smoother for non-trivial omega
        theta_mid = self.theta + ang / 2.0
        self.x += lin * math.cos(theta_mid)
        self.y += lin * math.sin(theta_mid)
        self.theta = _wrap_pi(self.theta + ang)

        return {
            "odom_x": round(self.x, 5),
            "odom_y": round(self.y, 5),
            "odom_theta_deg": round(math.degrees(self.theta), 3),
            "odom_linear_vel": round(lin / dt if dt > 0 else 0.0, 5),
            "odom_angular_vel": round(ang / dt if dt > 0 else 0.0, 5),
            "wheel_left_vel": round(vl, 5),
            "wheel_right_vel": round(vr, 5),
        }


if __name__ == "__main__":
    # ─ Self-test: straight-line trajectory, robot starts slightly off ─
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from trajectory import HermiteTrajectory

    # 5 m straight line over 5 s at 1 m/s
    traj = HermiteTrajectory(
        ts=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        xs=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ys=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    drv = DifferentialDriveTracker(traj)

    # Robot starts 10 cm to the right of the spline at t=0.
    x, y, yaw = 0.0, -0.10, 0.0
    AXLE = 0.4044
    WHEEL_R = 0.0985
    MAX_WHEEL = 6.28
    dt = 0.032
    print("t      x      y      yaw_deg  v_cmd  w_cmd  err_lat")
    for k in range(60):
        t = k * dt
        cmd = drv.step(t, x, y, yaw)
        vl, vr = wheel_velocities_for(cmd.v, cmd.omega, AXLE, WHEEL_R, MAX_WHEEL)
        # Simple kinematic forward-Euler: ignore wheel dynamics, integrate.
        v_eff = (vl + vr) * WHEEL_R / 2.0
        w_eff = (vr - vl) * WHEEL_R / AXLE
        x += v_eff * math.cos(yaw) * dt
        y += v_eff * math.sin(yaw) * dt
        yaw = _wrap_pi(yaw + w_eff * dt)
        if k % 6 == 0:
            print(f"{t:5.2f}  {x:6.3f} {y:+6.3f}  {math.degrees(yaw):+6.1f}  "
                  f"{cmd.v:5.2f}  {cmd.omega:+5.2f}  {cmd.pose_err_lat:+5.3f}")
    err_final = math.hypot(x - 1.92, y)  # at t=1.92s spline says x=1.92
    print(f"\n[final] pose=({x:.3f}, {y:.3f})  yaw={math.degrees(yaw):+.1f}")
