"""Time-parameterised 2D trajectory built from a sparse list of waypoints
(t_i, x_i, y_i) -- the original GT samples we must visit at exact times.

Strategy
--------
Piecewise cubic Hermite spline through every waypoint.
  - Position is C^0 + C^1 continuous and EXACTLY pins (t_i, x_i, y_i) by
    construction (endpoint-interpolating Hermite).
  - Tangent (= velocity) at each waypoint is set from a centred finite
    difference of neighbouring waypoints (so the spline derivative gives
    the speed/accel profile "implied by the waypoint geometry + timing",
    which is what the user asked for).
  - Heading at sim time t is the angle of the velocity vector at t (i.e.
    "direction toward the next densified waypoint" in the limit dt -> 0).

The class has no scipy dependency on purpose -- it must import inside the
Webots controller environment (which is just our project venv, but keep
it lean).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TrajectoryState:
    t: float
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    speed: float        # ||(vx, vy)||
    yaw: float          # atan2(vy, vx); set to last good yaw when speed ~= 0
    omega: float        # d(yaw)/dt finite-differenced from the spline


class HermiteTrajectory:
    """Piecewise cubic Hermite spline, evaluated by binary search on t.

    Construction is O(N). Each query is O(log N). For N <= ~50k waypoints
    (longest project paths) this is well below 1 ms per query on the
    target hardware.
    """

    def __init__(self, ts: list[float], xs: list[float], ys: list[float]):
        if len(ts) < 2:
            raise ValueError("need at least 2 waypoints")
        if len(ts) != len(xs) or len(ts) != len(ys):
            raise ValueError("ts / xs / ys length mismatch")
        for i in range(1, len(ts)):
            if ts[i] <= ts[i - 1]:
                raise ValueError(
                    f"timestamps must be strictly increasing "
                    f"(t[{i}]={ts[i]} <= t[{i-1}]={ts[i-1]})"
                )
        self.ts = list(ts)
        self.xs = list(xs)
        self.ys = list(ys)
        self.n = len(ts)
        # Pre-compute per-waypoint tangents (vx_i, vy_i) by centred FD.
        # End-points use one-sided FD.
        self.vxs: list[float] = [0.0] * self.n
        self.vys: list[float] = [0.0] * self.n
        for i in range(self.n):
            if i == 0:
                dt = ts[1] - ts[0]
                self.vxs[i] = (xs[1] - xs[0]) / dt
                self.vys[i] = (ys[1] - ys[0]) / dt
            elif i == self.n - 1:
                dt = ts[-1] - ts[-2]
                self.vxs[i] = (xs[-1] - xs[-2]) / dt
                self.vys[i] = (ys[-1] - ys[-2]) / dt
            else:
                dt = ts[i + 1] - ts[i - 1]
                self.vxs[i] = (xs[i + 1] - xs[i - 1]) / dt
                self.vys[i] = (ys[i + 1] - ys[i - 1]) / dt

    def _segment_index(self, t: float) -> int:
        """Return i such that ts[i] <= t <= ts[i+1] (clamped to valid range)."""
        if t <= self.ts[0]:
            return 0
        if t >= self.ts[-1]:
            return self.n - 2
        lo, hi = 0, self.n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        return lo

    @staticmethod
    def _hermite_basis(s: float, dt: float):
        """Cubic Hermite basis (h00, h10, h01, h11) evaluated at normalised
        parameter s in [0, 1]. h10 / h11 are scaled by dt so input tangents
        are in physical units (m/s, not per-unit-s)."""
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = (s3 - 2 * s2 + s) * dt
        h01 = -2 * s3 + 3 * s2
        h11 = (s3 - s2) * dt
        return h00, h10, h01, h11

    @staticmethod
    def _hermite_basis_deriv(s: float, dt: float):
        """First derivative wrt physical time t. d/dt = (1/dt) d/ds."""
        s2 = s * s
        # d/ds: h00'=6s^2-6s ; h10'=3s^2-4s+1 (no dt scale via chain rule below)
        h00s = 6 * s2 - 6 * s
        h10s = 3 * s2 - 4 * s + 1
        h01s = -6 * s2 + 6 * s
        h11s = 3 * s2 - 2 * s
        # Position uses h*(s) with h10/h11 carrying a factor dt; derivative
        # wrt t multiplies by 1/dt. Net: position-tangents stay physical.
        return h00s / dt, h10s, h01s / dt, h11s

    @staticmethod
    def _hermite_basis_deriv2(s: float, dt: float):
        """Second derivative wrt physical time t. d^2/dt^2 = (1/dt^2) d^2/ds^2."""
        h00ss = 12 * s - 6
        h10ss = 6 * s - 4
        h01ss = -12 * s + 6
        h11ss = 6 * s - 2
        return h00ss / (dt * dt), h10ss / dt, h01ss / (dt * dt), h11ss / dt

    def evaluate(self, t: float) -> TrajectoryState:
        """Return position, velocity, acceleration, yaw, omega at time t."""
        # Clamp queries past the ends to the endpoint values (no extrapolation).
        t_clamped = max(self.ts[0], min(self.ts[-1], t))
        i = self._segment_index(t_clamped)
        dt = self.ts[i + 1] - self.ts[i]
        s = (t_clamped - self.ts[i]) / dt

        h00, h10, h01, h11 = self._hermite_basis(s, dt)
        x = (h00 * self.xs[i] + h10 * self.vxs[i] +
             h01 * self.xs[i + 1] + h11 * self.vxs[i + 1])
        y = (h00 * self.ys[i] + h10 * self.vys[i] +
             h01 * self.ys[i + 1] + h11 * self.vys[i + 1])

        h00d, h10d, h01d, h11d = self._hermite_basis_deriv(s, dt)
        vx = (h00d * self.xs[i] + h10d * self.vxs[i] +
              h01d * self.xs[i + 1] + h11d * self.vxs[i + 1])
        vy = (h00d * self.ys[i] + h10d * self.vys[i] +
              h01d * self.ys[i + 1] + h11d * self.vys[i + 1])

        h00dd, h10dd, h01dd, h11dd = self._hermite_basis_deriv2(s, dt)
        ax = (h00dd * self.xs[i] + h10dd * self.vxs[i] +
              h01dd * self.xs[i + 1] + h11dd * self.vxs[i + 1])
        ay = (h00dd * self.ys[i] + h10dd * self.vys[i] +
              h01dd * self.ys[i + 1] + h11dd * self.vys[i + 1])

        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            yaw = math.atan2(vy, vx)
        else:
            # near-stationary -- look ahead a small dt to recover heading
            t_ahead = t_clamped + 0.05
            if t_ahead <= self.ts[-1]:
                ahead = self.evaluate_position(t_ahead)
                yaw = math.atan2(ahead[1] - y, ahead[0] - x)
            else:
                yaw = 0.0
        # omega = d(yaw)/dt = (vx*ay - vy*ax) / (vx^2 + vy^2)  (curvature * speed)
        denom = vx * vx + vy * vy
        omega = (vx * ay - vy * ax) / denom if denom > 1e-9 else 0.0

        return TrajectoryState(
            t=t, x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay,
            speed=speed, yaw=yaw, omega=omega,
        )

    def evaluate_position(self, t: float) -> tuple[float, float]:
        """Position-only fast path. Returns (x, y)."""
        t_clamped = max(self.ts[0], min(self.ts[-1], t))
        i = self._segment_index(t_clamped)
        dt = self.ts[i + 1] - self.ts[i]
        s = (t_clamped - self.ts[i]) / dt
        h00, h10, h01, h11 = self._hermite_basis(s, dt)
        x = (h00 * self.xs[i] + h10 * self.vxs[i] +
             h01 * self.xs[i + 1] + h11 * self.vxs[i + 1])
        y = (h00 * self.ys[i] + h10 * self.vys[i] +
             h01 * self.ys[i + 1] + h11 * self.vys[i + 1])
        return x, y


def feasibility(traj: HermiteTrajectory, max_speed: float, max_omega: float,
                  n_samples: int = 2000, yaw_window_s: float = 0.20) -> dict:
    """Sample the trajectory and report the peak required (speed, yaw_rate).

    Yaw rate is measured over a finite window (default 200 ms) rather than
    instantaneously: this matches what the differential-drive tracker
    actually has to achieve, and avoids the instantaneous-omega blow-up
    at near-zero-speed cusps from spline interpolation noise.

    Returns a dict for logging; downstream code decides whether to skip.
    """
    peak_speed = 0.0
    peak_yaw_rate = 0.0
    peak_accel = 0.0
    t_min, t_max = traj.ts[0], traj.ts[-1]
    last_yaw = None
    last_t = None
    for k in range(n_samples + 1):
        t = t_min + (t_max - t_min) * (k / n_samples)
        st = traj.evaluate(t)
        peak_speed = max(peak_speed, st.speed)
        peak_accel = max(peak_accel, math.hypot(st.ax, st.ay))
        if last_yaw is not None and (t - last_t) > 0:
            # accumulate dyaw across windows of >= yaw_window_s
            if (t - last_t) >= yaw_window_s or k == n_samples:
                dyaw = st.yaw - last_yaw
                # wrap to [-pi, pi]
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                rate = abs(dyaw) / (t - last_t)
                peak_yaw_rate = max(peak_yaw_rate, rate)
                last_yaw = st.yaw
                last_t = t
        else:
            last_yaw = st.yaw
            last_t = t
    return {
        "peak_speed_m_s": peak_speed,
        "peak_yaw_rate_rad_s": peak_yaw_rate,
        "peak_accel_m_s2": peak_accel,
        "feasible": (peak_speed <= max_speed and peak_yaw_rate <= max_omega),
        "limits": {"max_speed": max_speed, "max_yaw_rate": max_omega},
        "yaw_window_s": yaw_window_s,
    }


if __name__ == "__main__":
    # ─ Standalone self-test ─
    # Tiny synthetic path: straight x = t, y = 0 over t in [0, 1]
    tr = HermiteTrajectory([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0])
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = tr.evaluate(t)
        print(f"t={t:.2f}  x={s.x:.4f} y={s.y:.4f} "
              f"v={s.speed:.4f} yaw={math.degrees(s.yaw):+.2f}deg "
              f"omega={s.omega:+.4f}")
    print()
    # Endpoint pin check
    s0 = tr.evaluate(0.0)
    s1 = tr.evaluate(0.5)
    s2 = tr.evaluate(1.0)
    assert abs(s0.x - 0.0) < 1e-9 and abs(s1.x - 0.5) < 1e-9 and abs(s2.x - 1.0) < 1e-9
    print("[OK] endpoint pinning exact")
    # Feasibility report
    print(feasibility(tr, max_speed=2.0, max_omega=1.0))
