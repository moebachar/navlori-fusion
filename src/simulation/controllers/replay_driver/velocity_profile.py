"""Velocity profile from the real IMU (pipeline v2, user's algorithm).

Per press segment [T_j, T_{j+1}] (short, so integration drift stays
bounded):

  1. INTEGRATE the gravity-removed, world-frame horizontal acceleration
     into a velocity vector, starting from v = 0 at the press (RESET at
     every waypoint -- error never carries across segments):
         v_vec(t) = integral_{T_j}^{t} a_h(tau) dtau,   v_raw = |v_vec|
  2. LISSAGE: forward EMA smoothing (tau ~ 0.5 s).
  3. CORRECTION so the two hard constraints hold exactly:
         integral v dt = L_j   (the segment's path length)   and
         arrival at T_{j+1}    (by construction of the schedule)
     via a single scale c = L_j / integral(v_raw); a floor keeps the robot
     from stalling on zero-signal stretches and a cap at v_cap water-fills
     the excess into the free intervals (renormalised, so the integral
     stays exact).

The schedule s*(t) is the cumulative integral of the corrected profile;
s*(T_j) = S_j exactly for every press, so timing at the presses is met by
the *plan* -- the wheel-capped robot then tracks it and the residual is
measured, not hidden.

Pure python (runs inside the Webots controller).
"""
from __future__ import annotations

import math

GRAVITY = 9.81
GRID_HZ = 25.0


def world_horizontal_accel(imu_rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """(t[], ax[], ay[]) -- body accel rotated to world via roll/pitch/yaw
    (ZYX), gravity removed, z dropped. Android convention: a flat, still
    device reads (0, 0, +g)."""
    ts, axs, ays = [], [], []
    for r in imu_rows:
        try:
            t = float(r["sim_time"])
            ax, ay, az = float(r["accel_x"]), float(r["accel_y"]), float(r["accel_z"])
            ro = math.radians(float(r["roll_deg"]))
            pi_ = math.radians(float(r["pitch_deg"]))
            ya = math.radians(float(r["yaw_deg"]))
        except (KeyError, ValueError):
            continue
        cr, sr = math.cos(ro), math.sin(ro)
        cp, sp = math.cos(pi_), math.sin(pi_)
        cy, sy = math.cos(ya), math.sin(ya)
        # R = Rz(yaw) Ry(pitch) Rx(roll), world = R @ body
        wx = (cy * cp) * ax + (cy * sp * sr - sy * cr) * ay + (cy * sp * cr + sy * sr) * az
        wy = (sy * cp) * ax + (sy * sp * sr + cy * cr) * ay + (sy * sp * cr - cy * sr) * az
        ts.append(t)
        axs.append(wx)
        ays.append(wy)
    return ts, axs, ays


def _ema(ts: list[float], vs: list[float], tau: float) -> list[float]:
    out, e = [], vs[0] if vs else 0.0
    for i, v in enumerate(vs):
        dt = max(1e-4, ts[i] - ts[i - 1]) if i else 0.04
        a = dt / (tau + dt)
        e += a * (v - e)
        out.append(e)
    return out


class VelocityProfile:
    """v(t) and schedule s*(t) on a uniform grid over the whole path."""

    def __init__(self, grid_t: list[float], v: list[float], s_star: list[float],
                 stats: dict):
        self.grid_t = grid_t
        self.v_grid = v
        self.s_grid = s_star
        self.stats = stats

    def _idx(self, t: float) -> int:
        g = self.grid_t
        if t <= g[0]:
            return 0
        if t >= g[-1]:
            return len(g) - 1
        lo, hi = 0, len(g) - 1     # binary search: grid spacing is only
        while hi - lo > 1:         # per-segment uniform, not global
            mid = (lo + hi) // 2
            if g[mid] <= t:
                lo = mid
            else:
                hi = mid
        return lo

    def _interp(self, arr: list[float], t: float) -> float:
        i = self._idx(t)
        g = self.grid_t
        if i >= len(g) - 1:
            return arr[-1]
        f = (t - g[i]) / max(1e-9, g[i + 1] - g[i])
        f = max(0.0, min(1.0, f))
        return arr[i] + f * (arr[i + 1] - arr[i])

    def v(self, t: float) -> float:
        return self._interp(self.v_grid, t)

    def s_star(self, t: float) -> float:
        return self._interp(self.s_grid, t)


def build_profile(geom, imu_rows: list[dict], v_cap: float = 0.60,
                  smooth_tau: float = 0.5, floor_frac: float = 0.02) -> VelocityProfile:
    """geom: PathGeometry (has .times, .S). See module docstring."""
    its, iax, iay = world_horizontal_accel(imu_rows)

    grid_t: list[float] = []
    v_all: list[float] = []
    stats = {"n_segments": len(geom.times) - 1, "n_infeasible": 0,
             "n_no_imu": 0, "v_peak": 0.0, "integral_err_max": 0.0}

    for j in range(len(geom.times) - 1):
        t0, t1 = geom.times[j], geom.times[j + 1]
        Lj = geom.S[j + 1] - geom.S[j]
        dT = t1 - t0
        n = max(2, int(dT * GRID_HZ))
        seg_t = [t0 + dT * k / n for k in range(n)]        # [t0, t1)

        v_need = Lj / max(1e-9, dT)
        if v_need > 0.98 * v_cap:
            # segment demands more than the wheels can give -- no profile
            # can fix physics; run flat-out and record the infeasibility
            stats["n_infeasible"] += 1
            seg_v = [v_cap] * n
        else:
            # 1. integrate world-frame horizontal accel, RESET at the press
            i0 = next((i for i, tt in enumerate(its) if tt >= t0), None)
            vx = vy = 0.0
            raw_t, raw_v = [t0], [0.0]
            if i0 is not None:
                for i in range(i0, len(its)):
                    if its[i] > t1:
                        break
                    dt = its[i] - its[i - 1] if i > i0 else max(1e-3, its[i] - t0)
                    dt = max(1e-4, min(0.1, dt))
                    vx += iax[i] * dt
                    vy += iay[i] * dt
                    raw_t.append(its[i])
                    raw_v.append(math.hypot(vx, vy))
            if len(raw_t) < 4:
                stats["n_no_imu"] += 1
                seg_v = [v_need] * n                       # uniform fallback
            else:
                # 2. lissage
                sm = _ema(raw_t, raw_v, smooth_tau)
                # resample to the segment grid
                def interp(t):
                    if t <= raw_t[0]:
                        return sm[0]
                    if t >= raw_t[-1]:
                        return sm[-1]
                    lo, hi = 0, len(raw_t) - 1
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        if raw_t[mid] <= t:
                            lo = mid
                        else:
                            hi = mid
                    f = (t - raw_t[lo]) / max(1e-9, raw_t[lo + 1] - raw_t[lo])
                    return sm[lo] + f * (sm[lo + 1] - sm[lo])
                seg_v = [interp(t) for t in seg_t]
                # 3a. floor (never stall) then scale to meet the constraint
                mean_v = sum(seg_v) / n
                if mean_v < 1e-9:
                    seg_v = [v_need] * n
                else:
                    seg_v = [max(v, floor_frac * mean_v) for v in seg_v]
                    c = Lj / (sum(seg_v) * dT / n)
                    seg_v = [v * c for v in seg_v]
                # 3b. water-fill: enforce the cap and preserve the integral
                # exactly. Each pass clamps everything to v_cap (losing area)
                # then scales the sub-cap samples up to reabsorb the lost
                # area; converges to (all <= cap) AND (integral == Lj) when
                # the segment is feasible (mean <= cap, guaranteed here).
                for _ in range(12):
                    seg_v = [min(v, v_cap) for v in seg_v]
                    deficit = Lj - sum(seg_v) * dT / n
                    if abs(deficit) < 1e-9:
                        break
                    free = [k for k, v in enumerate(seg_v) if v < v_cap - 1e-9]
                    fa = sum(seg_v[k] for k in free) * dT / n
                    if fa < 1e-9:
                        break
                    sc = (fa + deficit) / fa
                    for k in free:
                        seg_v[k] *= sc

        grid_t += seg_t
        v_all += seg_v
        integ = sum(seg_v) * dT / n
        stats["integral_err_max"] = max(stats["integral_err_max"],
                                        abs(integ - Lj))
        stats["v_peak"] = max(stats["v_peak"], max(seg_v) if seg_v else 0.0)

    grid_t.append(geom.times[-1])
    v_all.append(v_all[-1] if v_all else 0.0)

    # schedule: cumulative integral, re-pinned to S_j at every press.
    # A press pin can sit below the running integral when an infeasible
    # segment overshot in raw terms; clamp non-decreasing so s*(t) stays
    # monotone (the tracker never asks the robot to reverse).
    s_star = [0.0]
    seg_j = 0
    for i in range(1, len(grid_t)):
        dt = grid_t[i] - grid_t[i - 1]
        nxt = s_star[-1] + v_all[i - 1] * dt
        while seg_j + 1 < len(geom.times) and grid_t[i] >= geom.times[seg_j + 1] - 1e-9:
            seg_j += 1
            nxt = geom.S[seg_j]             # exact pin at the press
        s_star.append(max(nxt, s_star[-1]))
    s_star[-1] = geom.L

    stats["feasible"] = stats["n_infeasible"] == 0

    return VelocityProfile(grid_t, v_all, s_star, stats)
