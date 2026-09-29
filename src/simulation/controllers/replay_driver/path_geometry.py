"""Smooth-spline path model for replay_driver (pipeline v2).

The waypoint presses are the only reliable anchors. The IMU heading turned out
to be phone ORIENTATION, not travel direction, so it can't reconstruct the true
between-press path on this dataset (it zig-zagged, inflating a 20 m gap into a
31 m walk). So we interpolate a smooth curve THROUGH the presses instead of
guessing the exact walked shape:

  * Centripetal Catmull-Rom spline through the waypoints -- passes exactly
    through every press, rounds the corners (no fake sharp kinks the robot
    can't drive), and the centripetal parameterisation prevents the loops/
    overshoot plain Catmull-Rom makes at sharp angles.
  * Length stays close to the straight-line distance, so required speeds stay
    feasible.

Honest framing: the shape BETWEEN presses is a smooth interpolation, not a
measurement -- this IMU can't measure it reliably. Presses (position + timing)
are exact.

Interface (unchanged): point_at(s), heading_at(s), project(x, y, hint),
lateral_offset(x, y, s), plus per-waypoint `times` and arc lengths `S`.
"""
from __future__ import annotations

import math

SAMPLES_PER_SEG = 24


class PathGeometry:
    def __init__(self, waypoints: list[tuple[float, float, float]],
                 imu_rows=None):        # imu_rows accepted but unused (see note)
        if len(waypoints) < 2:
            raise ValueError("need >= 2 waypoints")
        self.times = [w[0] for w in waypoints]
        self.wp = [(w[1], w[2]) for w in waypoints]

        dense, self._wp_index = self._catmull_rom(self.wp)
        self.dx = [p[0] for p in dense]
        self.dy = [p[1] for p in dense]
        self.dS = [0.0]
        for i in range(1, len(dense)):
            self.dS.append(self.dS[-1] + math.hypot(
                self.dx[i] - self.dx[i - 1], self.dy[i] - self.dy[i - 1]))
        self.L = self.dS[-1]
        self.S = [self.dS[i] for i in self._wp_index]

    # ---- centripetal Catmull-Rom through the waypoints ----
    @staticmethod
    def _lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def _catmull_rom(self, Q):
        if len(Q) < 3:
            # single segment: straight line, sampled
            dense = [self._lerp(Q[0], Q[-1], s / SAMPLES_PER_SEG)
                     for s in range(SAMPLES_PER_SEG)] + [Q[-1]]
            return dense, [0, len(dense) - 1]
        pad = [Q[0]] + list(Q) + [Q[-1]]          # phantom ends
        dense: list[tuple[float, float]] = []
        wp_index = []
        for k in range(len(Q) - 1):
            p0, p1, p2, p3 = pad[k], pad[k + 1], pad[k + 2], pad[k + 3]

            def knot(ti, a, b):
                return ti + max(1e-6, math.hypot(b[0] - a[0], b[1] - a[1])) ** 0.5
            t0 = 0.0
            t1 = knot(t0, p0, p1)
            t2 = knot(t1, p1, p2)
            t3 = knot(t2, p2, p3)
            wp_index.append(len(dense))            # p1 == Q[k] starts here
            for s in range(SAMPLES_PER_SEG):
                t = t1 + (t2 - t1) * s / SAMPLES_PER_SEG
                A1 = self._lerp(p0, p1, (t - t0) / (t1 - t0))
                A2 = self._lerp(p1, p2, (t - t1) / (t2 - t1))
                A3 = self._lerp(p2, p3, (t - t2) / (t3 - t2))
                B1 = self._lerp(A1, A2, (t - t0) / (t2 - t0))
                B2 = self._lerp(A2, A3, (t - t1) / (t3 - t1))
                dense.append(self._lerp(B1, B2, (t - t1) / (t2 - t1)))
        dense.append(Q[-1])
        wp_index.append(len(dense) - 1)
        return dense, wp_index

    # ---- arc-length queries ----
    def _seg(self, s: float) -> int:
        s = max(0.0, min(self.L, s))
        lo, hi = 0, len(self.dS) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.dS[mid] <= s:
                lo = mid
            else:
                hi = mid
        return lo

    def point_at(self, s: float) -> tuple[float, float]:
        s = max(0.0, min(self.L, s))
        i = self._seg(s)
        seg = self.dS[i + 1] - self.dS[i]
        f = 0.0 if seg < 1e-9 else (s - self.dS[i]) / seg
        return (self.dx[i] + f * (self.dx[i + 1] - self.dx[i]),
                self.dy[i] + f * (self.dy[i + 1] - self.dy[i]))

    def heading_at(self, s: float) -> float:
        i = self._seg(max(0.0, min(self.L - 1e-6, s)))
        return math.atan2(self.dy[i + 1] - self.dy[i],
                          self.dx[i + 1] - self.dx[i])

    def project(self, x: float, y: float, s_hint: float,
                window: float = 5.0) -> float:
        lo_s, hi_s = max(0.0, s_hint - window), min(self.L, s_hint + window)
        i0, i1 = self._seg(lo_s), self._seg(hi_s)
        best_s, best_d = s_hint, float("inf")
        for i in range(i0, i1 + 1):
            x1, y1 = self.dx[i], self.dy[i]
            x2, y2 = self.dx[i + 1], self.dy[i + 1]
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            f = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / L2))
            px, py = x1 + f * vx, y1 + f * vy
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best_d = d
                best_s = self.dS[i] + f * (self.dS[i + 1] - self.dS[i])
        return best_s

    def lateral_offset(self, x: float, y: float, s: float) -> float:
        px, py = self.point_at(s)
        h = self.heading_at(s)
        return -(x - px) * math.sin(h) + (y - py) * math.cos(h)
