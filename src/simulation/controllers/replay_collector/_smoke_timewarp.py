"""Headless gates for the IMU-shaped speed profile (iteration 4a).

Run from repo root:
    .venv\\Scripts\\python.exe src\\simulation\\controllers\\replay_collector\\_smoke_timewarp.py

Checks, in order:
  1. synthetic: anchors hit EXACTLY in time+position; robot ~stops where
     intensity is zero and catches up after; identity where intensity flat
  2. synthetic: water-fill v-cap respected without breaking anchor timing
  3. real F2 path_00 (4 raw presses): anchors exact, speed follows IMU
  4. real msiln path_00 (no raw file): kink-inferred anchors, endpoints exact
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[3]

from path_loader import anchor_times, build_trajectory, imu_intensity, load_path  # noqa: E402
from trajectory import HermiteTrajectory, IMUTimeWarp, WarpedTrajectory  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return ok


def synth(intensity_fn, v_max=None):
    """Base: x = t (1 m/s) over [0, 20], anchors at 0/10/20."""
    ts = [i * 0.1 for i in range(201)]
    base = HermiteTrajectory(ts, list(ts), [0.0] * len(ts))
    its = [i * 0.04 for i in range(501)]
    s = [intensity_fn(t) for t in its]
    r_max = [v_max / 1.0] * 2 if v_max else None
    warp = IMUTimeWarp([0.0, 10.0, 20.0], its, s, r_max_per_seg=r_max)
    return WarpedTrajectory(base, warp), base


def main() -> int:
    ok = True

    print("[1] synthetic: stop -> catch-up -> identity")
    tr, base = synth(lambda t: (0.0 if t < 4 else 1.0) if t < 10 else 1.0)
    for t_a in (0.0, 10.0, 20.0):
        du = abs(tr.warp.u(t_a) - t_a)
        wx, _ = tr.evaluate_position(t_a)
        bx, _ = base.evaluate_position(t_a)
        ok &= check(f"anchor t={t_a:.0f} exact", du < 1e-9 and abs(wx - bx) < 1e-9,
                    f"|u-t|={du:.2e}, |dx|={abs(wx-bx):.2e}")
    us = [tr.warp.u(t) for t in [i * 0.01 for i in range(2001)]]
    ok &= check("u monotone", all(b >= a - 1e-12 for a, b in zip(us, us[1:])))
    v_stop = max(tr.evaluate(t).speed for t in (1.0, 2.0, 3.0))
    v_go = tr.evaluate(7.0).speed
    v_id = tr.evaluate(15.0).speed
    ok &= check("near-stop while intensity=0", v_stop < 0.25, f"v={v_stop:.3f} m/s")
    ok &= check("catch-up while intensity high", v_go > 1.2, f"v={v_go:.3f} m/s")
    ok &= check("identity on flat segment", abs(v_id - 1.0) < 0.05, f"v={v_id:.3f} m/s")

    print("[2] synthetic: v-cap water-fill (spike, v_max=2.0)")
    tr, _ = synth(lambda t: (0.05 if t < 8 else 5.0) if t < 10 else 1.0, v_max=2.0)
    vpk = max(tr.evaluate(i * 0.02).speed for i in range(1001))
    ok &= check("peak speed <= v_max", vpk <= 2.0 + 1e-6, f"peak={vpk:.4f}")
    ok &= check("anchor t=10 still exact", abs(tr.warp.u(10.0) - 10.0) < 1e-9)

    print("[3] real F2 path_00")
    rp = load_path(REPO / "data/iln20_5d27099f_F2/path_00", 0)
    anchors, src = anchor_times(rp)
    tr, info = build_trajectory(rp, imu_speed_profile=True, v_max=2.5)
    ok &= check("profile enabled", info["enabled"], str(info))
    ok &= check("anchor source = waypoints_raw", src == "waypoints_raw",
                f"{src}, {len(anchors)} anchors")
    worst = 0.0
    for t_a in anchors:
        wx, wy = tr.evaluate_position(t_a)
        bx, by = tr.base.evaluate_position(t_a)
        worst = max(worst, math.hypot(wx - bx, wy - by))
    ok &= check("presses exact in position", worst < 1e-6, f"worst={worst:.2e} m")
    its, s = imu_intensity(rp)
    n = 300
    vs, ss = [], []
    for k in range(n):
        t = rp.t_start + rp.duration * (k + 0.5) / n
        vs.append(tr.evaluate(t).speed)
        i = min(range(len(its)), key=lambda j: abs(its[j] - t))
        ss.append(s[i])
    mv, ms = sum(vs) / n, sum(ss) / n
    num = sum((a - mv) * (b - ms) for a, b in zip(vs, ss))
    den = math.sqrt(sum((a - mv) ** 2 for a in vs) * sum((b - ms) ** 2 for b in ss))
    corr = num / den if den > 0 else 0.0
    ok &= check("speed follows IMU intensity", corr > 0.8, f"pearson r={corr:.3f}")
    shift = max(math.hypot(*(map(lambda a, b: a - b,
                                 tr.evaluate_position(rp.t_start + rp.duration * k / n),
                                 tr.base.evaluate_position(rp.t_start + rp.duration * k / n))))
                for k in range(n + 1))
    print(f"  [info] r in [{info['r_min']:.2f}, {info['r_max']:.2f}], "
          f"max label shift vs uniform {shift:.2f} m, "
          f"{info['n_clipped_intervals']} v-capped intervals")

    print("[4] real msiln path_00 (kink-inferred anchors)")
    rp2 = load_path(REPO / "data/msiln_site1_b1/path_00", 0)
    anchors2, src2 = anchor_times(rp2)
    tr2, info2 = build_trajectory(rp2, imu_speed_profile=True, v_max=2.5)
    ok &= check("profile enabled", info2["enabled"], str({k: info2[k] for k in
                ("anchor_source", "n_anchors", "r_min", "r_max")}))
    ok &= check("anchors inferred from GT kinks", src2 == "gt_kinks",
                f"{src2}, {len(anchors2)} anchors")
    e0 = math.hypot(*(map(lambda a, b: a - b,
                          tr2.evaluate_position(rp2.t_end),
                          tr2.base.evaluate_position(rp2.t_end))))
    ok &= check("endpoint exact", e0 < 1e-6, f"{e0:.2e} m")
    us2 = [tr2.warp.u(rp2.t_start + rp2.duration * k / 500) for k in range(501)]
    ok &= check("u monotone", all(b >= a - 1e-12 for a, b in zip(us2, us2[1:])))

    print(f"\n[smoke_timewarp] {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
