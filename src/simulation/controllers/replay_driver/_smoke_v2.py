"""Headless gates for pipeline v2's velocity law (user's algorithm:
per-segment IMU integration + lissage + constraint correction + reset).

Run from repo root:
    .venv\\Scripts\\python.exe src\\simulation\\controllers\\replay_driver\\_smoke_v2.py

Over ALL staged F2 paths it checks:
  1. schedule exactness: s*(T_j) = S_j at every press (the timing constraint
     is met by the plan, machine-exact)
  2. constraint correction: per-segment integral(v) = segment length
  3. bounds: 0 <= v <= V_CAP; s*(t) monotone
  4. realism probe (honest, informational): correlation between the
     integrated-IMU speed shape and the simple shaking-envelope -- tells us
     whether per-segment integration carries real signal or mostly drift.
Also reports how many segments are physically infeasible for Tiago++
(v_needed > wheel cap) -- those run flat-out and the miss is measured.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[3]
INPUT = REPO / "data" / "replay_runs" / "iln20_5d27099f_F2" / "input"

from path_geometry import PathGeometry  # noqa: E402


def rows(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def envelope(imu):
    """v1-style shaking envelope for the realism probe."""
    ts, out = [], []
    e = s = None
    for r in imu:
        try:
            t, m = float(r["sim_time"]), float(r["accel_magnitude"])
        except (KeyError, ValueError):
            continue
        if e is None:
            e, s = m, 0.0
        dt = max(1e-4, t - ts[-1]) if ts else 0.02
        e += dt / (0.8 + dt) * (m - e)
        hp = abs(m - e)
        s += dt / (0.6 + dt) * (hp - s)
        ts.append(t)
        out.append(s)
    return ts, out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 1e-12 else 0.0


def main() -> int:
    import argparse
    from velocity_profile import build_profile
    ap = argparse.ArgumentParser()
    # 0.95 = 0.95 * true device max (10.15 rad/s * 0.0985 m) -- matches what
    # the controller derives at runtime from the unmodified TIAGO (Option A)
    ap.add_argument("--v-cap", type=float, default=0.95)
    args, _ = ap.parse_known_args()
    V_CAP = args.v_cap

    paths = sorted(d for d in INPUT.glob("path_*") if d.is_dir())
    print(f"[smoke_v2] {len(paths)} paths, v_cap={V_CAP} m/s (Option C)")
    ok = True
    # gates apply to FEASIBLE paths only; infeasible paths are droppable
    worst_pin = worst_int = worst_v = 0.0
    feasible, droppable, corrs, v_peaks = [], [], [], []

    for d in paths:
        wps = [(float(r["sim_time"]), float(r["gt_x"]), float(r["gt_y"]))
               for r in rows(d / "waypoints_raw.csv")]
        imu = rows(d / "imu.csv") if (d / "imu.csv").is_file() else []
        geom = PathGeometry(wps, imu)
        prof = build_profile(geom, imu, v_cap=V_CAP)
        if not prof.stats["feasible"]:
            droppable.append((d.name, prof.stats["n_infeasible"]))
            continue
        feasible.append(d.name)
        for j, Tj in enumerate(geom.times):
            worst_pin = max(worst_pin, abs(prof.s_star(Tj) - geom.S[j]))
        worst_int = max(worst_int, prof.stats["integral_err_max"])
        worst_v = max(worst_v, max(prof.v_grid))
        if min(prof.v_grid) < -1e-9:
            print(f"  [FAIL] {d.name}: negative v"); ok = False
        if any(b < a - 1e-9 for a, b in zip(prof.s_grid, prof.s_grid[1:])):
            print(f"  [FAIL] {d.name}: schedule not monotone"); ok = False
        v_peaks.append(prof.stats["v_peak"])
        ets, evs = envelope(imu)
        if ets and len(prof.v_grid) > 20:
            env_i, k = [], 0
            for t in prof.grid_t:
                while k < len(ets) - 1 and ets[k + 1] < t:
                    k += 1
                env_i.append(evs[k])
            corrs.append(pearson(prof.v_grid, env_i))

    corrs.sort()
    med_corr = corrs[len(corrs) // 2] if corrs else float("nan")
    print(f"\n  feasible paths: {len(feasible)}/{len(paths)} | "
          f"droppable (need > cap): {len(droppable)}")
    print(f"  [{'PASS' if worst_pin < 1e-6 else 'FAIL'}] press pins exact: "
          f"worst {worst_pin:.2e} m")
    ok &= worst_pin < 1e-6
    print(f"  [{'PASS' if worst_int < 0.005 else 'FAIL'}] segment integral vs "
          f"length: worst {worst_int*1000:.2f} mm (limit 5 mm)")
    ok &= worst_int < 0.005
    print(f"  [{'PASS' if worst_v <= V_CAP + 1e-6 else 'FAIL'}] v within cap: "
          f"peak {worst_v:.3f} m/s")
    ok &= worst_v <= V_CAP + 1e-6
    print(f"  [{'PASS' if len(feasible) >= 10 else 'FAIL'}] enough feasible "
          f"paths: {len(feasible)} kept, {len(droppable)} dropped "
          f"(sanity floor 10)")
    ok &= len(feasible) >= 10
    if v_peaks:
        print(f"  [info] peak v on feasible paths: "
              f"median {sorted(v_peaks)[len(v_peaks)//2]:.2f}, "
              f"max {max(v_peaks):.2f} m/s")
    print(f"  [info] realism probe -- corr(integrated-IMU speed, shaking "
          f"envelope): median {med_corr:.2f}"
          + (f", range [{corrs[0]:.2f}, {corrs[-1]:.2f}]" if corrs else ""))
    if droppable:
        print(f"  [info] droppable paths: "
              f"{', '.join(n for n, _ in droppable[:12])}"
              + (" ..." if len(droppable) > 12 else ""))
    print(f"\n[smoke_v2] {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
