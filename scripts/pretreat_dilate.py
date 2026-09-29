"""Replay pipeline v2 -- time-axis dilation pre-treatment (Option B).

Real people walked faster (median ~0.9-1.1 m/s, bursts to 2) than the stock,
UNMODIFIED Tiago++ can drive (~1.0 m/s cap, and it tips if you crank it). So
we slow down each path's CLOCK until its real speed fits comfortably under the
stock cap: multiply every timestamp (waypoints, imu, wifi) by a per-path
factor D. Positions and sensor VALUES are untouched -- only the time labels
scale, and they all scale together so the four modalities stay perfectly
aligned. Speed and acceleration both drop by 1/D and 1/D^2, so the stock robot
drives every path smoothly, no tipping, nothing dropped.

D is chosen so the path's PEAK required speed (from the IMU speed profile along
the smooth spline) becomes <= target_frac * cap.

  data/replay_runs/<run>/input/         (real clock, from make_world)
    -> data/replay_runs/<run>/input_dilated/   (stretched clock)
       + dilation_manifest.json (per-path D)

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\pretreat_dilate.py --run iln20_5d27099f_F2
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "data" / "replay_runs"
DRIVER_DIR = REPO_ROOT / "src" / "simulation" / "controllers" / "replay_driver"
sys.path.insert(0, str(DRIVER_DIR))

from path_geometry import PathGeometry          # noqa: E402
from velocity_profile import build_profile       # noqa: E402

STOCK_CAP_MS = 10.1523 * 0.0985                  # ~1.00 m/s, stock Tiago++
WHEEL_R = 0.0985


def read_rows(p: Path) -> tuple[list[str], list[dict]]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return (r.fieldnames or []), list(r)


def write_rows(p: Path, cols: list[str], rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def dilate_csv(src: Path, dst: Path, D: float) -> None:
    cols, rows = read_rows(src)
    for r in rows:
        if "sim_time" in r and r["sim_time"] != "":
            r["sim_time"] = round(float(r["sim_time"]) * D, 4)
    write_rows(dst, cols, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name (e.g. iln20_5d27099f_F2)")
    ap.add_argument("--cap", type=float, default=STOCK_CAP_MS,
                    help="wheel speed cap in m/s (default stock ~1.0)")
    ap.add_argument("--target-frac", type=float, default=0.9,
                    help="dilate so peak speed <= this fraction of cap (0.9)")
    args = ap.parse_args()

    run = RUNS_ROOT / args.run
    src_in = run / "input"
    dst_in = run / "input_dilated"
    if not src_in.is_dir():
        sys.exit(f"input not found: {src_in} (run make_world.py first)")
    if dst_in.exists():
        shutil.rmtree(dst_in)
    if (src_in / "meta").is_dir():
        shutil.copytree(src_in / "meta", dst_in / "meta")

    target = args.target_frac * args.cap
    manifest = {"run": args.run, "cap_ms": args.cap, "target_ms": target,
                "paths": {}}
    paths = sorted(d for d in src_in.glob("path_*") if d.is_dir())
    print(f"[dilate] {len(paths)} paths, cap {args.cap:.2f} m/s, "
          f"target peak {target:.2f} m/s")
    for d in paths:
        wps = [(float(r["sim_time"]), float(r["gt_x"]), float(r["gt_y"]))
               for r in read_rows(d / "waypoints_raw.csv")[1]]
        imu = read_rows(d / "imu.csv")[1] if (d / "imu.csv").is_file() else []
        geom = PathGeometry(wps)                      # smooth spline
        prof = build_profile(geom, imu, v_cap=1e9)    # uncapped profile
        # Dilate for SUSTAINED speed (p90), not rare spikes: brief bursts
        # above cap just saturate the wheels for a moment and the tracker
        # catches up (only a sustained >4 m lag drops a path). Using the
        # absolute peak over-dilated a 96 s path to 578 s.
        vs = sorted(prof.v_grid)
        v_ref = vs[int(0.90 * (len(vs) - 1))] if vs else 0.0
        v_peak = prof.stats["v_peak"]
        D = max(1.0, v_ref / target) if v_ref > 1e-6 else 1.0

        dd = dst_in / d.name
        dd.mkdir(parents=True, exist_ok=True)
        for fn in ("waypoints_raw.csv", "imu.csv", "wifi.csv"):
            if (d / fn).is_file():
                dilate_csv(d / fn, dd / fn, D)
        if (d / "metadata.json").is_file():
            m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            m["dilation_factor"] = round(D, 4)
            m["v_peak_original_ms"] = round(v_peak, 3)
            (dd / "metadata.json").write_text(json.dumps(m, indent=2),
                                              encoding="utf-8")
        manifest["paths"][d.name] = {"D": round(D, 4),
                                     "v_peak_ms": round(v_peak, 3),
                                     "orig_dur_s": round(wps[-1][0] - wps[0][0], 1),
                                     "dilated_dur_s": round((wps[-1][0] - wps[0][0]) * D, 1)}
        print(f"  {d.name}: v_peak {v_peak:4.2f} m/s -> D {D:.2f} "
              f"(dur {manifest['paths'][d.name]['orig_dur_s']:.0f}s -> "
              f"{manifest['paths'][d.name]['dilated_dur_s']:.0f}s)", flush=True)

    (dst_in / "dilation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    Ds = [p["D"] for p in manifest["paths"].values()]
    print(f"[dilate] done -> {dst_in}\n[dilate] D range {min(Ds):.2f}-{max(Ds):.2f}, "
          f"median {sorted(Ds)[len(Ds)//2]:.2f}")


if __name__ == "__main__":
    main()
