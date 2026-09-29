"""Replay pipeline v2 -- SCRIPT 1 of 3: zone -> Webots world + drive inputs.

(2026-09-07 refactor. Script 2 = the Webots controller `replay_driver`
that consumes these inputs and produces GT+odometry+camera. Script 3 =
`package_run.py` that verifies and organises the final dataset.)

Given one zone (site/floor of the raw ILN 2.0 dump), this script:

  1. STAGES THE DRIVE INPUTS at data/replay_runs/<name>/input/ :
       meta/     floor_info.json, geojson_map.json, floor_image.png,
                 bssid_columns.json, dataset.json
       path_XX/  waypoints_raw.csv  (sim_time, gt_x, gt_y -- the real
                                     presses; drives timing + geometry)
                 imu.csv            (verbatim; drives the velocity profile,
                                     later stored as final original IMU)
                 wifi.csv           (verbatim; final original WiFi)
                 metadata.json      (trace id, t0, counts)
     No dense interpolated GT is fabricated here -- the controller's own
     drive produces the final GT.

  2. BUILDS THE WORLD src/simulation/worlds/<name>.wbt :
       walls from the geojson (corridors widened via --shrink-shops),
       ONE muted pastel per shop + neutral perimeter (coherent colouring),
       wall-aligned decorations, windows, ceiling, hanging signs,
       and the REAL Tiago++ PROTO with its physics untouched.
       Collision policy: every wall/decoration is visual-only
       (pass-through); ONLY the floor is solid, so the robot rolls
       normally and cannot crash into anything or fall off the map.

  3. GATES: world sanity (real Tiago++ present, zero debug geometry,
     zero collidable objects in the .wbt text) + geometric alignment
     (robot start on path wp0, floor extent, waypoint coverage).

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\make_world.py \\
        --site 5d27099f03f801723c32511d --floor F2
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time as pytime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
RAW_ROOT = REPO_ROOT / "data" / "iln20" / "data"
RUNS_ROOT = REPO_ROOT / "data" / "replay_runs"
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"

sys.path.insert(0, str(SCRIPTS))
from convert_iln20_floor import (  # noqa: E402
    build_wifi_rows, collect_bssids, merge_imu_streams, parse_trace,
)


def run_step(title: str, argv: list) -> None:
    print(f"\n{'='*64}\n  [{title}]\n{'='*64}", flush=True)
    r = subprocess.run([sys.executable] + [str(a) for a in argv], cwd=REPO_ROOT)
    if r.returncode != 0:
        sys.exit(f"[make_world] STOP: '{title}' failed (exit {r.returncode})")


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stage_inputs(src: Path, input_dir: Path) -> int:
    """Raw trace .txt files -> input/path_XX dirs. Returns n paths staged."""
    meta = input_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for f in ("floor_info.json", "geojson_map.json", "floor_image.png"):
        if (src / f).is_file():
            shutil.copy2(src / f, meta / f)

    traces = sorted((src / "path_data_files").glob("*.txt"))
    bssids = collect_bssids([str(t) for t in traces])
    (meta / "bssid_columns.json").write_text(
        json.dumps(bssids, indent=1), encoding="utf-8")
    print(f"[stage] {len(traces)} raw traces, {len(bssids)} unique BSSIDs")

    # order paths by their first event timestamp (deterministic, chronological)
    parsed = []
    for t in traces:
        d = parse_trace(str(t))
        if len(d["waypoint"]) < 2 or not d["accel"]:
            print(f"[stage]   skip {t.name}: "
                  f"{len(d['waypoint'])} waypoints, {len(d['accel'])} accel")
            continue
        t0 = min(s[0][0] for s in
                 (d["waypoint"], d["accel"], d["wifi"] or d["waypoint"]))
        parsed.append((t0, t.name, d))
    parsed.sort()

    n = 0
    for pid, (t0, trace_name, d) in enumerate(parsed):
        pdir = input_dir / f"path_{pid:02d}"
        pdir.mkdir(exist_ok=True)
        write_csv(pdir / "waypoints_raw.csv", ["sim_time", "gt_x", "gt_y"],
                  [{"sim_time": round((t - t0) / 1000.0, 4),
                    "gt_x": round(x, 5), "gt_y": round(y, 5)}
                   for t, x, y in d["waypoint"]])
        imu_rows = merge_imu_streams(d["accel"], d["gyro"], d["rotvec"], t0)
        if imu_rows:
            write_csv(pdir / "imu.csv", list(imu_rows[0].keys()), imu_rows)
        wifi_rows = build_wifi_rows(d["wifi"], bssids, t0)
        if wifi_rows:
            write_csv(pdir / "wifi.csv", list(wifi_rows[0].keys()), wifi_rows)
        (pdir / "metadata.json").write_text(json.dumps({
            "trace_file": trace_name, "t0_unix_ms": t0,
            "n_waypoints": len(d["waypoint"]), "n_imu": len(imu_rows),
            "n_wifi_scans": len(wifi_rows),
        }, indent=2), encoding="utf-8")
        n += 1
    return n


def sanity_gate(world: Path) -> None:
    """Real robot in, debug geometry out, nothing collidable but the floor."""
    text = world.read_text(encoding="utf-8")
    print(f"\n{'='*64}\n  [gate: world sanity]  {world.name}\n{'='*64}")
    ok = True
    for tok in ("DEF MARKER_", "DEF WP_", "DEF PS_", "ROBOT_PATHS",
                "boundingObject", "physics Physics"):
        c = text.count(tok)
        print(f"  [{'PASS' if c == 0 else 'FAIL'}] zero '{tok}': {c}")
        ok &= c == 0
    for tok in ("DEF TIAGO Tiago++", 'EXTERNPROTO', "DEF CEILING"):
        p = tok in text
        print(f"  [{'PASS' if p else 'FAIL'}] contains '{tok}'")
        ok &= p
    if not ok:
        sys.exit("[make_world] STOP: world sanity gate failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="24-char raw site id")
    ap.add_argument("--floor", required=True, help="e.g. B1 / F2")
    ap.add_argument("--name", default=None,
                    help="run/world name (default iln20_<site8>_<floor>)")
    ap.add_argument("--shrink-shops", type=float, default=0.75,
                    help="corridor widening in metres (default 0.75)")
    ap.add_argument("--wheel-max-ms", type=float, default=1.0,
                    help="TIAGO wheel speed cap in m/s (default 1.8; tunable)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--restage", action="store_true",
                    help="re-extract inputs even if input/ already exists")
    args = ap.parse_args()

    src = RAW_ROOT / args.site / args.floor
    if not (src / "path_data_files").is_dir():
        sys.exit(f"raw zone not found: {src}")
    name = args.name or f"iln20_{args.site[:8]}_{args.floor}"
    run_dir = RUNS_ROOT / name
    input_dir = run_dir / "input"
    world = WORLDS_DIR / f"{name}.wbt"

    # ── 1. stage drive inputs ──
    if input_dir.is_dir() and not args.restage:
        n = sum(1 for d in input_dir.glob("path_*") if d.is_dir())
        print(f"[make_world] input/ present ({n} paths) -- skipping staging "
              f"(--restage to force)")
    else:
        n = stage_inputs(src, input_dir)
        print(f"[stage] staged {n} paths -> {input_dir}")
    (input_dir / "meta" / "dataset.json").write_text(json.dumps({
        "name": name, "site_id": args.site, "floor_id": args.floor,
        "pipeline": "replay_v2", "staged": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_paths": sum(1 for d in input_dir.glob("path_*") if d.is_dir()),
        "drive_inputs": ["waypoints_raw.csv", "imu.csv"],
        "verbatim_modalities": ["wifi", "imu"],
    }, indent=2), encoding="utf-8")

    # ── 2. build the world (real Tiago++, per-shop palette, visual walls) ──
    run_step("build world", [SCRIPTS / "build_iln20_webots_world.py",
                             "--dataset-dir", input_dir, "--out-name", name,
                             "--robot", "tiago", "--clean",
                             "--shrink-shops", str(args.shrink_shops),
                             "--wheel-max-ms", str(args.wheel_max_ms),
                             "--seed", str(args.seed)])
    run_step("dedup walls", [SCRIPTS / "dedup_wbt_walls.py", "--world", world])
    run_step("ceiling", [SCRIPTS / "add_ceiling_to_wbt.py", "--world", world,
                         "--force"])

    # ── 3. gates ──
    sanity_gate(world)
    run_step("alignment gate", [SCRIPTS / "diagnose_world_alignment.py",
                                "--world", world, "--dataset", input_dir,
                                "--gate"])

    print(f"""
{'='*64}
  MAKE_WORLD COMPLETE
{'='*64}
  world  : {world}
  inputs : {input_dir}
  next   : script 2 (replay_driver controller) drives Tiago++ through
           each path_XX using waypoints_raw.csv timing + imu.csv velocity.
""")


if __name__ == "__main__":
    main()
