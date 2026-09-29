"""Replay pipeline v2 -- stage a drive: write replay_driver_config.json and
wire the world's Tiago++ to the replay_driver controller.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\stage_drive.py --run iln20_5d27099f_F2
        [--paths 0-4] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "data" / "replay_runs"
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"
DRIVER_DIR = REPO_ROOT / "src" / "simulation" / "controllers" / "replay_driver"


def parse_ids(spec: str) -> list[int]:
    out = []
    for c in spec.split(","):
        c = c.strip()
        if not c:
            continue
        if "-" in c:
            lo, hi = c.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(c))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name = world stem, e.g. iln20_5d27099f_F2")
    ap.add_argument("--paths", default=None, help="ids spec (default: all staged)")
    ap.add_argument("--fresh", action="store_true",
                    help="redo paths even if _done.json exists")
    ap.add_argument("--v-cap", type=float, default=1.5,
                    help="TIAGO wheel-speed cap in m/s (Option C; default 1.5 "
                         "keeps 55/62 F2 paths, 2.0 keeps 61)")
    args = ap.parse_args()

    run_dir = RUNS_ROOT / args.run
    # prefer the time-dilated inputs (pretreat_dilate.py) so the stock robot
    # can drive every path; fall back to raw input/ if not dilated yet
    dilated = run_dir / "input_dilated"
    input_dir = dilated if dilated.is_dir() else run_dir / "input"
    world = WORLDS_DIR / f"{args.run}.wbt"
    if not input_dir.is_dir():
        sys.exit(f"inputs missing: {input_dir} (run make_world.py first)")
    print(f"[stage] input: {input_dir.name}"
          + ("  (time-dilated)" if input_dir.name == "input_dilated" else
             "  (raw -- run pretreat_dilate.py for stock-robot feasibility)"))
    if not world.is_file():
        sys.exit(f"world missing: {world} (run make_world.py first)")

    ids = (parse_ids(args.paths) if args.paths else
           sorted(int(d.name[5:]) for d in input_dir.glob("path_*") if d.is_dir()))
    if not ids:
        sys.exit("no paths to stage")

    text = world.read_text(encoding="utf-8")
    m = re.search(r'DEF BUILD_STAMP Solid \{[^}]*?name "(build_\d+)"', text, re.DOTALL)
    stamp = m.group(1) if m else None

    # wire the robot's controller field (idempotent)
    tm = re.search(r'(DEF TIAGO Tiago\+\+ \{[^}]*?controller )"([^"]*)"', text, re.DOTALL)
    if not tm:
        sys.exit("no DEF TIAGO Tiago++ block in the world")
    if tm.group(2) != "replay_driver":
        world.write_text(text[:tm.start()] + tm.group(1) + '"replay_driver"'
                         + text[tm.end():], encoding="utf-8")
        print(f"[stage] controller: '{tm.group(2)}' -> 'replay_driver' "
              f"(RELOAD the world in Webots)")
    else:
        print("[stage] controller already 'replay_driver'")

    cfg = {
        "run_name": args.run,
        "input_dir": str(input_dir).replace("\\", "/"),
        "output_dir": str(run_dir / "output").replace("\\", "/"),
        "path_ids": ids,
        "rates_hz": {"ground_truth": 10.0, "odometry": 15.0, "camera": 5.0},
        "gains": {"k_v": 1.2, "k_head": 2.2, "k_lat": 1.2, "lookahead_m": 0.35},
        # Option C: raise TIAGO's wheel cap so it can match real walking
        # speed (stock 0.62 m/s leaves only 3/62 F2 paths feasible; 1.5 m/s
        # -> 55/62). Paths still exceeding it are dropped by package_run.
        "v_cap": args.v_cap,
        "resume": not args.fresh,
        "world_build_stamp": stamp,
    }
    (DRIVER_DIR / "replay_driver_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[stage] {len(ids)} paths staged | stamp {stamp} | resume={cfg['resume']}")
    print(f"[stage] Webots: File > Open/Reload World: {world}  then Play")


if __name__ == "__main__":
    main()
