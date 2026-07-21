"""Orchestrator for the replay_collector controller.

Writes replay_config.json next to the controller and patches the chosen
world file so TIAGO++'s `controller` field points at "replay_collector".
Then prints instructions for opening Webots.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\run_replay.py \\
        --world  src/simulation/worlds/iln20_5d27099f_F2.wbt \\
        --dataset data/iln20_5d27099f_F2 \\
        --paths  0 \\
        --output data/iln20_5d27099f_F2_replay
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROLLER_DIR = REPO_ROOT / "src" / "simulation" / "controllers" / "replay_collector"
CONFIG_PATH = CONTROLLER_DIR / "replay_config.json"


# ── world-file robot patch ──
TIAGO_HEAD_RE = re.compile(r"DEF TIAGO Tiago\+\+ \{")
RIG_CTRL_RE = re.compile(r'(DEF REPLAY_RIG Robot \{[^}]*?\bcontroller )"([^"]*)"',
                         re.DOTALL)

# Minimal kinematic replay rig. Deliberately contains NO Physics node and no
# boundingObject: a physics-less Robot is moved purely by supervisor field
# writes -- the constraint solver never touches it, so nothing can explode,
# drift, or detach (all of which the articulated Tiago++ PROTO did when
# pose-anchored). Camera matches the TIAGO head camera: 640x480, ~60 deg
# horizontal FOV, 1.19 m height, looking along +x (Webots FLU).
RIG_TEMPLATE = """DEF REPLAY_RIG Robot {{
  translation {x:.4f} {y:.4f} 0
  rotation 0 0 1 {yaw:.5f}
  name "replay_rig"
  controller "replay_collector"
  supervisor TRUE
  children [
    Pose {{
      translation 0 0 0.20
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.25 0.25 0.28 roughness 0.6 metalness 0.3 }}
          geometry Cylinder {{ radius 0.27 height 0.30 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 0.75
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.88 0.88 0.90 roughness 0.55 metalness 0.1 }}
          geometry Capsule {{ radius 0.16 height 0.70 }}
        }}
      ]
    }}
    Pose {{
      translation 0 0 1.16
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.30 0.30 0.33 roughness 0.5 metalness 0.2 }}
          geometry Sphere {{ radius 0.11 }}
        }}
      ]
    }}
    Camera {{
      translation 0.08 0 1.19
      name "head_front_camera"
      fieldOfView 1.0472
      width 640
      height 480
      far 50
    }}
  ]
}}
"""


def _block_end(text: str, brace_open_idx: int) -> int:
    depth = 0
    for i in range(brace_open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def parse_path_ids(spec: str) -> list[int]:
    """Parse "0", "0,2,4", "0-5", "0-3,7,9-12"."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def ensure_replay_rig(world: Path, dry_run: bool) -> tuple[bool, str]:
    """Replace the world's Tiago++ node with the kinematic REPLAY_RIG
    (keeping its start translation/rotation). Idempotent: if the rig is
    already present, just confirm its controller."""
    text = world.read_text(encoding="utf-8")

    if "DEF REPLAY_RIG Robot" in text:
        m = RIG_CTRL_RE.search(text)
        if m and m.group(2) == "replay_collector":
            return True, "REPLAY_RIG already present, controller ok"
        return False, "REPLAY_RIG present but controller field not readable"

    m = TIAGO_HEAD_RE.search(text)
    if not m:
        return False, "neither REPLAY_RIG nor DEF TIAGO Tiago++ found"
    end = _block_end(text, m.end() - 1)
    block = text[m.start():end]
    mt = re.search(r"translation\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", block)
    mr = re.search(r"rotation\s+0\s+0\s+1\s+(-?[\d.]+)", block)
    if not mt:
        return False, "TIAGO block has no parsable translation"
    rig = RIG_TEMPLATE.format(x=float(mt.group(1)), y=float(mt.group(2)),
                              yaw=float(mr.group(1)) if mr else 0.0)
    new_text = text[:m.start()] + rig + text[end:]
    # the Tiago++ PROTO is no longer used -- drop its EXTERNPROTO download
    new_text = re.sub(r'EXTERNPROTO "[^"]*Tiago\+\+\.proto"\n', "", new_text)
    if not dry_run:
        bak = world.with_suffix(world.suffix + ".bak-tiago")
        shutil.copy2(world, bak)
        world.write_text(new_text, encoding="utf-8")
    return True, "replaced Tiago++ with kinematic REPLAY_RIG (backup .bak-tiago)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="path to .wbt world")
    ap.add_argument("--dataset", required=True,
                    help="dataset root dir (containing path_NN subdirs)")
    ap.add_argument("--output", required=True,
                    help="output dir for replayed data")
    ap.add_argument("--paths", default="0",
                    help="path IDs spec, e.g. \"0\", \"0,2,4\", \"0-5\"")
    ap.add_argument("--gt-hz", type=float, default=10.0)
    ap.add_argument("--odom-hz", type=float, default=15.0)
    ap.add_argument("--camera-hz", type=float, default=5.0)
    ap.add_argument("--no-pose-anchor", action="store_true",
                    help="run with real differential-drive tracking; exact timing "
                         "only when path is feasible at TIAGO++'s wheel cap")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen; don't write config or patch world")
    args = ap.parse_args()

    world = Path(args.world).resolve()
    if not world.is_file():
        sys.exit(f"world not found: {world}")
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        sys.exit(f"dataset dir not found: {dataset}")
    if not CONTROLLER_DIR.is_dir():
        sys.exit(f"replay_collector controller dir not found: {CONTROLLER_DIR}")

    pids = parse_path_ids(args.paths)
    if not pids:
        sys.exit("--paths produced empty list")

    # Identify dataset name for metadata
    dataset_name = dataset.name

    cfg = {
        "_doc": "Generated by scripts/run_replay.py - edit and reload Webots to change.",
        "dataset_dir": str(dataset).replace("\\", "/"),
        "output_dir":  str(Path(args.output).resolve()).replace("\\", "/"),
        "path_ids": pids,
        "world_dataset_id": dataset_name,
        "rates_hz": {
            "ground_truth_dense": args.gt_hz,
            "odometry": args.odom_hz,
            "camera": args.camera_hz,
        },
        "drive": {
            "dt_lookahead_s": 0.20,
            "k_v": 1.0,
            "k_yaw": 2.5,
            "k_lat": 0.8,
        },
        "feasibility": {
            "max_speed_m_s": 0.62,
            "max_yaw_rate_rad_s": 3.07,
        },
        "pose_anchor": not args.no_pose_anchor,
        "hover_m": 0.01,
    }

    print(f"[run_replay] writing {CONFIG_PATH}")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
    else:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[run_replay] ensuring kinematic REPLAY_RIG in {world.name}")
    ok, msg = ensure_replay_rig(world, args.dry_run)
    if not ok:
        sys.exit(f"  ERROR: {msg}")
    print(f"  {msg}")

    print()
    print("Next steps:")
    print(f"  1. Open Webots and load:  {world}")
    print( "  2. Press Play (top toolbar).")
    print( "  3. Watch the controller console for per-path progress.")
    print(f"  Output will land in:   {cfg['output_dir']}")
    if cfg["pose_anchor"]:
        print( "  Pose-anchor: ON (exact timing always; per-step delta sub-cm).")
    else:
        print( "  Pose-anchor: OFF (real diff-drive; only feasible paths exact-timed).")


if __name__ == "__main__":
    main()
