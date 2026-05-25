"""Run DPVO inside the navlori_dpvo Docker container on selected NavLoRI paths.

For each path:
  1. Generate a calib.txt from camera_info.json (fx fy cx cy + 4 zero distortion).
  2. Mount the path's `camera/` PNG dir read-only into the container.
  3. Run DPVO's demo.py with --save_trajectory --plot.
  4. Output: <out_dir>/saved_trajectories/path_XX.txt  (TUM format)
            <out_dir>/trajectory_plots/path_XX.pdf

Usage (from repo root):
    python scripts/run_dpvo_paths.py                      # val + test paths
    python scripts/run_dpvo_paths.py --paths 13 17        # specific paths
    python scripts/run_dpvo_paths.py --stride 2           # subsample (default 1)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "async_collection"
DOCKER_IMAGE = "navlori_dpvo:latest"

VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]


def _winpath_to_docker(p: Path) -> str:
    """Map a Windows-form host path (e.g. X:\\foo) to Docker volume syntax.

    Docker Desktop on Windows accepts both `X:\\foo` and `/x/foo` for binds.
    Forward slashes are universally safer. We also lowercase the drive letter
    because some Docker mount paths choke on uppercase.
    """
    s = str(p.resolve()).replace("\\", "/")
    if len(s) > 2 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def _ensure_image() -> None:
    out = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(
            f"Docker image '{DOCKER_IMAGE}' not found. Build it first:\n"
            f"  cd external/dpvo_docker && docker build -t {DOCKER_IMAGE} ."
        )


def _write_calib(path_dir: Path, calib_dir: Path, pid: int) -> Path:
    info = json.loads((path_dir / "camera_info.json").read_text())
    # DPVO format: fx fy cx cy [k1 k2 p1 p2]   (Webots cam has zero distortion)
    line = (f"{info['fx']:.6f} {info['fy']:.6f} "
            f"{info['cx']:.6f} {info['cy']:.6f} 0 0 0 0\n")
    out = calib_dir / f"calib_path_{pid:02d}.txt"
    out.write_text(line)
    return out


def _stage_rgb_only(cam_dir: Path, staging: Path) -> int:
    """Hardlink (or copy) only the rgb_*.png frames into a clean dir.

    DPVO's `image_stream` globs every image extension in the input dir and
    sorts alphabetically — left to its own devices it picks up our
    `depth_*.png` files first, which silently corrupts the trajectory.
    Staging a frame-only dir is the cleanest fix.
    """
    staging.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(cam_dir.glob("rgb_*.png")):
        dst = staging / p.name
        if dst.exists():
            continue
        try:
            os.link(p, dst)            # NTFS hardlink — instant, no extra disk
        except OSError:
            shutil.copy2(p, dst)       # fallback if hardlinks not supported
        n += 1
    return n


def run_one(pid: int, out_dir: Path, stride: int, viz: bool = False) -> bool:
    pdir = DATA / f"path_{pid:02d}"
    cam_dir = pdir / "camera"
    if not cam_dir.exists():
        print(f"  skip path {pid}: {cam_dir} missing"); return False

    _write_calib(pdir, out_dir, pid)

    staging = out_dir / "_staged" / f"path_{pid:02d}"
    n_staged = _stage_rgb_only(cam_dir, staging)
    print(f"  staged {n_staged} rgb frames in {staging}")

    in_mount  = _winpath_to_docker(staging)
    out_mount = _winpath_to_docker(out_dir)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{in_mount}:/in:ro",
        "-v", f"{out_mount}:/out",
    ]
    
    if viz:
        # Pass the X11 display pointer from host to container
        cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
        
    cmd.extend([
        "-w", "/out",
        DOCKER_IMAGE,
        "conda", "run", "--no-capture-output", "-n", "dpvo",
        "python", "/DPVO/demo.py",
        "--network=/DPVO/dpvo.pth",
        "--config=/DPVO/config/default.yaml",
        "--imagedir=/in",
        f"--calib=/out/calib_path_{pid:02d}.txt",
        f"--stride={stride}",
        "--save_trajectory",
        "--plot",
        f"--name=path_{pid:02d}",
    ])
    
    if viz:
        cmd.insert(-1, "--viz")
    print(f"\n=== path {pid:02d} ===")
    print("  $ " + " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, text=True)
    print(f"  exit={res.returncode}  ({time.time()-t0:.1f}s)")
    return res.returncode == 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", type=int,
                   help="Path ids (default: val + test)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--viz", action="store_true", help="Launch Pangolin visualizer via X11")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_image()

    paths = args.paths or (VAL_PATHS + TEST_PATHS)
    out_dir = (Path(args.out_dir) if args.out_dir
               else ROOT / "runs" / f"dpvo_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")
    print(f"paths:   {paths}  (stride={args.stride})")

    ok = 0
    for pid in paths:
        if run_one(pid, out_dir, args.stride, args.viz):
            ok += 1

    print(f"\nDone. {ok}/{len(paths)} paths succeeded. Output: {out_dir}")
    print("Run scripts/eval_dpvo.py next to score the trajectories vs GT.")


if __name__ == "__main__":
    main()
