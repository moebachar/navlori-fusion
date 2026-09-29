"""Replay pipeline v2 -- SCRIPT 3 of 3: verify + assemble the final dataset.

Takes a finished Webots run (per-path ground_truth/odometry/camera from the
replay_driver, plus the dilated verbatim imu/wifi from input_dilated) and
produces a shippable 4-modality dataset:

  data/<run>_replay/
    meta/         floor_info, geojson, floor_image, bssid_columns, dataset.json
                  (dataset.json stamped with replay provenance + dilation)
    splits/       train/val/test .txt (deterministic 70/15/15 over passing paths)
    path_XX/
      ground_truth.csv   driven pose @10 Hz (the ACTUAL robot pose = the GT
                         the camera/odom correspond to)
      odometry.csv       synthesized wheel odometry @15 Hz
      camera.csv+camera/ RGB frames @5 Hz
      imu.csv            real IMU, verbatim, on the dilated clock
      wifi.csv           real WiFi, verbatim, on the dilated clock
      trajectory.png     GT path drawn on the floor plan
      metadata.json      dilation factor, press-hit errors, counts
    replay_manifest.json per-path verify verdicts
    videos/path_XX.mp4   (with --videos) 4-panel realtime viz

Gates per path (dropped from splits if failed, kept in manifest):
  done marker, camera count ~ duration*5Hz + non-blank, odometry present,
  worst waypoint-hit error < --max-press-err (default 0.8 m), imu/wifi present.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\package_run.py --run iln20_5d27099f_F2 [--videos]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time as pytime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "data" / "replay_runs"
WORLDS_DIR = REPO_ROOT / "src" / "simulation" / "worlds"
SCRIPTS = REPO_ROOT / "scripts"


def read_rows(p: Path):
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_path(out_dir: Path, in_dir: Path, max_press_err: float):
    """Return (ok, info) for one completed path."""
    info, ok = {}, True
    done = out_dir / "_done.json"
    if not done.is_file():
        return False, {"error": "not completed"}
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    dur = meta.get("duration_s", 0.0)
    info["duration_s"] = round(dur, 1)

    gt = read_rows(out_dir / "ground_truth.csv")
    od = read_rows(out_dir / "odometry.csv")
    cam = read_rows(out_dir / "camera.csv")
    info["gt_rows"], info["odom_rows"], info["frames"] = len(gt), len(od), len(cam)

    # camera count within 15% of duration*5Hz
    exp = dur * 5.0
    info["cam_ok"] = exp <= 0 or abs(len(cam) - exp) / exp <= 0.15
    ok &= info["cam_ok"]
    # non-blank sample
    blank = 0
    try:
        from PIL import Image
        import numpy as np
        step = max(1, len(cam) // 8)
        n = 0
        for r in cam[::step]:
            fp = out_dir / r["rgb_path"]
            if fp.is_file():
                n += 1
                if np.asarray(Image.open(fp).convert("L"), dtype=float).std() < 3:
                    blank += 1
        info["blank_frames"] = f"{blank}/{n}"
        ok &= (n == 0 or blank / n <= 0.25)
    except ImportError:
        info["blank_frames"] = "skipped"
    # waypoint-hit accuracy
    hits = meta.get("press_hits", [])
    worst = max((h["pos_err_m"] for h in hits), default=0.0)
    info["worst_press_m"] = round(worst, 3)
    info["press_ok"] = worst <= max_press_err
    ok &= info["press_ok"]
    ok &= len(od) > 0 and len(gt) > 0
    # verbatim inputs present
    info["imu"] = (in_dir / "imu.csv").is_file()
    info["wifi"] = (in_dir / "wifi.csv").is_file()
    ok &= info["imu"]
    return ok, info


def draw_trajectory(gt_rows, world_text, floor_wh, out_png: Path):
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    walls = []
    for m in re.finditer(
            r"DEF P?WALL_\d+ Solid \{\s*\n\s*translation\s+(-?[\d.]+)\s+(-?[\d.]+)"
            r"\s+(-?[\d.]+)\s*\n\s*rotation 0 0 1 (-?[\d.]+)", world_text):
        tail = world_text[m.end():m.end() + 400]
        mb = re.search(r"geometry Box \{ size (-?[\d.]+)", tail)
        if not mb:
            continue
        cx, cy, yaw, half = (float(m.group(1)), float(m.group(2)),
                             float(m.group(4)), float(mb.group(1)) / 2)
        dx, dy = math.cos(yaw) * half, math.sin(yaw) * half
        walls.append(((cx - dx, cx + dx), (cy - dy, cy + dy)))
    fig, ax = plt.subplots(figsize=(6, 6))
    for xs, ys in walls:
        ax.plot(xs, ys, "-", color="0.6", lw=0.8)
    gx = [float(r["gt_x"]) for r in gt_rows]
    gy = [float(r["gt_y"]) for r in gt_rows]
    ax.plot(gx, gy, "-", color="tab:red", lw=1.5)
    ax.plot(gx[0], gy[0], "o", color="green", ms=7)
    ax.plot(gx[-1], gy[-1], "s", color="black", ms=6)
    ax.set_aspect("equal")
    ax.set_title(out_png.parent.name)
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-press-err", type=float, default=0.8,
                    help="drop a path from splits if worst waypoint hit exceeds "
                         "this (m); still assembled + in manifest (default 0.8)")
    ap.add_argument("--videos", action="store_true",
                    help="also render the 4-panel realtime video per path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = RUNS_ROOT / args.run
    out_root = run_dir / "output"
    in_root = run_dir / "input_dilated"
    if not in_root.is_dir():
        in_root = run_dir / "input"
    world = WORLDS_DIR / f"{args.run}.wbt"
    final = REPO_ROOT / "data" / f"{args.run}_replay"
    if not out_root.is_dir():
        sys.exit(f"no run output: {out_root}")

    ids = sorted(int(d.name[5:]) for d in out_root.glob("path_*")
                 if d.is_dir() and (d / "_done.json").is_file())
    if not ids:
        sys.exit("no completed paths in output/")
    print(f"[package] {len(ids)} completed paths -> {final}")

    world_text = world.read_text(encoding="utf-8") if world.is_file() else ""
    fi = {}
    fip = in_root / "meta" / "floor_info.json"
    if fip.is_file():
        m = json.loads(fip.read_text(encoding="utf-8"))
        mi = m.get("map_info", m)
        fi = {"W": float(mi["width"]), "H": float(mi["height"])}

    manifest = {"run": args.run, "created": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
                "max_press_err_m": args.max_press_err, "paths": {}}
    passing = []
    for pid in ids:
        key = f"path_{pid:02d}"
        od, ind = out_root / key, in_root / key
        ok, info = verify_path(od, ind, args.max_press_err)
        manifest["paths"][key] = {"ok": ok, **info}
        fd = final / key
        fd.mkdir(parents=True, exist_ok=True)
        # assemble: driven gt/odom/camera + verbatim dilated imu/wifi
        for f in ("ground_truth.csv", "odometry.csv", "camera.csv", "metadata.json"):
            if (od / f).is_file():
                shutil.copy2(od / f, fd / f)
        if (od / "camera").is_dir():
            if (fd / "camera").exists():
                shutil.rmtree(fd / "camera")
            shutil.copytree(od / "camera", fd / "camera")
        for f in ("imu.csv", "wifi.csv"):
            if (ind / f).is_file():
                shutil.copy2(ind / f, fd / f)
        try:
            draw_trajectory(read_rows(od / "ground_truth.csv"), world_text, fi,
                            fd / "trajectory.png")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {key}: trajectory.png failed: {e}")
        print(f"  {key}: {'PASS' if ok else 'FAIL'} "
              f"press {info.get('worst_press_m','?')}m frames {info.get('frames','?')}"
              + ("" if ok else f"  <-- {[k for k in ('cam_ok','press_ok') if not info.get(k,True)]}"))
        if ok:
            passing.append(pid)

    # meta/
    (final / "meta").mkdir(exist_ok=True)
    if (in_root / "meta").is_dir():
        for f in (in_root / "meta").iterdir():
            if f.is_file():
                shutil.copy2(f, final / "meta" / f.name)
    dsj = final / "meta" / "dataset.json"
    base = json.loads(dsj.read_text(encoding="utf-8")) if dsj.is_file() else {}
    dil = {}
    dilp = in_root / "dilation_manifest.json"
    if dilp.is_file():
        dil = json.loads(dilp.read_text(encoding="utf-8"))
    base["name"] = final.name
    base["replay_v2"] = {
        "created": manifest["created"],
        "modalities_verbatim": ["wifi", "imu"],
        "modalities_synthesized": ["camera", "odometry", "ground_truth"],
        "clock": "time-dilated per path (stock Tiago++); see dilation factors",
        "geometry": "centripetal Catmull-Rom spline through real waypoints",
        "speed": "IMU walking-intensity profile along the spline",
        "n_passing": len(passing), "n_total": len(ids),
        "dilation_median": (sorted(p["D"] for p in dil.get("paths", {}).values())[
            len(dil.get("paths", {})) // 2] if dil.get("paths") else None),
    }
    dsj.write_text(json.dumps(base, indent=2), encoding="utf-8")

    # splits (deterministic 70/15/15 over passing paths)
    rng = random.Random(args.seed)
    sp = list(passing)
    rng.shuffle(sp)
    n = len(sp)
    ntr, nva = int(0.70 * n), int(0.15 * n)
    parts = {"train": sorted(sp[:ntr]), "val": sorted(sp[ntr:ntr + nva]),
             "test": sorted(sp[ntr + nva:])}
    (final / "splits").mkdir(exist_ok=True)
    for name, plist in parts.items():
        (final / "splits" / f"{name}.txt").write_text(
            "".join(f"path_{i:02d}\n" for i in plist), encoding="utf-8")

    manifest["n_passing"] = len(passing)
    manifest["splits"] = {k: len(v) for k, v in parts.items()}
    (final / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[package] {len(passing)}/{len(ids)} paths pass -> splits "
          f"train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}")
    print(f"[package] dataset -> {final}")

    if args.videos:
        run_ok = subprocess.run(
            [sys.executable, str(SCRIPTS / "render_replay_video.py"),
             "--replay", str(final), "--world", str(world)], cwd=REPO_ROOT)
        print(f"[package] videos {'done' if run_ok.returncode == 0 else 'FAILED'}"
              f" -> {final / 'videos'}")


if __name__ == "__main__":
    main()
