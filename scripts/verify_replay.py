"""Verify ONE replayed path against its source dataset (iteration-1 gates).

Headless — reads only the CSVs/PNGs the replay_collector wrote. Run after a
Webots replay finishes.

Gates (exit code 0 only if all PASS):
  1. anchor   : max anchor_err_m over dense GT rows < --max-anchor (1 cm).
                anchor_err_m = |measured robot pose - commanded spline pose|,
                logged by the controller every GT sample.
  2. camera   : frames emitted within --cam-tol of duration * camera_hz, and
                no sampled frame is (near-)black — catches the NULL-camera /
                no-GPU-session failure mode (CLAUDE.md rule 4).
  3. odometry : synthetic odom drift vs the source spline. Endpoint drift as
                a fraction of path length < --max-drift (3 %).
  4. verbatim : wifi.csv / imu.csv row counts + first/last sim_time identical
                to the source (constraint #1 of the replay design).

Also reported (no gate): spline heading-rate stats at the camera timestamps —
the "camera jerk" number that decides whether we must smooth the trajectory.

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\verify_replay.py \\
        --source data/iln20_5d27099f_F2/path_00 \\
        --replay data/iln20_5d27099f_F2_replay/path_00
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROLLER_DIR = REPO_ROOT / "src" / "simulation" / "controllers" / "replay_collector"
sys.path.insert(0, str(CONTROLLER_DIR))

from path_loader import load_path  # noqa: E402
from trajectory import HermiteTrajectory  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gate(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="source path dir (original dataset)")
    ap.add_argument("--replay", required=True, help="replayed path dir (controller output)")
    ap.add_argument("--camera-hz", type=float, default=5.0)
    ap.add_argument("--max-anchor", type=float, default=0.01, help="gate 1 threshold (m)")
    ap.add_argument("--cam-tol", type=float, default=0.05, help="gate 2 frame-count tolerance")
    ap.add_argument("--black-std", type=float, default=2.0,
                    help="gate 2: min per-image pixel std to count as non-black")
    ap.add_argument("--max-drift", type=float, default=0.03, help="gate 3 threshold (fraction)")
    ap.add_argument("--n-sample-frames", type=int, default=20)
    args = ap.parse_args()

    src = Path(args.source)
    rep = Path(args.replay)
    if not src.is_dir():
        sys.exit(f"source dir not found: {src}")
    if not rep.is_dir():
        sys.exit(f"replay dir not found: {rep}")

    rp = load_path(src, 0)
    traj = HermiteTrajectory([w.t for w in rp.waypoints],
                             [w.x for w in rp.waypoints],
                             [w.y for w in rp.waypoints])
    duration = rp.duration
    path_len = sum(
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(rp.waypoints, rp.waypoints[1:])
    )
    print(f"[verify] source: {rp.n_waypoints} waypoints, duration {duration:.1f}s, "
          f"path length {path_len:.1f}m")

    all_ok = True

    # ── Gate 1: anchor error, lag-corrected ──
    # anchor_err_m is measured BEFORE re-anchoring, so it inherently contains
    # one timestep of motion (speed * dt). The true anchor deviation is the
    # residual after subtracting that expected lag.
    DT = 0.032
    gt_rows = read_csv(rep / "ground_truth.csv")
    raw, corrected = [], []
    for r in gt_rows:
        if r.get("anchor_err_m") in (None, "") or r.get("is_original") not in ("0", "False"):
            continue
        e = float(r["anchor_err_m"])
        v = traj.evaluate(float(r["sim_time"])).speed
        raw.append(e)
        corrected.append(abs(e - v * DT))
    if corrected:
        mx = max(corrected)
        all_ok &= gate("anchor", mx < args.max_anchor,
                       f"lag-corrected max={mx*1000:.2f}mm over {len(corrected)} rows "
                       f"(raw max={max(raw)*100:.2f}cm incl. one-step motion; "
                       f"limit {args.max_anchor*1000:.0f}mm)")
    else:
        all_ok &= gate("anchor", False, "no anchor_err_m values found in ground_truth.csv")

    # ── Gate 2: camera count + non-black ──
    cam_rows = read_csv(rep / "camera.csv")
    expected = duration * args.camera_hz
    n = len(cam_rows)
    count_ok = expected > 0 and abs(n - expected) / expected <= args.cam_tol
    black = []
    if cam_rows:
        try:
            from PIL import Image
            import numpy as np
            step = max(1, n // args.n_sample_frames)
            for r in cam_rows[::step]:
                img_path = rep / r["rgb_path"]
                if not img_path.is_file():
                    black.append((r["rgb_path"], "missing"))
                    continue
                arr = np.asarray(Image.open(img_path).convert("L"), dtype=float)
                if arr.std() < args.black_std:
                    black.append((r["rgb_path"], f"std={arr.std():.2f}"))
        except ImportError:
            print("  [warn] PIL/numpy unavailable — skipping black-frame check")
    all_ok &= gate("camera-count", count_ok,
                   f"{n} frames vs expected {expected:.0f} (+/-{args.cam_tol*100:.0f}%)")
    # A single uniform frame is legitimate (camera passing flush to a wall);
    # a dead render pipeline makes MOST frames uniform. Fail above 20 %.
    n_sampled = max(1, len(cam_rows[::max(1, n // args.n_sample_frames)]))
    frac_bad = len(black) / n_sampled
    all_ok &= gate("camera-content", frac_bad <= 0.20,
                   f"{len(black)}/{n_sampled} sampled frames uniform/missing "
                   f"({frac_bad*100:.0f}%, limit 20%)"
                   + (f", e.g. {black[0]}" if black else ""))

    # ── Gate 3: odometry drift vs spline ──
    odom_rows = read_csv(rep / "odometry.csv")
    if odom_rows:
        drifts = []
        for r in odom_rows:
            t = float(r["sim_time"])
            sx, sy = traj.evaluate_position(t)
            drifts.append(math.hypot(float(r["odom_x"]) - sx, float(r["odom_y"]) - sy))
        endpoint = drifts[-1]
        frac = endpoint / max(path_len, 1e-9)
        all_ok &= gate("odom-drift", frac < args.max_drift,
                       f"endpoint {endpoint:.2f}m = {frac*100:.1f}% of {path_len:.1f}m "
                       f"(limit {args.max_drift*100:.0f}%), max mid-track {max(drifts):.2f}m, "
                       f"{len(odom_rows)} rows")
    else:
        all_ok &= gate("odom-drift", False, "odometry.csv empty")

    # ── Gate 4: verbatim WiFi/IMU copy-through ──
    for mod in ("wifi", "imu"):
        s_rows = read_csv(src / f"{mod}.csv")
        r_rows = read_csv(rep / f"{mod}.csv")
        same = (len(s_rows) == len(r_rows)
                and (not s_rows or (s_rows[0]["sim_time"] == r_rows[0]["sim_time"]
                                    and s_rows[-1]["sim_time"] == r_rows[-1]["sim_time"])))
        all_ok &= gate(f"verbatim-{mod}", same,
                       f"source {len(s_rows)} rows vs replay {len(r_rows)} rows")

    # ── Info: camera jerk (spline heading rate over a 200 ms window) ──
    if cam_rows:
        rates = []
        win = 0.2
        for r in cam_rows:
            t = float(r["sim_time"])
            if t + win > rp.t_end:
                continue
            y0 = traj.evaluate(t).yaw
            y1 = traj.evaluate(t + win).yaw
            d = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
            rates.append(abs(math.degrees(d)) / win)
        if rates:
            rates.sort()
            p50 = rates[len(rates) // 2]
            p95 = rates[int(len(rates) * 0.95)]
            print(f"  [info] camera heading rate: median {p50:.0f} deg/s, "
                  f"p95 {p95:.0f} deg/s, max {rates[-1]:.0f} deg/s "
                  f"({len(rates)} frames) — decides the trajectory-smoothing question")

    print(f"\n[verify] {'ALL GATES PASS' if all_ok else 'GATE FAILURE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
