"""Verify replayed paths against their source dataset (iteration-1 gates).

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

  one path:
    .venv\\Scripts\\python.exe scripts\\verify_replay.py \\
        --source data/iln20_5d27099f_F2/path_00 \\
        --replay data/iln20_5d27099f_F2_replay/path_00

  whole dataset (writes <replay>/replay_manifest.json):
    .venv\\Scripts\\python.exe scripts\\verify_replay.py --all \\
        --source data/iln20_5d27099f_F2 \\
        --replay data/iln20_5d27099f_F2_replay [--paths 0-4]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time as pytime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROLLER_DIR = REPO_ROOT / "src" / "simulation" / "controllers" / "replay_collector"
sys.path.insert(0, str(CONTROLLER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from path_loader import load_path  # noqa: E402
from trajectory import HermiteTrajectory  # noqa: E402
from run_replay import parse_path_ids  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_path(src: Path, rep: Path, path_id: int, args) -> tuple[bool, dict]:
    """Run all gates for one (source path dir, replay path dir) pair.
    Prints per-gate lines; returns (all_ok, results-dict for the manifest)."""
    results: dict = {"gates": {}, "info": {}}

    def gate(name: str, ok: bool, detail: str) -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        results["gates"][name] = {"ok": bool(ok), "detail": detail}
        return ok

    rp = load_path(src, path_id)
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
    results["info"].update(n_waypoints=rp.n_waypoints,
                           duration_s=round(duration, 2),
                           path_len_m=round(path_len, 2))

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
    n_sampled = max(1, len(cam_rows[::max(1, n // args.n_sample_frames)])) if cam_rows else 1
    frac_bad = len(black) / n_sampled
    all_ok &= gate("camera-content", frac_bad <= 0.20,
                   f"{len(black)}/{n_sampled} sampled frames uniform/missing "
                   f"({frac_bad*100:.0f}%, limit 20%)"
                   + (f", e.g. {black[0]}" if black else ""))
    results["info"]["frames"] = n

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
            results["info"]["heading_rate_deg_s"] = {
                "median": round(p50, 1), "p95": round(p95, 1),
                "max": round(rates[-1], 1)}

    return all_ok, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="source path dir; with --all: source dataset ROOT")
    ap.add_argument("--replay", required=True,
                    help="replay path dir; with --all: replay output ROOT")
    ap.add_argument("--all", action="store_true",
                    help="batch mode over path_XX subdirs; writes "
                         "replay_manifest.json into the replay root")
    ap.add_argument("--paths", default=None,
                    help="with --all: only verify these ids (e.g. '0-4', '0,2')")
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

    # ── Single-path mode (unchanged behavior) ──
    if not args.all:
        pid = int(src.name.split("_")[-1]) if src.name.startswith("path_") else 0
        all_ok, _ = verify_path(src, rep, pid, args)
        print(f"\n[verify] {'ALL GATES PASS' if all_ok else 'GATE FAILURE'}")
        return 0 if all_ok else 1

    # ── Batch mode ──
    if args.paths:
        ids = parse_path_ids(args.paths)
    else:
        ids = sorted(int(d.name[5:]) for d in rep.iterdir()
                     if d.is_dir() and d.name.startswith("path_")
                     and d.name[5:].isdigit())
    if not ids:
        sys.exit("batch mode: no path_XX dirs found in replay root "
                 "(and no --paths given)")

    per: dict[str, dict] = {}
    overall = True
    for pid in ids:
        key = f"path_{pid:02d}"
        s, r = src / key, rep / key
        print(f"\n{'='*56}\n  {key}\n{'='*56}")
        if not r.is_dir():
            print("  [FAIL] missing: replay dir does not exist")
            per[key] = {"ok": False, "error": "replay dir missing"}
            overall = False
            continue
        if not s.is_dir():
            print("  [FAIL] missing: source dir does not exist")
            per[key] = {"ok": False, "error": "source dir missing"}
            overall = False
            continue
        try:
            ok, res = verify_path(s, r, pid, args)
        except Exception as e:  # noqa: BLE001 — one broken path must not hide the rest
            print(f"  [FAIL] exception: {type(e).__name__}: {e}")
            ok, res = False, {"error": f"{type(e).__name__}: {e}"}
        res["ok"] = ok
        per[key] = res
        overall &= ok

    n_pass = sum(1 for v in per.values() if v.get("ok"))
    manifest = {
        "source": str(src),
        "replay": str(rep),
        "created": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": {"max_anchor_m": args.max_anchor,
                       "cam_tol": args.cam_tol,
                       "black_std": args.black_std,
                       "max_drift": args.max_drift,
                       "camera_hz": args.camera_hz},
        "n_paths": len(per),
        "n_pass": n_pass,
        "all_pass": overall,
        "paths": per,
    }
    mpath = rep / "replay_manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*56}")
    for key in sorted(per):
        v = per[key]
        if v.get("ok"):
            print(f"  [PASS] {key}")
        else:
            bad = v.get("error") or ", ".join(
                g for g, d in v.get("gates", {}).items() if not d["ok"])
            print(f"  [FAIL] {key}: {bad}")
    print(f"\n[verify] {n_pass}/{len(per)} paths pass -> "
          f"{'ALL GATES PASS' if overall else 'GATE FAILURE'}")
    print(f"[verify] manifest: {mpath}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
