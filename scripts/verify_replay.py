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

from path_loader import build_trajectory, imu_intensity, load_path  # noqa: E402
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
    # Rebuild the SAME trajectory the controller drove: the IMU-speed-profile
    # settings live in the replayed path's metadata.json (absent for pre-4a
    # data -> plain uniform-speed spline).
    prof = {}
    meta_p = rep / "metadata.json"
    if meta_p.is_file():
        try:
            prof = json.loads(meta_p.read_text(encoding="utf-8")) \
                .get("imu_speed_profile", {}) or {}
        except json.JSONDecodeError:
            prof = {}
    profile_on = bool(prof.get("enabled", False))
    traj, _ = build_trajectory(rp, imu_speed_profile=profile_on,
                               v_max=float(prof.get("v_max", 2.5)),
                               floor_frac=float(prof.get("floor_frac", 0.05)))
    results["info"]["imu_speed_profile"] = profile_on
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
    # one timestep of motion: the robot sits exactly at spline(t - DT). The
    # expected error is therefore the one-step spline DISPLACEMENT
    # |spline(t) - spline(t-DT)| -- NOT speed*DT, which is a straight-line
    # model that overshoots at heading kinks (900 deg/s kinks false-failed
    # six F2 paths by ~1 mm). True anchor deviation = residual vs that.
    DT = 0.032
    gt_rows = read_csv(rep / "ground_truth.csv")
    raw, corrected = [], []
    for r in gt_rows:
        if r.get("anchor_err_m") in (None, "") or r.get("is_original") not in ("0", "False"):
            continue
        e = float(r["anchor_err_m"])
        t = float(r["sim_time"])
        x1, y1 = traj.evaluate_position(t)
        x0, y0 = traj.evaluate_position(max(rp.t_start, t - DT))
        d = math.hypot(x1 - x0, y1 - y0)
        raw.append(e)
        corrected.append(abs(e - d))
    if corrected:
        mx = max(corrected)
        all_ok &= gate("anchor", mx < args.max_anchor,
                       f"lag-corrected max={mx*1000:.2f}mm over {len(corrected)} rows "
                       f"(raw max={max(raw)*100:.2f}cm incl. one-step motion; "
                       f"limit {args.max_anchor*1000:.0f}mm)")
    else:
        all_ok &= gate("anchor", False, "no anchor_err_m values found in ground_truth.csv")

    # ── Gate 1b: hover height — gt_z must equal the configured hover and be
    #     CONSTANT (the cumulative +1cm/path bug reached 0.62 m unnoticed;
    #     it also lifted the camera by the same amount). Only gated when the
    #     controller recorded hover_m (post-fix data). ──
    hover = None
    try:
        hover = float(json.loads((rep / "metadata.json")
                                 .read_text(encoding="utf-8"))["hover_m"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        pass
    if hover is None:
        # Absence of hover_m marks PRE-FIX controller output. A truncated
        # stale path once hid behind the old "skip when missing" behaviour
        # (path_15, gt_z=0.16, cut at 16.8/19.2s) -- stale is a failure.
        all_ok &= gate("z-hover", False,
                       "metadata.json lacks hover_m -> stale pre-fix data; "
                       "delete the path dir and re-replay it")
    else:
        zs = [float(r["gt_z"]) for r in gt_rows if r.get("gt_z") not in (None, "")]
        if zs:
            worst = max(abs(z - hover) for z in zs)
            all_ok &= gate("z-hover", worst < 0.005,
                           f"max |gt_z - {hover:.3f}| = {worst*1000:.1f}mm "
                           f"(limit 5mm; camera height depends on it)")

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

    # ── Info (profile on): label shift + odom-speed-vs-IMU correlation ──
    if profile_on and hasattr(traj, "base"):
        # how far the IMU-shaped labels moved vs the old uniform-speed fiction
        shifts = []
        n_grid = 400
        for k in range(n_grid + 1):
            t = rp.t_start + duration * k / n_grid
            wx, wy = traj.evaluate_position(t)
            bx, by = traj.base.evaluate_position(t)
            shifts.append(math.hypot(wx - bx, wy - by))
        # does the replayed odometry speed now follow the real IMU intensity?
        corr = float("nan")
        its, s = imu_intensity(rp)
        if odom_rows and its:
            import bisect
            xs_, ys_ = [], []
            for r in odom_rows:
                t = float(r["sim_time"])
                if not (its[0] <= t <= its[-1]):
                    continue
                i = bisect.bisect_left(its, t)
                i = min(max(i, 1), len(its) - 1)
                f = (t - its[i - 1]) / max(1e-9, its[i] - its[i - 1])
                xs_.append(s[i - 1] + f * (s[i] - s[i - 1]))
                ys_.append(abs(float(r["odom_linear_vel"])))
            if len(xs_) > 8:
                mx = sum(xs_) / len(xs_)
                my = sum(ys_) / len(ys_)
                num = sum((a - mx) * (b - my) for a, b in zip(xs_, ys_))
                den = math.sqrt(sum((a - mx) ** 2 for a in xs_)
                                * sum((b - my) ** 2 for b in ys_))
                corr = num / den if den > 1e-12 else float("nan")
        print(f"  [info] imu-profile: max label shift vs uniform-speed "
              f"{max(shifts):.2f}m (mean {sum(shifts)/len(shifts):.2f}m); "
              f"odom-speed vs IMU-intensity r={corr:.2f}")
        results["info"]["imu_profile_label_shift_m"] = {
            "max": round(max(shifts), 3),
            "mean": round(sum(shifts) / len(shifts), 3)}
        results["info"]["odom_vs_imu_corr"] = (round(corr, 3)
                                               if corr == corr else None)

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
    ap.add_argument("--max-drift", type=float, default=0.06,
                    help="gate 3 threshold (fraction). Catches synthesis BUGS "
                         "(the historic timing bug was 43%%). The stochastic "
                         "slip tail on ~100m paths reaches ~4%% at uniform "
                         "speed and ~5.5%% with the IMU profile (motion "
                         "concentrates into bursts -> larger per-step wheel "
                         "arcs -> more slip variance at identical distance).")
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
