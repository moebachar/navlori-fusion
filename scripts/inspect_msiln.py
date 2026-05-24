"""Inspect a Microsoft Indoor Location & Navigation (ILN 2.0) site.

Reads one site/floor (or all floors of a site) from the vendored
`indoor-location-competition-20` repo and reports the schema metrics
required by handoff/plans/PLAN_01:

- GT extent (m x m)
- GT (waypoint) rate (Hz) and step distribution (cm: med/p90/max)
- WiFi scan rate (Hz), WiFi NaN/missing %, APs visible (mean per scan) / total unique
- IMU rate (Hz) and any NaN/Inf
- Number of sessions/paths in the site, split by day if possible
  (sessions are inferred from the MongoDB ObjectId timestamp encoded in
   each trace filename; the first 4 bytes of an ObjectId are a Unix
   timestamp.)

Usage:
    python scripts/inspect_msiln.py --root <path-to-msiln20> --site site1
    python scripts/inspect_msiln.py --root <path-to-msiln20> --site site1 --floor F1

The script imports `read_data_file` from the vendored `io_f.py` so the
data parsing remains unmodified (Demand #3: vendored baselines stay as
upstream ships them).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _decode_objectid_ts(filename: str) -> dt.datetime | None:
    """Decode the embedded creation timestamp from a MongoDB ObjectId filename."""
    stem = Path(filename).stem
    if len(stem) < 8:
        return None
    try:
        return dt.datetime.utcfromtimestamp(int(stem[:8], 16))
    except ValueError:
        return None


def _rate_hz(timestamps_ms: np.ndarray) -> float:
    """Mean sample rate (Hz) from millisecond timestamps. NaN if fewer than 2 samples."""
    if timestamps_ms.size < 2:
        return float("nan")
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return float("nan")
    return (timestamps_ms.size - 1) / span_s


def _scan_intervals_hz(timestamps_ms: np.ndarray) -> float:
    """Mean inter-event rate for sparse events (WiFi scans)."""
    if timestamps_ms.size < 2:
        return float("nan")
    diffs_ms = np.diff(np.sort(timestamps_ms))
    diffs_ms = diffs_ms[diffs_ms > 0]
    if diffs_ms.size == 0:
        return float("nan")
    return 1000.0 / np.median(diffs_ms)


def inspect_floor(floor_dir: Path, read_data_file):
    """Return a dict of metrics aggregated over all traces in one floor."""
    pdf = floor_dir / "path_data_files"
    files = sorted(pdf.glob("*.txt"))
    if not files:
        return None

    gt_xy = []
    step_lens_cm = []
    waypoint_ts_all = []
    imu_rates = []
    wifi_scan_ts_per_file = []
    wifi_aps_per_scan = []
    wifi_unique_bssids = set()
    wifi_rssi_finite_total = 0
    wifi_rssi_total = 0
    imu_nan_count = 0
    imu_total_count = 0
    session_days = []
    session_dts = []

    for f in files:
        ts_dt = _decode_objectid_ts(f.name)
        if ts_dt is not None:
            session_dts.append(ts_dt)
            session_days.append(ts_dt.date())

        try:
            d = read_data_file(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to parse {f.name}: {e}", file=sys.stderr)
            continue

        # Waypoints (GT)
        wp = d.waypoint
        if wp.size:
            gt_xy.append(wp[:, 1:3])
            waypoint_ts_all.append(wp[:, 0].astype(np.int64))
            for i in range(1, len(wp)):
                step_lens_cm.append(
                    np.hypot(wp[i, 1] - wp[i - 1, 1], wp[i, 2] - wp[i - 1, 2]) * 100.0
                )

        # IMU (accelerometer chosen as primary rate; gyro/magn share scheduler)
        acce = d.acce
        if acce.size:
            imu_rates.append(_rate_hz(acce[:, 0].astype(np.int64)))
            arr = acce[:, 1:4]
            imu_nan_count += int(np.isnan(arr).sum() + np.isinf(arr).sum())
            imu_total_count += int(arr.size)

        # WiFi
        wifi = d.wifi
        if wifi.size:
            # wifi cols: [sys_ts, ssid, bssid, rssi, lastseen_ts]  (all strings)
            sys_ts = wifi[:, 0].astype(np.int64)
            # group rows by scan event (same sys_ts ~= one scan)
            unique_scans, counts = np.unique(sys_ts, return_counts=True)
            wifi_scan_ts_per_file.append(unique_scans)
            wifi_aps_per_scan.extend(counts.tolist())
            for bssid in wifi[:, 2].tolist():
                wifi_unique_bssids.add(bssid)
            # RSSI parse / NaN check
            for rs in wifi[:, 3].tolist():
                wifi_rssi_total += 1
                try:
                    v = float(rs)
                    if np.isfinite(v):
                        wifi_rssi_finite_total += 1
                except ValueError:
                    pass

    if not gt_xy:
        return None
    gt_xy_all = np.vstack(gt_xy)
    bbox_w = float(gt_xy_all[:, 0].max() - gt_xy_all[:, 0].min())
    bbox_h = float(gt_xy_all[:, 1].max() - gt_xy_all[:, 1].min())

    waypoint_ts_concat = np.concatenate(waypoint_ts_all) if waypoint_ts_all else np.array([], dtype=np.int64)
    gt_rate_hz = _rate_hz(np.sort(waypoint_ts_concat)) if waypoint_ts_concat.size else float("nan")

    wifi_scan_concat = (
        np.concatenate(wifi_scan_ts_per_file) if wifi_scan_ts_per_file else np.array([], dtype=np.int64)
    )
    wifi_scan_rate_hz = _scan_intervals_hz(wifi_scan_concat)

    days_sorted = sorted(set(session_days))
    days_count = defaultdict(int)
    for dd in session_days:
        days_count[str(dd)] += 1

    step_arr = np.array(step_lens_cm) if step_lens_cm else np.array([0.0])

    return {
        "floor": floor_dir.name,
        "n_paths": len(files),
        "gt": {
            "extent_w_m": bbox_w,
            "extent_h_m": bbox_h,
            "rate_hz": gt_rate_hz,
            "n_waypoints": int(waypoint_ts_concat.size),
            "step_cm_median": float(np.median(step_arr)),
            "step_cm_p90": float(np.percentile(step_arr, 90)),
            "step_cm_max": float(step_arr.max()),
        },
        "imu": {
            "rate_hz_mean": float(np.mean(imu_rates)) if imu_rates else float("nan"),
            "rate_hz_median": float(np.median(imu_rates)) if imu_rates else float("nan"),
            "nan_inf_count": imu_nan_count,
            "total_samples": imu_total_count,
        },
        "wifi": {
            "scan_rate_hz_median": float(wifi_scan_rate_hz),
            "aps_per_scan_mean": float(np.mean(wifi_aps_per_scan)) if wifi_aps_per_scan else float("nan"),
            "aps_per_scan_median": float(np.median(wifi_aps_per_scan)) if wifi_aps_per_scan else float("nan"),
            "unique_bssids_total": len(wifi_unique_bssids),
            "rssi_parseable_pct": (
                100.0 * wifi_rssi_finite_total / wifi_rssi_total if wifi_rssi_total else float("nan")
            ),
            "rssi_total_rows": wifi_rssi_total,
        },
        "sessions": {
            "n_distinct_days": len(days_sorted),
            "first_day": str(days_sorted[0]) if days_sorted else None,
            "last_day": str(days_sorted[-1]) if days_sorted else None,
            "paths_per_day": dict(days_count),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to msiln20 repo root (must contain io_f.py and data/)")
    ap.add_argument("--site", default="site1", help="Site folder under data/ (default: site1)")
    ap.add_argument("--floor", default=None, help="Single floor (e.g. F1); default: scan all floors of the site")
    ap.add_argument("--out", default=None, help="Optional path to write the human-readable report")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "io_f.py").exists():
        print(f"ERROR: io_f.py not found under {root}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))
    from io_f import read_data_file  # noqa: PLC0415

    site_dir = root / "data" / args.site
    if not site_dir.exists():
        print(f"ERROR: site dir not found: {site_dir}", file=sys.stderr)
        return 2

    if args.floor:
        floors = [site_dir / args.floor]
    else:
        floors = sorted(p for p in site_dir.iterdir() if p.is_dir())

    reports = []
    for fdir in floors:
        if not (fdir / "path_data_files").exists():
            continue
        print(f"[inspect] {fdir.relative_to(root)} ...", flush=True)
        r = inspect_floor(fdir, read_data_file)
        if r is not None:
            r["site"] = args.site
            reports.append(r)

    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append(f"Microsoft ILN 2.0 schema inspection -- {args.site}")
    summary_lines.append("=" * 78)
    total_paths = sum(r["n_paths"] for r in reports)
    all_days = set()
    for r in reports:
        first = r["sessions"]["first_day"]
        last = r["sessions"]["last_day"]
        if first:
            all_days.add(first)
        if last:
            all_days.add(last)
    summary_lines.append(
        f"floors scanned: {len(reports)}    total traces: {total_paths}    "
        f"distinct day span: {len(all_days)} dates"
    )
    summary_lines.append("")

    for r in reports:
        summary_lines.append(f"--- {r['floor']} ({r['n_paths']} paths) ---")
        gt = r["gt"]
        summary_lines.append(
            f"  GT extent : {gt['extent_w_m']:.1f} m x {gt['extent_h_m']:.1f} m   "
            f"rate ~ {gt['rate_hz']:.2f} Hz  ({gt['n_waypoints']} waypoints)"
        )
        summary_lines.append(
            f"  step len  : median {gt['step_cm_median']:.1f} cm   "
            f"p90 {gt['step_cm_p90']:.1f} cm   max {gt['step_cm_max']:.1f} cm"
        )
        imu = r["imu"]
        summary_lines.append(
            f"  IMU acc   : {imu['rate_hz_mean']:.1f} Hz mean   "
            f"NaN/Inf {imu['nan_inf_count']}/{imu['total_samples']} samples"
        )
        wifi = r["wifi"]
        summary_lines.append(
            f"  WiFi      : scan {wifi['scan_rate_hz_median']:.2f} Hz median   "
            f"{wifi['aps_per_scan_mean']:.0f} APs/scan mean   "
            f"{wifi['unique_bssids_total']} unique BSSIDs   "
            f"RSSI parse {wifi['rssi_parseable_pct']:.1f}%"
        )
        ses = r["sessions"]
        summary_lines.append(
            f"  Sessions  : {ses['n_distinct_days']} distinct days  "
            f"[{ses['first_day']} ... {ses['last_day']}]"
        )
        summary_lines.append("")

    summary_lines.append("=" * 78)
    summary_lines.append("RAW REPORT (JSON):")
    summary_lines.append("=" * 78)
    summary_lines.append(json.dumps(reports, indent=2, default=str))

    text = "\n".join(summary_lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
