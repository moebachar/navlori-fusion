"""Convert ONE ILN 2.0 floor into the project's async_collection format.

Source layout (per ILN 2.0):
  iln20/data/<site>/<floor>/
    floor_info.json, floor_image.png, geojson_map.json
    path_data_files/<trace>.txt
      one TSV row per sensor event; types we use: TYPE_ACCELEROMETER,
      TYPE_GYROSCOPE, TYPE_ROTATION_VECTOR, TYPE_WIFI, TYPE_WAYPOINT.

Output layout (matches data/async_collection/):
  data/iln20_<site8>_<floor>/
    meta/
      dataset.json       - site_id, floor_id, splits, AP vocabulary, etc.
      floor_info.json    - copied (width/height in metres)
      geojson_map.json   - copied (NOTE: Web Mercator coords, not waypoint frame)
      floor_image.png    - copied
      bssid_columns.json - ordered list of BSSIDs that became wifi_rssi_* cols
    splits/
      train.txt, val.txt, test.txt   - one path_XX per line
    path_00/ ... path_NN/
      imu.csv             - sim_time, accel_x/y/z, accel_magnitude, gyro_x/y/z,
                            gyro_magnitude, roll_deg, pitch_deg, yaw_deg
      wifi.csv            - sim_time, wifi_visible_count, wifi_strongest_rssi,
                            wifi_strongest_mac, wifi_rssi_<bssid_no_colons>...
      ground_truth.csv    - sim_time, gt_x, gt_y, gt_z, gt_heading_rad,
                            gt_heading_deg, path_id, waypoint_idx
                            (dense 10 Hz, linearly interpolated from raw waypoints)
      waypoints_raw.csv   - sim_time, gt_x, gt_y  (original landmark presses,
                            for Webots replay)
      odometry.csv        - empty header (Webots will populate)
      camera.csv          - empty header (Webots will populate)
      camera/             - empty dir
      metadata.json       - trace_id, t0_unix_ms, sensor counts, etc.

Time convention: every CSV's sim_time is SECONDS from the path's first event
timestamp (t0). Source timestamps are Unix epoch ms.

Coordinate convention: waypoint (x, y) kept in the source frame (metres from
floor-plan corner). NO y-flip. Webots reconstruction will adapt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration (CLI overrides via argparse)
# ---------------------------------------------------------------------------
DEFAULT_SITE = "5d27099f03f801723c32511d"
DEFAULT_FLOOR = "F2"
DEFAULT_SRC_ROOT = r"X:\navlori-fusion\data\iln20\data"
DEFAULT_OUT_ROOT = r"X:\navlori-fusion\data"
DEFAULT_SEED = 42
DEFAULT_GT_HZ = 10.0     # match project Webots GT cadence


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_trace(path: str) -> dict:
    """Read one ILN trace.txt; return per-type lists of (ts_ms, fields...)."""
    accel, gyro, rotvec, wifi, waypoint = [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 2:
                continue
            ts, typ = c[0], c[1]
            try:
                t = int(ts)
            except ValueError:
                continue
            if typ == "TYPE_ACCELEROMETER" and len(c) >= 5:
                try:
                    accel.append((t, float(c[2]), float(c[3]), float(c[4])))
                except ValueError:
                    pass
            elif typ == "TYPE_GYROSCOPE" and len(c) >= 5:
                try:
                    gyro.append((t, float(c[2]), float(c[3]), float(c[4])))
                except ValueError:
                    pass
            elif typ == "TYPE_ROTATION_VECTOR" and len(c) >= 5:
                try:
                    rotvec.append((t, float(c[2]), float(c[3]), float(c[4])))
                except ValueError:
                    pass
            elif typ == "TYPE_WIFI" and len(c) >= 5:
                # cols: ts, TYPE, ssid, bssid, rssi, frequency, lastSeenTs
                try:
                    wifi.append((t, c[3], int(c[4])))
                except ValueError:
                    pass
            elif typ == "TYPE_WAYPOINT" and len(c) >= 4:
                try:
                    waypoint.append((t, float(c[2]), float(c[3])))
                except ValueError:
                    pass
    accel.sort(); gyro.sort(); rotvec.sort(); wifi.sort(); waypoint.sort()
    return {
        "accel": accel, "gyro": gyro, "rotvec": rotvec,
        "wifi": wifi, "waypoint": waypoint,
    }


# ---------------------------------------------------------------------------
# Sensor conversions
# ---------------------------------------------------------------------------
def quat_xyz_to_euler_deg(qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    """Android rotation vector (x, y, z of unit quat) -> roll/pitch/yaw degrees."""
    # w = sqrt(1 - x² - y² - z²), clamped for floating-point safety
    qw2 = max(0.0, 1.0 - (qx * qx + qy * qy + qz * qz))
    qw = math.sqrt(qw2)
    # standard ZYX (yaw, pitch, roll) Tait-Bryan
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def merge_imu_streams(accel, gyro, rotvec, t0_ms: int) -> list[dict]:
    """Join 3 IMU streams on EXACT timestamp (they sample together at 50 Hz).
    Returns rows ready for imu.csv (sim_time in seconds, all fields populated).
    """
    gyro_by_ts = {t: (x, y, z) for t, x, y, z in gyro}
    rot_by_ts = {t: (x, y, z) for t, x, y, z in rotvec}
    rows = []
    for t, ax, ay, az in accel:
        g = gyro_by_ts.get(t)
        r = rot_by_ts.get(t)
        if g is None or r is None:
            continue  # require all three streams present (lockstep dataset)
        gx, gy, gz = g
        roll, pitch, yaw = quat_xyz_to_euler_deg(*r)
        rows.append({
            "sim_time": round((t - t0_ms) / 1000.0, 4),
            "accel_x": round(ax, 5), "accel_y": round(ay, 5), "accel_z": round(az, 5),
            "accel_magnitude": round(math.sqrt(ax*ax + ay*ay + az*az), 5),
            "gyro_x":  round(gx, 6), "gyro_y":  round(gy, 6), "gyro_z":  round(gz, 6),
            "gyro_magnitude":  round(math.sqrt(gx*gx + gy*gy + gz*gz), 6),
            "roll_deg":  round(roll,  3),
            "pitch_deg": round(pitch, 3),
            "yaw_deg":   round(yaw,   3),
        })
    return rows


def collect_bssids(trace_paths: list[str]) -> list[str]:
    """First pass: union of all BSSIDs across all traces. Stable sort."""
    seen = set()
    for p in trace_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 5 and c[1] == "TYPE_WIFI":
                    seen.add(c[3])
    return sorted(seen)


def build_wifi_rows(wifi_entries, bssids: list[str], t0_ms: int) -> list[dict]:
    """Group TYPE_WIFI by scan timestamp; one CSV row per scan.

    Missing AP in a scan -> -200 dBm (matches project's "no signal" convention).
    """
    by_scan = defaultdict(dict)   # {ts_ms: {bssid: rssi}}
    for t, bssid, rssi in wifi_entries:
        by_scan[t][bssid] = rssi
    rows = []
    for ts in sorted(by_scan.keys()):
        scan = by_scan[ts]
        # strongest visible at -85 threshold (matches Webots collector default)
        visible = [r for r in scan.values() if r > -85]
        strongest_mac = max(scan, key=scan.get) if scan else ""
        row = {
            "sim_time": round((ts - t0_ms) / 1000.0, 4),
            "wifi_visible_count": len(visible),
            "wifi_strongest_rssi": round(scan.get(strongest_mac, -200), 1) if strongest_mac else -200,
            "wifi_strongest_mac": strongest_mac,
        }
        for b in bssids:
            row[f"wifi_rssi_{b.replace(':', '')}"] = round(scan.get(b, -200), 1)
        rows.append(row)
    return rows


def interpolate_ground_truth(waypoints, t_start_ms: int, t_end_ms: int,
                              t0_ms: int, gt_hz: float, path_id: int):
    """Linear interpolation of sparse waypoints to a dense GT grid.

    Returns (dense_rows, raw_rows). Each dense row has heading derived from the
    finite-diff direction of motion at that point. waypoint_idx is the index of
    the last raw waypoint at or before that time.
    """
    raw_rows = [
        {"sim_time": round((t - t0_ms) / 1000.0, 4),
         "gt_x": round(x, 5), "gt_y": round(y, 5)}
        for t, x, y in waypoints
    ]
    if len(waypoints) < 2:
        return [], raw_rows

    wp_t = [w[0] for w in waypoints]
    wp_x = [w[1] for w in waypoints]
    wp_y = [w[2] for w in waypoints]

    # dense time grid (ms)
    step_ms = max(1, int(1000.0 / gt_hz))
    grid = list(range(max(t_start_ms, wp_t[0]), min(t_end_ms, wp_t[-1]) + 1, step_ms))
    if not grid:
        return [], raw_rows

    dense = []
    last_xy = None
    j = 0
    for t in grid:
        # advance j so that wp_t[j] <= t < wp_t[j+1]
        while j + 1 < len(wp_t) and wp_t[j + 1] < t:
            j += 1
        if j + 1 >= len(wp_t):
            # at or past the last waypoint
            x, y = wp_x[-1], wp_y[-1]
        else:
            t0, t1 = wp_t[j], wp_t[j + 1]
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            x = wp_x[j] + f * (wp_x[j + 1] - wp_x[j])
            y = wp_y[j] + f * (wp_y[j + 1] - wp_y[j])
        heading_rad = 0.0
        if last_xy is not None:
            dx, dy = x - last_xy[0], y - last_xy[1]
            if dx * dx + dy * dy > 1e-8:
                heading_rad = math.atan2(dy, dx)
        last_xy = (x, y)
        # find last raw waypoint at or before this t
        wp_idx = j if wp_t[j] <= t else max(0, j - 1)
        dense.append({
            "sim_time": round((t - t0_ms) / 1000.0, 4),
            "gt_x": round(x, 5), "gt_y": round(y, 5), "gt_z": 0.0,
            "gt_heading_rad": round(heading_rad, 5),
            "gt_heading_deg": round(math.degrees(heading_rad), 3),
            "path_id": path_id,
            "waypoint_idx": wp_idx,
        })
    return dense, raw_rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(path: str, cols: list[str], rows: list[dict]) -> int:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=DEFAULT_SITE)
    ap.add_argument("--floor", default=DEFAULT_FLOOR)
    ap.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--gt-hz", type=float, default=DEFAULT_GT_HZ)
    args = ap.parse_args()

    src = os.path.join(args.src_root, args.site, args.floor)
    pdir = os.path.join(src, "path_data_files")
    if not os.path.isdir(pdir):
        sys.exit(f"source not found: {pdir}")

    out_name = f"iln20_{args.site[:8]}_{args.floor}"
    out = os.path.join(args.out_root, out_name)
    if os.path.isdir(out):
        print(f"[warn] {out} exists; will overwrite per-path contents")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out, "splits"), exist_ok=True)

    # ----- meta: copy source artefacts -----
    for f in ("floor_info.json", "geojson_map.json", "floor_image.png"):
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out, "meta", f))

    # ----- order traces deterministically (by trace ID = ObjectId timestamp) -----
    trace_files = sorted(f for f in os.listdir(pdir) if f.endswith(".txt"))
    print(f"[iln20] {args.site}/{args.floor}: {len(trace_files)} traces")

    # ----- pass 1: collect global BSSID vocabulary -----
    print(f"[iln20] pass 1: scanning all traces for BSSID vocabulary...", flush=True)
    bssids = collect_bssids([os.path.join(pdir, f) for f in trace_files])
    print(f"[iln20]   {len(bssids)} unique BSSIDs across all traces", flush=True)
    with open(os.path.join(out, "meta", "bssid_columns.json"), "w") as fh:
        json.dump({"bssids": bssids,
                   "n_bssids": len(bssids),
                   "column_format": "wifi_rssi_<bssid_no_colons>"}, fh, indent=2)

    # ----- splits (random, fixed seed) -----
    n = len(trace_files)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    n_test = n - n_train - n_val
    pids = list(range(n))
    random.Random(args.seed).shuffle(pids)
    split = {
        "train": sorted(pids[:n_train]),
        "val":   sorted(pids[n_train:n_train + n_val]),
        "test":  sorted(pids[n_train + n_val:]),
    }
    print(f"[iln20] split (seed={args.seed}): train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}")
    for name, ids in split.items():
        with open(os.path.join(out, "splits", f"{name}.txt"), "w") as fh:
            for pid in ids:
                fh.write(f"path_{pid:02d}\n")

    # ----- column orders (match async_collector.py) -----
    imu_cols = ["sim_time", "accel_x", "accel_y", "accel_z", "accel_magnitude",
                "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude",
                "roll_deg", "pitch_deg", "yaw_deg"]
    odom_cols = ["sim_time", "odom_x", "odom_y", "odom_theta_deg",
                 "odom_linear_vel", "odom_angular_vel",
                 "wheel_left_vel", "wheel_right_vel"]
    wifi_cols = (["sim_time", "wifi_visible_count", "wifi_strongest_rssi",
                  "wifi_strongest_mac"]
                 + [f"wifi_rssi_{b.replace(':', '')}" for b in bssids])
    gt_cols = ["sim_time", "gt_x", "gt_y", "gt_z", "gt_heading_rad",
               "gt_heading_deg", "path_id", "waypoint_idx"]
    cam_cols = ["sim_time", "frame_id", "rgb_path", "depth_path"]
    raw_wp_cols = ["sim_time", "gt_x", "gt_y"]

    # ----- pass 2: per-path conversion -----
    per_path_meta = {}
    for pid, fn in enumerate(trace_files):
        path_dir = os.path.join(out, f"path_{pid:02d}")
        os.makedirs(path_dir, exist_ok=True)
        os.makedirs(os.path.join(path_dir, "camera"), exist_ok=True)
        trace_id = fn[:-4]
        events = parse_trace(os.path.join(pdir, fn))
        # earliest event = t0 for this path
        all_ts = ([t for t, *_ in events["accel"]]
                  + [t for t, *_ in events["gyro"]]
                  + [t for t, *_ in events["rotvec"]]
                  + [t for t, *_ in events["wifi"]]
                  + [t for t, *_ in events["waypoint"]])
        if not all_ts:
            print(f"  [{pid:02d}] {trace_id}: EMPTY trace, skipping", flush=True)
            continue
        t0_ms = min(all_ts)
        t_end_ms = max(all_ts)

        imu_rows = merge_imu_streams(events["accel"], events["gyro"],
                                      events["rotvec"], t0_ms)
        wifi_rows = build_wifi_rows(events["wifi"], bssids, t0_ms)
        gt_dense, gt_raw = interpolate_ground_truth(
            events["waypoint"], t0_ms, t_end_ms, t0_ms, args.gt_hz, pid)

        n_imu  = write_csv(os.path.join(path_dir, "imu.csv"), imu_cols, imu_rows)
        n_wifi = write_csv(os.path.join(path_dir, "wifi.csv"), wifi_cols, wifi_rows)
        n_gt   = write_csv(os.path.join(path_dir, "ground_truth.csv"), gt_cols, gt_dense)
        n_wpr  = write_csv(os.path.join(path_dir, "waypoints_raw.csv"), raw_wp_cols, gt_raw)
        # empty stubs for Webots to fill
        n_odo  = write_csv(os.path.join(path_dir, "odometry.csv"), odom_cols, [])
        n_cam  = write_csv(os.path.join(path_dir, "camera.csv"), cam_cols, [])

        meta = {
            "path_id": pid,
            "trace_id": trace_id,
            "site_id": args.site,
            "floor_id": args.floor,
            "source": "iln20",
            "t0_unix_ms": t0_ms,
            "duration_s": round((t_end_ms - t0_ms) / 1000.0, 3),
            "samples": {"imu": n_imu, "wifi": n_wifi,
                         "ground_truth": n_gt, "waypoints_raw": n_wpr,
                         "odometry": n_odo, "camera": n_cam},
            "n_raw_waypoints": len(events["waypoint"]),
            "n_bssids_in_trace": len({b for _, b, _ in events["wifi"]}),
        }
        with open(os.path.join(path_dir, "metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        per_path_meta[f"path_{pid:02d}"] = meta
        print(f"  [{pid:02d}] {trace_id}: dur={meta['duration_s']:>6.1f}s "
              f"imu={n_imu:>5} wifi={n_wifi:>3} wp_raw={n_wpr:>2} "
              f"gt_dense={n_gt:>4} (skipped odom/cam)", flush=True)

    # ----- top-level dataset metadata -----
    info = json.load(open(os.path.join(out, "meta", "floor_info.json")))
    mi = info.get("map_info", info)
    dataset_meta = {
        "name": out_name,
        "source": "Microsoft / XYZ10 Indoor Location Competition 2.0 (Kaggle)",
        "license": "MIT (https://github.com/location-competition/indoor-location-competition-20)",
        "site_id": args.site,
        "floor_id": args.floor,
        "floor_width_m": mi["width"],
        "floor_height_m": mi["height"],
        "splits": split,
        "split_seed": args.seed,
        "split_ratio": [0.70, 0.15, 0.15],
        "gt_interp_hz": args.gt_hz,
        "modalities_present": ["imu", "wifi", "ground_truth", "waypoints_raw"],
        "modalities_pending_from_webots": ["odometry", "camera"],
        "n_paths": len(per_path_meta),
        "n_bssids": len(bssids),
        "coordinate_frame": "waypoints in METRES, origin at floor-plan corner "
                            "(matches floor_info width/height). NO y-flip. "
                            "GeoJSON is in Web Mercator (EPSG:3857) — needs "
                            "separate transform before use in Webots.",
        "asyncolloection_compatibility": (
            "CSVs match async_collector.py output column names. "
            "FusionDataset reads this dir like any other async_collection."),
    }
    with open(os.path.join(out, "meta", "dataset.json"), "w") as fh:
        json.dump(dataset_meta, fh, indent=2)

    print(f"\n[iln20] DONE. Output: {out}")
    print(f"        {len(per_path_meta)} paths converted.")
    print(f"        WiFi vocabulary: {len(bssids)} BSSIDs.")
    print(f"        Splits: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
