"""Convert Microsoft Indoor Location & Navigation 2.0 to navlori-fusion format.

Mirrors `scripts/convert_ipin2024.py` in output layout, CLI surface, and
`metadata.json` schema. One-floor mode (default: site1/B1, recommended by
RESULT_01).

Source format (one .txt per trace, vendored io_f.py groups it for us)
---------------------------------------------------------------------
- Unix-ms timestamps, multiple `TYPE_*` rows per timestamp.
- `TYPE_WAYPOINT`  -> ground truth (x, y) in floor-local metres (no projection).
- `TYPE_ACCELEROMETER` / `TYPE_GYROSCOPE` / `TYPE_MAGNETIC_FIELD` /
  `TYPE_ROTATION_VECTOR` -> IMU rows.
- `TYPE_WIFI` -> rows tagged by BSSID + RSSI; rows with the same sys_ts
  belong to one scan.

Demand #3: `io_f.py` is imported via `importlib.util.spec_from_file_location`
from the cloned starter repo; the vendored source is never modified.

Splits
------
Each trace filename is a MongoDB ObjectId; the first 8 hex chars are a
Unix timestamp of creation. We group traces by UTC date and assign the
day boundaries deterministically (earliest two days -> train, second-to-last
-> val, last -> test) unless `--split-spec` overrides.

Outputs
-------
``<out_root>/msiln_<site>_<floor>/``
    ├── path_NN/
    │     ├── imu.csv          — accel+gyro+ahrs merged @ ~50 Hz
    │     ├── wifi.csv         — per-scan RSSI fingerprints
    │     ├── odometry.csv     — header-only stub (phone dataset)
    │     ├── ground_truth.csv — waypoint anchors linearly interp @ ~10 Hz
    │     ├── metadata.json
    │     └── trajectory.png
    ├── ap_vocab.json
    ├── split.json
    ├── metadata.json
    └── convert_msiln.py        — archived copy
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── tunables (mirror IPIN converter where applicable) ───────────────────────
IMU_TARGET_HZ = 50          # msiln native is 50 Hz; keep it
IMU_NN_TOL_S = 0.05         # nearest-neighbor tolerance when merging gyro/magn onto accel
GT_INTERP_HZ = 10           # resample interpolated waypoints to 10 Hz (IPIN convention)
MIN_WAYPOINTS_PER_TRACE = 3 # drop traces with fewer than this many anchors
MIN_TRACE_SECONDS = 5.0     # drop very short traces

IMU_STUB = ("sim_time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,"
            "roll_deg,pitch_deg,yaw_deg\n")
WIFI_STUB = "sim_time,wifi_visible_count,wifi_strongest_rssi,wifi_strongest_mac\n"
ODOM_STUB = ("sim_time,odom_x,odom_y,odom_theta_deg,"
             "odom_linear_vel,odom_angular_vel,wheel_left_vel,wheel_right_vel\n")


def _load_vendored_io(msiln_root: Path):
    """Import io_f.read_data_file from the cloned competition repo unmodified."""
    src = msiln_root / "io_f.py"
    if not src.exists():
        raise FileNotFoundError(f"vendored io_f.py not found at {src}")
    spec = importlib.util.spec_from_file_location("msiln20_io_f", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.read_data_file


def _decode_day(filename: str) -> dt.date | None:
    """First 8 hex chars of an ObjectId encode a Unix timestamp (s)."""
    stem = Path(filename).stem
    if len(stem) < 8:
        return None
    try:
        return dt.datetime.utcfromtimestamp(int(stem[:8], 16)).date()
    except ValueError:
        return None


# ── per-trace builders ──────────────────────────────────────────────────────

def build_gt_df(waypoints_ms_xy: np.ndarray, t_lo_ms: float) -> pd.DataFrame:
    """Linearly interpolate waypoint anchors at GT_INTERP_HZ.

    waypoints_ms_xy: (K, 3) -> [unix_ms, x_m, y_m].
    Returns df with sim_time in seconds (re-zeroed to t_lo_ms).
    """
    if len(waypoints_ms_xy) < 2:
        return pd.DataFrame(columns=["sim_time", "gt_x", "gt_y"])
    ts_ms = waypoints_ms_xy[:, 0].astype(np.float64)
    x = waypoints_ms_xy[:, 1].astype(np.float64)
    y = waypoints_ms_xy[:, 2].astype(np.float64)

    t0_ms, t1_ms = ts_ms[0], ts_ms[-1]
    duration_s = (t1_ms - t0_ms) / 1000.0
    n = max(int(duration_s * GT_INTERP_HZ) + 1, 2)
    dense_ms = np.linspace(t0_ms, t1_ms, n)
    xd = np.interp(dense_ms, ts_ms, x)
    yd = np.interp(dense_ms, ts_ms, y)
    return pd.DataFrame({
        "sim_time": np.round((dense_ms - t_lo_ms) / 1000.0, 3),
        "gt_x": np.round(xd, 4),
        "gt_y": np.round(yd, 4),
    })


def _ahrs_to_euler_deg(ahrs: np.ndarray) -> np.ndarray:
    """msiln TYPE_ROTATION_VECTOR is (qx, qy, qz); reconstruct qw, then RPY.

    ahrs cols: [ts_ms, qx, qy, qz]. Returns (N, 3) of [roll, pitch, yaw] in degrees.
    Handles edge case where qx^2+qy^2+qz^2 > 1 by clipping.
    """
    if len(ahrs) == 0:
        return np.empty((0, 3))
    qx, qy, qz = ahrs[:, 1], ahrs[:, 2], ahrs[:, 3]
    norm_sq = qx * qx + qy * qy + qz * qz
    qw = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(norm_sq, 1.0)))
    # Standard quaternion -> RPY (Z-Y-X intrinsic)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.stack([roll, pitch, yaw], axis=-1))


def _nearest_merge(base_ms: np.ndarray, other: np.ndarray, tol_s: float) -> np.ndarray:
    """Merge `other[:, 1:]` onto base_ms (ms) by nearest-ts within tol_s.

    other: (M, K+1), first column timestamp ms. Returns (N, K) with NaN
    where no match within tol.
    """
    if len(other) == 0:
        return np.full((len(base_ms), other.shape[1] - 1 if other.ndim > 1 else 1), np.nan)
    o_ts = other[:, 0]
    idx = np.searchsorted(o_ts, base_ms)
    out = np.full((len(base_ms), other.shape[1] - 1), np.nan)
    tol_ms = tol_s * 1000.0
    for i, t in enumerate(base_ms):
        best = None
        best_dt = tol_ms
        if idx[i] > 0:
            dtm = abs(o_ts[idx[i] - 1] - t)
            if dtm <= best_dt:
                best, best_dt = idx[i] - 1, dtm
        if idx[i] < len(o_ts):
            dtm = abs(o_ts[idx[i]] - t)
            if dtm <= best_dt:
                best, best_dt = idx[i], dtm
        if best is not None:
            out[i] = other[best, 1:]
    return out


def build_imu_df(acce: np.ndarray, gyro: np.ndarray, ahrs: np.ndarray,
                 t_lo_ms: float, t_hi_ms: float) -> pd.DataFrame:
    """Build imu.csv with accel as base time grid, decimated to IMU_TARGET_HZ."""
    if len(acce) == 0:
        return pd.DataFrame(columns=["sim_time", "accel_x", "accel_y", "accel_z",
                                     "gyro_x", "gyro_y", "gyro_z",
                                     "roll_deg", "pitch_deg", "yaw_deg"])
    mask = (acce[:, 0] >= t_lo_ms) & (acce[:, 0] <= t_hi_ms)
    a = acce[mask]
    if len(a) == 0:
        return pd.DataFrame(columns=["sim_time", "accel_x", "accel_y", "accel_z",
                                     "gyro_x", "gyro_y", "gyro_z",
                                     "roll_deg", "pitch_deg", "yaw_deg"])
    native_hz = max(len(a) / max((a[-1, 0] - a[0, 0]) / 1000.0, 1e-6), 1.0)
    step = max(int(round(native_hz / IMU_TARGET_HZ)), 1)
    a_ds = a[::step]

    gyr = _nearest_merge(a_ds[:, 0], gyro, IMU_NN_TOL_S) if len(gyro) else \
        np.full((len(a_ds), 3), np.nan)

    if len(ahrs):
        ahrs_mask = (ahrs[:, 0] >= t_lo_ms) & (ahrs[:, 0] <= t_hi_ms)
        ahrs_in = ahrs[ahrs_mask]
        rpy_full = _ahrs_to_euler_deg(ahrs_in) if len(ahrs_in) else \
            np.full((0, 3), np.nan)
        if len(ahrs_in):
            # nearest-merge RPY onto a_ds timestamps
            rpy_with_ts = np.concatenate([ahrs_in[:, [0]], rpy_full], axis=1)
            rpy = _nearest_merge(a_ds[:, 0], rpy_with_ts, IMU_NN_TOL_S)
        else:
            rpy = np.full((len(a_ds), 3), np.nan)
    else:
        rpy = np.full((len(a_ds), 3), np.nan)

    sim_time_s = (a_ds[:, 0] - t_lo_ms) / 1000.0
    df = pd.DataFrame({
        "sim_time": np.round(sim_time_s, 4),
        "accel_x": a_ds[:, 1], "accel_y": a_ds[:, 2], "accel_z": a_ds[:, 3],
        "gyro_x": gyr[:, 0], "gyro_y": gyr[:, 1], "gyro_z": gyr[:, 2],
        "roll_deg": rpy[:, 0], "pitch_deg": rpy[:, 1], "yaw_deg": rpy[:, 2],
    })
    return df


def build_wifi_df(wifi_raw: np.ndarray, t_lo_ms: float, t_hi_ms: float,
                  bssid_vocab: list[str]) -> pd.DataFrame:
    """Group wifi rows by sys_ts (one scan) and pivot to per-BSSID columns.

    wifi_raw cols (strings): [sys_ts, ssid, bssid, rssi, lastseen_ts]
    """
    if len(wifi_raw) == 0:
        return pd.DataFrame()
    ts = wifi_raw[:, 0].astype(np.int64)
    mask = (ts >= t_lo_ms) & (ts <= t_hi_ms)
    if not mask.any():
        return pd.DataFrame()
    ts_f = ts[mask]
    bssid = wifi_raw[mask, 2]
    try:
        rssi = wifi_raw[mask, 3].astype(np.float32)
    except ValueError:
        rssi = np.array([float(r) for r in wifi_raw[mask, 3].tolist()], dtype=np.float32)

    rssi_cols = [f"wifi_rssi_{m}" for m in bssid_vocab]
    mac_to_col = {m: i for i, m in enumerate(bssid_vocab)}

    # Group by sys_ts
    unique_ts, inv = np.unique(ts_f, return_inverse=True)
    rows = []
    for s_idx, scan_ts in enumerate(unique_ts):
        sel = inv == s_idx
        macs_here = bssid[sel]
        rssis_here = rssi[sel]
        row = {c: np.nan for c in rssi_cols}
        visible = 0
        best_rssi = np.nan
        best_mac = ""
        for m, r in zip(macs_here, rssis_here):
            col = mac_to_col.get(m)
            if col is None:
                continue
            # Keep max RSSI if a BSSID duplicates within one scan
            if np.isnan(row[rssi_cols[col]]) or r > row[rssi_cols[col]]:
                row[rssi_cols[col]] = r
            visible += 1
            if np.isnan(best_rssi) or r > best_rssi:
                best_rssi, best_mac = float(r), m
        if visible == 0:
            continue
        row_out = {
            "sim_time": round((float(scan_ts) - t_lo_ms) / 1000.0, 3),
            "wifi_visible_count": visible,
            "wifi_strongest_rssi": best_rssi,
            "wifi_strongest_mac": best_mac,
        }
        row_out.update(row)
        rows.append(row_out)
    cols = ["sim_time", "wifi_visible_count", "wifi_strongest_rssi",
            "wifi_strongest_mac"] + rssi_cols
    return pd.DataFrame(rows, columns=cols)


def write_path(out_dir: Path, gt_df: pd.DataFrame,
               imu_df: pd.DataFrame, wifi_df: pd.DataFrame,
               src_file: Path, site: str, floor: str,
               path_id: int, native_split: str,
               day: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(gt_df["sim_time"].max() - gt_df["sim_time"].min())

    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)
    if len(imu_df) > 0:
        imu_df.to_csv(out_dir / "imu.csv", index=False)
    else:
        (out_dir / "imu.csv").write_text(IMU_STUB, encoding="utf-8")
    if len(wifi_df) > 0:
        wifi_df.to_csv(out_dir / "wifi.csv", index=False)
    else:
        (out_dir / "wifi.csv").write_text(WIFI_STUB, encoding="utf-8")
    (out_dir / "odometry.csv").write_text(ODOM_STUB, encoding="utf-8")

    meta = {
        "dataset": "MS_ILN_2.0",
        "source_file": src_file.name,
        "site": site,
        "floor": floor,
        "survey_day": day,
        "path_id": path_id,
        "native_split": native_split,
        "duration_s": round(duration, 2),
        "n_gt_samples": len(gt_df),
        "gt_rate_hz": round(len(gt_df) / duration, 2) if duration > 0 else 0,
        "gt_sim_time_range": [
            round(float(gt_df["sim_time"].min()), 2),
            round(float(gt_df["sim_time"].max()), 2),
        ],
        "n_wifi_scans": len(wifi_df),
        "wifi_rate_hz": round(len(wifi_df) / duration, 3) if duration > 0 else 0,
        "n_imu_samples": len(imu_df),
        "imu_rate_hz": round(len(imu_df) / duration, 1) if duration > 0 else 0,
        "x_range_m": [round(float(gt_df["gt_x"].min()), 3),
                      round(float(gt_df["gt_x"].max()), 3)],
        "y_range_m": [round(float(gt_df["gt_y"].min()), 3),
                      round(float(gt_df["gt_y"].max()), 3)],
        "modalities_available": (
            ["ground_truth"]
            + (["wifi"] if len(wifi_df) > 0 else [])
            + (["imu"] if len(imu_df) > 0 else [])
        ),
        "notes": {
            "odometry": "not available (smartphone dataset)",
            "camera": "not available (smartphone dataset)",
            "imu_target_hz": IMU_TARGET_HZ,
            "gt_interp_hz": GT_INTERP_HZ,
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(6, 6))
    t = gt_df["sim_time"].values
    sc = ax.scatter(gt_df["gt_x"], gt_df["gt_y"], c=t, cmap="viridis", s=4, linewidths=0)
    ax.plot(float(gt_df["gt_x"].iloc[0]), float(gt_df["gt_y"].iloc[0]),
            "go", ms=8, label="start")
    ax.plot(float(gt_df["gt_x"].iloc[-1]), float(gt_df["gt_y"].iloc[-1]),
            "rs", ms=8, label="end")
    plt.colorbar(sc, ax=ax, label="sim_time (s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"path_{path_id:02d}  |  {site}/{floor}  |  {native_split}  |  {day}  |  "
        f"{src_file.stem[:8]}…\n"
        f"GT: {len(gt_df)} pts @ {meta['gt_rate_hz']} Hz  |  "
        f"WiFi: {meta['n_wifi_scans']} scans @ {meta['wifi_rate_hz']} Hz  |  "
        f"IMU: {meta['n_imu_samples']} rows @ {meta['imu_rate_hz']} Hz"
    )
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=100)
    plt.close(fig)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msiln-root", type=Path,
                    default=Path(r"C:\Users\FabLab\AppData\Local\Temp\msiln20"),
                    help="Path to cloned competition starter repo "
                         "(must contain io_f.py and data/<site>/<floor>).")
    ap.add_argument("--site", default="site1", help="Site folder under data/ (default: site1)")
    ap.add_argument("--floor", default="B1", help="Floor under site (default: B1)")
    ap.add_argument("--out-root", type=Path, default=Path("data"),
                    help="Where to write msiln_<site>_<floor>/.")
    ap.add_argument("--split-spec", default=None,
                    help="Override split assignment. Format: "
                         "'train=YYYY-MM-DD,YYYY-MM-DD;val=YYYY-MM-DD;test=YYYY-MM-DD'. "
                         "If omitted, earliest two days -> train, then val -> 2nd-to-last day, "
                         "test -> last day.")
    args = ap.parse_args()

    msiln_root: Path = args.msiln_root
    site = args.site
    floor = args.floor
    site_floor_dir = msiln_root / "data" / site / floor / "path_data_files"
    if not site_floor_dir.exists():
        print(f"ERROR: {site_floor_dir} not found", file=sys.stderr)
        return

    read_data_file = _load_vendored_io(msiln_root)
    files = sorted(site_floor_dir.glob("*.txt"))
    if not files:
        print(f"ERROR: no .txt traces under {site_floor_dir}", file=sys.stderr)
        return

    dataset_name = f"msiln_{site}_{floor.lower()}"
    out_dir = args.out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"MS ILN 2.0 -> {out_dir}  (site={site} floor={floor})", flush=True)
    print(f"  source: {site_floor_dir} ({len(files)} traces)", flush=True)

    # ── pass 1: parse + collect waypoints & bssids; group by day ────────────
    @dataclass
    class TraceParsed:
        path: Path
        day: dt.date
        acce: np.ndarray
        gyro: np.ndarray
        ahrs: np.ndarray
        wifi: np.ndarray
        waypoint: np.ndarray

    traces: list[TraceParsed] = []
    bssid_set: set[str] = set()
    skipped_short = 0
    skipped_no_day = 0

    for f in files:
        day = _decode_day(f.name)
        if day is None:
            skipped_no_day += 1
            continue
        try:
            d = read_data_file(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] parse fail {f.name}: {e}", file=sys.stderr, flush=True)
            continue
        wp = d.waypoint
        if len(wp) < MIN_WAYPOINTS_PER_TRACE:
            skipped_short += 1
            continue
        dur_s = (wp[-1, 0] - wp[0, 0]) / 1000.0
        if dur_s < MIN_TRACE_SECONDS:
            skipped_short += 1
            continue
        traces.append(TraceParsed(
            path=f, day=day,
            acce=d.acce, gyro=d.gyro, ahrs=d.ahrs,
            wifi=d.wifi, waypoint=wp,
        ))
        if len(d.wifi):
            for m in d.wifi[:, 2].tolist():
                bssid_set.add(m)

    print(f"  parsed {len(traces)} usable traces (skipped {skipped_short} short, "
          f"{skipped_no_day} no-day)", flush=True)
    bssid_vocab = sorted(bssid_set)
    print(f"  BSSIDs observed: {len(bssid_vocab)}", flush=True)

    # ── split assignment ────────────────────────────────────────────────────
    by_day: dict[dt.date, list[TraceParsed]] = {}
    for t in traces:
        by_day.setdefault(t.day, []).append(t)
    sorted_days = sorted(by_day)
    if args.split_spec:
        day_to_split: dict[dt.date, str] = {}
        for part in args.split_spec.split(";"):
            split_name, days_csv = part.split("=", 1)
            for d_str in days_csv.split(","):
                day_to_split[dt.date.fromisoformat(d_str.strip())] = split_name.strip()
    else:
        # default: oldest two days -> train, then val=second-to-last, test=last
        day_to_split = {}
        if len(sorted_days) >= 4:
            for d_ in sorted_days[:-2]:
                day_to_split[d_] = "train"
            day_to_split[sorted_days[-2]] = "val"
            day_to_split[sorted_days[-1]] = "test"
        elif len(sorted_days) == 3:
            day_to_split[sorted_days[0]] = "train"
            day_to_split[sorted_days[1]] = "val"
            day_to_split[sorted_days[2]] = "test"
        else:
            # fall back to within-day random (would be a leak; not what plan wants)
            print(f"  [warn] only {len(sorted_days)} distinct days — cross-session split degenerate")
            for i, d_ in enumerate(sorted_days):
                day_to_split[d_] = ("train", "val", "test")[i]

    print(f"  day mapping:")
    for d_ in sorted_days:
        print(f"    {d_} -> {day_to_split.get(d_, '(unassigned)')}  "
              f"({len(by_day[d_])} traces)", flush=True)

    # ── pass 2: deterministic ordering, emit paths ──────────────────────────
    split_order = {"train": 0, "val": 1, "test": 2}
    def _key(t: TraceParsed):
        return (split_order.get(day_to_split.get(t.day, "test"), 99),
                t.day.isoformat(), t.path.name)
    traces.sort(key=_key)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_path: list[dict] = []
    for path_id, tr in enumerate(traces):
        split = day_to_split.get(tr.day, "test")
        t_lo_ms = float(tr.waypoint[0, 0])
        t_hi_ms = float(tr.waypoint[-1, 0])
        gt_df = build_gt_df(tr.waypoint, t_lo_ms)
        if len(gt_df) < 2:
            continue
        imu_df = build_imu_df(tr.acce, tr.gyro, tr.ahrs, t_lo_ms, t_hi_ms)
        wifi_df = build_wifi_df(tr.wifi, t_lo_ms, t_hi_ms, bssid_vocab)
        out_path = out_dir / f"path_{path_id:02d}"
        meta = write_path(
            out_path, gt_df, imu_df, wifi_df, tr.path,
            site, floor, path_id, split, tr.day.isoformat(),
        )
        splits[split].append(path_id)
        per_path.append(meta)
        if path_id % 25 == 0 or path_id == len(traces) - 1:
            print(f"    path_{path_id:02d}  [{split}]  {tr.day}  "
                  f"{meta['duration_s']:.1f}s  GT={meta['n_gt_samples']}  "
                  f"WiFi={meta['n_wifi_scans']}  IMU={meta['n_imu_samples']}",
                  flush=True)

    # ── dataset-level files ─────────────────────────────────────────────────
    (out_dir / "ap_vocab.json").write_text(
        json.dumps({m: i for i, m in enumerate(bssid_vocab)}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "dataset": "MS_ILN_2.0",
        "site": site,
        "floor": floor,
        "n_paths": sum(len(v) for v in splits.values()),
        "n_aps": len(bssid_vocab),
        "splits": {k: len(v) for k, v in splits.items()},
        "day_mapping": {d_.isoformat(): day_to_split.get(d_, "?") for d_ in sorted_days},
        "total_duration_s": round(sum(m["duration_s"] for m in per_path), 2),
        "modalities": ["wifi", "imu", "ground_truth"],
        "imu_target_hz": IMU_TARGET_HZ,
        "gt_interp_hz": GT_INTERP_HZ,
        "split_mode": "cross-session-day",
        "source_starter_repo": "https://github.com/location-competition/indoor-location-competition-20",
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), out_dir / "convert_msiln.py")

    n_emitted = sum(len(v) for v in splits.values())
    print(f"\n  wrote {n_emitted} paths to {out_dir}", flush=True)
    print(f"  splits: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}  (skipped {skipped_short} short, "
          f"{skipped_no_day} no-day)", flush=True)


if __name__ == "__main__":
    main()
