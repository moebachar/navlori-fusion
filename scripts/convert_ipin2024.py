"""Convert IPIN 2024 Track 3 raw logs to navlori-fusion dataset format.

Single-floor mode.  Walks the 'GetSensorData' Android logs in
``01 Logfiles/{01 Training, 02 Testing, 03 Scoring}`` and extracts a
per-floor dataset at ``<out_root>/ipin2024_floor<N>/`` with the same layout
as ``data/async_collection/``.

Ground truth
------------
Training trials have sparse POSI lines embedded in the .txt (10-26 waypoints
per run).  Testing + Scoring trials get GT from the competition's evaluation
CSVs under ``03 Evaluation/GT/`` (scoring .txt have **no** POSI — blind test).
GT is linearly interpolated between waypoints in Lat/Lon space then projected
to local metric XY with an equirectangular tangent plane at the floor
centroid.

Floor segmentation
------------------
Trials 51-53 (and the testing/scoring trials) visit multiple floors in a
single run.  Consecutive POSI with the same floor id form a *segment*.  Each
segment becomes one ``path_NN`` in that floor's dataset.

Timestamp convention
--------------------
t0 = earliest AppTimestamp across {ACCE, GYRO, AHRS, WIFI, POSI} within the
source file.  sim_time = raw_app_ts - t0 (always >= 0).  Per-segment, we
further re-zero t0 to the first GT time in the segment so segments match the
async_collection convention of ~0-based sim_time.

Outputs
-------
``<out_root>/ipin2024_floor<N>/``
    ├── path_NN/
    │     ├── imu.csv          — ACCE+GYRO+AHRS merged @ ~32 Hz
    │     ├── wifi.csv         — per-scan RSSI fingerprints
    │     ├── odometry.csv     — header-only stub (not available)
    │     ├── ground_truth.csv — POSI interpolated @ ~10 Hz
    │     ├── metadata.json
    │     └── trajectory.png
    ├── ap_vocab.json
    ├── split.json             — native train/val/test attribution
    ├── metadata.json
    ├── convert_ipin2024.py
    └── configs_snippet.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── tunables ────────────────────────────────────────────────────────────────
IMU_TARGET_HZ = 32        # resample AHRS-driven IMU to ~32 Hz
IMU_NN_TOL_S = 0.05       # nearest-neighbor tolerance when merging ACCE/GYRO onto AHRS
WIFI_SCAN_GAP_S = 2.5     # WIFI lines within this gap are the same "scan"
GT_INTERP_HZ = 10         # resample interpolated POSI to 10 Hz
MIN_POSI_PER_SEGMENT = 3  # drop segments with fewer than this many waypoints
MIN_SEGMENT_SECONDS = 5.0 # drop very short segments
EARTH_R = 6_371_000.0     # meters

# IPIN paths
TRAIN_DIR = "01 Logfiles/01 IPIN2024_T3_TrainingTrials (training)"
VAL_DIR   = "01 Logfiles/02 IPIN2024_T3_TestingTrials (validation)"
TEST_DIR  = "01 Logfiles/03 IPIN2024_T3_ScoringTrials (final test)"
GT_DIR    = "03 Evaluation/GT"

# Map integer floor id to readable label
FLOOR_LABEL = {0: "0", -1: "-1", -2: "-2"}

# ── parsing helpers ─────────────────────────────────────────────────────────

@dataclass
class TrialData:
    """Everything parsed out of one .txt file."""
    source: Path
    split: str           # "train" | "val" | "test"
    acce: np.ndarray     # (N, 4) [app_ts, ax, ay, az]
    gyro: np.ndarray     # (N, 4) [app_ts, gx, gy, gz]
    ahrs: np.ndarray     # (N, 4) [app_ts, pitch, roll, yaw]
    wifi: list[tuple[float, str, float]]   # [(app_ts, mac_lower, rssi), ...]
    posi: np.ndarray     # (N, 4) [ts, lat, lon, floor_int]


def _parse_floor(raw) -> int | None:
    """POSI / GT-CSV floor values are inconsistent across sources.

    Seen so far: ``'0'``, ``'00'``, ``'-1'``, ``'-2'``, numpy floats like
    ``0.0`` coming out of pandas when the column is typed as float.
    """
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def parse_trial(txt_path: Path, split: str) -> TrialData:
    """Stream-parse one .txt logfile into arrays."""
    acce, gyro, ahrs = [], [], []
    wifi: list[tuple[float, str, float]] = []
    posi = []

    with txt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("%") or line[0].isspace():
                continue
            parts = line.rstrip("\r\n").split(";")
            tag = parts[0]
            try:
                if tag == "ACCE" and len(parts) >= 6:
                    acce.append((float(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))
                elif tag == "GYRO" and len(parts) >= 6:
                    gyro.append((float(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))
                elif tag == "AHRS" and len(parts) >= 6:
                    # AHRS;appTs;sensorTs;PitchX;RollY;YawZ;Quat2;Quat3;Quat4;Accuracy
                    ahrs.append((float(parts[1]), float(parts[3]), float(parts[4]), float(parts[5])))
                elif tag == "WIFI" and len(parts) >= 7:
                    # WIFI;appTs;sensorTs;SSID;MAC;freq;RSS
                    mac = parts[4].strip().lower().replace(":", "")
                    if mac:
                        wifi.append((float(parts[1]), mac, float(parts[6])))
                elif tag == "POSI" and len(parts) >= 7:
                    # POSI;ts;counter;lat;lon;floor;building
                    fl = _parse_floor(parts[5])
                    if fl is not None:
                        posi.append((float(parts[1]), float(parts[3]), float(parts[4]), fl))
            except (ValueError, IndexError):
                continue

    return TrialData(
        source=txt_path,
        split=split,
        acce=np.array(acce, dtype=np.float64) if acce else np.empty((0, 4)),
        gyro=np.array(gyro, dtype=np.float64) if gyro else np.empty((0, 4)),
        ahrs=np.array(ahrs, dtype=np.float64) if ahrs else np.empty((0, 4)),
        wifi=wifi,
        posi=np.array(posi, dtype=np.float64) if posi else np.empty((0, 4)),
    )


def load_eval_gt(trial_name: str, gt_dir: Path) -> np.ndarray | None:
    """Load GT from ``03 Evaluation/GT/GT_<trial_name>.csv`` (time, lon, lat, floor, wp_id).

    Returns array [ts, lat, lon, floor_int] or None if missing.
    """
    gt_path = gt_dir / f"GT_{trial_name}.csv"
    if not gt_path.exists():
        return None
    df = pd.read_csv(gt_path, header=None, names=["ts", "lon", "lat", "floor", "wp"])
    out = []
    for _, row in df.iterrows():
        fl = _parse_floor(str(row["floor"]))
        if fl is None:
            continue
        out.append((float(row["ts"]), float(row["lat"]), float(row["lon"]), fl))
    return np.array(out, dtype=np.float64) if out else None


# ── segmentation ────────────────────────────────────────────────────────────

def floor_segments(posi: np.ndarray, target_floor: int) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx_exclusive), ...] of runs on target_floor."""
    if len(posi) == 0:
        return []
    floors = posi[:, 3].astype(int)
    segments = []
    i = 0
    n = len(posi)
    while i < n:
        if floors[i] != target_floor:
            i += 1
            continue
        j = i
        while j < n and floors[j] == target_floor:
            j += 1
        segments.append((i, j))
        i = j
    return segments


# ── geodetic projection ─────────────────────────────────────────────────────

def build_projector(ref_lat: float, ref_lon: float):
    """Return (lat, lon) -> (x_m, y_m) equirectangular at ref_lat, ref_lon."""
    lat0_rad = math.radians(ref_lat)
    m_per_deg_lat = math.pi / 180 * EARTH_R
    m_per_deg_lon = m_per_deg_lat * math.cos(lat0_rad)

    def project(lat: np.ndarray, lon: np.ndarray):
        x = (lon - ref_lon) * m_per_deg_lon
        y = (lat - ref_lat) * m_per_deg_lat
        return x, y

    return project


# ── sensor merging ──────────────────────────────────────────────────────────

def _nearest_merge(base_ts: np.ndarray, other: np.ndarray, tol: float) -> np.ndarray:
    """Merge `other[:, 1:]` onto `base_ts` by nearest-ts within tol.

    other: (M, K+1), first column is timestamp. Returns (N, K) with NaN where
    no match within tol.
    """
    if len(other) == 0:
        return np.full((len(base_ts), other.shape[1] - 1), np.nan)
    o_ts = other[:, 0]
    idx = np.searchsorted(o_ts, base_ts)
    out = np.full((len(base_ts), other.shape[1] - 1), np.nan)
    for i, t in enumerate(base_ts):
        candidates = []
        if idx[i] > 0:
            candidates.append(idx[i] - 1)
        if idx[i] < len(o_ts):
            candidates.append(idx[i])
        best = None
        best_dt = tol
        for c in candidates:
            dt = abs(o_ts[c] - t)
            if dt <= best_dt:
                best_dt = dt
                best = c
        if best is not None:
            out[i] = other[best, 1:]
    return out


def build_imu_df(trial: TrialData, t_lo: float, t_hi: float) -> pd.DataFrame:
    """Build imu.csv rows using AHRS as the time grid, resampled to IMU_TARGET_HZ."""
    ahrs = trial.ahrs[(trial.ahrs[:, 0] >= t_lo) & (trial.ahrs[:, 0] <= t_hi)]
    if len(ahrs) == 0:
        return pd.DataFrame(columns=["sim_time", "accel_x", "accel_y", "accel_z",
                                     "gyro_x", "gyro_y", "gyro_z",
                                     "roll_deg", "pitch_deg", "yaw_deg"])
    # Decimate AHRS to target rate
    native_hz = max(len(ahrs) / max(ahrs[-1, 0] - ahrs[0, 0], 1e-6), 1.0)
    step = max(int(round(native_hz / IMU_TARGET_HZ)), 1)
    ahrs_ds = ahrs[::step]

    acc = _nearest_merge(ahrs_ds[:, 0], trial.acce, IMU_NN_TOL_S)   # (N, 3)
    gyr = _nearest_merge(ahrs_ds[:, 0], trial.gyro, IMU_NN_TOL_S)   # (N, 3)

    sim_time = ahrs_ds[:, 0] - t_lo
    # IPIN AHRS layout is PitchX, RollY, YawZ → map to our (roll, pitch, yaw)
    pitch = ahrs_ds[:, 1]
    roll  = ahrs_ds[:, 2]
    yaw   = ahrs_ds[:, 3]

    df = pd.DataFrame({
        "sim_time": np.round(sim_time, 4),
        "accel_x": acc[:, 0], "accel_y": acc[:, 1], "accel_z": acc[:, 2],
        "gyro_x":  gyr[:, 0], "gyro_y":  gyr[:, 1], "gyro_z":  gyr[:, 2],
        "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw,
    })
    # Drop rows where BOTH acc and gyro are missing (kept if any signal present)
    keep = ~(df[["accel_x", "gyro_x"]].isna().all(axis=1))
    return df.loc[keep].reset_index(drop=True)


def build_wifi_df(trial: TrialData, t_lo: float, t_hi: float,
                  bssid_vocab: list[str]) -> pd.DataFrame:
    """Group WIFI lines into scans (per-BSSID max RSSI) and emit one row per scan."""
    if not trial.wifi:
        return pd.DataFrame()
    rows = [(t, m, r) for (t, m, r) in trial.wifi if t_lo <= t <= t_hi]
    rows.sort(key=lambda x: x[0])
    if not rows:
        return pd.DataFrame()

    scans: list[dict] = []
    cur_mac: dict[str, float] = {}
    cur_start = rows[0][0]
    last_ts = cur_start

    for ts, mac, rssi in rows:
        if ts - last_ts > WIFI_SCAN_GAP_S and cur_mac:
            scans.append({"t_mid": 0.5 * (cur_start + last_ts), "rssi": dict(cur_mac)})
            cur_mac = {}
            cur_start = ts
        cur_mac[mac] = max(cur_mac.get(mac, -200.0), rssi)
        last_ts = ts
    if cur_mac:
        scans.append({"t_mid": 0.5 * (cur_start + last_ts), "rssi": dict(cur_mac)})

    rssi_cols = [f"wifi_rssi_{m}" for m in bssid_vocab]
    mac_to_col = {m: f"wifi_rssi_{m}" for m in bssid_vocab}
    data = []
    for s in scans:
        readings: dict[str, float] = s["rssi"]
        row = {c: np.nan for c in rssi_cols}
        for mac, r in readings.items():
            if mac in mac_to_col:
                row[mac_to_col[mac]] = r
        visible = [(m, r) for m, r in readings.items() if m in mac_to_col]
        if visible:
            best_mac, best_rssi = max(visible, key=lambda x: x[1])
        else:
            best_mac, best_rssi = "", np.nan
        row_out = {
            "sim_time": round(s["t_mid"] - t_lo, 3),
            "wifi_visible_count": len(visible),
            "wifi_strongest_rssi": best_rssi,
            "wifi_strongest_mac": best_mac,
        }
        row_out.update(row)
        data.append(row_out)
    cols = ["sim_time", "wifi_visible_count", "wifi_strongest_rssi",
            "wifi_strongest_mac"] + rssi_cols
    return pd.DataFrame(data, columns=cols)


def build_gt_df(posi_seg: np.ndarray, t_lo: float, project) -> pd.DataFrame:
    """Interpolate POSI waypoints to GT_INTERP_HZ and project to metric XY."""
    ts = posi_seg[:, 0]
    lat = posi_seg[:, 1]
    lon = posi_seg[:, 2]
    if len(ts) < 2:
        return pd.DataFrame(columns=["sim_time", "gt_x", "gt_y"])

    t_start, t_end = ts[0], ts[-1]
    n = max(int((t_end - t_start) * GT_INTERP_HZ) + 1, 2)
    dense_ts = np.linspace(t_start, t_end, n)
    dense_lat = np.interp(dense_ts, ts, lat)
    dense_lon = np.interp(dense_ts, ts, lon)
    x, y = project(dense_lat, dense_lon)

    return pd.DataFrame({
        "sim_time": np.round(dense_ts - t_lo, 3),
        "gt_x": np.round(x, 4),
        "gt_y": np.round(y, 4),
    })


# ── trial -> segments pipeline ──────────────────────────────────────────────

def _trial_name(txt: Path) -> str:
    return txt.stem


_REP_SUFFIX = re.compile(r"_repetition\d+$")


def _trial_group_key(txt: Path) -> str:
    """Strip the ``_repetitionNN`` suffix so all reps of one trial share a key.

    Testing / Scoring trials have no repetition suffix and map to themselves.
    """
    return _REP_SUFFIX.sub("", txt.stem)


def chunk_segment(
    gt_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    wifi_df: pd.DataFrame,
    chunk_seconds: float,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Split a (gt, imu, wifi) triple into ~chunk_seconds sub-segments.

    Each sub-segment has its sim_time re-zeroed to start at 0.  Sub-segments
    with fewer than 2 GT rows are dropped.
    """
    if len(gt_df) < 2:
        return []
    t_min = float(gt_df["sim_time"].min())
    t_max = float(gt_df["sim_time"].max())
    duration = t_max - t_min
    if duration <= 0:
        return []
    n_chunks = max(1, int(round(duration / chunk_seconds)))
    boundaries = np.linspace(t_min, t_max, n_chunks + 1)

    out: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_chunks):
        lo = float(boundaries[i])
        hi = float(boundaries[i + 1])
        # Inclusive boundary on the last chunk so we don't drop the final GT row
        if i == n_chunks - 1:
            gt_mask = (gt_df["sim_time"] >= lo) & (gt_df["sim_time"] <= hi)
        else:
            gt_mask = (gt_df["sim_time"] >= lo) & (gt_df["sim_time"] < hi)

        gt_chunk = gt_df.loc[gt_mask].reset_index(drop=True).copy()
        if len(gt_chunk) < 2:
            continue

        if len(imu_df):
            imu_mask = (imu_df["sim_time"] >= lo) & (imu_df["sim_time"] <= hi)
            imu_chunk = imu_df.loc[imu_mask].reset_index(drop=True).copy()
        else:
            imu_chunk = imu_df.copy()
        if len(wifi_df):
            wifi_mask = (wifi_df["sim_time"] >= lo) & (wifi_df["sim_time"] <= hi)
            wifi_chunk = wifi_df.loc[wifi_mask].reset_index(drop=True).copy()
        else:
            wifi_chunk = wifi_df.copy()

        # Re-zero sim_time so the chunk looks like a normal path_NN
        chunk_t0 = float(gt_chunk["sim_time"].iloc[0])
        gt_chunk["sim_time"] = (gt_chunk["sim_time"] - chunk_t0).round(3)
        if len(imu_chunk):
            imu_chunk["sim_time"] = (imu_chunk["sim_time"] - chunk_t0).round(4)
            # Drop any rows that landed before the first GT (rare edge case)
            imu_chunk = imu_chunk.loc[imu_chunk["sim_time"] >= 0].reset_index(drop=True)
        if len(wifi_chunk):
            wifi_chunk["sim_time"] = (wifi_chunk["sim_time"] - chunk_t0).round(3)
            wifi_chunk = wifi_chunk.loc[wifi_chunk["sim_time"] >= 0].reset_index(drop=True)

        out.append((gt_chunk, imu_chunk, wifi_chunk))
    return out


def discover_trials(raw_root: Path) -> list[tuple[Path, str]]:
    """Return [(txt_path, split), ...] across the three IPIN split dirs."""
    out: list[tuple[Path, str]] = []
    for split, sub in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        d = raw_root / sub
        if not d.exists():
            print(f"  [warn] missing: {d}")
            continue
        for p in sorted(d.glob("*.txt")):
            out.append((p, split))
    return out


def gather_posi_for_trial(trial: TrialData, raw_root: Path) -> np.ndarray:
    """POSI from .txt if non-empty, else fall back to evaluation GT CSV."""
    if len(trial.posi) > 0:
        return trial.posi
    gt = load_eval_gt(_trial_name(trial.source), raw_root / GT_DIR)
    return gt if gt is not None else np.empty((0, 4))


def build_bssid_vocab(trials: Iterable[TrialData], target_floor: int,
                      raw_root: Path) -> list[str]:
    """Collect every BSSID observed during *any* target-floor segment."""
    macs: set[str] = set()
    for trial in trials:
        posi = gather_posi_for_trial(trial, raw_root)
        segs = floor_segments(posi, target_floor)
        if not segs:
            continue
        if not trial.wifi:
            continue
        # A wifi reading is in a segment if its ts falls in any [t_start, t_end]
        span_ranges = [(posi[a, 0], posi[b - 1, 0]) for a, b in segs]
        for t, mac, _ in trial.wifi:
            if any(lo <= t <= hi for lo, hi in span_ranges):
                macs.add(mac)
    return sorted(macs)


# ── writers (mirror convert_imuwifine style) ────────────────────────────────

IMU_STUB = ("sim_time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,"
            "roll_deg,pitch_deg,yaw_deg\n")
WIFI_STUB = "sim_time,wifi_visible_count,wifi_strongest_rssi,wifi_strongest_mac\n"
ODOM_STUB = ("sim_time,odom_x,odom_y,odom_theta_deg,"
             "odom_linear_vel,odom_angular_vel,wheel_left_vel,wheel_right_vel\n")


def write_path(out_dir: Path, gt_df: pd.DataFrame,
               imu_df: pd.DataFrame, wifi_df: pd.DataFrame,
               src_file: Path, floor: int, path_id: int,
               native_split: str, seg_index: int) -> dict:
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
        "dataset": "IPIN2024_Track3",
        "source_file": src_file.name,
        "segment_index": seg_index,
        "floor": floor,
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
            "wifi_scan_gap_s": WIFI_SCAN_GAP_S,
            "gt_interp_hz": GT_INTERP_HZ,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
        f"path_{path_id:02d}  |  floor {floor}  |  {native_split}  |  "
        f"{src_file.stem} [seg {seg_index}]\n"
        f"GT: {len(gt_df)} pts @ {meta['gt_rate_hz']} Hz  |  "
        f"WiFi: {meta['n_wifi_scans']} scans @ {meta['wifi_rate_hz']} Hz  |  "
        f"IMU: {meta['n_imu_samples']} rows @ {meta['imu_rate_hz']} Hz"
    )
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=100)
    plt.close(fig)
    return meta


def write_dataset_summary(out_dir: Path, floor: int,
                          splits: dict[str, list[int]],
                          n_aps: int, per_path: list[dict],
                          ref_lat: float, ref_lon: float,
                          dataset_name: str,
                          split_mode: str = "trial-out",
                          chunk_seconds: float | None = None,
                          split_seed: int | None = None) -> None:
    total = sum(len(v) for v in splits.values())
    duration_sum = sum(m["duration_s"] for m in per_path)
    summary = {
        "dataset": "IPIN2024_Track3",
        "floor": floor,
        "n_paths": total,
        "n_aps": n_aps,
        "splits": {k: len(v) for k, v in splits.items()},
        "total_duration_s": round(duration_sum, 2),
        "modalities": ["wifi", "imu", "ground_truth"],
        "reference_lat": ref_lat,
        "reference_lon": ref_lon,
        "imu_target_hz": IMU_TARGET_HZ,
        "wifi_scan_gap_s": WIFI_SCAN_GAP_S,
        "gt_interp_hz": GT_INTERP_HZ,
        "split_mode": split_mode,
    }
    if split_mode == "within-trial":
        summary["chunk_seconds"] = chunk_seconds
        summary["split_seed"] = split_seed
    (out_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8")

    # Hydra config snippet
    floor_lbl = FLOOR_LABEL[floor]
    train = splits.get("train", [])
    val   = splits.get("val", [])
    test  = splits.get("test", [])
    if split_mode == "within-trial":
        header = (
            f"# IPIN 2024 Track 3 - floor {floor_lbl} (within-trial split).\n"
            f"# {total} chunks total (~{chunk_seconds:.0f}s each, seed={split_seed}, "
            f"split: train={len(train)} / val={len(val)} / test={len(test)}).\n"
            f"# Train and val are drawn from the SAME trials so sensor cadence and\n"
            f"# device characteristics match across splits - good for evaluating\n"
            f"# whether per-modality encoders can regress (x, y) from short windows.\n"
        )
    else:
        header = (
            f"# IPIN 2024 Track 3 - floor {floor_lbl}.\n"
            f"# {total} paths total (native split: train={len(train)} / val={len(val)} / test={len(test)}).\n"
        )
    snippet = f"""{header}# Modalities: wifi + imu (no odom, no camera - smartphone dataset).
# GT is POSI waypoints linearly interpolated at {GT_INTERP_HZ} Hz then projected
# to local XY via equirectangular tangent plane at the floor centroid.
data:
  name: {dataset_name}
  root: ${{oc.env:NAVLORI_DATA_ROOT,data}}
  collection_dir: {dataset_name}
  source: ipin2024
  modalities: [wifi, imu]

  split:
    train_paths: {train}
    val_paths:   {val}
    test_paths:  {test}

  preprocessing:
    normalize: true
    wifi_pca: 128   # reduce {n_aps} APs -> 128 dims

  windows:
    imu: 32    # ~1 s at {IMU_TARGET_HZ} Hz
    wifi: 1    # single scan
"""
    (out_dir / "configs_snippet.yaml").write_text(snippet, encoding="utf-8")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floor", type=int, required=True, choices=[0, -1, -2],
                    help="Floor id to extract.")
    ap.add_argument("--raw-root", type=Path,
                    default=Path("data/2024_IPIN_Competition_Track03"),
                    help="Path to extracted IPIN dataset root.")
    ap.add_argument("--out-root", type=Path, default=Path("data"),
                    help="Where to write ipin2024_floor<N>/.")
    ap.add_argument(
        "--keep-all-repetitions", action="store_true",
        help=(
            "Keep every ``_repetitionNN`` file as its own path.  Default is to "
            "dedupe: within each (trial, segment) group, keep only the rep "
            "with the most WiFi scans (tiebreaker: longest duration).  "
            "Testing/Scoring trials have no reps and are unaffected."
        ),
    )
    ap.add_argument(
        "--split-mode", choices=["trial-out", "within-trial"], default="trial-out",
        help=(
            "Split granularity.  'trial-out' (default) keeps each (trial, "
            "segment) as one path and uses the IPIN native train/val/test "
            "split.  'within-trial' chunks each segment into ~CHUNK_SECONDS "
            "sub-segments and stratifies them across train/val/test with a "
            "seeded shuffle, so train and val come from the same trials and "
            "share sensor characteristics.  Output goes to "
            "<out_root>/ipin2024_floor<N>_intra/ to keep the original dataset "
            "untouched."
        ),
    )
    ap.add_argument(
        "--chunk-seconds", type=float, default=15.0,
        help="Sub-segment duration in seconds for --split-mode within-trial.",
    )
    ap.add_argument(
        "--split-ratios", type=str, default="0.7,0.15,0.15",
        help=(
            "Comma-separated train,val,test fractions for --split-mode "
            "within-trial.  Must sum to 1."
        ),
    )
    ap.add_argument(
        "--split-seed", type=int, default=42,
        help="RNG seed for the within-trial shuffle.",
    )
    args = ap.parse_args()

    ratios = [float(x) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error(
            f"--split-ratios must be three comma-separated floats summing to 1, "
            f"got {args.split_ratios!r}."
        )

    raw_root: Path = args.raw_root
    out_root: Path = args.out_root
    floor = args.floor
    floor_lbl = FLOOR_LABEL[floor]
    out_suffix = "_intra" if args.split_mode == "within-trial" else ""
    dataset_name = f"ipin2024_floor{floor_lbl}{out_suffix}"
    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"IPIN 2024 Track 3 -> {out_dir}  (floor={floor}, split-mode={args.split_mode})")
    print(f"  raw root: {raw_root}")

    # 1. Discover and parse all trials
    trials_meta = discover_trials(raw_root)
    print(f"  found {len(trials_meta)} .txt logfiles")
    trials: list[TrialData] = []
    for txt, split in trials_meta:
        t = parse_trial(txt, split)
        trials.append(t)

    # 2. Build BSSID vocabulary restricted to target-floor segments
    bssid_vocab = build_bssid_vocab(trials, floor, raw_root)
    print(f"  floor-{floor_lbl} BSSIDs observed: {len(bssid_vocab)}")
    if not bssid_vocab:
        print("  [abort] no BSSIDs on this floor; nothing to write.")
        return

    # 3. Build reference lat/lon = centroid of all floor-target POSI
    all_lat, all_lon = [], []
    for trial in trials:
        posi = gather_posi_for_trial(trial, raw_root)
        if len(posi) == 0:
            continue
        mask = posi[:, 3].astype(int) == floor
        all_lat.extend(posi[mask, 1].tolist())
        all_lon.extend(posi[mask, 2].tolist())
    if not all_lat:
        print("  [abort] no POSI on this floor.")
        return
    ref_lat = float(np.median(all_lat))
    ref_lon = float(np.median(all_lon))
    project = build_projector(ref_lat, ref_lon)
    print(f"  reference: lat={ref_lat:.8f} lon={ref_lon:.8f}")

    # 4. Collect candidate segments (pre-dedupe)
    @dataclass
    class Candidate:
        trial: TrialData
        seg_index: int
        gt_df: pd.DataFrame
        imu_df: pd.DataFrame
        wifi_df: pd.DataFrame

        @property
        def score(self) -> tuple:
            # Lexicographic: more WiFi scans > more IMU rows > longer trajectory
            dur = float(self.gt_df["sim_time"].max() - self.gt_df["sim_time"].min())
            return (len(self.wifi_df), len(self.imu_df), dur)

    candidates: list[Candidate] = []
    dropped = 0

    for trial in trials:
        posi = gather_posi_for_trial(trial, raw_root)
        segments = floor_segments(posi, floor)
        for seg_idx, (a, b) in enumerate(segments):
            posi_seg = posi[a:b]
            if len(posi_seg) < MIN_POSI_PER_SEGMENT:
                dropped += 1
                continue
            t_lo = float(posi_seg[0, 0])
            t_hi = float(posi_seg[-1, 0])
            if (t_hi - t_lo) < MIN_SEGMENT_SECONDS:
                dropped += 1
                continue
            gt_df = build_gt_df(posi_seg, t_lo, project)
            if len(gt_df) < 2:
                dropped += 1
                continue
            imu_df = build_imu_df(trial, t_lo, t_hi)
            wifi_df = build_wifi_df(trial, t_lo, t_hi, bssid_vocab)
            candidates.append(Candidate(trial, seg_idx, gt_df, imu_df, wifi_df))

    # 4b. Dedupe repetitions: keep the richest rep per (trial-group, segment)
    deduped = 0
    if not args.keep_all_repetitions:
        best_per_group: dict[tuple[str, int], Candidate] = {}
        for cand in candidates:
            key = (_trial_group_key(cand.trial.source), cand.seg_index)
            cur = best_per_group.get(key)
            if cur is None or cand.score > cur.score:
                best_per_group[key] = cand
        deduped = len(candidates) - len(best_per_group)
        candidates = list(best_per_group.values())

    # Deterministic ordering: split (train, val, test), then source name, then seg
    split_order = {"train": 0, "val": 1, "test": 2}
    candidates.sort(key=lambda c: (split_order[c.trial.split],
                                   c.trial.source.stem, c.seg_index))

    if not candidates:
        print("  [abort] no usable segments.")
        return

    # 5. Emit paths
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_path: list[dict] = []
    n_chunks_total = 0

    if args.split_mode == "trial-out":
        for path_id, cand in enumerate(candidates):
            out_path = out_dir / f"path_{path_id:02d}"
            meta = write_path(
                out_path, cand.gt_df, cand.imu_df, cand.wifi_df, cand.trial.source,
                floor, path_id, cand.trial.split, cand.seg_index,
            )
            splits[cand.trial.split].append(path_id)
            per_path.append(meta)
            print(f"    path_{path_id:02d}  [{cand.trial.split}]  "
                  f"{cand.trial.source.stem} seg{cand.seg_index}  "
                  f"{meta['duration_s']:.1f}s  "
                  f"GT={meta['n_gt_samples']}  WiFi={meta['n_wifi_scans']}  "
                  f"IMU={meta['n_imu_samples']}")
    else:
        # within-trial: chunk every candidate, then seeded-shuffle and partition
        @dataclass
        class Chunk:
            source: Path
            seg_index: int
            chunk_index: int
            gt_df: pd.DataFrame
            imu_df: pd.DataFrame
            wifi_df: pd.DataFrame

        chunks: list[Chunk] = []
        for cand in candidates:
            sub = chunk_segment(cand.gt_df, cand.imu_df, cand.wifi_df,
                                args.chunk_seconds)
            for ci, (g, i, w) in enumerate(sub):
                chunks.append(Chunk(cand.trial.source, cand.seg_index, ci, g, i, w))

        n_chunks_total = len(chunks)
        if not chunks:
            print("  [abort] no chunks produced (segments shorter than chunk size).")
            return

        # Deterministic shuffle, then ratio partition
        rng = random.Random(args.split_seed)
        order = list(range(len(chunks)))
        rng.shuffle(order)
        n = len(order)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        # Guarantee at least one sample per split when feasible
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        train_idx = set(order[:n_train])
        val_idx = set(order[n_train:n_train + n_val])
        # remainder = test

        for path_id, ch in enumerate(chunks):
            if path_id in train_idx:
                assigned = "train"
            elif path_id in val_idx:
                assigned = "val"
            else:
                assigned = "test"
            out_path = out_dir / f"path_{path_id:02d}"
            meta = write_path(
                out_path, ch.gt_df, ch.imu_df, ch.wifi_df, ch.source,
                floor, path_id, assigned, ch.seg_index,
            )
            meta["chunk_index"] = ch.chunk_index
            splits[assigned].append(path_id)
            per_path.append(meta)
            print(f"    path_{path_id:02d}  [{assigned}]  "
                  f"{ch.source.stem} seg{ch.seg_index} chunk{ch.chunk_index}  "
                  f"{meta['duration_s']:.1f}s  "
                  f"GT={meta['n_gt_samples']}  WiFi={meta['n_wifi_scans']}  "
                  f"IMU={meta['n_imu_samples']}")

    # 5. Dataset-level files
    (out_dir / "ap_vocab.json").write_text(
        json.dumps({m: i for i, m in enumerate(bssid_vocab)}, indent=2),
        encoding="utf-8",
    )
    write_dataset_summary(
        out_dir, floor, splits, len(bssid_vocab),
        per_path, ref_lat, ref_lon,
        dataset_name=dataset_name,
        split_mode=args.split_mode,
        chunk_seconds=args.chunk_seconds if args.split_mode == "within-trial" else None,
        split_seed=args.split_seed if args.split_mode == "within-trial" else None,
    )
    # Archive a copy of the script for reproducibility
    shutil.copy2(Path(__file__), out_dir / "convert_ipin2024.py")

    n_emitted = sum(len(v) for v in splits.values())
    print()
    print(f"  wrote {n_emitted} paths to {out_dir}")
    print(f"  splits: train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}  "
          f"(dropped {dropped} short/empty segments, "
          f"deduped {deduped} repetitions"
          + (f", chunked into {n_chunks_total} pieces" if args.split_mode == "within-trial" else "")
          + ")")
    print(f"  config snippet: {out_dir / 'configs_snippet.yaml'}")


if __name__ == "__main__":
    main()
