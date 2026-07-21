"""MSILN site1/B1 loader for fusion-baseline integration (PLAN_40).

Loads per-trajectory (timestamp, accel, gyro, ahrs, wifi-snapshot, gt-waypoints)
from the async_collection format (``data/msiln_site1_b1/path_NN/``) and
produces TWO output shapes:

* ``load_msiln_paths_for_imuwifine(path_ids, ap_vocab)`` →
  list of per-path numpy structs with WiFi+IMU resampled onto a common 10 Hz
  grid, in the format the IMUWiFine baseline (LSTM-fusion) consumes.
* ``load_msiln_paths_for_competition(path_ids)`` →
  list of per-path dicts with raw async timestamps, in the format the
  MSILN-competition PDR + WiFi-snapshot baseline consumes (matches their
  ``ReadData`` namedtuple semantics).

The official cross-session splits live in ``configs/data/msiln_site1_b1.yaml``:
train = paths 0-93, val = 94-127, test = 128-132.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MSILN_ROOT = PROJECT_ROOT / "data" / "msiln_site1_b1"

# MSILN cross-session splits (mirrors configs/data/msiln_site1_b1.yaml)
TRAIN_PATHS = list(range(0, 94))
VAL_PATHS = list(range(94, 128))
TEST_PATHS = list(range(128, 133))


def load_ap_vocab() -> dict[str, int]:
    """Return the per-BSSID -> index map (1419 APs for MSILN site1/B1)."""
    return json.loads((MSILN_ROOT / "ap_vocab.json").read_text())


# ---------------------------------------------------------------------------
# IMUWiFine-style consumer: per-path, common-rate (10 Hz default) tensor
# ---------------------------------------------------------------------------

def _interp(t_query: np.ndarray, t_known: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """1D linear interpolation per column. vals shape (N, K)."""
    out = np.empty((len(t_query), vals.shape[1]), dtype=np.float32)
    for k in range(vals.shape[1]):
        out[:, k] = np.interp(t_query, t_known, vals[:, k])
    return out


def _wifi_snapshot_at(t_query: np.ndarray, wifi_df: pd.DataFrame,
                      rssi_cols: list[str], no_signal: float = -100.0) -> np.ndarray:
    """For each query timestamp, take the most-recent WiFi scan's RSSI vector.

    Missing APs (NaN) are filled with ``no_signal``. RSSI values are kept
    in dBm space (the IMUWiFine normalize_wifi step shifts + scales).
    """
    n_aps = len(rssi_cols)
    out = np.full((len(t_query), n_aps), no_signal, dtype=np.float32)
    if wifi_df.empty:
        return out
    wifi_t = wifi_df["sim_time"].values.astype(np.float64)
    wifi_rssi = wifi_df[rssi_cols].values.astype(np.float32)
    wifi_rssi = np.where(np.isnan(wifi_rssi), no_signal, wifi_rssi)
    # For each t_query, find latest scan with sim_time <= t
    idx = np.searchsorted(wifi_t, t_query, side="right") - 1
    valid = idx >= 0
    out[valid] = wifi_rssi[idx[valid]]
    return out


def load_msiln_paths_for_imuwifine(
    path_ids: Iterable[int],
    ap_vocab: dict[str, int] | None = None,
    target_hz: float = 10.0,
    no_signal: float = -100.0,
) -> tuple[list[dict], list[str]]:
    """Load + resample MSILN paths onto a common 10 Hz grid.

    Returns ``(paths, rssi_cols)``:
    - ``paths`` is a list of dicts, one per loaded path, each with keys:
      ``path_id``, ``t`` (T,), ``wifi`` (T, n_aps), ``imu`` (T, 6),
      ``ahrs`` (T, 3), ``gt`` (T, 2). All ndarrays.
    - ``rssi_cols`` is the AP column order (length == n_aps).
    """
    if ap_vocab is None:
        ap_vocab = load_ap_vocab()
    rssi_cols = [f"wifi_rssi_{mac}" for mac in ap_vocab.keys()]

    out = []
    for pid in path_ids:
        pdir = MSILN_ROOT / f"path_{pid:02d}"
        if not pdir.is_dir():
            continue
        gt = pd.read_csv(pdir / "ground_truth.csv")
        imu = pd.read_csv(pdir / "imu.csv")
        wifi = pd.read_csv(pdir / "wifi.csv")

        # Resample onto a common time grid at target_hz
        t_start = max(float(gt["sim_time"].min()), float(imu["sim_time"].min()))
        t_end = min(float(gt["sim_time"].max()), float(imu["sim_time"].max()))
        if t_end - t_start < 1.0:
            continue
        step = 1.0 / target_hz
        t_grid = np.arange(t_start, t_end, step, dtype=np.float64)
        if len(t_grid) < 2:
            continue

        imu_t = imu["sim_time"].values.astype(np.float64)
        imu_vals = imu[["accel_x", "accel_y", "accel_z",
                          "gyro_x", "gyro_y", "gyro_z"]].values.astype(np.float32)
        imu_resamp = _interp(t_grid, imu_t, imu_vals)

        ahrs_vals = imu[["roll_deg", "pitch_deg", "yaw_deg"]].values.astype(np.float32)
        ahrs_resamp = _interp(t_grid, imu_t, ahrs_vals)

        gt_t = gt["sim_time"].values.astype(np.float64)
        gt_vals = gt[["gt_x", "gt_y"]].values.astype(np.float32)
        gt_resamp = _interp(t_grid, gt_t, gt_vals)

        # WiFi rssi_cols may not all be present per-path → reindex with NaN
        wifi_present = [c for c in rssi_cols if c in wifi.columns]
        for c in rssi_cols:
            if c not in wifi.columns:
                wifi[c] = np.nan
        wifi_resamp = _wifi_snapshot_at(t_grid, wifi, rssi_cols, no_signal=no_signal)

        out.append({
            "path_id": int(pid),
            "t":    t_grid.astype(np.float32),
            "wifi": wifi_resamp,
            "imu":  imu_resamp,
            "ahrs": ahrs_resamp,
            "gt":   gt_resamp,
        })
    return out, rssi_cols


# ---------------------------------------------------------------------------
# Competition PDR consumer: raw async timestamps, no resampling
# ---------------------------------------------------------------------------

def load_msiln_paths_for_competition(path_ids: Iterable[int]) -> list[dict]:
    """Load MSILN paths into the format the competition PDR functions expect.

    The competition's ``compute_step_positions(acce, ahrs, posi)`` expects
    numpy arrays of shape ``(N, K+1)`` where the first column is the
    timestamp (ms) and the remaining columns are the sensor channels.
    We assemble those columns from our async_collection format.

    Returns a list of per-path dicts with keys: ``path_id``, ``acce`` (N, 4),
    ``ahrs`` (N, 4 — rotation vector x/y/z + acc), ``magn`` (N, 4 — zeros if
    no magnetometer), ``waypoints`` (N, 3) timestamp + x + y, and ``wifi``
    (N, 5) timestamp + ssid + bssid + rssi + frequency-or-zero.
    """
    out = []
    for pid in path_ids:
        pdir = MSILN_ROOT / f"path_{pid:02d}"
        if not pdir.is_dir():
            continue
        gt = pd.read_csv(pdir / "ground_truth.csv")
        imu = pd.read_csv(pdir / "imu.csv")
        wifi = pd.read_csv(pdir / "wifi.csv")

        # The competition code wants timestamps in milliseconds (Android-style)
        # but our sim_time is in seconds — multiply by 1000.
        ts_imu_ms = (imu["sim_time"].values * 1000.0).astype(np.float64)
        ts_wifi_ms = (wifi["sim_time"].values * 1000.0).astype(np.float64) if not wifi.empty else np.array([])
        ts_gt_ms = (gt["sim_time"].values * 1000.0).astype(np.float64)

        acce = np.column_stack([
            ts_imu_ms,
            imu["accel_x"].values, imu["accel_y"].values, imu["accel_z"].values,
        ])
        # ahrs = rotation vector (x, y, z) — synthesize from roll/pitch/yaw
        # (the competition only needs heading info; we'll pass yaw in z slot)
        roll = np.deg2rad(imu["roll_deg"].values)
        pitch = np.deg2rad(imu["pitch_deg"].values)
        yaw = np.deg2rad(imu["yaw_deg"].values)
        # Build quaternion-equivalent rotation vector (small-angle approx OK
        # for heading-only PDR; we just need yaw to roll through compute_headings)
        ahrs = np.column_stack([ts_imu_ms, roll, pitch, yaw])

        magn = np.column_stack([ts_imu_ms, np.zeros_like(roll), np.zeros_like(roll), np.zeros_like(roll)])

        waypoints = np.column_stack([ts_gt_ms, gt["gt_x"].values, gt["gt_y"].values])

        # WiFi: timestamp + ssid + bssid + rssi (strongest per scan-time)
        # We collapse our wide-format per-time-tick into per-(timestamp, bssid) tuples
        if not wifi.empty:
            rssi_cols = [c for c in wifi.columns if c.startswith("wifi_rssi_")]
            wifi_rows = []
            for _, row in wifi.iterrows():
                t = float(row["sim_time"]) * 1000.0
                for c in rssi_cols:
                    rssi = row[c]
                    if pd.isna(rssi):
                        continue
                    bssid = c.replace("wifi_rssi_", "")
                    wifi_rows.append([t, "msiln_ap", bssid, float(rssi), 0.0])
            wifi_arr = np.array(wifi_rows, dtype=object) if wifi_rows else np.zeros((0, 5))
        else:
            wifi_arr = np.zeros((0, 5))

        out.append({
            "path_id": int(pid),
            "acce": acce,
            "ahrs": ahrs,
            "magn": magn,
            "waypoints": waypoints,
            "wifi": wifi_arr,
            "ibeacon": np.zeros((0, 3)),
        })
    return out


__all__ = [
    "MSILN_ROOT",
    "TRAIN_PATHS", "VAL_PATHS", "TEST_PATHS",
    "load_ap_vocab",
    "load_msiln_paths_for_imuwifine",
    "load_msiln_paths_for_competition",
]
