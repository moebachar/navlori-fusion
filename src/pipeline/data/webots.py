"""Webots Tiago async_collection — 18-path 4-modality fusion dataset.

Sensor rates (per CLAUDE.md):
  IMU ~31 Hz, Odometry ~15 Hz, Ground Truth ~10 Hz,
  WiFi ~1 Hz, Camera ~5 Hz.

Canonical split (CLAUDE.md):
  train = paths [1, 3-12]   (11 paths)
  val   = paths [2, 13, 14] (3 paths)
  test  = paths [15, 16, 17] (3 paths)

Path 0 is empty / failed-collection and excluded from all splits.
"""
from __future__ import annotations

from ._common import collect_path_metadata, not_applicable, path_to, summarise_path_lengths

DATASET_NAME = "webots"
COLLECTION_DIR = "async_collection"
CONFIG_NAME = "simulation"


def load(modalities=None, K=4, batch_size=128, **kwargs):
    """Build the canonical FusionDataModule for Webots.

    Returns the ``FusionDataModule`` instance (with ``train_ds``,
    ``val_ds``, ``test_ds`` attributes). For ad-hoc inspection use
    ``stats()`` instead.
    """
    from src.pipeline.fusion.builder import build_datamodule, load_config
    cfg = load_config(CONFIG_NAME)
    if modalities is not None:
        cfg.dataset.modalities = list(modalities)
    cfg.temporal.n_instants = int(K)
    cfg.data.batch_size = int(batch_size)
    return build_datamodule(cfg)


def stats() -> dict:
    """Return summary statistics for the Webots dataset."""
    collection = path_to(f"data/{COLLECTION_DIR}")
    per_path = collect_path_metadata(collection)
    return {
        "name": DATASET_NAME,
        "collection_dir": str(collection.relative_to(path_to("."))),
        "modalities_available": ["wifi", "imu", "camera", "odom"],
        "splits": {
            "train": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "val": [2, 13, 14],
            "test": [15, 16, 17],
        },
        "sensor_rates_hz": {"imu": 31, "odom": 15, "gt": 10, "wifi": 1, "camera": 5},
        "n_paths_total": len(per_path),
        **summarise_path_lengths(per_path),
        "known_caveats": [
            "Path 0 is empty / failed-collection; excluded.",
            "Webots WiFi is GPR-synthesised, not measured — sub-metre MAE optimistic vs real-world (CLAUDE.md honest finding #3).",
        ],
        "source_result": "RESULT_06+ (fusion); RESULT_03/04 (per-leg encoder audits)",
    }


def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    """Return raw + preprocessed paired samples for the given modality."""
    import pandas as pd
    import numpy as np
    if modality not in {"wifi", "imu", "camera", "odom"}:
        return not_applicable(modality, DATASET_NAME)
    sample_path = path_to(f"data/{COLLECTION_DIR}/path_01")
    if modality == "wifi":
        wifi = pd.read_csv(sample_path / "wifi.csv")
        rssi_cols = [c for c in wifi.columns if c.startswith("wifi_rssi_")]
        raw = wifi[rssi_cols].iloc[:n_samples].values.astype(np.float32)
        # Anchor2Vec preprocessing: NaN -> -100 (no signal) then affine to [0,1].
        clean = np.where(np.isnan(raw), -100.0, raw).clip(-100, 0)
        preprocessed = (clean + 100.0) / 100.0
        return {
            "raw": raw,
            "preprocessed": preprocessed,
            "description_raw": f"raw RSSI dBm in [-100, 0]; NaN = AP not visible (n={raw.shape[0]} scan(s), {raw.shape[1]} APs)",
            "description_preprocessed": "NaN -> -100 (no signal), then affine (x+100)/100 to [0, 1]",
            "preprocessing_pipeline": ["nan_fill(-100)", "clip(-100, 0)", "affine: (x+100)/100"],
        }
    if modality == "imu":
        imu = pd.read_csv(sample_path / "imu.csv")
        cols_dev = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        raw = imu[cols_dev].head(32).values.astype(np.float32)
        return {
            "raw": raw,
            "preprocessed": raw,
            "description_raw": f"raw 6-ch device-frame (accel xyz + gyro xyz), 32-step window (~1 s)",
            "description_preprocessed": "z-score per channel using train-set statistics (computed by FusionDataModule)",
            "preprocessing_pipeline": ["window 32 steps", "z-score with train-set mean/std"],
        }
    if modality == "camera":
        return {
            "raw": None,
            "preprocessed": None,
            "description_raw": "RGB image 480x640 at 5 Hz; sample camera_*.png",
            "description_preprocessed": "ImageNet-normalised, then DPVOMotion 2x-0.5 affine to [-1, 1]",
            "preprocessing_pipeline": ["resize/crop", "to_tensor", "ImageNet norm", "DPVO 2x-0.5"],
            "note": "Camera preprocessing visualisation deferred to plot helpers (raw PNG + normalised array).",
        }
    if modality == "odom":
        odom = pd.read_csv(sample_path / "odometry.csv")
        raw_cols = ["odom_x", "odom_y", "odom_theta_deg",
                     "odom_linear_vel", "odom_angular_vel",
                     "wheel_left_vel", "wheel_right_vel"]
        raw = odom[raw_cols].head(16).values.astype(np.float32)
        delta = np.diff(raw, axis=0)
        return {
            "raw": raw,
            "preprocessed": delta,
            "description_raw": "7-column odometry (x, y, theta_deg, linear/angular vel, wheel L/R) at 15 Hz",
            "description_preprocessed": "P-B Δ-features (frame-to-frame deltas; RESULT_04 winner over P-A raw normalisation)",
            "preprocessing_pipeline": ["window 16 steps", "Δ-features (diff over time)", "z-score with train-set mean/std"],
        }
    return not_applicable(modality, DATASET_NAME)


__all__ = ["load", "stats", "preprocessing_demo"]
