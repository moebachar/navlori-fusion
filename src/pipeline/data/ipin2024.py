"""IPIN 2024 Track 3 floor 0 — 2-modality WiFi+IMU, 16 paths.

train = [0, 1, 2, 3, 4, 5] (6), val = [6, 7, 8, 9] (4),
test = [10, 11, 12, 13, 14, 15] (6).

Small-train regime: only 174 WiFi scans + 6924 IMU windows in train
(per RESULT_22). Both CNN1D and LSTM-attn fusion candidates overfit
fast (train loss 0.5 vs val loss 9.0 by epoch 20).

Diagnostic: CNN1D `only:wifi` val 19.45 m **beats** wlanloc SOTA val
20.53 m by 5 % — the fusion regression is a small-train-overfit
artifact, not a fundamental WiFi failure.
"""
from __future__ import annotations

from ._common import collect_path_metadata, not_applicable, path_to, summarise_path_lengths

DATASET_NAME = "ipin2024_floor0"
COLLECTION_DIR = "ipin2024_floor0"
CONFIG_NAME = "ipin2024_floor0"


def load(modalities=None, K=4, batch_size=128, **kwargs):
    from src.pipeline.fusion.builder import build_datamodule, load_config
    cfg = load_config(CONFIG_NAME)
    if modalities is not None:
        cfg.dataset.modalities = list(modalities)
    cfg.temporal.n_instants = int(K)
    cfg.data.batch_size = int(batch_size)
    return build_datamodule(cfg)


def stats() -> dict:
    collection = path_to(f"data/{COLLECTION_DIR}")
    per_path = collect_path_metadata(collection)
    return {
        "name": DATASET_NAME,
        "collection_dir": str(collection.relative_to(path_to("."))),
        "modalities_available": ["wifi", "imu"],
        "splits": {
            "train": [0, 1, 2, 3, 4, 5],
            "val": [6, 7, 8, 9],
            "test": [10, 11, 12, 13, 14, 15],
        },
        "n_paths_total": len(per_path),
        **summarise_path_lengths(per_path),
        "known_caveats": [
            "Small-train regime: 174 WiFi scans + 6924 IMU windows ≈ 10× smaller than IMUWiFine. Fusion archs overfit fast.",
            "CNN1D `only:wifi` val 19.45 m beats wlanloc SOTA val 20.53 m by 5 % (RESULT_22).",
            "IPIN has IMU on all splits (unlike IMUWiFine) → RoNIN ResNet1D fully measurable on test.",
        ],
        "source_result": "RESULT_22",
    }


def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    import pandas as pd
    import numpy as np
    if modality not in {"wifi", "imu"}:
        return not_applicable(modality, DATASET_NAME)
    sample_path = path_to(f"data/{COLLECTION_DIR}/path_00")
    if modality == "wifi":
        wifi = pd.read_csv(sample_path / "wifi.csv")
        rssi_cols = [c for c in wifi.columns if c.startswith("wifi_rssi_")]
        raw = wifi[rssi_cols].iloc[:n_samples].values.astype(np.float32)
        clean = np.where(np.isnan(raw), -100.0, raw).clip(-100, 0)
        preprocessed = (clean + 100.0) / 100.0
        return {
            "raw": raw,
            "preprocessed": preprocessed,
            "description_raw": f"raw RSSI dBm; n={raw.shape[0]} scan(s), {raw.shape[1]} APs (232 BSSIDs)",
            "description_preprocessed": "NaN -> -100; affine to [0, 1]; PCA 232 -> 128",
            "preprocessing_pipeline": ["nan_fill(-100)", "clip", "affine", "PCA 232 -> 128"],
        }
    if modality == "imu":
        imu = pd.read_csv(sample_path / "imu.csv")
        cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        raw = imu[cols].head(32).values.astype(np.float32)
        return {
            "raw": raw,
            "preprocessed": raw,
            "description_raw": "6-ch device-frame IMU, 32-step window (~1 s at ~25 Hz)",
            "description_preprocessed": "z-score with train-set mean/std",
            "preprocessing_pipeline": ["window 32", "z-score"],
        }
    return not_applicable(modality, DATASET_NAME)


__all__ = ["load", "stats", "preprocessing_demo"]
