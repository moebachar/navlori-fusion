"""Microsoft Indoor Localization 2.0 — site1/B1 cross-session.

133 traces total; train = paths 0-93 (Nov 24 surveyor), val =
94-127 (Nov 25), test = 128-132 (Dec 5+6). Cross-session, 2-modality
(WiFi+IMU).

RESULT_15: path 130 (786 samples, 28 % of test, WiFi-dense) pulls
kNN test mean down — a documented per-path-composition property.
"""
from __future__ import annotations

from ._common import collect_path_metadata, not_applicable, path_to, summarise_path_lengths

DATASET_NAME = "msiln_site1_b1"
COLLECTION_DIR = "msiln_site1_b1"
CONFIG_NAME = "msiln_site1_b1"


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
            "train": list(range(0, 94)),
            "val": list(range(94, 128)),
            "test": list(range(128, 133)),
        },
        "n_paths_total": len(per_path),
        **summarise_path_lengths(per_path),
        "known_caveats": [
            "Cross-session train (Nov 24) / val (Nov 25) / test (Dec 5+6).",
            "Test path 130 (786 samples, ~28 % of test, WiFi-dense) dominates kNN test mean — RESULT_15.",
            "WiFi RSSI fingerprints drift across sessions; gate (c)-1 partial only.",
        ],
        "source_result": "RESULT_15",
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
            "description_raw": f"raw RSSI dBm; n={raw.shape[0]} scan(s), {raw.shape[1]} APs",
            "description_preprocessed": "NaN -> -100; affine to [0, 1]",
            "preprocessing_pipeline": ["nan_fill(-100)", "clip", "affine"],
        }
    if modality == "imu":
        imu = pd.read_csv(sample_path / "imu.csv")
        cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        raw = imu[cols].head(32).values.astype(np.float32)
        return {
            "raw": raw,
            "preprocessed": raw,
            "description_raw": "6-ch device-frame IMU, 32-step window",
            "description_preprocessed": "z-score with train-set mean/std",
            "preprocessing_pipeline": ["window 32", "z-score"],
        }
    return not_applicable(modality, DATASET_NAME)


__all__ = ["load", "stats", "preprocessing_demo"]
