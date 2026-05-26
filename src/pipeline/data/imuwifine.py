"""IMUWiFine floor 4 — 2-modality WiFi+IMU, 80 paths.

train = paths 0-39 (40), val = 40-59 (20), test = 60-79 (20).

**Two raw formats coexist** (`scripts/convert_imuwifine.py:42-52`):
- train+val from Android logger format (WiFi @ 0.31 Hz, IMU @ 30 Hz)
- test from a separate campaign (WiFi @ 5.65 Hz, **NO IMU**, different
  physical y-range 1.2-1.6 m vs 0-5 m on train+val).

RESULT_20 audit verdict: failure mode 3 (legitimate cross-session
shift, no code bug). Fusion's test column on this dataset is
effectively WiFi-only inference (IMU windows zero-pad).
"""
from __future__ import annotations

from ._common import collect_path_metadata, not_applicable, path_to, summarise_path_lengths

DATASET_NAME = "imuwifine_floor4"
COLLECTION_DIR = "imuwifine_floor4"
CONFIG_NAME = "imuwifine"


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
            "train": list(range(0, 40)),
            "val": list(range(40, 60)),
            "test": list(range(60, 80)),
        },
        "n_paths_total": len(per_path),
        **summarise_path_lengths(per_path),
        "known_caveats": [
            "**Test paths lack IMU by dataset design** (RESULT_20 audit). Fusion test = WiFi-only inference.",
            "Two raw formats: train+val Android logger (WiFi 0.31 Hz, IMU 30 Hz); test header-less (WiFi 5.65-6.57 Hz, no IMU).",
            "Test physical region constrained to y=1.2-1.6 m vs train+val 0-5 m (cross-session, separate campaign).",
            "Val/test gap +408 % on CNN1D is failure mode 3 (legitimate cross-session shift, not code bug).",
        ],
        "source_result": "RESULT_19, RESULT_20",
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
            "description_raw": f"raw RSSI dBm; n={raw.shape[0]} scan(s), {raw.shape[1]} APs (343 BSSIDs on floor 4)",
            "description_preprocessed": "NaN -> -100; affine to [0, 1]; then wifi_pca to 128 dims (FusionDataset)",
            "preprocessing_pipeline": ["nan_fill(-100)", "clip", "affine", "PCA 343 -> 128"],
        }
    if modality == "imu":
        imu_p = sample_path / "imu.csv"
        if not imu_p.exists() or imu_p.stat().st_size < 100:
            return {
                "raw": None,
                "preprocessed": None,
                "description_raw": "IMU absent on this path (train+val have IMU; test paths do not)",
                "description_preprocessed": "n/a",
                "preprocessing_pipeline": [],
                "note": "See RESULT_20 audit; IMUWiFine test paths lack IMU by dataset design.",
            }
        imu = pd.read_csv(imu_p)
        cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        raw = imu[cols].head(32).values.astype(np.float32)
        return {
            "raw": raw,
            "preprocessed": raw,
            "description_raw": "6-ch device-frame IMU @ 30 Hz, 32-step window (~1 s after downsample from 192 Hz)",
            "description_preprocessed": "z-score per channel using train-set statistics",
            "preprocessing_pipeline": ["downsample 192 Hz -> 32 Hz", "window 32 steps", "z-score"],
        }
    return not_applicable(modality, DATASET_NAME)


__all__ = ["load", "stats", "preprocessing_demo"]
