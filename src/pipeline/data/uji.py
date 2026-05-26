"""UJIIndoorLoc — per-scan WiFi benchmark, no temporal axis.

Data: ``data/uji_indoorloc/{trainingData,validationData}.csv``
- 19937 train scans + 1111 val scans
- 520 APs (WAP001 .. WAP520)
- Labels: (LONGITUDE, LATITUDE, BUILDINGID, FLOOR)
- "Not detected" sentinel = 100 (RSSI; outside the [-104, 0] range)

K=1 + M=1 degenerate row of the main-results table (RESULT_24).
Per-scan input means the temporal aggregators (CNN1D / LSTM-attn)
have no axis to operate on and collapse to encoder + head.
"""
from __future__ import annotations

from pathlib import Path

from ._common import not_applicable, path_to


DATASET_NAME = "uji_indoorloc"
DATA_DIR = "uji_indoorloc"


def load(split: str = "validation", **kwargs):
    """Return (X_rssi, Y_xy_buildfloor) numpy arrays."""
    import pandas as pd
    import numpy as np
    root = path_to(f"data/{DATA_DIR}")
    csv_name = {"training": "trainingData.csv", "train": "trainingData.csv",
                "validation": "validationData.csv", "val": "validationData.csv"}[split]
    df = pd.read_csv(root / csv_name)
    waps = [c for c in df.columns if c.startswith("WAP")]
    X = df[waps].values.astype(np.float32)
    Y = df[["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"]].copy()
    return X, Y


def stats() -> dict:
    import pandas as pd
    root = path_to(f"data/{DATA_DIR}")
    train_p = root / "trainingData.csv"
    val_p = root / "validationData.csv"
    n_train = len(pd.read_csv(train_p)) if train_p.is_file() else 0
    n_val = len(pd.read_csv(val_p)) if val_p.is_file() else 0
    return {
        "name": DATASET_NAME,
        "data_dir": str(root.relative_to(path_to("."))) if root.is_dir() else f"data/{DATA_DIR}",
        "modalities_available": ["wifi"],
        "splits": {"train": n_train, "validation": n_val},
        "n_aps": 520,
        "wap_columns": "WAP001 .. WAP520",
        "not_detected_sentinel": 100,
        "rssi_valid_range_dbm": [-104, 0],
        "label_columns": ["LONGITUDE", "LATITUDE", "BUILDINGID", "FLOOR"],
        "label_frame": "geographic (longitude/latitude in metres after centring)",
        "known_caveats": [
            "Per-scan data — NO temporal axis. K=1 + M=1 degenerate row of the main-results table (RESULT_24 α7).",
            "wlan_localization global SOTA val mean Euclidean 15.17 m; Anchor2Vec 8.69 m (RESULT_01); CNN1D 8.72, LSTM-attn 8.43 (RESULT_24).",
            "No test split — `validationData.csv` is the benchmark. Per-scan distribution reported instead of per-trajectory smoothness.",
        ],
        "source_result": "RESULT_01 (per-leg WiFi audit), RESULT_24 (K=1 M=1 main-table row)",
    }


def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    import pandas as pd
    import numpy as np
    if modality != "wifi":
        return not_applicable(modality, DATASET_NAME)
    root = path_to(f"data/{DATA_DIR}")
    df = pd.read_csv(root / "validationData.csv").head(n_samples)
    waps = [c for c in df.columns if c.startswith("WAP")]
    raw = df[waps].values.astype(np.float32)
    # Anchor2Vec / wlan_localization preprocessing: 100 sentinel -> -100, then affine.
    cleaned = np.where(raw == 100.0, -100.0, raw).clip(-100, 0)
    preprocessed = (cleaned + 100.0) / 100.0
    return {
        "raw": raw,
        "preprocessed": preprocessed,
        "description_raw": f"raw RSSI: 100 = not detected, [-104, 0] = detected dBm (n={raw.shape[0]} scan(s), {raw.shape[1]} APs)",
        "description_preprocessed": "100 -> -100; clip [-100, 0]; affine to [0, 1]",
        "preprocessing_pipeline": ["sentinel 100 -> -100 (no signal)", "clip(-100, 0)", "affine: (x+100)/100"],
    }


__all__ = ["load", "stats", "preprocessing_demo"]
