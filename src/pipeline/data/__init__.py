"""Data loading and preprocessing.

Two layers:
- Low-level ``FusionDataModule`` / ``FusionDataset`` for the
  temporally-windowed fusion datasets (Webots / MSILN / IMUWiFine
  / IPIN).
- High-level **dataset factory** (post-run-2 consolidation,
  PLAN_27) — ``load_dataset(name)`` / ``dataset_stats(name)`` /
  ``preprocessing_demo(name, modality)`` — uniform API across all
  7 datasets used in the run-2 main-results table. The factory
  dispatches to per-dataset modules ``webots`` / ``msiln`` /
  ``imuwifine`` / ``ipin2024`` / ``ronin_canonical`` / ``tartanair``
  / ``uji``.
"""

from .datamodule import FusionDataModule
from .dataset import (
    IMU_COLS,
    MODALITY_DIMS,
    ODOM_COLS,
    FusionDataset,
)
from .factory import (
    dataset_stats,
    list_datasets,
    load_dataset,
    preprocessing_demo,
)

__all__ = [
    # low-level
    "FusionDataModule",
    "FusionDataset",
    "IMU_COLS",
    "ODOM_COLS",
    "MODALITY_DIMS",
    # high-level factory
    "list_datasets",
    "load_dataset",
    "dataset_stats",
    "preprocessing_demo",
]
