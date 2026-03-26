"""DataModule — creates train/val/test DataLoaders with shared normalization.

Usage with Hydra config::

    dm = FusionDataModule(cfg.data)
    dm.setup()
    for batch in dm.train_dataloader():
        ...

Usage standalone::

    dm = FusionDataModule.from_paths(
        data_dir="data/async_collection",
        train_paths=[1,3,4,5,6,7,8,9,10,11,12],
        val_paths=[2,13,14],
        test_paths=[15,16,17],
    )
    dm.setup()
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import FusionDataset


class FusionDataModule:
    """Manages train/val/test splits with consistent normalization.

    Normalization stats are always computed from the training set and
    shared across val/test to prevent data leakage.
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_paths: list[int],
        val_paths: list[int],
        test_paths: list[int],
        modalities: list[str] | None = None,
        windows: dict[str, int] | None = None,
        normalize: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        camera_transform=None,
        wifi_pca: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.modalities = modalities or ["imu", "odom", "wifi", "camera"]
        self.windows = windows
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.camera_transform = camera_transform
        self.wifi_pca = wifi_pca

        self.train_ds: FusionDataset | None = None
        self.val_ds: FusionDataset | None = None
        self.test_ds: FusionDataset | None = None

    @classmethod
    def from_hydra(cls, cfg) -> "FusionDataModule":
        """Create from a Hydra DictConfig (configs/data/simulation.yaml)."""
        return cls(
            data_dir=cfg.data.root + "/async_collection",
            train_paths=list(cfg.data.split.train_paths),
            val_paths=list(cfg.data.split.val_paths),
            test_paths=list(cfg.data.split.test_paths),
            modalities=list(cfg.data.modalities),
            normalize=cfg.data.preprocessing.normalize,
            batch_size=cfg.training.batch_size,
        )

    @classmethod
    def from_paths(
        cls,
        data_dir: str | Path = "data/async_collection",
        train_paths: list[int] | None = None,
        val_paths: list[int] | None = None,
        test_paths: list[int] | None = None,
        **kwargs,
    ) -> "FusionDataModule":
        """Quick constructor with sensible defaults for 18 paths."""
        return cls(
            data_dir=data_dir,
            train_paths=train_paths or [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            val_paths=val_paths or [2, 13, 14],
            test_paths=test_paths or [15, 16, 17],
            **kwargs,
        )

    def setup(self) -> None:
        """Load data and compute normalization stats from training set."""
        # Training set — compute stats and fit PCA (if requested)
        self.train_ds = FusionDataset(
            data_dir=self.data_dir,
            path_ids=self.train_paths,
            modalities=self.modalities,
            windows=self.windows,
            normalize=self.normalize,
            stats=None,  # will compute internally
            camera_transform=self.camera_transform,
            wifi_pca=self.wifi_pca,
        )

        # Share training stats and PCA model with val/test (no leakage)
        shared_stats = self.train_ds.stats
        shared_pca = self.train_ds._wifi_pca_model

        self.val_ds = FusionDataset(
            data_dir=self.data_dir,
            path_ids=self.val_paths,
            modalities=self.modalities,
            windows=self.windows,
            normalize=self.normalize,
            stats=shared_stats,
            camera_transform=self.camera_transform,
            wifi_pca=self.wifi_pca,
            wifi_pca_model=shared_pca,
        )

        self.test_ds = FusionDataset(
            data_dir=self.data_dir,
            path_ids=self.test_paths,
            modalities=self.modalities,
            windows=self.windows,
            normalize=self.normalize,
            stats=shared_stats,
            camera_transform=self.camera_transform,
            wifi_pca=self.wifi_pca,
            wifi_pca_model=shared_pca,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "Call setup() first"
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "Call setup() first"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None, "Call setup() first"
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def summary(self) -> str:
        """Print dataset sizes and feature dimensions."""
        lines = ["Fusion DataModule Summary", "=" * 40]
        for name, ds in [("Train", self.train_ds), ("Val", self.val_ds), ("Test", self.test_ds)]:
            if ds is not None:
                lines.append(f"  {name}: {len(ds)} samples")
        if self.train_ds is not None:
            lines.append(f"  Modalities: {self.modalities}")
            lines.append(f"  Feature dims: {self.train_ds.feature_dims}")
            lines.append(f"  Windows: { {m: self.train_ds.windows[m] for m in self.modalities} }")
            if self.wifi_pca:
                pca = self.train_ds._wifi_pca_model
                ev = pca.explained_variance_ratio_.sum() * 100
                lines.append(f"  WiFi PCA: 117 → {self.wifi_pca} dims ({ev:.1f}% variance)")
            if self.train_ds.stats:
                for mod, s in self.train_ds.stats.items():
                    lines.append(f"  {mod} mean range: [{s['mean'].min():.3f}, {s['mean'].max():.3f}]")
                    lines.append(f"  {mod} std range:  [{s['std'].min():.3f}, {s['std'].max():.3f}]")
        return "\n".join(lines)
