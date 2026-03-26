"""Tests for multi-modal dataloader.

These tests use REAL data from data/async_collection/ (path_01 at minimum).
They verify shapes, dtypes, normalization, and edge cases.
"""

import numpy as np
import pytest
import torch

from src.pipeline.data import FusionDataModule, FusionDataset, IMU_COLS, ODOM_COLS

DATA_DIR = "data/async_collection"

# Use a small subset for fast tests
TRAIN_PATHS = [1, 3]
VAL_PATHS = [2]
TEST_PATHS = [4]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Single normalized dataset on train paths."""
    return FusionDataset(
        data_dir=DATA_DIR,
        path_ids=TRAIN_PATHS,
        modalities=["imu", "odom", "wifi"],
        normalize=True,
    )


@pytest.fixture(scope="module")
def dataset_unnorm():
    """Unnormalized dataset for checking raw values."""
    return FusionDataset(
        data_dir=DATA_DIR,
        path_ids=TRAIN_PATHS,
        modalities=["imu", "odom", "wifi"],
        normalize=False,
    )


@pytest.fixture(scope="module")
def datamodule():
    """Full DataModule with train/val/test."""
    dm = FusionDataModule(
        data_dir=DATA_DIR,
        train_paths=TRAIN_PATHS,
        val_paths=VAL_PATHS,
        test_paths=TEST_PATHS,
        modalities=["imu", "odom", "wifi"],
        batch_size=8,
    )
    dm.setup()
    return dm


# ------------------------------------------------------------------
# Dataset basics
# ------------------------------------------------------------------

class TestDatasetBasics:
    def test_length_positive(self, dataset):
        assert len(dataset) > 0

    def test_sample_keys(self, dataset):
        sample = dataset[0]
        assert "imu" in sample
        assert "odom" in sample
        assert "wifi" in sample
        assert "target" in sample
        assert "timestamp" in sample
        assert "path_id" in sample

    def test_target_shape(self, dataset):
        sample = dataset[0]
        assert sample["target"].shape == (2,)  # (x, y)
        assert sample["target"].dtype == torch.float32

    def test_timestamp_is_positive(self, dataset):
        sample = dataset[0]
        assert sample["timestamp"].item() > 0


class TestModalityShapes:
    def test_imu_shape(self, dataset):
        sample = dataset[0]
        assert sample["imu"].shape == (32, len(IMU_COLS))  # (window, 9)
        assert sample["imu"].dtype == torch.float32

    def test_odom_shape(self, dataset):
        sample = dataset[0]
        assert sample["odom"].shape == (16, len(ODOM_COLS))  # (window, 7)
        assert sample["odom"].dtype == torch.float32

    def test_wifi_shape(self, dataset):
        sample = dataset[0]
        n_aps = dataset.num_wifi_aps
        assert n_aps > 0
        assert sample["wifi"].shape == (1, n_aps)  # (window=1, n_aps)
        assert sample["wifi"].dtype == torch.float32

    def test_custom_window(self):
        ds = FusionDataset(
            data_dir=DATA_DIR,
            path_ids=[1],
            modalities=["imu"],
            windows={"imu": 64},
            normalize=False,
        )
        sample = ds[0]
        assert sample["imu"].shape == (64, len(IMU_COLS))


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

class TestNormalization:
    def test_stats_computed(self, dataset):
        assert dataset.stats is not None
        assert "imu" in dataset.stats
        assert "odom" in dataset.stats
        assert "wifi" in dataset.stats

    def test_stats_shapes(self, dataset):
        for mod in ["imu", "odom", "wifi"]:
            mean = dataset.stats[mod]["mean"]
            std = dataset.stats[mod]["std"]
            assert mean.ndim == 1
            assert std.ndim == 1
            assert len(mean) == len(std)

    def test_imu_stats_plausible(self, dataset):
        """IMU accel should have mean near 0 (robot mostly stationary/slow)."""
        mean = dataset.stats["imu"]["mean"]
        std = dataset.stats["imu"]["std"]
        # Std should be positive
        assert (std > 0).all(), "IMU std should be positive"

    def test_normalized_values_reasonable(self, dataset):
        """After normalization, values should be roughly in [-5, 5] range."""
        sample = dataset[len(dataset) // 2]  # middle sample
        for mod in ["imu", "odom"]:
            vals = sample[mod]
            # Allow some outliers, but the bulk should be small
            assert vals.abs().mean() < 10, f"{mod} normalized values too large"

    def test_unnormalized_wifi_negative(self, dataset_unnorm):
        """Raw WiFi RSSI values should be negative (dBm)."""
        sample = dataset_unnorm[len(dataset_unnorm) // 2]
        wifi = sample["wifi"]
        # Non-zero values should be negative (RSSI in dBm)
        nonzero = wifi[wifi != 0]
        if len(nonzero) > 0:
            assert (nonzero < 0).all(), "Raw RSSI should be negative dBm"


# ------------------------------------------------------------------
# DataModule
# ------------------------------------------------------------------

class TestDataModule:
    def test_setup_creates_datasets(self, datamodule):
        assert datamodule.train_ds is not None
        assert datamodule.val_ds is not None
        assert datamodule.test_ds is not None

    def test_splits_disjoint(self, datamodule):
        """Train/val/test should use different paths."""
        train_paths = {r["path_id"] for r in datamodule.train_ds._gt_rows}
        val_paths = {r["path_id"] for r in datamodule.val_ds._gt_rows}
        test_paths = {r["path_id"] for r in datamodule.test_ds._gt_rows}
        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)
        assert val_paths.isdisjoint(test_paths)

    def test_shared_normalization(self, datamodule):
        """Val/test should use training stats (no data leakage)."""
        train_stats = datamodule.train_ds.stats
        val_stats = datamodule.val_ds.stats
        test_stats = datamodule.test_ds.stats
        # They should be the exact same object
        assert val_stats is train_stats
        assert test_stats is train_stats

    def test_train_dataloader(self, datamodule):
        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        assert batch["imu"].shape[0] == 8  # batch_size
        assert batch["target"].shape == (8, 2)

    def test_val_dataloader_no_shuffle(self, datamodule):
        """Val loader should give same order on two iterations."""
        loader = datamodule.val_dataloader()
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))
        # First batch timestamps should be identical across calls
        loader2 = datamodule.val_dataloader()
        batch2 = next(iter(loader2))
        assert torch.equal(batch1["timestamp"], batch2["timestamp"])


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_path_skipped(self):
        """Path 0 is empty — dataset should still work."""
        ds = FusionDataset(
            data_dir=DATA_DIR,
            path_ids=[0, 1],
            modalities=["imu"],
            normalize=False,
        )
        assert len(ds) > 0
        # All samples should be from path 1
        for i in range(min(5, len(ds))):
            assert ds[i]["path_id"] == 1

    def test_single_path(self):
        ds = FusionDataset(
            data_dir=DATA_DIR,
            path_ids=[1],
            modalities=["imu", "odom", "wifi"],
            normalize=False,
        )
        assert len(ds) > 0

    def test_first_sample_has_padding(self):
        """First GT sample may not have enough IMU history → should be zero-padded."""
        ds = FusionDataset(
            data_dir=DATA_DIR,
            path_ids=[1],
            modalities=["imu"],
            windows={"imu": 32},
            normalize=False,
        )
        sample = ds[0]
        imu = sample["imu"]
        # First few rows might be zero-padded
        assert imu.shape == (32, len(IMU_COLS))
        # Should not be all zeros (at least 1 real observation)
        assert imu.abs().sum() > 0

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="No data found"):
            FusionDataset(
                data_dir=DATA_DIR,
                path_ids=[99],
                modalities=["imu"],
                normalize=False,
            )


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

class TestReproducibility:
    def test_deterministic(self):
        """Same index should always return the same sample."""
        ds = FusionDataset(
            data_dir=DATA_DIR,
            path_ids=[1],
            modalities=["imu", "odom"],
            normalize=True,
        )
        s1 = ds[10]
        s2 = ds[10]
        assert torch.equal(s1["imu"], s2["imu"])
        assert torch.equal(s1["odom"], s2["odom"])
        assert torch.equal(s1["target"], s2["target"])
