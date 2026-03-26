"""Async multi-modal dataset for indoor localization.

Each sample is anchored to a ground-truth timestamp.  For that instant the
dataset gathers the most recent *window* of observations from every requested
modality, producing fixed-size tensors ready for per-modality encoders.

Window sizes are modality-specific because sensor rates differ:
  IMU  ~31 Hz → 32 samples ≈ 1 s
  Odom ~15 Hz → 16 samples ≈ 1 s
  WiFi  ~1 Hz →  1 sample  (single scan, 117-dim RSSI vector)
  Camera ~0.5 Hz → 1 frame  (loaded lazily)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Column groups – keep in sync with async_collector.py output
# ---------------------------------------------------------------------------
IMU_COLS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "roll_deg", "pitch_deg", "yaw_deg",
]  # 9 features (drop magnitudes – they're derived)

ODOM_COLS = [
    "odom_x", "odom_y", "odom_theta_deg",
    "odom_linear_vel", "odom_angular_vel",
    "wheel_left_vel", "wheel_right_vel",
]  # 7 features

GT_COLS = ["gt_x", "gt_y"]  # regression targets

# Default window sizes (number of past observations to include)
DEFAULT_WINDOWS = {
    "imu": 32,   # ~1 s at 31 Hz
    "odom": 16,  # ~1 s at 15 Hz
    "wifi": 1,   # single scan (sparse modality)
    "camera": 1, # single frame
}

# Features dimensions per modality
MODALITY_DIMS = {
    "imu": len(IMU_COLS),       # 9
    "odom": len(ODOM_COLS),     # 7
    "wifi": None,               # set dynamically from CSV header
    "camera": (3, 224, 224),    # after resize + RGB conversion
}

# Map modality short names to actual CSV filenames
_CSV_FILENAMES = {
    "imu": "imu.csv",
    "odom": "odometry.csv",
    "wifi": "wifi.csv",
    "camera": "camera.csv",
}


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a sensor CSV, set sim_time as float64 index."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "sim_time" in df.columns:
        df["sim_time"] = df["sim_time"].astype(np.float64)
    return df


def _get_wifi_rssi_cols(df: pd.DataFrame) -> list[str]:
    """Extract the wifi_rssi_* column names (per-AP RSSI values)."""
    return [c for c in df.columns if c.startswith("wifi_rssi_")]


class FusionDataset(Dataset):
    """Multi-modal async dataset for indoor robot localization.

    Parameters
    ----------
    data_dir : path to ``data/async_collection/``
    path_ids : list of path indices to include (e.g. [1, 2, 3])
    modalities : subset of ["imu", "odom", "wifi", "camera"]
    windows : per-modality window sizes (overrides DEFAULT_WINDOWS)
    normalize : whether to z-score normalize (requires stats)
    stats : dict of {modality: {mean: ndarray, std: ndarray}} — if None
            and normalize=True, stats are computed from this dataset
    camera_transform : torchvision transform for camera images (optional)
    """

    def __init__(
        self,
        data_dir: str | Path,
        path_ids: list[int],
        modalities: list[str] | None = None,
        windows: dict[str, int] | None = None,
        normalize: bool = True,
        stats: dict | None = None,
        camera_transform=None,
        wifi_pca: int | None = None,
        wifi_pca_model=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["imu", "odom", "wifi", "camera"]
        self.windows = {**DEFAULT_WINDOWS, **(windows or {})}
        self.normalize = normalize
        self.stats = stats
        self.camera_transform = camera_transform
        self.wifi_pca = wifi_pca
        self._wifi_pca_model = wifi_pca_model

        # ------------------------------------------------------------------
        # Load all path data into memory (CSVs are small, ~4 MB total)
        # ------------------------------------------------------------------
        self._gt_rows: list[dict] = []   # each entry → one sample
        self._modality_data: dict[str, list[pd.DataFrame]] = {m: [] for m in self.modalities}
        self._path_indices: list[int] = []  # maps sample → internal path index

        self._wifi_rssi_cols: list[str] | None = None  # set on first wifi load

        for path_id in sorted(path_ids):
            pdir = self.data_dir / f"path_{path_id:02d}"
            if not pdir.exists():
                continue

            # Ground truth (defines timeline)
            gt_df = _load_csv(pdir / "ground_truth.csv")
            if len(gt_df) == 0:
                continue  # skip empty paths (e.g. path_00)

            gt_times = gt_df["sim_time"].values
            gt_xy = gt_df[GT_COLS].values.astype(np.float32)

            # Load modality dataframes
            mod_dfs: dict[str, pd.DataFrame] = {}
            for mod in self.modalities:
                csv_path = pdir / _CSV_FILENAMES.get(mod, f"{mod}.csv")
                if csv_path.exists():
                    mod_dfs[mod] = _load_csv(csv_path)
                else:
                    mod_dfs[mod] = pd.DataFrame()

            # Set wifi column names from first non-empty wifi dataframe
            if "wifi" in mod_dfs and self._wifi_rssi_cols is None:
                wifi_df = mod_dfs["wifi"]
                if len(wifi_df) > 0:
                    self._wifi_rssi_cols = _get_wifi_rssi_cols(wifi_df)

            # Store path-level data
            pidx = len(self._path_indices)  # not used per-sample, we store per path
            path_internal_idx = len(self._modality_data.get("imu", []))
            for mod in self.modalities:
                self._modality_data[mod].append(mod_dfs.get(mod, pd.DataFrame()))

            # Build per-sample index: one entry per GT timestamp
            for i in range(len(gt_times)):
                self._gt_rows.append({
                    "time": gt_times[i],
                    "target": gt_xy[i],
                    "path_idx": path_internal_idx,
                    "path_id": path_id,
                    "path_dir": str(pdir),
                })

        # Validate
        if len(self._gt_rows) == 0:
            raise ValueError(f"No data found for path_ids={path_ids} in {self.data_dir}")

        # Set wifi dim
        if "wifi" in self.modalities:
            if self._wifi_rssi_cols is not None:
                # Fit PCA if requested and no pre-fitted model was passed
                if self.wifi_pca and self._wifi_pca_model is None:
                    self._wifi_pca_model = self._fit_wifi_pca(self.wifi_pca)
                MODALITY_DIMS["wifi"] = self.wifi_pca or len(self._wifi_rssi_cols)
            else:
                raise ValueError("WiFi requested but no wifi_rssi_* columns found")

        # Compute normalization stats if needed
        if self.normalize and self.stats is None:
            self.stats = self._compute_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._gt_rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._gt_rows[idx]
        t = row["time"]
        pidx = row["path_idx"]

        sample = {
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "timestamp": torch.tensor(t, dtype=torch.float64),
            "path_id": row["path_id"],
        }

        for mod in self.modalities:
            if mod == "camera":
                sample[mod] = self._get_camera(pidx, t, row["path_dir"])
            else:
                sample[mod] = self._get_window(mod, pidx, t)

        return sample

    @property
    def feature_dims(self) -> dict[str, int | tuple]:
        """Return feature dimensionality per modality."""
        dims = {}
        for mod in self.modalities:
            dims[mod] = MODALITY_DIMS[mod]
        return dims

    @property
    def num_wifi_aps(self) -> int:
        return len(self._wifi_rssi_cols) if self._wifi_rssi_cols else 0

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _get_window(self, mod: str, path_idx: int, t: float) -> torch.Tensor:
        """Get a fixed-size window of the most recent observations before t."""
        df = self._modality_data[mod][path_idx]
        win = self.windows[mod]

        if mod == "wifi":
            cols = self._wifi_rssi_cols
        elif mod == "imu":
            cols = IMU_COLS
        elif mod == "odom":
            cols = ODOM_COLS
        else:
            raise ValueError(f"Unknown modality: {mod}")

        # Output feature dim (may differ from len(cols) if PCA is applied)
        n_out = self.wifi_pca if (mod == "wifi" and self._wifi_pca_model) else len(cols)

        if len(df) == 0:
            return torch.zeros(win, n_out, dtype=torch.float32)

        times = df["sim_time"].values
        # Find observations at or before time t
        mask = times <= t + 1e-6  # small epsilon for float comparison
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            # No data before t — return zeros (will be masked in training)
            return torch.zeros(win, n_out, dtype=torch.float32)

        # Take the last `win` observations
        start = max(0, valid_indices[-1] - win + 1)
        end = valid_indices[-1] + 1
        chunk = df.iloc[start:end][cols].values.astype(np.float32)

        # Apply WiFi PCA before padding (so padding stays as zeros)
        if mod == "wifi" and self._wifi_pca_model is not None:
            chunk = self._apply_wifi_pca(chunk)

        # Pad at the front if not enough history
        if len(chunk) < win:
            pad = np.zeros((win - len(chunk), n_out), dtype=np.float32)
            chunk = np.concatenate([pad, chunk], axis=0)

        tensor = torch.from_numpy(chunk)

        # Normalize
        if self.normalize and self.stats and mod in self.stats:
            mean = torch.tensor(self.stats[mod]["mean"], dtype=torch.float32)
            std = torch.tensor(self.stats[mod]["std"], dtype=torch.float32)
            tensor = (tensor - mean) / (std + 1e-8)

        return tensor

    def _get_camera(self, path_idx: int, t: float, path_dir: str) -> torch.Tensor:
        """Load most recent camera frame before time t."""
        from PIL import Image
        from torchvision import transforms as T

        df = self._modality_data["camera"][path_idx]
        h, w = 224, 224  # target size

        if len(df) == 0:
            return torch.zeros(3, h, w, dtype=torch.float32)

        times = df["sim_time"].values
        mask = times <= t + 1e-6
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return torch.zeros(3, h, w, dtype=torch.float32)

        row = df.iloc[valid_indices[-1]]
        rgb_path = Path(path_dir) / row["rgb_path"]

        if not rgb_path.exists():
            return torch.zeros(3, h, w, dtype=torch.float32)

        img = Image.open(rgb_path).convert("RGB")

        if self.camera_transform is not None:
            img = self.camera_transform(img)
        else:
            # Default: resize + to tensor + ImageNet normalize
            default_tf = T.Compose([
                T.Resize((h, w)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            img = default_tf(img)

        return img

    # ------------------------------------------------------------------
    # WiFi PCA
    # ------------------------------------------------------------------

    def _fit_wifi_pca(self, n_components: int):
        """Fit a PCA model on all WiFi RSSI data in this dataset."""
        from sklearn.decomposition import PCA

        all_vals = []
        for df in self._modality_data["wifi"]:
            if len(df) > 0 and all(c in df.columns for c in self._wifi_rssi_cols):
                all_vals.append(df[self._wifi_rssi_cols].values.astype(np.float32))
        if not all_vals:
            raise ValueError("No WiFi data available to fit PCA")
        arr = np.concatenate(all_vals, axis=0)
        # Replace NaN with -100 (no signal)
        arr = np.nan_to_num(arr, nan=-100.0)
        pca = PCA(n_components=n_components)
        pca.fit(arr)
        return pca

    def _apply_wifi_pca(self, data: np.ndarray) -> np.ndarray:
        """Project WiFi RSSI through the fitted PCA model."""
        data = np.nan_to_num(data, nan=-100.0)
        return self._wifi_pca_model.transform(data).astype(np.float32)

    # ------------------------------------------------------------------
    # Normalization statistics
    # ------------------------------------------------------------------

    def _compute_stats(self) -> dict:
        """Compute per-modality mean and std from all data in this dataset."""
        stats = {}
        for mod in self.modalities:
            if mod == "camera":
                continue  # camera uses ImageNet stats

            if mod == "wifi":
                cols = self._wifi_rssi_cols
            elif mod == "imu":
                cols = IMU_COLS
            elif mod == "odom":
                cols = ODOM_COLS
            else:
                continue

            # Concatenate all path dataframes for this modality
            all_vals = []
            for df in self._modality_data[mod]:
                if len(df) > 0 and all(c in df.columns for c in cols):
                    all_vals.append(df[cols].values.astype(np.float32))

            if all_vals:
                arr = np.concatenate(all_vals, axis=0)
                # If WiFi PCA is active, compute stats in PCA space
                if mod == "wifi" and self._wifi_pca_model is not None:
                    arr = self._apply_wifi_pca(arr)
                stats[mod] = {
                    "mean": arr.mean(axis=0),
                    "std": arr.std(axis=0),
                }
            else:
                n = (self.wifi_pca if mod == "wifi" and self._wifi_pca_model else len(cols))
                stats[mod] = {
                    "mean": np.zeros(n, dtype=np.float32),
                    "std": np.ones(n, dtype=np.float32),
                }

        return stats
