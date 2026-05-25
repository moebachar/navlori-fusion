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

# M4: world-frame IMU. Body-frame accel makes the same physical motion look
# different per heading, crippling the motion leg (autopsy Probe 5 / M4 test:
# body 12% skill -> world-only 52%). In "world" frame we rotate the horizontal
# accel by yaw and feed [ax_world, ay_world, gyro_x, gyro_y, gyro_z] (5 feats).
IMU_WORLD_N_FEATURES = 5


def _imu_to_world(chunk: np.ndarray) -> np.ndarray:
    """Body-frame IMU window (..., 9 IMU_COLS) -> world-frame (..., 5).

    Output cols: [ax_world, ay_world, gyro_x, gyro_y, gyro_z]. Horizontal
    accel is yaw-rotated into the world frame; vertical accel and the raw
    orientation degrees are dropped (they confused the model — M4 test).
    """
    ax, ay = chunk[..., 0], chunk[..., 1]
    yaw = np.deg2rad(chunk[..., 8])
    axw = np.cos(yaw) * ax - np.sin(yaw) * ay
    ayw = np.sin(yaw) * ax + np.cos(yaw) * ay
    return np.stack([axw, ayw, chunk[..., 3], chunk[..., 4], chunk[..., 5]], axis=-1)

ODOM_COLS = [
    "odom_theta_deg",
    "odom_linear_vel", "odom_angular_vel",
    "wheel_left_vel", "wheel_right_vel",
]  # 5 features — odom_x / odom_y removed: they are absolute wheel-odometry
# position estimates, i.e. the target leaking into the input. The encoder
# was reading position straight off them (audit finding A4, 2026-05-20).

GT_COLS = ["gt_x", "gt_y"]  # regression targets

# Per-frame DPVO ``imap`` pool dimensionality (matches dpvo.net.DIM).
DPVO_FEATURE_DIM = 384

# Default window sizes (number of past observations to include)
DEFAULT_WINDOWS = {
    "imu": 32,         # ~1 s at 31 Hz
    "odom": 16,        # ~1 s at 15 Hz
    "wifi": 1,         # single scan (sparse modality)
    "camera": 1,       # single frame
    "vision_dpvo": 4,  # 4 most-recent DPVO hidden states
}

# Features dimensions per modality
MODALITY_DIMS = {
    "imu": len(IMU_COLS),                  # 9
    "odom": len(ODOM_COLS),                # 5 (after dropping odom_x, odom_y)
    "wifi": None,                          # set dynamically from CSV header
    "camera": (3, 480, 640),               # after resize + RGB conversion
    "vision_dpvo": DPVO_FEATURE_DIM,       # pre-extracted DPVO imap pool
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
    camera_stride : spacing between camera frames in a window. ``stride=1``
        (default) returns consecutive frames; ``stride=5`` returns every
        fifth frame so a pair spans ~1 s instead of ~0.2 s. Used by motion
        encoders (DPVO) that need visible inter-frame displacement.
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
        camera_stride: int = 1,
        wifi_pca: int | None = None,
        wifi_pca_model=None,
        wifi_norm: str = "whiten",
        wifi_max_stale_s: float | None = None,
        imu_frame: str = "body",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["imu", "odom", "wifi", "camera"]
        self.windows = {**DEFAULT_WINDOWS, **(windows or {})}
        self.normalize = normalize
        self.stats = stats
        self.camera_transform = camera_transform
        self.camera_stride = max(1, int(camera_stride))
        # WiFi encoding mode (autopsy Probe 4):
        #   "whiten" — PCA then per-component z-score (legacy; DESTROYS signal:
        #              z-scoring PCA components amplifies noise to signal scale)
        #   "raw"    — -100 fill + fixed affine scale, NO PCA, NO z-score
        #              (metric-preserving; ~4x better WiFi-kNN on real data)
        self.wifi_norm = wifi_norm
        # "raw" mode ignores PCA entirely — the whitening was the problem, and
        # PCA-rotation alone bought nothing (Probe 4 E vs B).
        self.wifi_pca = None if wifi_norm == "raw" else wifi_pca
        self._wifi_pca_model = None if wifi_norm == "raw" else wifi_pca_model
        # Max seconds a carried-forward WiFi scan is treated as a live fix
        # (M2). None = no cap (legacy behavior).
        self.wifi_max_stale_s = wifi_max_stale_s
        # IMU frame (M4): "body" (raw 9 cols) or "world" (5 world-frame feats).
        self.imu_frame = imu_frame

        # ------------------------------------------------------------------
        # Load all path data into memory (CSVs are small, ~4 MB total)
        # ------------------------------------------------------------------
        self._gt_rows: list[dict] = []   # each entry → one sample
        # Skip CSV bookkeeping for non-tabular modalities — they don't have a CSV.
        _csv_mods = [m for m in self.modalities if m in _CSV_FILENAMES]
        self._modality_data: dict[str, list[pd.DataFrame]] = {m: [] for m in _csv_mods}
        self._path_indices: list[int] = []  # maps sample → internal path index

        self._wifi_rssi_cols: list[str] | None = None  # set on first wifi load
        # path_idx → DPVO feature payload (only populated if vision_dpvo is requested).
        self._dpvo_features: list[dict | None] = []

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
            for mod in _csv_mods:
                csv_path = pdir / _CSV_FILENAMES.get(mod, f"{mod}.csv")
                if csv_path.exists():
                    mod_dfs[mod] = _load_csv(csv_path)
                else:
                    mod_dfs[mod] = pd.DataFrame()

            # Load DPVO features if requested. Path may legitimately lack the
            # file (extraction not yet run for that path) — sample's window
            # falls back to zeros.
            if "vision_dpvo" in self.modalities:
                feat_path = pdir / "dpvo_features.pt"
                if feat_path.exists():
                    payload = torch.load(feat_path, weights_only=True, map_location="cpu")
                    self._dpvo_features.append(payload)
                else:
                    print(f"[FusionDataset] WARNING: missing {feat_path}; using zeros.")
                    self._dpvo_features.append(None)

            # Set wifi column names from first non-empty wifi dataframe
            if "wifi" in mod_dfs and self._wifi_rssi_cols is None:
                wifi_df = mod_dfs["wifi"]
                if len(wifi_df) > 0:
                    self._wifi_rssi_cols = _get_wifi_rssi_cols(wifi_df)

            # Store path-level data. path_internal_idx is the index into the
            # per-modality dataframe lists. Use any CSV-backed modality's
            # current length (they grow in lockstep below). Prior to this
            # patch we keyed off 'imu' specifically and broke silently when
            # imu wasn't loaded.
            path_internal_idx = (
                len(self._modality_data[_csv_mods[0]]) if _csv_mods
                else len(self._dpvo_features) - (1 if "vision_dpvo" in self.modalities else 0)
            )
            for mod in _csv_mods:
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
        # Precompute all non-camera windows into a single stacked tensor
        # per modality for O(1) __getitem__ (pure tensor slice, no Python loop).
        # ------------------------------------------------------------------
        self._cache: dict[str, torch.Tensor] = {}
        tabular_mods = [m for m in self.modalities if m != "camera"]
        for mod in tabular_mods:
            if mod == "vision_dpvo":
                windows = [self._get_dpvo_window(r["path_idx"], r["time"])
                           for r in self._gt_rows]
            else:
                windows = [self._get_window(mod, r["path_idx"], r["time"])
                           for r in self._gt_rows]
            self._cache[mod] = torch.stack(windows)  # (N, window, features)

        # Pre-stack targets and timestamps
        self._targets = torch.tensor(
            np.array([r["target"] for r in self._gt_rows], dtype=np.float32),
        )
        self._timestamps = torch.tensor(
            np.array([r["time"] for r in self._gt_rows], dtype=np.float64),
        )

        # Free raw dataframes — no longer needed
        for mod in tabular_mods:
            if mod in self._modality_data:
                self._modality_data[mod] = []
        # DPVO payloads are large (~MB per path); release after caching.
        if "vision_dpvo" in self.modalities:
            self._dpvo_features = [None] * len(self._dpvo_features)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._gt_rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._gt_rows[idx]

        sample = {
            "target": self._targets[idx],
            "timestamp": self._timestamps[idx],
            "path_id": row["path_id"],
        }

        for mod in self.modalities:
            if mod == "camera":
                sample[mod] = self._get_camera(
                    row["path_idx"], row["time"], row["path_dir"]
                )
            else:
                sample[mod] = self._cache[mod][idx]

        return sample

    def get_pair_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample motion-supervision tensors.

        For each sample ``i`` whose camera pair spans csv indices
        ``[end_i - camera_stride, end_i]``, returns:

        * ``target_prev`` : ``(N, 2)`` float — GT ``(x, y)`` at the time of the
          previous frame in the pair.
        * ``delta`` : ``(N, 2)`` float — ``self._targets[i] - target_prev[i]``,
          i.e. the position change supervising a motion encoder.
        * ``valid`` : ``(N,)`` bool — True iff a previous frame existed for
          this sample and a matching GT row was found.

        Invalid rows (typically the first ``camera_stride`` samples per path)
        have zeros in ``target_prev`` / ``delta`` and ``valid=False``.
        """
        assert "camera" in self.modalities, \
            "get_pair_targets requires the camera modality"
        n = len(self._gt_rows)
        target_prev = torch.zeros((n, 2), dtype=torch.float32)
        valid = torch.zeros(n, dtype=torch.bool)

        # Per-path (sorted_times, sorted_xy) for fast nearest-time lookup.
        by_path: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for r in self._gt_rows:
            pidx = r["path_idx"]
            t_arr, xy_list = by_path.setdefault(pidx, ([], []))
            t_arr.append(float(r["time"]))
            xy_list.append(r["target"])
        by_path = {p: (np.asarray(t), np.stack(xy))
                   for p, (t, xy) in by_path.items()}

        stride = self.camera_stride
        for i, row in enumerate(self._gt_rows):
            pidx = row["path_idx"]
            cam_df = self._modality_data["camera"][pidx]
            if len(cam_df) == 0:
                continue
            cam_t = cam_df["sim_time"].values
            mask = cam_t <= row["time"] + 1e-6
            valid_idx = np.where(mask)[0]
            if len(valid_idx) == 0:
                continue
            end = int(valid_idx[-1])
            prev_idx = end - stride
            if prev_idx < 0:
                continue
            t_prev = float(cam_t[prev_idx])
            gt_t_arr, gt_xy = by_path[pidx]
            j = int(np.argmin(np.abs(gt_t_arr - t_prev)))
            # Tolerance: GT @ ~10 Hz → 0.1 s spacing; 0.5 s is plenty of slack
            if abs(gt_t_arr[j] - t_prev) > 0.5:
                continue
            target_prev[i] = torch.from_numpy(gt_xy[j].astype(np.float32))
            valid[i] = True

        delta = self._targets - target_prev
        delta[~valid] = 0.0
        return target_prev, delta, valid

    def get_tensors(self, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (X, y) stacked tensors for a tabular modality.

        Used by EncoderTrainer to build a TensorDataset directly, bypassing
        __getitem__ overhead entirely.

        Returns:
            X: (N, window, features) float32 tensor
            y: (N, 2) float32 tensor of (x, y) positions
        """
        if modality not in self._cache:
            raise KeyError(f"Modality '{modality}' not in cache. Camera requires __getitem__.")
        return self._cache[modality], self._targets

    def get_targets(self, mode: str = "position",
                    lookback_s: float = 1.0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample regression target + validity mask.

        Parameters
        ----------
        mode
            * ``"position"``  — returns the absolute ``(x, y)`` target for
              each sample and an all-True validity mask. Same as
              ``self._targets``; preserved for callers that want a uniform
              API.
            * ``"displacement"`` — returns
              ``delta_i = self._targets[i] - gt(t_i - lookback_s)``
              where the lookback finds the **nearest GT row in the same
              path** whose timestamp is at or before ``t_i - lookback_s``.
              If no such row exists (sample falls within the first
              ``lookback_s`` of its path), the displacement is set to zero
              and the validity mask is False — the trainer should mask
              the loss on those samples.
        lookback_s
            Seconds to look back for the displacement reference. Match the
            encoder's effective temporal window (e.g. 1.0 s for IMU/Odom,
            ``camera_stride * (1 / camera_rate)`` for DPVO motion pairs).

        Returns
        -------
        target : ``(N, 2)`` float32
        valid  : ``(N,)`` bool
        """
        if mode == "position":
            return self._targets, torch.ones(len(self._targets), dtype=torch.bool)
        if mode != "displacement":
            raise ValueError(f"mode must be 'position' or 'displacement', got {mode!r}")

        n = len(self._gt_rows)
        delta = torch.zeros((n, 2), dtype=torch.float32)
        valid = torch.zeros(n, dtype=torch.bool)
        times = self._timestamps.cpu().numpy()
        targets = self._targets.cpu().numpy()

        # Bucket sample indices by path; lookback is in-path only.
        by_path: dict[int, list[int]] = {}
        for i, r in enumerate(self._gt_rows):
            by_path.setdefault(r["path_id"], []).append(i)

        for pid, idx_list in by_path.items():
            idx_arr = np.asarray(idx_list)
            t_path = times[idx_arr]
            xy_path = targets[idx_arr]
            order = np.argsort(t_path)
            t_sorted = t_path[order]
            xy_sorted = xy_path[order]
            idx_sorted = idx_arr[order]

            # For each sample, find the latest in-path sample whose time
            # is <= t_i - lookback_s. searchsorted gives the insertion
            # index; the previous index (if any) is our reference.
            t_query = t_sorted - lookback_s
            ref_pos = np.searchsorted(t_sorted, t_query, side="right") - 1
            ok = ref_pos >= 0
            for k in np.where(ok)[0]:
                ref = ref_pos[k]
                delta[idx_sorted[k]] = torch.from_numpy(
                    (xy_sorted[k] - xy_sorted[ref]).astype(np.float32))
                valid[idx_sorted[k]] = True

        return delta, valid

    @property
    def feature_dims(self) -> dict[str, int | tuple]:
        """Return feature dimensionality per modality."""
        dims = {}
        for mod in self.modalities:
            if mod == "imu" and self.imu_frame == "world":
                dims[mod] = IMU_WORLD_N_FEATURES   # M4: world-frame (5)
            else:
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

        # Output feature dim (differs from len(cols) for WiFi PCA or world IMU)
        if mod == "wifi" and self._wifi_pca_model:
            n_out = self.wifi_pca
        elif mod == "imu" and self.imu_frame == "world":
            n_out = IMU_WORLD_N_FEATURES
        else:
            n_out = len(cols)

        if len(df) == 0:
            return torch.zeros(win, n_out, dtype=torch.float32)

        times = df["sim_time"].values
        # Find observations at or before time t
        mask = times <= t + 1e-6  # small epsilon for float comparison
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            # No data before t — return zeros (will be masked in training)
            return torch.zeros(win, n_out, dtype=torch.float32)

        # WiFi staleness cap (M2): if the most recent scan is older than
        # wifi_max_stale_s, treat WiFi as ABSENT (zeros -> unavailable) rather
        # than feeding an ancient scan as a live fix. Without this, one scan
        # is carried forward for up to ~2748 samples / 275 s (autopsy Probe 2),
        # poisoning training (one input, thousands of different targets) and
        # making the model trust a fix that points where you were minutes ago.
        if (mod == "wifi" and self.wifi_max_stale_s is not None
                and (t - times[valid_indices[-1]]) > self.wifi_max_stale_s):
            return torch.zeros(win, n_out, dtype=torch.float32)

        # Take the last `win` observations
        start = max(0, valid_indices[-1] - win + 1)
        end = valid_indices[-1] + 1
        chunk = df.iloc[start:end][cols].values.astype(np.float32)

        if mod == "wifi" and self.wifi_norm == "raw":
            # Non-whitening encoding (autopsy Probe 4): -100 fill + a single
            # fixed affine, NO PCA, NO per-AP z-score. Maps [-100,-30] -> [0,0.7];
            # a missing/-100 AP -> 0, which also matches the front-pad zeros
            # (so "no signal" is one consistent value). Distance-preserving.
            chunk = np.nan_to_num(chunk, nan=-100.0)
            chunk = (chunk + 100.0) / 100.0
        elif mod == "wifi" and self._wifi_pca_model is not None:
            # Legacy whiten path: PCA here, per-component z-score applied below.
            chunk = self._apply_wifi_pca(chunk)
        elif mod == "imu" and self.imu_frame == "world":
            # M4: yaw-rotate horizontal accel into the world frame (9 -> 5).
            chunk = _imu_to_world(chunk)

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

    def _get_dpvo_window(self, path_idx: int, t: float) -> torch.Tensor:
        """Window of the most recent DPVO ``imap`` pools at or before ``t``.

        Output shape: ``(windows['vision_dpvo'], DPVO_FEATURE_DIM)``. Uses the
        same ``camera_stride`` as raw camera windows so the temporal gap
        between successive features matches what the encoder saw on disk.
        """
        win = self.windows.get("vision_dpvo", DEFAULT_WINDOWS["vision_dpvo"])
        payload = (self._dpvo_features[path_idx]
                   if path_idx < len(self._dpvo_features) else None)
        if payload is None:
            return torch.zeros(win, DPVO_FEATURE_DIM, dtype=torch.float32)

        times = payload["sim_time"].numpy() if torch.is_tensor(payload["sim_time"]) \
                else np.asarray(payload["sim_time"])
        feats: torch.Tensor = payload["features"]                # (N_frames, DIM)
        mask = times <= t + 1e-6
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return torch.zeros(win, DPVO_FEATURE_DIM, dtype=torch.float32)

        end = int(valid[-1])
        raw = [end - k * self.camera_stride for k in reversed(range(win))]
        keep = [i for i in raw if i >= 0]
        chunk = feats[keep].to(torch.float32)                    # (k, DIM)
        if chunk.shape[0] < win:
            pad = torch.zeros(win - chunk.shape[0], DPVO_FEATURE_DIM,
                              dtype=torch.float32)
            chunk = torch.cat([pad, chunk], dim=0)
        return chunk

    def _get_camera(self, path_idx: int, t: float, path_dir: str) -> torch.Tensor:
        """Load camera frame(s) ending at time t.

        Returns ``(3, H, W)`` when ``windows['camera'] == 1`` (default — keeps
        existing single-frame encoders working) and ``(W, 3, H, W)`` when
        ``windows['camera'] > 1`` (used by motion encoders that need a pair
        / short clip of consecutive frames). Pre-pads with zeros if there
        aren't enough frames yet.
        """
        from PIL import Image
        from torchvision import transforms as T

        df = self._modality_data["camera"][path_idx]
        h, w = 480, 640  # target size
        win = self.windows.get("camera", 1)

        def _zeros() -> torch.Tensor:
            return (torch.zeros(3, h, w, dtype=torch.float32) if win == 1
                    else torch.zeros(win, 3, h, w, dtype=torch.float32))

        if len(df) == 0:
            return _zeros()

        times = df["sim_time"].values
        mask = times <= t + 1e-6
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return _zeros()

        if self.camera_transform is not None:
            tf = self.camera_transform
        else:
            tf = T.Compose([
                T.Resize((h, w)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        def _load_one(row_idx: int) -> torch.Tensor:
            row = df.iloc[row_idx]
            rgb_path = Path(path_dir) / row["rgb_path"]
            if not rgb_path.exists():
                return torch.zeros(3, h, w, dtype=torch.float32)
            return tf(Image.open(rgb_path).convert("RGB"))

        if win == 1:
            return _load_one(int(valid_indices[-1]))

        end = int(valid_indices[-1])
        # Frames at indices [end - (win-1)*stride, ..., end - stride, end].
        # Stride > 1 widens the temporal gap between frames in the window
        # without changing the camera sample rate.
        raw_indices = [end - k * self.camera_stride for k in reversed(range(win))]
        keep = [i for i in raw_indices if i >= 0]
        frames = [_load_one(i) for i in keep]
        # Pre-pad with zeros if fewer frames are available than the window.
        if len(frames) < win:
            pad = [torch.zeros(3, h, w, dtype=torch.float32)
                   for _ in range(win - len(frames))]
            frames = pad + frames
        return torch.stack(frames, dim=0)  # (win, 3, H, W)

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
            if mod == "vision_dpvo":
                continue  # DPVO features are normalized inside the encoder (LayerNorm)
            if mod == "wifi" and self.wifi_norm == "raw":
                continue  # raw WiFi is fixed-affine scaled in _get_window; no z-score

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
                # M4: stats must be computed on the world-frame features the
                # encoder actually sees, not the raw 9 body-frame cols.
                if mod == "imu" and self.imu_frame == "world":
                    arr = _imu_to_world(arr)
                stats[mod] = {
                    "mean": arr.mean(axis=0),
                    "std": arr.std(axis=0),
                }
            else:
                if mod == "wifi" and self._wifi_pca_model:
                    n = self.wifi_pca
                elif mod == "imu" and self.imu_frame == "world":
                    n = IMU_WORLD_N_FEATURES
                else:
                    n = len(cols)
                stats[mod] = {
                    "mean": np.zeros(n, dtype=np.float32),
                    "std": np.ones(n, dtype=np.float32),
                }

        return stats
