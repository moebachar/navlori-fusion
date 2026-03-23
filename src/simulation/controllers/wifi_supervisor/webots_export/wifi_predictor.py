
"""
NavLoRI WiFi Predictor for Webots
==================================
Lightweight predictor that loads pre-computed grid or GP models
to simulate WiFi RSSI at any (x, y) position.

Usage in Webots controller:
    from wifi_predictor import WiFiPredictor
    predictor = WiFiPredictor("path/to/webots_export")
    rssi = predictor.predict(x=2.5, y=-3.1)
    # rssi is a dict: {"AA:BB:CC:DD:EE:FF": -67.3, ...}
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


class WiFiPredictor:
    """Fast WiFi RSSI predictor using pre-computed grid interpolation."""

    def __init__(self, export_dir: str, method: str = "grid"):
        """
        Args:
            export_dir: Path to webots_export directory
            method: "grid" (fast, uses pre-computed grid) or "gp" (exact, slower)
        """
        self.export_dir = Path(export_dir)
        self.method = method

        with open(self.export_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.ap_names = self.metadata["ap_list"]
        self.norm = self.metadata["normalization"]

        if method == "grid":
            self._load_grid()
        elif method == "gp":
            self._load_gp()

    def _load_grid(self):
        """Load pre-computed RSSI grid for interpolation."""
        data = np.load(self.export_dir / "rssi_grid.npz", allow_pickle=True)
        self.x_grid = data["x_grid"]
        self.y_grid = data["y_grid"]
        self.rssi_mean = data["rssi_mean"]   # (ny, nx, n_aps)
        self.rssi_std = data["rssi_std"]

        # Build interpolators for each AP
        self.interpolators_mean = []
        self.interpolators_std = []
        for i in range(len(self.ap_names)):
            interp_mean = RegularGridInterpolator(
                (self.y_grid, self.x_grid),
                self.rssi_mean[:, :, i],
                method="linear",
                bounds_error=False,
                fill_value=-200.0,
            )
            interp_std = RegularGridInterpolator(
                (self.y_grid, self.x_grid),
                self.rssi_std[:, :, i],
                method="linear",
                bounds_error=False,
                fill_value=10.0,
            )
            self.interpolators_mean.append(interp_mean)
            self.interpolators_std.append(interp_std)

    def _load_gp(self):
        """Load GP models for exact prediction (requires gpytorch)."""
        import pickle
        with open(self.export_dir / "gp_models.pkl", "rb") as f:
            self.gp_data = pickle.load(f)

    def predict(self, x: float, y: float, add_noise: bool = True) -> dict:
        """
        Predict WiFi RSSI at position (x, y).

        Args:
            x, y: Robot position in meters (original coordinate system)
            add_noise: If True, add Gaussian noise based on predicted uncertainty

        Returns:
            dict mapping AP MAC address → predicted RSSI (dBm)
        """
        if self.method == "grid":
            return self._predict_grid(x, y, add_noise)
        else:
            return self._predict_gp(x, y)

    def _predict_grid(self, x: float, y: float, add_noise: bool) -> dict:
        """Fast prediction via grid interpolation (~0.1ms)."""
        result = {}
        point = np.array([[y, x]])  # RegularGridInterpolator expects (y, x) order

        for i, ap_name in enumerate(self.ap_names):
            mean_rssi = float(self.interpolators_mean[i](point)[0])
            if mean_rssi <= -150:  # Out of bounds
                continue

            if add_noise:
                std_rssi = float(self.interpolators_std[i](point)[0])
                mean_rssi += np.random.normal(0, std_rssi)

            # Clamp to realistic RSSI range
            result[ap_name] = float(np.clip(mean_rssi, -100, -20))

        return result

    def predict_batch(self, positions: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Batch prediction for multiple positions.

        Args:
            positions: (N, 2) array of (x, y) positions
            add_noise: Add realistic noise

        Returns:
            (N, num_aps) array of RSSI values
        """
        N = len(positions)
        rssi = np.full((N, len(self.ap_names)), -200.0)
        points = np.column_stack([positions[:, 1], positions[:, 0]])  # (y, x) order

        for i in range(len(self.ap_names)):
            rssi[:, i] = self.interpolators_mean[i](points)
            if add_noise:
                std = self.interpolators_std[i](points)
                rssi[:, i] += np.random.normal(0, std)

        rssi = np.clip(rssi, -100, -20)
        rssi[rssi <= -150] = -200  # Mark out-of-bounds as missing
        return rssi

    def get_ap_names(self) -> list:
        return self.ap_names
