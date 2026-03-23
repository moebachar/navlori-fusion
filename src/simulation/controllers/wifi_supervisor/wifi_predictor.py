"""
NavLoRI WiFi Predictor for Webots
==================================
Lightweight predictor that loads pre-computed GPR grid
to simulate WiFi RSSI at any (x, y) position.

Loads rssi_grid.npz and uses bilinear interpolation for ~0.1ms predictions.

Usage:
    from wifi_predictor import WiFiPredictor
    predictor = WiFiPredictor("/path/to/webots_export")
    rssi = predictor.predict(x=2.5, y=-3.1)
    # rssi is a dict: {"AA:BB:CC:DD:EE:FF": -67.3, ...}
"""

import json
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


# The 3 APs that diverged during training (near-zero lengthscales, MAE > 100)
DIVERGED_APS = {
    "34:15:93:5c:d3:21",
    "34:15:93:5c:d3:24",
    "34:15:93:9c:69:c2",
}


class WiFiPredictor:
    """Fast WiFi RSSI predictor using pre-computed grid interpolation."""

    def __init__(self, export_dir: str):
        """
        Args:
            export_dir: Path to webots_export directory containing rssi_grid.npz
        """
        self.export_dir = Path(export_dir)

        # Load metadata
        with open(self.export_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.norm = self.metadata.get("normalization", {})

        # Load grid
        data = np.load(self.export_dir / "rssi_grid.npz", allow_pickle=True)
        self.x_grid = data["x_grid"]
        self.y_grid = data["y_grid"]
        all_ap_names = list(data["ap_names"])
        rssi_mean = data["rssi_mean"]    # (ny, nx, n_aps)
        rssi_std = data["rssi_std"]

        # Filter out diverged APs
        valid_idx = [i for i, ap in enumerate(all_ap_names) if ap not in DIVERGED_APS]
        self.ap_names = [all_ap_names[i] for i in valid_idx]
        self.rssi_mean = rssi_mean[:, :, valid_idx]
        self.rssi_std = rssi_std[:, :, valid_idx]
        self.num_aps = len(self.ap_names)

        # Spatial bounds
        self.x_min = float(data.get("x_min", self.x_grid[0]))
        self.x_max = float(data.get("x_max", self.x_grid[-1]))
        self.y_min = float(data.get("y_min", self.y_grid[0]))
        self.y_max = float(data.get("y_max", self.y_grid[-1]))

        # Build interpolators
        self._interp_mean = []
        self._interp_std = []
        for i in range(self.num_aps):
            self._interp_mean.append(RegularGridInterpolator(
                (self.y_grid, self.x_grid),
                self.rssi_mean[:, :, i],
                method="linear",
                bounds_error=False,
                fill_value=-200.0,
            ))
            self._interp_std.append(RegularGridInterpolator(
                (self.y_grid, self.x_grid),
                self.rssi_std[:, :, i],
                method="linear",
                bounds_error=False,
                fill_value=10.0,
            ))

        print(f"[WiFiPredictor] Loaded {self.num_aps} APs "
              f"(excluded {len(DIVERGED_APS)} diverged), "
              f"grid {len(self.x_grid)}x{len(self.y_grid)}, "
              f"bounds X[{self.x_min:.1f}, {self.x_max:.1f}] "
              f"Y[{self.y_min:.1f}, {self.y_max:.1f}]")

    def predict(self, x: float, y: float, add_noise: bool = True) -> dict:
        """
        Predict WiFi RSSI at position (x, y) in meters.

        Args:
            x, y: Robot position in the original coordinate system
            add_noise: Add Gaussian noise based on GP uncertainty

        Returns:
            dict: AP MAC → RSSI in dBm (only APs with signal > -100 dBm)
        """
        point = np.array([[y, x]])  # interpolator expects (y, x)
        result = {}

        for i, ap_name in enumerate(self.ap_names):
            mean_rssi = float(self._interp_mean[i](point)[0])

            if mean_rssi <= -150:  # out of bounds / no coverage
                continue

            if add_noise:
                std_rssi = float(self._interp_std[i](point)[0])
                mean_rssi += np.random.normal(0, min(std_rssi, 8.0))

            result[ap_name] = float(np.clip(mean_rssi, -100, -20))

        return result

    def predict_array(self, x: float, y: float, add_noise: bool = True) -> np.ndarray:
        """
        Same as predict() but returns a fixed-length array for all APs.
        Missing APs get -200.

        Returns:
            (num_aps,) numpy array of RSSI values
        """
        point = np.array([[y, x]])
        rssi = np.full(self.num_aps, -200.0)

        for i in range(self.num_aps):
            val = float(self._interp_mean[i](point)[0])
            if val > -150:
                if add_noise:
                    std = float(self._interp_std[i](point)[0])
                    val += np.random.normal(0, min(std, 8.0))
                rssi[i] = np.clip(val, -100, -20)

        return rssi

    def get_top_n(self, x: float, y: float, n: int = 15,
                  add_noise: bool = True) -> list:
        """
        Get the N strongest APs at position (x, y).

        Returns:
            List of (ap_name, rssi_dbm) tuples, sorted strongest first
        """
        scan = self.predict(x, y, add_noise=add_noise)
        sorted_aps = sorted(scan.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_aps[:n]

    def is_in_bounds(self, x: float, y: float, margin: float = 0.5) -> bool:
        """Check if position is within the mapped area."""
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin)

    def get_bounds(self) -> dict:
        """Return spatial bounds of the WiFi map."""
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "y_min": self.y_min, "y_max": self.y_max,
        }