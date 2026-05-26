"""Odometry encoder — lightweight 1D CNN on temporal windows.

Same architecture as the IMU encoder but with fewer channels, matching the
lower dimensionality (7 features) and rate (~15 Hz) of wheel odometry.

Odometry is a relative sensor: it tracks displacement and heading changes
from encoder ticks. The CNN learns to extract motion patterns (turns,
straight segments, acceleration) from short temporal windows.

Input:  (batch, window=16, 7) — ~1s of odom_xy, theta, velocities, wheel speeds
Output: (batch, embed_dim)
"""

import torch
import torch.nn as nn

from .base import BaseEncoder
from .imu import ConvBlock1D


class OdomCNN(BaseEncoder):
    """1D CNN encoder for odometry temporal windows.

    Parameters
    ----------
    in_features : int
        Number of input features per timestep (default 7).
    embed_dim : int
        Output embedding dimension (default 128).
    channels : tuple
        Channel progression for conv blocks (default (16, 32, 64)).
    kernel_size : int
        Convolution kernel size (default 3).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 7,
        embed_dim: int = 128,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim)
        self.in_features = in_features

        # Build conv stack
        layers = []
        ch_in = in_features
        for ch_out in channels:
            layers.append(ConvBlock1D(ch_in, ch_out, kernel_size, dropout))
            ch_in = ch_out
        self.conv_stack = nn.Sequential(*layers)

        # Projection to embed_dim
        self.head = nn.Sequential(
            nn.Linear(channels[-1], embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window, in_features) — e.g. (B, 16, 7)

        Returns:
            (batch, embed_dim)
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (B, in_features, window)

        x = self.conv_stack(x)  # (B, channels[-1], window)

        # Global average pooling over time
        x = x.mean(dim=2)  # (B, channels[-1])

        x = self.head(x)  # (B, embed_dim)

        return x

    @torch.no_grad()
    def demo_forward(self, raw_input):
        """Per-encoder introspection (notebook §0). Returns conv stack
        activations as the ``intermediate``."""
        import numpy as np
        x = raw_input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        self.eval()
        x_t = x.transpose(1, 2)
        conv_out = self.conv_stack(x_t)
        encoded = self.head(conv_out.mean(dim=2))
        return {
            "raw": x.detach().cpu().numpy(),
            "preprocessed": x.detach().cpu().numpy(),
            "intermediate": conv_out.detach().cpu().numpy(),
            "encoded": encoded.detach().cpu().numpy(),
            "description": (
                f"OdomCNN: 1D-CNN over window={x.shape[1]} "
                f"({self.in_features}-channel input) -> "
                f"{conv_out.shape[1]} channels -> {self.embed_dim}-d token. "
                f"P-B Δ-features (RESULT_04 winner) recommended at the dataset stage."
            ),
        }

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "odom",
            "shape": (16, self.in_features),
            "dtype": "float32",
        }
