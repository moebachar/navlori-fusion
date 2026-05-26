"""IMU encoder — 1D CNN on temporal windows.

Reference: Brossard et al., "AI-IMU Dead-Reckoning" — CNN learns IMU noise
covariance from data, outperforming hand-tuned EKFs. We adapt the approach
for embedding extraction rather than covariance estimation.

Architecture:
    1. Three 1D conv blocks with increasing channels (temporal feature extraction)
    2. Global average pooling (window → single vector)
    3. Linear projection to embed_dim

Input:  (batch, window=32, 9) — ~1s of accel_xyz, gyro_xyz, roll/pitch/yaw
Output: (batch, embed_dim)
"""

import torch
import torch.nn as nn

from .base import BaseEncoder


class ConvBlock1D(nn.Module):
    """Conv1D → BatchNorm → GELU → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IMUCNN(BaseEncoder):
    """1D CNN encoder for IMU temporal windows.

    Parameters
    ----------
    in_features : int
        Number of input features per timestep (default 9).
    embed_dim : int
        Output embedding dimension (default 128).
    channels : tuple
        Channel progression for conv blocks (default (32, 64, 128)).
    kernel_size : int
        Convolution kernel size (default 3).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 9,
        embed_dim: int = 128,
        channels: tuple[int, ...] = (32, 64, 128),
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
            x: (batch, window, in_features) — e.g. (B, 32, 9)

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
        """Per-encoder introspection (notebook §0). Returns the conv
        stack's final activations (pre-pooling) as the
        ``intermediate``."""
        import numpy as np
        x = raw_input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        self.eval()
        # Transpose for conv1d
        x_t = x.transpose(1, 2)
        conv_out = self.conv_stack(x_t)         # (B, C, T)
        encoded = self.head(conv_out.mean(dim=2))
        return {
            "raw": x.detach().cpu().numpy(),
            "preprocessed": x.detach().cpu().numpy(),
            "intermediate": conv_out.detach().cpu().numpy(),  # (B, channels[-1], window)
            "encoded": encoded.detach().cpu().numpy(),
            "description": (
                f"IMUCNN: 1D-CNN ({self.in_features}->{conv_out.shape[1]} ch) over "
                f"window={x.shape[1]} ({self.in_features}-channel input) -> global avg "
                f"pool -> {self.embed_dim}-d token."
            ),
        }

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "imu",
            "shape": (32, self.in_features),
            "dtype": "float32",
        }
