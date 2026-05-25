"""DPVO full-pipeline encoder.

Trains a small MLP head over **upstream-DPVO hidden states** that have been
pre-extracted offline. The heavy lifting (running SLAM end-to-end on each
path) happens in ``scripts/extract_dpvo_features.py`` inside a docker
container — the result is one ``dpvo_features.pt`` per path, shaped
``(N_camera_frames, 384)``: the per-frame mean of the patch-level ``imap``
context features that DPVO's update operator consumes.

At training time this encoder consumes a short window of those cached
features and projects them to the project-wide ``embed_dim`` (default 128).
At inference time the same code path applies — only the upstream feature
producer is swapped from "offline cache" to "live DPVO instance" (see
``DPVOOnlineRunner`` in the docs).

Input:  (batch, window=4, 384) — pre-extracted DPVO ``imap`` pool per frame
Output: (batch, embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder

DPVO_DIM = 384


class DPVOFullEncoder(BaseEncoder):
    """MLP head over pre-extracted DPVO hidden states.

    Parameters
    ----------
    in_dim : int
        Dim of the cached DPVO feature (DPVO's ``DIM`` = 384).
    embed_dim : int
        Output dimension.
    hidden : int
        Width of the MLP hidden layer.
    pool : {"last", "mean"}
        Pool strategy across the time window. ``"last"`` keeps the most
        recent frame's feature (matches online-runner behavior); ``"mean"``
        averages across the window for a smoother token.
    dropout : float
        Dropout in the MLP head.
    """

    def __init__(
        self,
        in_dim: int = DPVO_DIM,
        embed_dim: int = 128,
        hidden: int = 256,
        pool: str = "last",
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim)
        if pool not in {"last", "mean"}:
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        self.in_dim = in_dim
        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window, in_dim) — pre-extracted DPVO imap pool, or
               (batch, in_dim) if a window has already been collapsed.

        Returns:
            (batch, embed_dim)
        """
        if x.dim() == 2:
            return self.head(x)
        if self.pool == "last":
            z = x[:, -1]
        else:
            z = x.mean(dim=1)
        return self.head(z)

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "vision_dpvo",
            "shape": (None, self.in_dim),  # window length is configurable
            "dtype": "float32",
        }
