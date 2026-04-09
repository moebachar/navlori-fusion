"""WiFi RSSI encoder — Anchor2Vec (AaTs-inspired).

Reference: Yin et al., "All-embracing Transformer (AaTs)" — Anchor2Vec tokenization
projects RSS fingerprints through k learned anchor embeddings into a d-dimensional
token, avoiding the artificial 2D reshaping that CNNs require.

Architecture:
    1. Anchor projection: RSSI (n_aps,) → similarity to k anchors → (k,)
    2. Anchor embedding: weighted sum of k anchor vectors → (embed_dim,)
    3. MLP head: refine with LayerNorm + nonlinearity

Input:  (batch, 1, n_aps) — single WiFi scan (window=1)
Output: (batch, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder


class Anchor2Vec(BaseEncoder):
    """WiFi RSSI encoder using learned anchor projections.

    Parameters
    ----------
    n_aps : int
        Number of access points (input features per scan).
    embed_dim : int
        Output embedding dimension.
    n_anchors : int
        Number of learned anchor points (k in the paper). Default 64.
    dropout : float
        Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        n_aps: int,
        embed_dim: int = 128,
        n_anchors: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim)
        self.n_aps = n_aps
        self.n_anchors = n_anchors

        # Anchor projection: each anchor is a learnable n_aps-dim vector
        # Similarity between input RSSI and each anchor gives a soft assignment
        self.anchors = nn.Parameter(torch.randn(n_anchors, n_aps) * 0.02)

        # Temperature for softmax over anchor similarities
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Each anchor has a learnable embedding vector
        self.anchor_embeddings = nn.Parameter(torch.randn(n_anchors, embed_dim) * 0.02)

        # MLP refinement head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, window=1, n_aps) — WiFi RSSI scan

        Returns:
            (batch, embed_dim)
        """
        # Squeeze window dim: (batch, 1, n_aps) → (batch, n_aps)
        if x.ndim == 3:
            x = x.squeeze(1)

        # Step 1: Compute similarity to each anchor
        # x: (batch, n_aps), anchors: (k, n_aps) → sim: (batch, k)
        sim = F.linear(x, self.anchors)  # (batch, k)
        weights = F.softmax(sim / (self.temperature.abs() + 1e-6), dim=-1)  # (batch, k)

        # Step 2: Weighted sum of anchor embeddings
        # weights: (batch, k), anchor_embeddings: (k, embed_dim) → (batch, embed_dim)
        token = torch.matmul(weights, self.anchor_embeddings)  # (batch, embed_dim)

        # Step 3: Refine
        token = token + self.head(token)  # residual connection

        return token

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "wifi",
            "shape": (1, self.n_aps),
            "dtype": "float32",
        }
