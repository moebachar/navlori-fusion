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

    @torch.no_grad()
    def demo_forward(self, raw_input):
        """Per-encoder introspection for the notebook §0 preprocessing demo.

        Returns a dict with keys ``raw`` / ``preprocessed`` /
        ``intermediate`` (anchor attention weights) / ``encoded`` /
        ``description``. ``raw_input`` may be a numpy array or torch
        tensor of shape (n_aps,) or (1, n_aps) or (B, 1, n_aps).
        """
        import numpy as np
        x = raw_input
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        x = x.to(next(self.parameters()).device)
        self.eval()
        # Step 1: anchor similarity (softmax weights = intermediate)
        if x.ndim == 3:
            x_sq = x.squeeze(1)
        else:
            x_sq = x
        sim = F.linear(x_sq, self.anchors)
        weights = F.softmax(sim / (self.temperature.abs() + 1e-6), dim=-1)
        # Step 2-3: token + head refinement (re-running forward path)
        encoded = self.forward(x)
        return {
            "raw": x_sq.detach().cpu().numpy(),
            "preprocessed": x_sq.detach().cpu().numpy(),
            "intermediate": weights.detach().cpu().numpy(),
            "encoded": encoded.detach().cpu().numpy(),
            "description": (
                f"Anchor2Vec: {x_sq.shape[1]} APs -> {self.n_anchors} learned anchors "
                f"(softmax-attention weights are the intermediate) -> "
                f"{self.embed_dim}-d token via weighted sum of anchor embeddings + MLP head."
            ),
        }

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "wifi",
            "shape": (1, self.n_aps),
            "dtype": "float32",
        }
