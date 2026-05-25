"""WiFi set-transformer encoder (sparse-observed forward).

Per-AP / per-BSSID token transformer with a CLS readout, per
[Lazaro et al. 2025, arXiv:2506.00656]. Each AP gets its own learnable
embedding; per-scan tokens are (bssid_embed, rssi_proj) projected to the
shared embedding dim. A 2-layer ``nn.TransformerEncoder`` (batch_first)
runs self-attention over CLS + the *observed* APs only.

The first pass at this encoder (iter_05) built all 1419 tokens per scan
and masked out the ~92% unobserved positions. The mask suppressed
their softmax contribution but the tokens still consumed full
``O(N²)`` attention activations and OOM'd at every batch size on the
8 GB project GPU. This version (iter_06, PLAN_06) selects each row's
top-`max_observed_per_scan` observed APs via ``torch.gather`` and runs
attention only over the kept tokens (~127 mean / 256 cap → ~120×
cheaper than dense).

Input contract matches :class:`Anchor2Vec`:
    forward(x)         x.shape == (batch, window=1, n_aps) or (batch, n_aps)
    output             (batch, embed_dim)
    input_spec         {"modality": "wifi", "shape": (1, n_aps), "dtype": "float32"}

Input values are assumed to be in ``[0, 1]`` (the dataset's ``wifi_norm:
raw`` path scales ``rssi`` by ``(rssi + 100) / 100`` and treats missing
APs as ``0``). The mask threshold ``epsilon`` distinguishes "AP unseen"
(value ~0) from "AP seen with very weak RSSI" (value > epsilon).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder


class WiFiSetTransformer(BaseEncoder):
    """Per-AP transformer encoder over the WiFi RSSI scan.

    Parameters
    ----------
    n_aps : int
        Number of access points (BSSIDs in the dataset vocabulary).
    embed_dim : int
        Output embedding dim (shared with the fusion model).
    bssid_dim : int
        Dim of the per-BSSID learnable embedding (default 32).
    n_layers : int
        Transformer encoder depth (default 2).
    n_heads : int
        Attention heads (default 4).
    ff_mult : int
        Feedforward hidden multiplier (default 4 -> ``4 * embed_dim``).
    dropout : float
        Transformer dropout (default 0.1).
    epsilon : float
        Mask threshold; positions with ``x <= epsilon`` are treated as
        missing APs and masked out of attention. Default 0.005 (≈ rssi
        ≤ -99.5 in the raw scaling).
    max_observed_per_scan : int
        Hard cap on the number of observed-AP tokens kept per scan
        (defensive against the rare scan that sees > 256 BSSIDs). The
        weakest-RSSI observed APs are dropped first. Default 256.
    """

    def __init__(
        self,
        n_aps: int,
        embed_dim: int = 128,
        bssid_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        epsilon: float = 0.005,
        max_observed_per_scan: int = 256,
    ):
        super().__init__(embed_dim)
        self.n_aps = n_aps
        self.epsilon = float(epsilon)
        self.max_observed_per_scan = int(max_observed_per_scan)

        # Per-BSSID learnable embedding table.
        self.bssid_embed = nn.Embedding(n_aps, bssid_dim)
        nn.init.normal_(self.bssid_embed.weight, std=0.02)

        # Project (bssid_embed, rssi_scalar) -> embed_dim per-AP token.
        self.token_proj = nn.Sequential(
            nn.Linear(bssid_dim + 1, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Unmaskable CLS token used for readout (never masked → softmax safe).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Stock PyTorch transformer encoder; batch_first so input shape
        # is (B, seq, embed_dim).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

        # Pre-compute the AP index buffer once (saves index_select per fwd).
        self.register_buffer(
            "_ap_idx",
            torch.arange(n_aps, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one WiFi scan per batch element via sparse-observed gather.

        Selects each row's top-``max_observed_per_scan`` strongest-RSSI
        observed APs (where ``x > epsilon``), builds tokens only for
        those, and runs attention over CLS + the kept tokens. Attention
        cost is ``O((k+1)²)`` where k is the per-batch max observed
        count (≤ ``max_observed_per_scan``), not ``O(n_aps²)``.

        Parameters
        ----------
        x : (B, 1, n_aps) or (B, n_aps)
            RSSI scan, scaled to ``[0, 1]`` with 0 = missing.
        """
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, n_aps)
        B, N = x.shape

        observed = x > self.epsilon                          # (B, N) bool
        # Sort key: observed APs come first (offset +10), ordered by RSSI.
        sort_keys = observed.float() * 10.0 + x
        _, sort_idx = sort_keys.sort(dim=1, descending=True, stable=True)

        # Per-batch trim to max observed across the batch (always ≥ 1 so
        # the encoder has something to feed besides CLS).
        n_obs_max = int(observed.sum(dim=1).max().item())
        keep = max(1, min(n_obs_max, self.max_observed_per_scan))
        kept_idx = sort_idx[:, :keep]                        # (B, keep)

        # Gather BSSID id (= position index) and RSSI value for kept slots.
        obs_bssid = kept_idx                                 # (B, keep)
        obs_rssi = x.gather(1, kept_idx)                     # (B, keep)
        obs_mask = obs_rssi <= self.epsilon                  # (B, keep), True if padding

        bssid_emb = self.bssid_embed(obs_bssid)              # (B, keep, bssid_dim)
        tokens = self.token_proj(
            torch.cat([bssid_emb, obs_rssi.unsqueeze(-1)], dim=-1)
        )                                                    # (B, keep, D)

        # Prepend CLS (never masked → softmax safe even if every kept slot
        # is padding for some row, e.g. a totally-empty scan).
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, keep+1, D)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, obs_mask], dim=1)

        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.out_norm(out[:, 0])

    @property
    def input_spec(self) -> dict:
        return {
            "modality": "wifi",
            "shape": (1, self.n_aps),
            "dtype": "float32",
        }
