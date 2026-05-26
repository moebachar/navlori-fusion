"""PLAN_21 — Transformer-from-scratch candidate (MoTTransformer).

Modality-of-Time Transformer designed AFTER RESULT_17/18 evidence:
  - Per-modality learnable modality embedding (no time positional embed).
  - 3-layer transformer encoder, 2-head, FFN dim = 2D (not 4D).
  - **ALiBi temporal bias** on attention: bias[i,j] = -slope_h * |t_i - t_j|.
    Bias only on temporal axis; modality-modality identical-time pairs
    get no positional bias (modality embedding handles modality identity).
  - Single learnable-query cross-attention readout (1-head).
  - MLP head D -> 64 -> 2 -> (x, y).
  - No CLS, no PositionQuery, no anchor/motion machinery.
  - Param budget ~0.48 M (parity with CNN1D / LSTM-attn for fair compare).

Honours the same forward signature as FusionTransformer so the
FusionTrainer / bakeoff CANDIDATES registry work unchanged:
    forward(inputs, avail, dt=None, query_dt=None) -> (B, 2)

NaN safety: if a sample has all tokens masked (all modalities × all
instants invalid after dropouts), the row's mask is forcibly unmasked
on token 0 so softmax stays defined. This is the same NaN-trick the
incumbent solves with a never-masked CLS, just placed differently.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ALiBiSelfAttention(nn.Module):
    """Multi-head self-attention with ALiBi temporal bias.

    Parameters
    ----------
    embed_dim : int
        Token feature dim.
    n_heads : int
        Number of attention heads.
    bias_matrix : (n_heads, S, S) float
        Pre-computed temporal-distance bias (added before softmax). Built
        once at FusionBlock init for the fixed token grid (K * M).
    dropout : float
        Dropout rate on attention weights.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        alibi_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, S, D); alibi_bias: (n_heads, S, S); key_padding_mask: (B, S) bool, True=ignore.
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # scores: (B, H, S, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + alibi_bias.unsqueeze(0)  # broadcast over B
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class _MoTLayer(nn.Module):
    """Pre-norm transformer encoder layer with ALiBi attention + 2D FFN."""

    def __init__(self, embed_dim: int, n_heads: int, ff_mult: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _ALiBiSelfAttention(embed_dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, alibi_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), alibi_bias, key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class MoTTransformer(nn.Module):
    """Modality-of-Time Transformer (PLAN_21).

    Same interface as FusionTransformer:
        forward(inputs, avail, dt=None, query_dt=None) -> (B, 2)

    ``encoders`` is a dict mapping modality name -> per-modality encoder
    (Anchor2Vec / IMUCNN / DPVOMotionEncoder / OdomCNN). The encoders
    return (B*K, embed_dim); the model reshapes to (B, K, embed_dim) and
    adds learnable modality embeddings before the encoder stack.

    Parameters
    ----------
    encoders : dict[str, nn.Module]
    embed_dim : int (D)
    n_layers : int (default 3)
    n_heads : int (default 2)
    ff_mult : int (default 2; smaller than incumbent's 4)
    dropout : float
    n_instants : int (K, default 4)
    alibi_slopes : list[float] | None
        Per-head ALiBi slopes (positive values; larger = sharper decay).
        Default: [1.0, 0.5] for n_heads=2; learnable parameters either way.
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        n_instants: int = 4,
        alibi_slopes: list[float] | None = None,
        # Accept and ignore extras (use_time, depth, readout, etc.) so we
        # can be built from incumbent_kwargs without surprise.
        **kwargs,
    ):
        super().__init__()
        self.modalities = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.embed_dim = embed_dim
        self.n_instants = n_instants
        self.n_modalities = len(self.modalities)
        self.n_heads = n_heads

        # Learnable modality embeddings (one (D,) per modality).
        self.modality_emb = nn.Parameter(
            torch.randn(self.n_modalities, embed_dim) * 0.02
        )

        # Per-head ALiBi slopes — learnable, init to [1.0, 0.5, 0.25, ...]
        # so head 0 has sharper temporal locality, later heads broader.
        if alibi_slopes is None:
            alibi_slopes = [1.0 / (2 ** h) for h in range(n_heads)]
        assert len(alibi_slopes) == n_heads
        self.alibi_slopes = nn.Parameter(torch.tensor(alibi_slopes, dtype=torch.float32))

        # Encoder stack: 3 layers x 2 heads.
        self.layers = nn.ModuleList([
            _MoTLayer(embed_dim, n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Cross-attention readout — single learnable query (1, D),
        # 1-head attention attending to the (K*M) encoder outputs.
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.readout_attn = nn.MultiheadAttention(
            embed_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.readout_norm = nn.LayerNorm(embed_dim)

        # Head: D -> 64 -> 2 (xy).
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # Pre-compute temporal-distance index matrix once (depends on K, M):
        # for token i at position (t_i, m_i) and j at (t_j, m_j),
        # dist[i,j] = |t_i - t_j|. The bias is `-slope_h * dist[i,j]`.
        S = n_instants * self.n_modalities
        t = torch.arange(S) // self.n_modalities       # (S,) instant index per token
        dist = (t.unsqueeze(0) - t.unsqueeze(1)).abs().float()  # (S, S)
        self.register_buffer("_alibi_dist", dist, persistent=False)

    def _alibi_bias(self) -> torch.Tensor:
        # Build (n_heads, S, S) bias from current learnable slopes.
        # Slopes are kept positive via abs() so the bias is monotonically
        # non-positive in distance — preserves ALiBi's intent under updates.
        slopes = self.alibi_slopes.abs().view(self.n_heads, 1, 1)
        return -slopes * self._alibi_dist.unsqueeze(0)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        avail: dict[str, torch.Tensor],
        dt: dict[str, torch.Tensor] | None = None,
        query_dt: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor:
        # ===== Stage 1 — Encode + add modality embeddings + reshape =====
        # We flatten (B, K, M, D) to (B, K*M, D) with token-i at (t=i//M, m=i%M).
        per_mod_tokens: list[torch.Tensor] = []
        per_mod_pad: list[torch.Tensor] = []
        K = self.n_instants
        M = self.n_modalities
        for mi, mod in enumerate(self.modalities):
            x = inputs[mod]                                 # (B, K, *window)
            B = x.shape[0]
            z = self.encoders[mod](x.flatten(0, 1))         # (B*K, D)
            z = z.view(B, K, self.embed_dim)
            z = z + self.modality_emb[mi]                   # broadcast over K
            per_mod_tokens.append(z)                        # (B, K, D)
            per_mod_pad.append(~avail[mod])                 # (B, K)
        # Stack to (B, K, M, D) then reshape to (B, K*M, D) with t-major order.
        tokens_kmd = torch.stack(per_mod_tokens, dim=2)     # (B, K, M, D)
        pad_km = torch.stack(per_mod_pad, dim=2)            # (B, K, M)
        tokens = tokens_kmd.reshape(B, K * M, self.embed_dim)
        pad = pad_km.reshape(B, K * M)                      # True = ignore

        # NaN safety: if ALL tokens of a row are masked, force-unmask token 0
        # so attention softmax stays defined. (The incumbent uses a never-
        # masked CLS for the same purpose.)
        all_masked = pad.all(dim=1)
        if all_masked.any():
            pad = pad.clone()
            pad[all_masked, 0] = False

        # ===== Stage 2 — Self-attention encoder with ALiBi temporal bias =====
        alibi = self._alibi_bias().to(tokens.dtype)         # (H, S, S)
        x = tokens
        for layer in self.layers:
            x = layer(x, alibi, key_padding_mask=pad)
        x = self.norm(x)

        # ===== Stage 3 — Single-query cross-attention readout =====
        q = self.query.expand(B, -1, -1)                    # (B, 1, D)
        attn_out, _ = self.readout_attn(q, x, x, key_padding_mask=pad)
        pooled = self.readout_norm(attn_out.squeeze(1) + q.squeeze(1))

        # ===== Stage 4 — Position head =====
        return self.head(pooled)
