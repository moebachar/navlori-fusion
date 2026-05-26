"""PLAN_16 architecture bake-off — 3 alternative aggregator backbones
that drop in to the FusionTransformer's `self.encoder` slot.

Each candidate subclasses ``FusionTransformer`` and only replaces the
nn.TransformerEncoder layer stack (and the source-key-padding-mask
plumbing). Everything else — encoders, CLS, PositionQuery readout,
position head, modality/time embeddings — is inherited unchanged so
the comparison is apples-to-apples on the aggregator alone.

Candidates:
  - ``LSTMAttnFusion``   — bidirectional LSTM over the (1 + K·M)-token
                            sequence; mask-aware via packed sequence is
                            avoided (mask just zeroes padded outputs).
  - ``TCNFusion``         — 3-layer 1D dilated conv stack (kernel 3,
                            dilations [1, 2, 4]) on the token sequence
                            treated as channels-first 1D.
  - ``CNN1DFusion``       — 3-layer plain 1D conv stack (kernel 3, no
                            dilation); the minimum-baseline candidate.

A fourth candidate (Transformer-from-scratch) is documented in the
PLAN_16 narrative but skipped this iteration per the plan's "cut to 3
if overrun" provision (4 candidates × 30 epochs × ~5 min each blows
the 75-min compute budget).

The bake-off keeps the incumbent's PositionQuery cross-attention
readout (NOT the cls/decomposed readouts) so the differentiator is
strictly the aggregator.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mot_transformer import MoTTransformer
from .transformer import FusionTransformer


class _MaskedBiLSTM(nn.Module):
    """BiLSTM over (B, S, D) with per-position mask applied to outputs.

    Mask-aware packing skipped (CPU overhead on the project's batch
    sizes); we just zero masked positions in the output to prevent
    them polluting the downstream attention's keys.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        # BiLSTM hidden_dim=64 each direction → concat 128 → linear back to
        # embed_dim for residual-friendly output.
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D); key_padding_mask: (B, S) True = ignore.
        out, _ = self.lstm(x)
        out = self.proj(out)
        # Residual + norm (mirrors the transformer's GELU-MLP residual pattern).
        out = self.norm(x + out)
        # Zero masked positions so they don't propagate signal.
        keep = (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        return out * keep


class _DilatedTCN(nn.Module):
    """3-layer 1D dilated conv stack (kernel 3, dilations [1, 2, 4]).

    Receptive field = 1 + 2·(1+2+4) = 15 token positions; covers
    K·M = 4·4 = 16 + 1 CLS comfortably.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        for d in (1, 2, 4):
            layers.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=3,
                                     padding=d, dilation=d))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) → (B, D, S) for Conv1d → (B, S, D) back.
        h = x.transpose(1, 2)
        h = self.layers(h)
        h = h.transpose(1, 2)
        out = self.norm(x + h)
        keep = (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        return out * keep


class _PlainCNN1D(nn.Module):
    """3-layer plain 1D conv (no dilation) — minimum-baseline candidate."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.layers(h)
        h = h.transpose(1, 2)
        out = self.norm(x + h)
        keep = (~key_padding_mask).unsqueeze(-1).to(out.dtype)
        return out * keep


def _swap_encoder(model: FusionTransformer, new_encoder: nn.Module) -> FusionTransformer:
    """Replace ``model.encoder`` (the nn.TransformerEncoder) with a drop-in
    aggregator that has the same ``(x, key_padding_mask) -> x`` signature.

    The FusionTransformer's forward calls
    ``self.encoder(x, src_key_padding_mask=full_pad)``. We wrap the
    drop-in so it accepts the keyword arg name `src_key_padding_mask`.
    """

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, src_key_padding_mask=None):
            return self.m(x, src_key_padding_mask)

    model.encoder = _Wrap(new_encoder)
    return model


def build_lstm_attn(incumbent_kwargs: dict, encoders: dict) -> FusionTransformer:
    """LSTM-with-attention candidate. Inherits all incumbent kwargs
    except the aggregator swap; PositionQuery readout retained."""
    m = FusionTransformer(encoders=encoders, **incumbent_kwargs)
    m = _swap_encoder(m, _MaskedBiLSTM(embed_dim=incumbent_kwargs["embed_dim"],
                                        hidden_dim=incumbent_kwargs["embed_dim"]))
    return m


def build_tcn(incumbent_kwargs: dict, encoders: dict) -> FusionTransformer:
    """TCN dilated-conv candidate."""
    m = FusionTransformer(encoders=encoders, **incumbent_kwargs)
    m = _swap_encoder(m, _DilatedTCN(embed_dim=incumbent_kwargs["embed_dim"],
                                      dropout=incumbent_kwargs.get("dropout", 0.1)))
    return m


def build_cnn1d(incumbent_kwargs: dict, encoders: dict) -> FusionTransformer:
    """Plain 1D-CNN over instants candidate."""
    m = FusionTransformer(encoders=encoders, **incumbent_kwargs)
    m = _swap_encoder(m, _PlainCNN1D(embed_dim=incumbent_kwargs["embed_dim"],
                                      dropout=incumbent_kwargs.get("dropout", 0.1)))
    return m


def build_incumbent(incumbent_kwargs: dict, encoders: dict) -> FusionTransformer:
    """Run-1 FusionTransformer baseline (no aggregator swap)."""
    return FusionTransformer(encoders=encoders, **incumbent_kwargs)


def build_mot_transformer(incumbent_kwargs: dict, encoders: dict) -> MoTTransformer:
    """PLAN_21 transformer-from-scratch candidate. Designed AFTER RESULT_17/18.

    3-layer, 2-head, ALiBi temporal bias, FFN dim=2D, single-query cross-attn
    readout, MLP head D->64->2. ~0.48 M params (parity vs CNN1D 0.51 M /
    LSTM-attn 0.57 M).
    """
    return MoTTransformer(
        encoders=encoders,
        embed_dim=incumbent_kwargs["embed_dim"],
        n_layers=3,
        n_heads=2,
        ff_mult=2,
        dropout=float(incumbent_kwargs.get("dropout", 0.1)),
        n_instants=4,
    )


CANDIDATES = {
    "incumbent": build_incumbent,
    "lstm_attn": build_lstm_attn,
    "tcn": build_tcn,
    "cnn1d": build_cnn1d,
    "mot_transformer": build_mot_transformer,
}
