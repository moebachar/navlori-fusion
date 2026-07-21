"""Fusion architectures and aggregators.

Four fusion architectures benchmarked in run-2 (plus the incumbent
from run-1) — see `handoff/SUMMARY.md` and `RESULT_17/18/21` for
the verdicts:

| arch              | params  | val   | test  | source        | verdict                              |
|-------------------|--------:|------:|------:|---------------|--------------------------------------|
| incumbent         | 1.55 M  | 0.394 | 0.417 | RESULT_06/13/14 | over-parameterised baseline      |
| **cnn1d**         | 0.51 M  | 0.282 | 0.339 | RESULT_17/18  | **Phase B WINNER**                   |
| lstm_attn         | 0.57 M  | 0.301 | 0.340 | RESULT_17/18  | runner-up; dead-reckoning regime     |
| tcn               | ~0.51 M | —     | —     | RESULT_16     | bake-off candidate; no distinct find |
| mot_transformer   | 0.74 M  | 0.594 | 0.608 | RESULT_21     | γ5 — WORST; honest negative result   |

The 4 bake-off candidates (cnn1d / lstm_attn / tcn / mot_transformer)
share the FusionTransformer's encoder + readout pipeline; what
differs is the aggregator block over the (K=4 instants × M
modalities) token sequence. Use ``build_arch(name)`` to construct
any of them.
"""
from __future__ import annotations

import inspect
from typing import Any

from .base import BaseFusion
from .bakeoff import CANDIDATES
from .transformer import ContinuousTimeEncoding, FusionTransformer

DEFAULT_CONFIG = {
    "K": 4,
    "M_max": 4,
    "D": 128,
    "modality_dropout": 0.4,
    "instant_dropout": 0.45,
}


def list_archs() -> list[str]:
    """Return the canonical fusion architecture names accepted by
    ``build_arch``."""
    return list(CANDIDATES)


def build_arch(name: str, encoders: dict | None = None,
                dataset: str = "simulation", **overrides) -> Any:
    """Construct the named fusion architecture.

    Parameters
    ----------
    name : str
        One of ``list_archs()``.
    encoders : dict[str, nn.Module] or None
        Per-modality encoders dict. If ``None``, auto-built from
        ``dataset`` config (default ``"simulation"`` = Webots 4-mod).
    dataset : str
        Hydra dataset config name (used only when ``encoders is None``).
    **overrides
        Override ``incumbent_kwargs`` (e.g. ``depth``, ``n_heads``,
        ``dropout``, ``readout``, ``absolute_modalities``).

    Returns
    -------
    nn.Module — ready to wrap in ``FusionTrainer``.
    """
    if name not in CANDIDATES:
        raise KeyError(
            f"Unknown architecture {name!r}. Available: {list_archs()}"
        )

    if encoders is None:
        from .builder import build_datamodule, build_encoders, extract_vision_tokens, load_config
        cfg = load_config(dataset)
        dm = build_datamodule(cfg)
        encoders, _vision = build_encoders(cfg, dm)
        # Capture defaults from the dataset config
        cfg_kwargs = dict(
            embed_dim=int(cfg.model.embed_dim),
            depth=int(cfg.model.depth),
            n_heads=int(cfg.model.n_heads),
            ff_mult=int(cfg.model.ff_mult),
            dropout=float(cfg.model.dropout),
            use_time=bool(cfg.model.use_time),
            readout=str(cfg.model.readout),
            absolute_modalities=list(
                cfg.model.get("absolute_modalities", None) or ["wifi"]),
        )
    else:
        cfg_kwargs = dict(
            embed_dim=128,
            depth=4,
            n_heads=8,
            ff_mult=4,
            dropout=0.1,
            use_time=True,
            readout="query",
            absolute_modalities=["wifi"],
        )
    cfg_kwargs.update(overrides)
    return CANDIDATES[name](cfg_kwargs, encoders)


__all__ = [
    "BaseFusion", "ContinuousTimeEncoding", "FusionTransformer",
    "CANDIDATES",
    "list_archs", "build_arch", "DEFAULT_CONFIG",
]
