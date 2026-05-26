"""Training-curve plotter for the publication notebook (PLAN_34 take #4).

Every ``train_*`` helper in ``src.pipeline.training.inline_encoders`` (and
``FusionTrainer.fit`` via ``FusionHistory``) returns a history object with
some subset of ``train_loss`` / ``val_loss`` / ``val_mae`` per epoch. This
plotter renders whatever is present:

- loss panel: ``train_loss`` (+ ``val_loss`` if present);
- MAE panel: ``val_mae`` if present (otherwise the loss panel stands alone).

Closed-form fits (e.g. the DPVOMotion linear head) have no per-epoch history;
callers should skip the plot and print a one-line note instead.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ._style import set_paper_style


def _as_dict(history) -> dict:
    """Accept a dict OR a dataclass (FusionHistory) and return a dict."""
    if isinstance(history, dict):
        return history
    out = {}
    for key in ("train_loss", "val_loss", "val_mae"):
        val = getattr(history, key, None)
        if val is not None:
            out[key] = val
    return out


def plot_training_curves(history, title: str = "",
                         save_to: str | Path | None = None):
    """Render train/val loss + val MAE curves from a training history.

    Parameters
    ----------
    history : dict | FusionHistory
        Must expose ``train_loss`` and optionally ``val_loss`` / ``val_mae``
        as per-epoch lists.
    title : str
        Prefix for the panel titles (e.g. the encoder/arch name).
    save_to : path, optional
        If given, ``fig.savefig(save_to)``.

    Returns
    -------
    matplotlib.figure.Figure | None
        ``None`` if the history has no plottable per-epoch series.
    """
    set_paper_style()
    h = _as_dict(history)
    train_loss = h.get("train_loss")
    val_loss = h.get("val_loss")
    val_mae = h.get("val_mae")

    has_loss = bool(train_loss)
    has_mae = bool(val_mae)
    if not has_loss and not has_mae:
        return None

    n_panels = int(has_loss) + int(has_mae)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.2),
                             squeeze=False)
    axes = axes[0]
    col = 0

    if has_loss:
        ax = axes[col]; col += 1
        ax.plot(train_loss, label="train", color="#1f77b4", lw=1.6)
        if val_loss:
            ax.plot(val_loss, label="val", color="#d62728", lw=1.6)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(f"{title} — loss".strip(" —"))
        ax.legend(); ax.grid(True, alpha=0.3)

    if has_mae:
        ax = axes[col]
        ax.plot(val_mae, label="val MAE", color="#2ca02c", lw=1.6)
        best = min(val_mae)
        best_ep = val_mae.index(best)
        ax.axhline(best, ls="--", color="grey", alpha=0.6)
        ax.scatter([best_ep], [best], color="#2ca02c", zorder=5,
                   label=f"best {best:.3f} m @ ep {best_ep}")
        ax.set_xlabel("epoch"); ax.set_ylabel("val MAE (m)")
        ax.set_title(f"{title} — val MAE".strip(" —"))
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["plot_training_curves"]
