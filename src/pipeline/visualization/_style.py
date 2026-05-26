"""Shared matplotlib style for run-2 figures.

Centralised so the notebook + scripts both produce visually
consistent paper-strength plots.
"""
from __future__ import annotations

import matplotlib as mpl


PAPER_FONT_SIZE = 10
PAPER_TITLE_SIZE = 11
PAPER_FIG_DPI = 110

# Color palette used across run-2 figures: stable mapping for the
# main 4 architectures + per-leg SOTA.
COLOR_PALETTE = {
    "incumbent":         "#4c4c4c",
    "cnn1d":             "#1f77b4",
    "lstm_attn":         "#2ca02c",
    "tcn":               "#9467bd",
    "mot_transformer":   "#e377c2",
    "wlan_localization": "#ff7f0e",
    "ronin_resnet1d":    "#d62728",
    "tartanvo":          "#8c564b",
    "dpvomotion":        "#17becf",
    "anchor2vec":        "#bcbd22",
    "imucnn":            "#7f7f7f",
}


def set_paper_style():
    """Apply the run-2 default paper-figure rcParams."""
    mpl.rcParams.update({
        "font.size": PAPER_FONT_SIZE,
        "axes.titlesize": PAPER_TITLE_SIZE,
        "axes.labelsize": PAPER_FONT_SIZE,
        "xtick.labelsize": PAPER_FONT_SIZE - 1,
        "ytick.labelsize": PAPER_FONT_SIZE - 1,
        "legend.fontsize": PAPER_FONT_SIZE - 1,
        "figure.dpi": PAPER_FIG_DPI,
        "figure.figsize": (7.2, 4.5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def color_for(label: str) -> str:
    """Return a stable color for known method labels; else default."""
    return COLOR_PALETTE.get(label.lower(), "#000000")


__all__ = ["set_paper_style", "color_for", "COLOR_PALETTE",
           "PAPER_FONT_SIZE", "PAPER_TITLE_SIZE", "PAPER_FIG_DPI"]
