"""Paper-figure helpers for evaluation results.

- ``plot_staleness_curve``: WiFi staleness sweep (RESULT_14 style).
- ``plot_subset_eval_bar``: per-subset MAE bars (`only:wifi`,
  `only:imu`, ..., `full`) — surfaces dead-reckoning regime.
- ``plot_main_results_heatmap``: 6-row × N-arch heatmap of main-
  results-table MAE values (bonus visualisation).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._style import color_for, set_paper_style


def plot_staleness_curve(lags, mae_values, label: str | None = None,
                          slope: float | None = None,
                          title: str = "WiFi staleness sweep",
                          save_to: str | Path | None = None):
    """Plot test MAE vs WiFi staleness lag (in seconds).

    Parameters
    ----------
    lags : iterable of int (instants).
    mae_values : iterable of float (m).
    label : str, optional label for the legend (e.g. "CNN1D").
    slope : float, optional pre-computed linear-fit slope (m/s).
    title : str.
    save_to : path-like, optional.
    """
    set_paper_style()
    lags = np.asarray(list(lags))
    mae = np.asarray(list(mae_values))
    secs = lags * 0.9
    fig, ax = plt.subplots(figsize=(7, 4.5))
    color = color_for(label) if label else "C0"
    ax.plot(secs, mae, "o-", color=color, lw=2, markersize=7,
            label=label or "MAE")
    if slope is not None:
        ax.text(0.02, 0.95, f"slope = {slope:.4f} m/s", transform=ax.transAxes,
                fontsize=9, va="top", color="dimgrey")
    ax.set_xlabel("WiFi staleness (s)")
    ax.set_ylabel("test MAE (m)")
    ax.set_title(title)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="C3 gate 0.5 m")
    ax.legend(loc="best")
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


def plot_subset_eval_bar(subset_dict: dict, title: str = "Subset eval",
                          save_to: str | Path | None = None):
    """Plot per-subset MAE as horizontal bars.

    ``subset_dict`` maps subset-name -> float (MAE in metres).
    Example: ``{"only:wifi": 0.42, "only:imu": 0.34, ..., "full": 0.34}``.
    """
    set_paper_style()
    names = list(subset_dict.keys())
    values = [float(v) for v in subset_dict.values()]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1))
    bars = ax.barh(names, values, color="steelblue")
    # Highlight `full` if present
    for b, n in zip(bars, names):
        if n == "full" or "wifi+imu+camera+odom" in n:
            b.set_color("#1f77b4")
        elif n.startswith("only:"):
            b.set_color("#999999")
    ax.set_xlabel("MAE (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    for b, v in zip(bars, values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


def plot_main_results_heatmap(table_df, value_col: str = "test_mae",
                               title: str = "Main results — test MAE (m)",
                               save_to: str | Path | None = None):
    """Plot a heatmap of main-results-table values.

    Expects a DataFrame indexed by dataset, columns = methods, cell
    values = MAE (or whatever ``value_col`` semantically means).
    """
    set_paper_style()
    import pandas as pd
    if not isinstance(table_df, pd.DataFrame):
        raise TypeError("table_df must be a pandas DataFrame")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    vals = table_df.values.astype(float)
    # Row-normalise by row max (so a method that's best on a row appears darker).
    norms = vals / np.nanmax(vals, axis=1, keepdims=True)
    im = ax.imshow(norms, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table_df.columns)))
    ax.set_xticklabels(table_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table_df.index)))
    ax.set_yticklabels(table_df.index)
    # Cell annotations
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row-normalised MAE")
    fig.tight_layout()
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["plot_staleness_curve", "plot_subset_eval_bar",
           "plot_main_results_heatmap"]
