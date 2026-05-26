"""Per-trajectory plot helpers (predicted vs ground-truth).

Used by both the notebook §1-2 cells AND the iteration training
wrappers' ``--save-plots`` paths. Promoted from the
``runs/overnight/run2_iter_*/test_paths/`` pattern.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._style import set_paper_style


def plot_per_trajectory(pred_xy, gt_xy, path_id: int,
                        suffix: str = "", save_to: str | Path | None = None):
    """Plot predicted vs GT 2D trajectory for one path.

    Parameters
    ----------
    pred_xy : (N, 2) array of predicted (x, y).
    gt_xy : (N, 2) array of GT (x, y).
    path_id : int, the path identifier (used in the title).
    suffix : str, extra title text (e.g. "(CNN1D K=4 4-mod)").
    save_to : path-like, optional.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()
    pred_xy = np.asarray(pred_xy); gt_xy = np.asarray(gt_xy)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", label="GT", lw=1.5)
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "r-", label="pred", lw=1.0, alpha=0.75)
    ax.scatter(gt_xy[0, 0], gt_xy[0, 1], c="green", s=40, marker="o", label="start")
    ax.set_aspect("equal")
    ax.set_title(f"path_{int(path_id):02d}{(' ' + suffix) if suffix else ''}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="best")
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["plot_per_trajectory"]
