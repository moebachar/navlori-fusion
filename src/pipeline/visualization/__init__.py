"""Run-2 visualization package.

Plotters for the run-2 walkthrough notebook (PLAN_30) + scripts.
Every function returns a ``matplotlib.figure.Figure`` so callers can
both display inline (Jupyter) and save to disk (`fig.savefig(...)`).
"""

from .dataset_overview import plot_dataset_overview
from .evaluation import (
    plot_main_results_heatmap,
    plot_staleness_curve,
    plot_subset_eval_bar,
)
from .preprocessing import plot_preprocessing_demo
from .publication import (
    plot_gt_trajectory,
    plot_modality_samples,
    plot_preprocessing_influence,
)
from .trajectory import plot_per_trajectory
from .training_curves import plot_training_curves
from ._style import COLOR_PALETTE, color_for, set_paper_style

__all__ = [
    "plot_dataset_overview",
    "plot_per_trajectory",
    "plot_staleness_curve",
    "plot_subset_eval_bar",
    "plot_main_results_heatmap",
    "plot_preprocessing_demo",
    "plot_gt_trajectory",
    "plot_modality_samples",
    "plot_preprocessing_influence",
    "plot_training_curves",
    "COLOR_PALETTE", "color_for", "set_paper_style",
]
