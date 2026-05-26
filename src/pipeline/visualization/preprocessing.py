"""Side-by-side raw / preprocessed sample figure for the notebook §0.

Accepts a ``demo_dict`` from ``src.pipeline.data.preprocessing_demo``
and renders an appropriate 2-panel figure based on the modality.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._style import set_paper_style


def plot_preprocessing_demo(demo_dict: dict, modality: str,
                             title: str | None = None,
                             save_to: str | Path | None = None):
    """Render a 2-panel raw / preprocessed figure for one modality.

    Returns the matplotlib Figure. If ``demo_dict['raw']`` is None
    (because the modality isn't supported), a 1-panel info figure is
    returned with the demo's ``note`` text.
    """
    set_paper_style()
    raw = demo_dict.get("raw")
    pre = demo_dict.get("preprocessed")
    desc_raw = demo_dict.get("description_raw", "raw")
    desc_pre = demo_dict.get("description_preprocessed", "preprocessed")
    note = demo_dict.get("note")
    if raw is None and pre is None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.axis("off")
        ax.text(0.02, 0.5,
                (note or f"{modality}: no preprocessing demo available."),
                transform=ax.transAxes, fontsize=10, va="center", family="monospace")
        fig.suptitle(title or f"preprocessing demo — {modality} (n/a)",
                     fontsize=11)
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if modality == "wifi" and isinstance(raw, np.ndarray):
        # Raw RSSI: histogram of dBm values across all APs (NaN-aware).
        flat_raw = raw.ravel()
        valid = flat_raw[~np.isnan(flat_raw)] if np.issubdtype(flat_raw.dtype, np.floating) else flat_raw
        axes[0].hist(valid, bins=30, color="steelblue", alpha=0.8)
        axes[0].set_title("raw RSSI (dBm)")
        axes[0].set_xlabel("dBm"); axes[0].set_ylabel("count")
        flat_pre = np.asarray(pre).ravel()
        axes[1].hist(flat_pre, bins=30, color="darkorange", alpha=0.8)
        axes[1].set_title("preprocessed RSSI (affine to [0,1])")
        axes[1].set_xlabel("value"); axes[1].set_ylabel("count")
    elif modality == "imu" and isinstance(raw, np.ndarray):
        # 6-channel time-series for raw + preprocessed.
        ch_names = ["accel_x", "accel_y", "accel_z",
                    "gyro_x", "gyro_y", "gyro_z"]
        for c in range(min(6, raw.shape[1])):
            axes[0].plot(raw[:, c], label=ch_names[c], lw=1.0, alpha=0.7)
            axes[1].plot(np.asarray(pre)[:, c], label=ch_names[c], lw=1.0, alpha=0.7)
        axes[0].set_title("raw 6-ch IMU"); axes[0].set_xlabel("step")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].set_title("preprocessed (z-score or world-frame)")
        axes[1].set_xlabel("step"); axes[1].legend(fontsize=7, ncol=2)
    elif modality == "odom" and isinstance(raw, np.ndarray):
        for c in range(min(raw.shape[1], 4)):
            axes[0].plot(raw[:, c], label=f"col{c}", lw=1.0, alpha=0.7)
        axes[0].set_title("raw odometry"); axes[0].legend(fontsize=7)
        if isinstance(pre, np.ndarray):
            for c in range(min(pre.shape[1], 4)):
                axes[1].plot(pre[:, c], label=f"Δcol{c}", lw=1.0, alpha=0.7)
            axes[1].set_title("Δ-features (P-B)"); axes[1].legend(fontsize=7)
    elif modality == "camera":
        # Camera demos return path strings, not arrays. Try to load the
        # first image inline.
        if isinstance(raw, list) and raw:
            try:
                img = plt.imread(raw[0])
                axes[0].imshow(img); axes[0].set_title("raw RGB")
                axes[0].axis("off")
            except Exception:
                axes[0].axis("off")
                axes[0].text(0.5, 0.5, str(raw[0]), ha="center", va="center")
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "raw: n/a", ha="center", va="center")
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, desc_pre, wrap=True, ha="center", va="center",
                     fontsize=9, family="monospace")
    else:
        axes[0].axis("off"); axes[1].axis("off")
        axes[0].text(0.02, 0.95, desc_raw, transform=axes[0].transAxes,
                     va="top", fontsize=9, family="monospace")
        axes[1].text(0.02, 0.95, desc_pre, transform=axes[1].transAxes,
                     va="top", fontsize=9, family="monospace")

    fig.suptitle(title or f"preprocessing — {modality}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["plot_preprocessing_demo"]
