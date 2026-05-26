"""Per-dataset multi-panel overview figure.

Used by the notebook §0 pre-section. Renders up to 4 panels depending
on which modalities the dataset has:
- Panel A: trajectory map (if any path-based dataset).
- Panel B: RSSI distribution (if WiFi present).
- Panel C: IMU channel histograms (if IMU present).
- Panel D: per-path duration distribution (if path-based).

Designed to handle the heterogeneity of our 7 datasets: per-scan UJI
collapses the trajectory panel; Camera-only TartanAir skips B/C; etc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._style import set_paper_style
from src.pipeline.data import dataset_stats


def _try_load_path_metadata(name: str) -> list[dict]:
    """Read each ``data/<collection>/path_*/metadata.json`` (if any)."""
    s = dataset_stats(name)
    coll = s.get("collection_dir") or s.get("data_dir") or ""
    if not coll:
        return []
    from src.pipeline.data._common import collect_path_metadata, path_to
    return collect_path_metadata(path_to(coll))


def _try_sample_rssi(name: str, n_paths: int = 3) -> np.ndarray | None:
    """Read a small sample of WiFi RSSI for the dataset (if applicable)."""
    s = dataset_stats(name)
    coll = s.get("collection_dir") or s.get("data_dir") or ""
    if not coll or "wifi" not in s.get("modalities_available", []):
        if name == "uji_indoorloc":
            from src.pipeline.data._common import path_to
            df = pd.read_csv(path_to("data/uji_indoorloc/validationData.csv"))
            waps = [c for c in df.columns if c.startswith("WAP")]
            arr = df[waps].head(200).values.astype(float)
            # 100 sentinel -> nan for histogram
            arr = np.where(arr == 100, np.nan, arr)
            return arr
        return None
    from src.pipeline.data._common import path_to
    rows = []
    cnt = 0
    for path_dir in sorted(path_to(coll).glob("path_*")):
        if cnt >= n_paths:
            break
        wifi_p = path_dir / "wifi.csv"
        if not wifi_p.exists():
            continue
        try:
            wifi = pd.read_csv(wifi_p)
            cols = [c for c in wifi.columns if c.startswith("wifi_rssi_")]
            if not cols:
                continue
            rows.append(wifi[cols].values.astype(float))
            cnt += 1
        except Exception:
            continue
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def _try_sample_imu(name: str, n_paths: int = 2) -> np.ndarray | None:
    """Read a small sample of IMU 6-ch data."""
    s = dataset_stats(name)
    coll = s.get("collection_dir") or s.get("data_dir") or ""
    if "imu" not in s.get("modalities_available", []):
        return None
    if not coll:
        return None
    from src.pipeline.data._common import path_to
    rows = []
    cnt = 0
    for path_dir in sorted(path_to(coll).glob("path_*")):
        if cnt >= n_paths:
            break
        imu_p = path_dir / "imu.csv"
        if not imu_p.exists() or imu_p.stat().st_size < 100:
            continue
        try:
            imu = pd.read_csv(imu_p)
            cols = ["accel_x", "accel_y", "accel_z",
                    "gyro_x", "gyro_y", "gyro_z"]
            if not all(c in imu.columns for c in cols):
                continue
            rows.append(imu[cols].head(2000).values.astype(float))
            cnt += 1
        except Exception:
            continue
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def plot_dataset_overview(name: str, save_to: str | Path | None = None):
    """Render a multi-panel overview for a dataset.

    Parameters
    ----------
    name : str
        Dataset name from ``list_datasets()``.
    save_to : path-like, optional
        If provided, save the figure here at the default DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()
    s = dataset_stats(name)
    mods = s.get("modalities_available", [])
    per_path = _try_load_path_metadata(name)
    rssi = _try_sample_rssi(name)
    imu = _try_sample_imu(name)

    # Decide panel layout.
    panels = []
    if per_path and any("x_range_m" in m or "gt_x" in m for m in per_path):
        panels.append("traj")
    if "wifi" in mods and rssi is not None:
        panels.append("rssi")
    if "imu" in mods and imu is not None:
        panels.append("imu")
    if per_path:
        panels.append("durations")
    if not panels:
        panels = ["info"]
    n = len(panels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for i, p in enumerate(panels):
        ax = axes_flat[i]
        if p == "traj":
            for m in per_path:
                xs = []
                # Per-path metadata has x_range / y_range, not actual trajectory
                # values. As a lightweight overview, plot a colored bar from
                # min..max in (x, y) space — gives the spatial footprint without
                # needing to load every CSV.
                xr = m.get("x_range_m")
                yr = m.get("y_range_m")
                if xr is None or yr is None:
                    continue
                pid = m.get("path_id", -1)
                ax.plot([xr[0], xr[1]], [yr[0], yr[1]], "-",
                        alpha=0.5, lw=2, label=f"p{int(pid):02d}" if pid < 6 else None)
            ax.set_aspect("equal")
            ax.set_title("Spatial footprint (per-path x/y ranges)")
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            if len(per_path) <= 6:
                ax.legend(fontsize=7, loc="best", ncol=2)
        elif p == "rssi":
            vals = rssi[~np.isnan(rssi)].ravel()
            if len(vals) > 0:
                ax.hist(vals, bins=40, color="steelblue", alpha=0.8)
                ax.set_title(f"RSSI distribution (sample n={len(vals)})")
                ax.set_xlabel("RSSI (dBm)"); ax.set_ylabel("count")
        elif p == "imu":
            channels = ["accel_x", "accel_y", "accel_z",
                        "gyro_x", "gyro_y", "gyro_z"]
            for c in range(6):
                ax.hist(imu[:, c], bins=40, alpha=0.5, label=channels[c])
            ax.set_title("IMU 6-channel sample distribution")
            ax.set_xlabel("value (raw units)"); ax.set_ylabel("count")
            ax.legend(fontsize=7, ncol=2)
        elif p == "durations":
            durs = [m.get("duration_s", 0) for m in per_path]
            ax.bar(range(len(durs)), durs, color="seagreen")
            ax.set_title(f"Per-path duration (n={len(durs)})")
            ax.set_xlabel("path index"); ax.set_ylabel("duration (s)")
        elif p == "info":
            ax.axis("off")
            mods_str = ", ".join(mods)
            txt = (
                f"{s.get('name', name)}\n\n"
                f"modalities: {mods_str}\n\n"
            )
            for k in ("n_paths_total", "n_frames", "splits"):
                if k in s:
                    txt += f"{k}: {s[k]}\n"
            caveats = s.get("known_caveats", [])
            if caveats:
                txt += "\nknown caveats:\n  - " + "\n  - ".join(caveats[:3])
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, family="monospace")
    for i in range(len(panels), len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(f"{name} — overview", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = ["plot_dataset_overview"]
