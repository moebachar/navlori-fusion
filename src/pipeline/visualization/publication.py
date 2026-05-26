"""Publication-figure helpers for the run-2 walkthrough notebook (PLAN_34).

Three per-dataset figures replace the distribution histograms (dropped per
take #1):

- ``plot_gt_trajectory(name)`` — ground-truth (x, y) path of one sample
  sequence (or a position scatter for per-scan datasets like UJI).
- ``plot_modality_samples(name)`` — one short window of each available raw
  modality (WiFi RSSI bars / IMU 6-ch / camera frame / odom columns).
- ``plot_preprocessing_influence(name, modality)`` — raw → preprocessed for
  the dataset's primary modality, built on ``preprocessing_demo``.

All three are defensive: they return ``None`` (and the caller skips) when the
underlying data isn't on disk, so the notebook never errors on a missing
dataset.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._style import set_paper_style


# --------------------------------------------------------------------------
# Ground-truth trajectory
# --------------------------------------------------------------------------

def plot_gt_trajectory(name: str, save_to: str | Path | None = None):
    """Plot the ground-truth (x, y) trajectory of one sample sequence.

    For per-scan WiFi datasets (UJI) there is no trajectory; plots a position
    scatter of the validation fingerprints instead.
    """
    set_paper_style()
    from src.pipeline.data import dataset_stats
    from src.pipeline.data._common import path_to
    import pandas as pd

    s = dataset_stats(name)
    coll = s.get("collection_dir") or s.get("data_dir") or ""

    xy = None
    label = ""
    if name == "ronin_canonical":
        try:
            from src.pipeline.data import load_dataset
            ds = load_dataset(name)
            gp = ds.gt_pos
            seq0 = gp[0] if isinstance(gp, (list, tuple)) else gp
            xy = np.asarray(seq0)[:, :2]
            label = "one RoNIN unseen sequence"
        except Exception:
            return None
    elif name == "tartanair_hospital":
        try:
            pose_p = path_to("data/tartanair_hospital/P000/pose_left.txt")
            poses = np.loadtxt(pose_p)
            xy = poses[:, :2]
            label = "TartanAir hospital P000 (NED x,y)"
        except Exception:
            return None
    elif name == "uji_indoorloc":
        try:
            df = pd.read_csv(path_to("data/uji_indoorloc/validationData.csv"))
            xy = df[["LONGITUDE", "LATITUDE"]].values.astype(float)
            label = "UJI validation fingerprints (per-scan, no path)"
        except Exception:
            return None
    else:
        # Path-based collections (webots, imuwifine, msiln, ipin): read the
        # longest ground_truth.csv.
        if not coll:
            return None
        best = None
        for pdir in sorted(path_to(coll).glob("path_*")):
            gt_p = pdir / "ground_truth.csv"
            if not gt_p.exists():
                continue
            try:
                gt = pd.read_csv(gt_p)
                xcol = "gt_x" if "gt_x" in gt.columns else gt.columns[1]
                ycol = "gt_y" if "gt_y" in gt.columns else gt.columns[2]
                arr = gt[[xcol, ycol]].values.astype(float)
                if best is None or len(arr) > len(best[1]):
                    best = (pdir.name, arr)
            except Exception:
                continue
        if best is None:
            return None
        xy = best[1]
        label = f"{name} — {best[0]} (longest path)"

    if xy is None or len(xy) < 2:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    is_scatter = name == "uji_indoorloc"
    if is_scatter:
        ax.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.5, color="#1f77b4")
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-", color="#1f77b4", lw=1.4, alpha=0.85)
        ax.scatter(xy[0, 0], xy[0, 1], color="green", s=45, zorder=5, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], color="red", s=45, zorder=5, label="end")
        ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Ground truth — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Raw modality samples
# --------------------------------------------------------------------------

def plot_modality_samples(name: str, save_to: str | Path | None = None):
    """One short window of each available raw modality for the dataset."""
    set_paper_style()
    from src.pipeline.data import dataset_stats
    from .dataset_overview import _try_sample_rssi, _try_sample_imu

    from src.pipeline.data._common import path_to
    import pandas as pd
    s = dataset_stats(name)
    mods = s.get("modalities_available", [])

    panels = []  # list of (title, draw_fn)

    if "wifi" in mods:
        rssi = _try_sample_rssi(name, n_paths=1)
        if (rssi is None or not rssi.size) and name == "uji_indoorloc":
            try:
                df = pd.read_csv(path_to("data/uji_indoorloc/validationData.csv"))
                waps = [c for c in df.columns if c.startswith("WAP")]
                arr = df[waps].head(1).values.astype(float)
                rssi = np.where(arr == 100, np.nan, arr)
            except Exception:
                rssi = None
        if rssi is not None and rssi.size:
            scan = rssi[0]
            def draw_wifi(ax, scan=scan):
                vals = np.where(np.isnan(scan), -100.0, scan)
                ax.bar(range(len(vals)), vals, color="#1f77b4", width=1.0)
                ax.set_xlabel("AP index"); ax.set_ylabel("RSSI (dBm)")
            panels.append(("WiFi — one RSSI scan", draw_wifi))

    if "imu" in mods:
        imu = _try_sample_imu(name, n_paths=1)
        if (imu is None or not imu.size) and name == "ronin_canonical":
            # RoNIN: pull the 6-channel feature window from the dataset object.
            try:
                from src.pipeline.data import load_dataset
                ds = load_dataset(name)
                feats = ds.features
                seq0 = feats[0] if isinstance(feats, (list, tuple)) else feats
                imu = np.asarray(seq0)[:200, :3]
            except Exception:
                imu = None
        if imu is not None and imu.size:
            win = imu[:200]
            def draw_imu(ax, win=win):
                for j in range(min(3, win.shape[1])):
                    ax.plot(win[:, j], lw=0.9, label=["x", "y", "z"][j])
                ax.set_xlabel("sample"); ax.set_ylabel("gyro/accel ch")
                ax.legend(fontsize=7, ncol=3)
            panels.append(("IMU — 200-sample window", draw_imu))

    if "camera" in mods:
        img = None
        try:
            from PIL import Image
            cam_dir = path_to("data/tartanair_hospital/P000/image_left")
            frames = sorted(cam_dir.glob("*.png"))
            if frames:
                img = np.asarray(Image.open(str(frames[0])).convert("RGB"))
        except Exception:
            img = None
        if img is not None:
            def draw_cam(ax, img=img):
                ax.imshow(img); ax.axis("off")
            panels.append(("Camera — one frame", draw_cam))

    if "odom" in mods:
        coll = s.get("collection_dir") or s.get("data_dir") or ""
        odo_win = None
        if coll:
            for pdir in sorted(path_to(coll).glob("path_*")):
                op = pdir / "odometry.csv"
                if op.exists():
                    try:
                        odo = pd.read_csv(op)
                        cols = [c for c in ["odom_linear_vel", "odom_angular_vel"]
                                if c in odo.columns]
                        if cols:
                            odo_win = odo[cols].head(200).values.astype(float)
                            odo_lbls = cols
                            break
                    except Exception:
                        continue
        if odo_win is not None:
            def draw_odom(ax, win=odo_win, lbls=odo_lbls):
                for j, lbl in enumerate(lbls):
                    ax.plot(win[:, j], lw=1.0, label=lbl)
                ax.set_xlabel("sample"); ax.set_ylabel("odom")
                ax.legend(fontsize=7)
            panels.append(("Odometry — 200-sample window", draw_odom))

    if not panels:
        return None

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False)
    axes = axes[0]
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{name} — raw modality samples", fontsize=11)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Preprocessing influence
# --------------------------------------------------------------------------

def plot_preprocessing_influence(name: str, modality: str,
                                 save_to: str | Path | None = None):
    """Raw → preprocessed for the dataset's primary modality.

    Built on ``preprocessing_demo(name, modality)`` (RESULT_27); emphasizes
    *what preprocessing changes* with a before/after pair.
    """
    set_paper_style()
    from src.pipeline.data import preprocessing_demo
    try:
        demo = preprocessing_demo(name, modality)
    except Exception:
        return None
    raw = np.asarray(demo.get("raw"))
    pre = np.asarray(demo.get("preprocessed"))
    if raw is None or pre is None or raw.size == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    raw0 = raw[0] if raw.ndim > 1 else raw
    pre0 = pre[0] if pre.ndim > 1 else pre

    if modality == "wifi":
        axes[0].bar(range(len(raw0)), np.where(np.isnan(raw0), -100.0, raw0),
                    color="#7f7f7f", width=1.0)
        axes[0].set_ylabel("RSSI (dBm)")
        axes[1].bar(range(len(pre0)), pre0, color="#1f77b4", width=1.0)
        axes[1].set_ylabel("normalized [0,1]")
        for ax in axes:
            ax.set_xlabel("AP index")
    else:
        axes[0].plot(raw0, color="#7f7f7f", lw=1.0)
        axes[1].plot(pre0, color="#1f77b4", lw=1.0)
        for ax in axes:
            ax.set_xlabel("feature / sample")

    axes[0].set_title(f"raw — {demo.get('description_raw', '')}", fontsize=8)
    axes[1].set_title(f"preprocessed — {demo.get('description_preprocessed', '')}",
                      fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name} — {modality} preprocessing influence", fontsize=11)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
    return fig


__all__ = [
    "plot_gt_trajectory",
    "plot_modality_samples",
    "plot_preprocessing_influence",
]
