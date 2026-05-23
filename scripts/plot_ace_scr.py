"""Generate the full plot deck for the ACE SCR pipeline.

Outputs into a `plots/` subdirectory of the chosen run. Figures:

    01_training_curves.png     - train/val loss + val median/p95 3D err per epoch
    02_pnp_error_cdf.png       - CDF of (x, y) euclidean error (val + test)
    03_per_path_metrics.png    - per-path median + p95 (x, y) bar chart
    04_baseline_comparison.png - DINOv2 probe / ACE probe / ACE SCR bar chart
    05_trajectories.png        - predicted vs GT trajectories overlaid on walls
    06_scene_pointcloud.png    - all predicted world XYZ points, bird's-eye view

Usage:
    python scripts/plot_ace_scr.py                # latest ace_scr_* run
    python scripts/plot_ace_scr.py --run runs/ace_scr_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Polygon as MplPolygon


# Wall geometry lifted from async_collector.py (same floor plan).
_RAW_WALLS = [
    (1.48, -1.47, 2.2327, 0.2, 2.7),
    (1.27, -3.35, -2.5454, 0.2, 2.707),
    (-0.61, -6.52, 2.2301, 0.2, 6.9709),
    (-10.33, -5.40, 2.15, 0.2, 7.5),
    (-2.16, -10.77, -2.6475, 0.2, 5.0),
    (1.10, -11.53, -1.0, 0.2, 5.1671),
    (3.99, -11.34, -2.6168, 0.2, 3.0),
    (-1.06, -16.10, 2.1194, 0.2, 13.8),
    (-8.01, -18.25, 0.6777, 0.2, 1.9),
    (-6.55, -16.13, -1.0151, 0.2, 5.1559),
    (-5.62, -12.74, 0.5114, 0.2, 4.6),
    (-8.43, -11.76, 2.1214, 0.2, 3.8606),
    (-12.77, -8.61, 0.5831, 0.2, 2.9),
    (-10.79, -11.60, 0.5831, 0.2, 2.7),
    (-6.30, -4.47, -2.482, 0.2, 3.1004),
    (-4.05, -4.58, -0.8897, 0.2, 3.6),
    (-3.72, -1.98, 0.5912, 0.2, 3.4447),
    (-2.75, 1.41, -0.8125, 0.2, 6.0),
    (0.84, 1.52, -2.5178, 0.2, 5.3),
]


def _wall_polygon(tx, ty, angle, sx, sy):
    hx, hy = sx / 2, sy / 2
    ca, sa = math.cos(angle), math.sin(angle)
    return [(tx + ca * cx - sa * cy, ty + sa * cx + ca * cy)
            for cx, cy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def _draw_walls(ax, color: str = "#888", alpha: float = 0.55) -> None:
    for tx, ty, angle, sx, sy in _RAW_WALLS:
        ax.add_patch(MplPolygon(
            _wall_polygon(tx, ty, angle, sx, sy),
            closed=True, facecolor=color, edgecolor="#333",
            linewidth=0.5, alpha=alpha,
        ))


def _pick_run(override: str | None) -> Path:
    if override:
        return Path(override)
    runs = sorted(
        r for r in (ROOT / "runs").glob("ace_scr_*")
        if (r / "head.pt").exists()
        and not r.name.startswith("ace_scr_overfit")
        and not r.name.startswith("ace_scr_smoketest")
    )
    if not runs:
        raise SystemExit("No production ACE SCR run found.")
    return runs[-1]


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_training_curves(run_dir: Path, out: Path) -> None:
    hist = json.loads((run_dir / "history.json").read_text())
    epochs = np.arange(len(hist["train_loss"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(epochs, hist["train_loss"], label="train", lw=1.8, color="#1f77b4")
    axes[0].plot(epochs, hist["val_loss"],   label="val",   lw=1.8, color="#d62728")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("masked-L1 loss (m)")
    axes[0].set_title("Training / val loss")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    med = np.array(hist["val_median_err"])
    p95 = np.array(hist["val_p95_err"])
    axes[1].plot(epochs, med, color="#2ca02c", lw=1.8, label="val median")
    axes[1].axvline(hist["best_epoch"], color="#2ca02c",
                    linestyle="--", alpha=0.4, label=f"best @ {hist['best_epoch']}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("3D err (m)")
    axes[1].set_title(f"Val median — best {hist['best_val_median']:.3f} m")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, p95, color="#ff7f0e", lw=1.8, label="val p95")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("3D err (m)")
    axes[2].set_title("Val p95")
    axes[2].legend(); axes[2].grid(alpha=0.25)

    fig.suptitle(f"{run_dir.name}  —  {hist['elapsed_sec']:.0f} s total", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _cdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_pnp_error_cdf(run_dir: Path, out: Path) -> None:
    val = pd.read_csv(run_dir / "pnp_eval_val_frames.csv")
    tst = pd.read_csv(run_dir / "pnp_eval_test_frames.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for df, label, color in [(val, "val", "#1f77b4"), (tst, "test", "#d62728")]:
        errs = df.loc[df["ok"] & df["euclid_xy_to_gt"].notna(), "euclid_xy_to_gt"].values
        if not errs.size:
            continue
        xs, ys = _cdf(errs)
        axes[0].plot(xs, ys, label=f"{label} (n={len(errs)})", lw=2, color=color)
    axes[0].set_xlabel("(x, y) euclidean error to GT (m)")
    axes[0].set_ylabel("fraction of frames ≤ x")
    axes[0].set_title("PnP (x, y) error — CDF")
    axes[0].axvline(1.0, color="#555", linestyle=":", alpha=0.6, label="1 m")
    axes[0].axvline(0.5, color="#aaa", linestyle=":", alpha=0.6, label="0.5 m")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    combined = np.concatenate([
        val.loc[val["ok"], "euclid_xy_to_gt"].values,
        tst.loc[tst["ok"], "euclid_xy_to_gt"].values,
    ])
    axes[0].set_xlim(0, np.percentile(combined, 99))

    cm = plt.get_cmap("tab10")
    all_df = pd.concat([val.assign(split="val"), tst.assign(split="test")],
                       ignore_index=True)
    for i, pid in enumerate(sorted(all_df["path_id"].unique())):
        sub = all_df[(all_df["path_id"] == pid) & all_df["ok"]]
        if sub.empty:
            continue
        xs, ys = _cdf(sub["euclid_xy_to_gt"].values)
        split = sub["split"].iloc[0]
        axes[1].plot(xs, ys, label=f"path {pid} ({split})",
                     lw=1.6, color=cm(i % 10))
    axes[1].set_xlabel("(x, y) euclidean error to GT (m)")
    axes[1].set_ylabel("fraction of frames ≤ x")
    axes[1].set_title("Per-path CDF")
    axes[1].axvline(1.0, color="#555", linestyle=":", alpha=0.6)
    axes[1].axvline(0.5, color="#aaa", linestyle=":", alpha=0.6)
    axes[1].legend(ncol=2, fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(0, 10)

    fig.suptitle("Robot (x, y) localisation error from frozen ACE trunk + PnP",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_per_path_metrics(run_dir: Path, out: Path) -> None:
    val = json.loads((run_dir / "pnp_eval_val.json").read_text())["per_path"]
    tst = json.loads((run_dir / "pnp_eval_test.json").read_text())["per_path"]
    rows = []
    for pid, d in val.items():
        rows.append((int(pid), "val", d["euclid_xy_to_gt"]["median"],
                     d["euclid_xy_to_gt"]["p95"], d["n_ok"]))
    for pid, d in tst.items():
        rows.append((int(pid), "test", d["euclid_xy_to_gt"]["median"],
                     d["euclid_xy_to_gt"]["p95"], d["n_ok"]))
    rows.sort()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(rows))
    meds = [r[2] for r in rows]
    p95s = [r[3] for r in rows]
    colors = ["#1f77b4" if r[1] == "val" else "#d62728" for r in rows]

    ax.bar(xs - 0.18, meds, width=0.36, color=colors)
    ax.bar(xs + 0.18, p95s, width=0.36, color=colors, alpha=0.45)
    for i, (pid, split, med, p95, n) in enumerate(rows):
        ax.text(i - 0.18, med, f"{med:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.18, p95, f"{p95:.1f}", ha="center", va="bottom", fontsize=8,
                color="#555")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p{r[0]}\n({r[1]})" for r in rows])
    ax.set_ylabel("(x, y) error (m)")
    ax.set_title("Per-path PnP (x, y) error — median (solid) + p95 (pale)")
    ax.axhline(1.0, color="#555", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", label="val"),
        Patch(facecolor="#d62728", label="test"),
    ], loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_baseline_comparison(out: Path) -> None:
    methods = [
        ("DINOv2+LoRA\nlinear probe",    3.64,  "#8c564b"),
        ("ACE\nlinear probe",            3.49,  "#ff7f0e"),
        ("ACE SCR + PnP\n(val, ours)",   0.714, "#1f77b4"),
        ("ACE SCR + PnP\n(test, ours)",  0.709, "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(methods))
    ys = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    bars = ax.bar(xs, ys, color=colors, edgecolor="#222", linewidth=0.6)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.05, f"{y:.2f} m",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel("(x, y) error (m)")
    ax.set_title("Robot (x, y) localisation error — vision-only methods")
    ax.axhline(1.0, color="#555", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.98, 0.98,
            "Linear probe: MAE\nACE SCR+PnP: median euclid",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_trajectories(run_dir: Path, data_dir: Path, out: Path) -> None:
    val = pd.read_csv(run_dir / "pnp_eval_val_frames.csv")
    tst = pd.read_csv(run_dir / "pnp_eval_test_frames.csv")
    paths = sorted(set(val["path_id"]) | set(tst["path_id"]))
    n = len(paths)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.5),
                             squeeze=False)
    for k, pid in enumerate(paths):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        _draw_walls(ax)

        gt_csv = Path(data_dir) / f"path_{pid:02d}" / "ground_truth.csv"
        if gt_csv.exists():
            gt = pd.read_csv(gt_csv)
            ax.plot(gt["gt_x"], gt["gt_y"],
                    color="#1f77b4", lw=2.2, alpha=0.9, label="GT")

        df = pd.concat([val[val["path_id"] == pid], tst[tst["path_id"] == pid]],
                       ignore_index=True)
        df = df[df["ok"]]
        if not df.empty:
            ax.scatter(df["pred_cam_x"], df["pred_cam_y"],
                       s=6, color="#d62728", alpha=0.7, label="PnP")

        ax.set_aspect("equal")
        ax.set_xlim(-14.5, 5.5); ax.set_ylim(-19.5, 3.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        med = (df["euclid_xy_to_gt"].median() if not df.empty else float("nan"))
        split = "val" if pid in [2, 13, 14] else "test"
        ax.set_title(f"path {pid:02d} ({split})  —  median err {med:.2f} m",
                     fontsize=11)
        ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(loc="upper right", fontsize=9)

    for k in range(len(paths), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    fig.suptitle("Predicted vs ground-truth trajectories — val + test", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_scene_pointcloud(run_dir: Path, data_dir: Path, out: Path,
                          max_points: int = 200_000) -> None:
    from src.pipeline.data.scr_dataset import SCRDataset
    from src.pipeline.encoders import ACEScrRegressor
    from torch.utils.data import DataLoader

    train_path_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    val_path_ids = [2, 13, 14]

    scene_ds = SCRDataset(data_dir=str(data_dir), path_ids=train_path_ids,
                          stride=8, preload_depth=False)
    mean_xyz = torch.from_numpy(scene_ds.mean_camera_translation)
    del scene_ds

    ds = SCRDataset(data_dir=str(data_dir), path_ids=val_path_ids,
                    stride=8, preload_depth=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACEScrRegressor(
        mean_xyz=mean_xyz,
        weights_path=str(ROOT / "runs" / "_weights" / "ace_encoder_pretrained.pt"),
    ).to(device).eval()
    model.head.load_state_dict(torch.load(run_dir / "head.pt",
                                           weights_only=True, map_location=device))

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    errs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            pred = model(rgb).cpu().numpy()
            gt = batch["scene_coords"].numpy()
            valid = batch["valid_mask"].numpy()
            d = np.linalg.norm(pred - gt, axis=1)
            for i in range(rgb.shape[0]):
                m = valid[i]
                xs.append(pred[i, 0][m].ravel())
                ys.append(pred[i, 1][m].ravel())
                zs.append(pred[i, 2][m].ravel())
                errs.append(d[i][m].ravel())
    X = np.concatenate(xs); Y = np.concatenate(ys)
    Z = np.concatenate(zs); E = np.concatenate(errs)

    if len(X) > max_points:
        idx = np.random.default_rng(0).choice(len(X), max_points, replace=False)
        X, Y, Z, E = X[idx], Y[idx], Z[idx], E[idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    _draw_walls(axes[0])
    sc = axes[0].scatter(X, Y, c=Z, s=1.5, cmap="viridis",
                         vmin=0, vmax=2.5, alpha=0.45, linewidths=0)
    plt.colorbar(sc, ax=axes[0], shrink=0.8, label="predicted z (m)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-14.5, 5.5); axes[0].set_ylim(-19.5, 3.5)
    axes[0].set_title(f"Val scene-coord cloud — coloured by height  (n={len(X):,})")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].grid(alpha=0.15)

    _draw_walls(axes[1])
    sc2 = axes[1].scatter(X, Y, c=np.clip(E, 0, 3), s=1.5, cmap="magma",
                          alpha=0.5, linewidths=0)
    plt.colorbar(sc2, ax=axes[1], shrink=0.8,
                 label="3D error (m, clipped @ 3)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-14.5, 5.5); axes[1].set_ylim(-19.5, 3.5)
    axes[1].set_title("Same points — coloured by 3D error")
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    axes[1].grid(alpha=0.15)

    fig.suptitle("Scene reconstruction from predicted scene coords (val split)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--data-dir", default=str(ROOT / "data" / "async_collection"))
    p.add_argument("--skip-pointcloud", action="store_true")
    args = p.parse_args()

    run_dir = _pick_run(args.run)
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    print(f"run  : {run_dir}")
    print(f"plots: {out_dir}\n")

    plot_training_curves(run_dir, out_dir / "01_training_curves.png")
    plot_pnp_error_cdf(run_dir, out_dir / "02_pnp_error_cdf.png")
    plot_per_path_metrics(run_dir, out_dir / "03_per_path_metrics.png")
    plot_baseline_comparison(out_dir / "04_baseline_comparison.png")
    plot_trajectories(run_dir, Path(args.data_dir), out_dir / "05_trajectories.png")
    if not args.skip_pointcloud:
        plot_scene_pointcloud(run_dir, Path(args.data_dir),
                              out_dir / "06_scene_pointcloud.png")


if __name__ == "__main__":
    main()
