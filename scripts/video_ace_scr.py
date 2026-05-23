"""Render an mp4 of the ACE SCR pipeline running on one path.

Each video frame has four panels, side-by-side:
    top-left    : input RGB
    top-middle  : GT scene-coord map (XYZ → RGB)
    top-right   : predicted scene-coord map (XYZ → RGB)
    bottom      : bird's-eye view with walls, GT trajectory, and the
                  rolling sequence of PnP camera-centre predictions so far.

Usage (from repo root):
    python scripts/video_ace_scr.py                 # latest run, path 13 (best val)
    python scripts/video_ace_scr.py --path 14 --fps 8
    python scripts/video_ace_scr.py --run runs/ace_scr_... --path 15 --fps 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Polygon as MplPolygon

from src.pipeline.data.scr_dataset import SCRDataset
from src.pipeline.encoders import ACEScrRegressor


# Walls (same as async_collector / plot_ace_scr)
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


def _draw_walls(ax, color="#888", alpha=0.55) -> None:
    for tx, ty, angle, sx, sy in _RAW_WALLS:
        ax.add_patch(MplPolygon(
            _wall_polygon(tx, ty, angle, sx, sy), closed=True,
            facecolor=color, edgecolor="#333", linewidth=0.5, alpha=alpha,
        ))


def _sc_to_rgb(sc: np.ndarray, valid: np.ndarray,
               lo_hi: tuple[np.ndarray, np.ndarray] | None = None,
               ) -> np.ndarray:
    rgb = np.zeros(sc.shape[1:] + (3,), dtype=np.float32)
    for c in range(3):
        if lo_hi is not None:
            lo, hi = lo_hi[0][c], lo_hi[1][c]
        else:
            ch = sc[c][valid] if valid.any() else sc[c].ravel()
            if ch.size == 0:
                continue
            lo, hi = np.percentile(ch, [1, 99])
        if hi - lo < 1e-6:
            continue
        rgb[..., c] = np.clip((sc[c] - lo) / (hi - lo), 0, 1)
    rgb[~valid] = 0.0
    return rgb


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None)
    p.add_argument("--path", type=int, default=13, help="Path id (default 13 = best val)")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "async_collection"))
    p.add_argument("--weights",
                   default=str(ROOT / "runs" / "_weights" / "ace_encoder_pretrained.pt"))
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--step", type=int, default=1,
                   help="Sub-sample factor on frames (1 = every frame).")
    p.add_argument("--every", type=int, default=1)
    return p.parse_args()


def _fig_to_array(fig) -> np.ndarray:
    """Rasterise a matplotlib figure to (H, W, 3) uint8."""
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def main() -> None:
    args = _parse_args()

    run_dir = _pick_run(args.run)
    ckpt = run_dir / "head.pt"
    print(f"run dir: {run_dir}")
    print(f"head:    {ckpt}")

    # Scene centre (match trainer).
    train_path_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    scene_ds = SCRDataset(data_dir=args.data_dir, path_ids=train_path_ids,
                          stride=8, preload_depth=False)
    mean_xyz = torch.from_numpy(scene_ds.mean_camera_translation)
    del scene_ds

    ds = SCRDataset(data_dir=args.data_dir, path_ids=[args.path],
                    stride=8, preload_depth=False)
    print(f"path {args.path}: {len(ds)} frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACEScrRegressor(
        mean_xyz=mean_xyz, weights_path=args.weights,
    ).to(device).eval()
    model.head.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))

    # PnP pose per frame: reuse the per-frame CSV if present, else compute.
    pnp_csv_candidates = [
        run_dir / "pnp_eval_val_frames.csv",
        run_dir / "pnp_eval_test_frames.csv",
    ]
    pnp_df = None
    for p in pnp_csv_candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["path_id"] == args.path]
        if not df.empty:
            pnp_df = df.reset_index(drop=True)
            print(f"loaded PnP CSV for path {args.path}: {len(pnp_df)} rows from {p.name}")
            break
    if pnp_df is None:
        raise SystemExit(f"No PnP CSV for path {args.path}. Run eval_ace_scr_pnp.py --split val/test.")

    # GT trajectory for context.
    gt_df = pd.read_csv(Path(args.data_dir) / f"path_{args.path:02d}" / "ground_truth.csv")

    # Match each dataset frame to its PnP row by sim_time (they're generated
    # from the same source so equality works; fall back to nearest).
    pnp_times = pnp_df["sim_time"].values

    # Pre-scan GT coord range for consistent XYZ→RGB mapping across frames.
    # Compute once on ~32 sampled frames rather than all of them.
    print("computing global XYZ colour range (from GT over whole path)...")
    gt_all = []
    probe_idx = np.linspace(0, len(ds) - 1, min(32, len(ds)), dtype=int)
    for i in probe_idx:
        s = ds[int(i)]
        sc = s["scene_coords"].numpy()
        valid = s["valid_mask"].numpy()
        if valid.any():
            gt_all.append(sc[:, valid].reshape(3, -1))
    gt_all = np.concatenate(gt_all, axis=1)
    lo = np.percentile(gt_all, 1, axis=1)
    hi = np.percentile(gt_all, 99, axis=1)
    print(f"  XYZ range lo={lo}  hi={hi}")

    frames_idx = list(range(0, len(ds), max(1, args.step)))
    out_mp4 = run_dir / f"video_ace_scr_path{args.path:02d}.mp4"
    print(f"rendering {len(frames_idx)} frames to {out_mp4} @ {args.fps} fps")

    writer = imageio.get_writer(out_mp4, fps=args.fps, codec="libx264",
                                 quality=8, pixelformat="yuv420p",
                                 macro_block_size=None)

    # Rolling PnP history (camera centre trail).
    pnp_xy_trail: list[tuple[float, float]] = []

    try:
        for k, i in enumerate(frames_idx):
            s = ds[i]
            sim_time = float(s["sim_time"])
            rgb = s["rgb"]  # (3, H, W)
            gt_sc = s["scene_coords"].numpy()
            valid = s["valid_mask"].numpy()

            with torch.no_grad():
                pred = model(rgb.unsqueeze(0).to(device)).cpu().numpy()[0]

            # PnP row with closest sim_time (should usually match exactly).
            j = int(np.argmin(np.abs(pnp_times - sim_time)))
            row = pnp_df.iloc[j]
            if bool(row["ok"]):
                pnp_xy_trail.append((float(row["pred_cam_x"]),
                                      float(row["pred_cam_y"])))

            err = np.linalg.norm(pred - gt_sc, axis=0)
            valid_err = err[valid]
            med = float(np.median(valid_err)) if valid_err.size else float("nan")

            # --- draw composite figure ---
            fig = plt.figure(figsize=(13.5, 8.5), dpi=100)
            gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15],
                                  hspace=0.18, wspace=0.06)
            ax_rgb = fig.add_subplot(gs[0, 0])
            ax_gt  = fig.add_subplot(gs[0, 1])
            ax_pr  = fig.add_subplot(gs[0, 2])
            ax_map = fig.add_subplot(gs[1, :])

            ax_rgb.imshow(rgb.permute(1, 2, 0).numpy())
            ax_rgb.set_title(f"RGB  (path {args.path:02d} frame {i}/{len(ds)-1}  "
                             f"t={sim_time:.1f}s)")
            ax_rgb.axis("off")

            ax_gt.imshow(_sc_to_rgb(gt_sc, valid, lo_hi=(lo, hi)))
            ax_gt.set_title("GT scene coord  (XYZ→RGB)")
            ax_gt.axis("off")

            ax_pr.imshow(_sc_to_rgb(pred, valid, lo_hi=(lo, hi)))
            err_str = "n/a" if math.isnan(med) else f"{med:.2f} m"
            ax_pr.set_title(f"Predicted scene coord   (this-frame median = {err_str})")
            ax_pr.axis("off")

            # Bird's-eye map
            _draw_walls(ax_map)
            ax_map.plot(gt_df["gt_x"], gt_df["gt_y"], color="#1f77b4",
                        lw=1.8, alpha=0.7, label="GT trajectory")
            if pnp_xy_trail:
                tx = [p[0] for p in pnp_xy_trail]
                ty = [p[1] for p in pnp_xy_trail]
                ax_map.plot(tx, ty, color="#d62728", lw=1.2, alpha=0.6,
                            label="PnP trail")
                ax_map.scatter(tx[-1], ty[-1], s=90, color="#d62728",
                                edgecolor="black", linewidth=0.7, zorder=5,
                                label="PnP (now)")
            ax_map.scatter(row["gt_x"], row["gt_y"], s=90, color="#1f77b4",
                           edgecolor="black", linewidth=0.7, zorder=5,
                           label="GT (now)")
            ax_map.set_aspect("equal")
            ax_map.set_xlim(-14.5, 5.5); ax_map.set_ylim(-19.5, 3.5)
            ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
            ax_map.grid(alpha=0.2)
            euc = row["euclid_xy_to_gt"] if bool(row["ok"]) else float("nan")
            ax_map.set_title(
                f"Floor-plan view — rolling PnP camera-centre estimate  "
                f"|  current (x,y) err = "
                f"{'PnP failed' if math.isnan(euc) else f'{euc:.2f} m'}"
            )
            ax_map.legend(loc="upper right", fontsize=9)

            frame_img = _fig_to_array(fig)
            plt.close(fig)
            writer.append_data(frame_img)

            if (k % max(1, len(frames_idx) // 10)) == 0:
                print(f"  {k+1}/{len(frames_idx)}  t={sim_time:.1f}s")
    finally:
        writer.close()

    print(f"\nsaved: {out_mp4}")


if __name__ == "__main__":
    main()
