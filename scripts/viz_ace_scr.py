"""Visualise ACE SCR predictions on any frame from any path.

Usage:
    python scripts/viz_ace_scr.py                              # newest run, middle frame, path_01 (train) and path_2 (val)
    python scripts/viz_ace_scr.py --run runs/ace_scr_20260424_140000
    python scripts/viz_ace_scr.py --path 14 --frame 50

Saves a 2x3 panel PNG into the chosen run's directory. The figure has:
    row 1: RGB | GT scene coord (XYZ→RGB) | predicted scene coord
    row 2: validity mask | per-pixel 3D error (m) | text summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline.data.scr_dataset import SCRDataset
from src.pipeline.encoders import ACEScrRegressor


def _sc_to_rgb(sc: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map XYZ scene coords to RGB using 1st-99th percentile stretch."""
    rgb = np.zeros(sc.shape[1:] + (3,), dtype=np.float32)
    for c in range(3):
        ch = sc[c][valid]
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
    runs = sorted((ROOT / "runs").glob("ace_scr_*"))
    runs = [r for r in runs if (r / "head.pt").exists()
            and not r.name.startswith("ace_scr_overfit")
            and not r.name.startswith("ace_scr_smoketest")]
    if not runs:
        # Fall back to any ace_scr run with head.pt
        runs = [r for r in sorted((ROOT / "runs").glob("ace_scr_*"))
                if (r / "head.pt").exists()]
    if not runs:
        raise SystemExit("No ACE SCR runs with head.pt found. Train first.")
    return runs[-1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="Run dir (default: latest ace_scr_*)")
    p.add_argument("--path", type=int, default=2, help="Path id to visualise")
    p.add_argument("--frame", type=int, default=None, help="Frame index (default: middle)")
    p.add_argument("--weights",
                   default=str(ROOT / "runs" / "_weights" / "ace_encoder_pretrained.pt"))
    p.add_argument("--data-dir", default=str(ROOT / "data" / "async_collection"))
    p.add_argument("--mean-source", default=None,
                   help="Path id to compute scene-centre mean from "
                        "(default: use train split 1,3..12 like the trainer).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = _pick_run(args.run)
    ckpt = run_dir / "head.pt"
    print(f"run dir: {run_dir}")
    print(f"head checkpoint: {ckpt}")

    # Re-derive the scene centre the way the trainer did: from the training split.
    train_path_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    scene_ds = SCRDataset(
        data_dir=args.data_dir,
        path_ids=train_path_ids if args.mean_source is None else [int(args.mean_source)],
        stride=8,
        preload_depth=False,
    )
    mean_xyz = torch.from_numpy(scene_ds.mean_camera_translation)
    print(f"scene-centre mean (world xyz): {mean_xyz.tolist()}")
    del scene_ds

    # Target path
    ds = SCRDataset(
        data_dir=args.data_dir,
        path_ids=[args.path],
        stride=8,
        preload_depth=False,
    )
    idx = args.frame if args.frame is not None else len(ds) // 2
    idx = max(0, min(idx, len(ds) - 1))
    sample = ds[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACEScrRegressor(
        mean_xyz=mean_xyz, weights_path=args.weights,
    ).to(device).eval()
    model.head.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))

    rgb = sample["rgb"]
    gt_sc = sample["scene_coords"].numpy()
    valid = sample["valid_mask"].numpy()

    with torch.no_grad():
        pred = model(rgb.unsqueeze(0).to(device)).cpu().numpy()[0]

    err = np.linalg.norm(pred - gt_sc, axis=0)
    err_masked = np.where(valid, err, np.nan)
    ve = err[valid]
    med = float(np.median(ve)) if ve.size else float("nan")
    p95 = float(np.percentile(ve, 95)) if ve.size else float("nan")
    print(f"path_{args.path:02d} frame {idx}/{len(ds)}  sim_time={sample['sim_time']:.1f}s")
    print(f"  target: ({sample['target'][0]:.2f}, {sample['target'][1]:.2f})")
    print(f"  3D err: median={med:.3f}m  p95={p95:.3f}m  n={ve.size}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(rgb.permute(1, 2, 0).numpy())
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_sc_to_rgb(gt_sc, valid))
    axes[0, 1].set_title("GT scene coord (XYZ→RGB)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(_sc_to_rgb(pred, valid))
    axes[0, 2].set_title("Predicted scene coord")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(valid, cmap="gray")
    axes[1, 0].set_title(f"Validity  ({valid.sum()}/{valid.size})")
    axes[1, 0].axis("off")

    vmax = float(np.nanpercentile(err_masked, 95)) if ve.size else 1.0
    im = axes[1, 1].imshow(err_masked, vmin=0, vmax=max(vmax, 0.01), cmap="magma")
    axes[1, 1].set_title("Per-pixel 3D error (m)")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].axis("off")
    split_kind = ("train" if args.path in train_path_ids
                  else "val" if args.path in [2, 13, 14]
                  else "test" if args.path in [15, 16, 17]
                  else "unknown")
    summary = (
        f"Run: {run_dir.name}\n"
        f"Path: {args.path:02d} ({split_kind})\n"
        f"Frame: {idx}/{len(ds)}  t={sample['sim_time']:.1f}s\n"
        f"Robot target: ({sample['target'][0]:.2f}, {sample['target'][1]:.2f})\n"
        f"Valid: {int(valid.sum())}/{valid.size}\n\n"
        f"Median: {med:.3f} m\n"
        f"Mean:   {ve.mean() if ve.size else float('nan'):.3f} m\n"
        f"p95:    {p95:.3f} m\n"
        f"Max:    {ve.max() if ve.size else float('nan'):.3f} m"
    )
    axes[1, 2].text(0.02, 0.98, summary, va="top", family="monospace", fontsize=10)

    plt.suptitle(
        f"ACE SCR — {run_dir.name} — path_{args.path:02d} ({split_kind}) frame {idx}",
        y=1.02,
    )
    plt.tight_layout()

    out = run_dir / f"scr_viz_path{args.path:02d}_frame{idx:04d}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
