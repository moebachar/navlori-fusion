"""Train the ACE scene-specific head on NavLoRI data.

Uses all training paths by default (per configs/data/simulation.yaml split).

Usage (from repo root):
    python scripts/train_ace_scr.py
    python scripts/train_ace_scr.py --epochs 50 --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.data.scr_dataset import SCRDataset
from src.pipeline.encoders import ACEScrRegressor
from src.pipeline.training import SCRTrainer


# Splits — mirror configs/data/simulation.yaml
TRAIN_PATHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data" / "async_collection"))
    p.add_argument("--weights",
                   default=str(ROOT / "runs" / "_weights" / "ace_encoder_pretrained.pt"))
    p.add_argument("--run-dir", default=str(ROOT / "runs"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preload-depth", action="store_true",
                   help="Decode every depth PNG into memory up-front.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("--- datasets ---")
    train_ds = SCRDataset(
        data_dir=args.data_dir,
        path_ids=TRAIN_PATHS,
        preload_depth=args.preload_depth,
    )
    val_ds = SCRDataset(
        data_dir=args.data_dir,
        path_ids=VAL_PATHS,
        preload_depth=args.preload_depth,
    )
    print(f"  train: {len(train_ds)} frames across paths {TRAIN_PATHS}")
    print(f"  val:   {len(val_ds)} frames across paths {VAL_PATHS}")
    mean_xyz = torch.from_numpy(train_ds.mean_camera_translation)
    print(f"  scene centre (mean train cam): {mean_xyz.tolist()}")

    print("\n--- model ---")
    regressor = ACEScrRegressor(
        mean_xyz=mean_xyz,
        num_head_blocks=1,
        use_homogeneous=True,
        weights_path=args.weights,
    )
    n_train = sum(p.numel() for p in regressor.head.parameters())
    print(f"  trainable head params: {n_train/1e6:.2f}M")

    print("\n--- training ---")
    trainer = SCRTrainer(
        regressor=regressor,
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        run_dir=args.run_dir,
    )
    history = trainer.fit(epochs=args.epochs, verbose=True)

    print("\n--- evaluation ---")
    results = trainer.evaluate()
    o = results["overall"]
    print(f"  overall  median={o['median']:.3f}m  "
          f"mean={o['mean']:.3f}m  p95={o['p95']:.3f}m  n={o['n']}")
    print("  per path:")
    for pid, s in sorted(results["per_path"].items(), key=lambda kv: int(kv[0])):
        print(f"    path {pid:>2}: median={s['median']:.3f}m  "
              f"p95={s['p95']:.3f}m  n={s['n']}")

    print(f"\nartifacts: {trainer.run_path}")


if __name__ == "__main__":
    main()
