"""Smoke test for Action 3: modality-balanced loss + post-fit diagnostics.

Verifies:
  1. ``FusionTrainer(modality_balanced_loss=True)`` constructs and trains
     without crashing; a 3-epoch run on simulation finishes.
  2. ``meta.json`` records the new flags.
  3. Post-fit diagnostics print the subset table and (when the baseline
     file exists) the baseline comparison block.
  4. Modality-balanced training actually runs the leave-one-out branch
     (we can't easily assert this without instrumenting, but we verify
     it doesn't make the model NaN and that loss still decreases).

Run: ``.venv/Scripts/python.exe scripts/_smoke_action3.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data.datamodule import FusionDataModule  # noqa: E402
from src.pipeline.encoders import Anchor2Vec, IMUCNN, OdomCNN  # noqa: E402
from src.pipeline.fusion.transformer import FusionTransformer  # noqa: E402
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402


def main():
    mods = ["imu", "odom", "wifi"]
    dm = FusionDataModule.from_paths(modalities=mods, batch_size=128)
    dm.setup()

    n_aps = dm.train_ds.num_wifi_aps
    encoders = {
        "imu": IMUCNN(in_features=9, embed_dim=128),
        "odom": OdomCNN(in_features=5, embed_dim=128),
        "wifi": Anchor2Vec(n_aps=n_aps, embed_dim=128),
    }
    model = FusionTransformer(encoders, embed_dim=128, depth=2, n_heads=4)

    baselines_path = ROOT / "runs" / "baselines" / "simulation" / "baselines.json"
    assert baselines_path.exists(), \
        f"Need Action 1 baselines first: {baselines_path}"

    trainer = FusionTrainer(
        model, dm, mods,
        lr=1e-3, modality_dropout=0.4, patience=10,
        modality_balanced_loss=True, modality_balanced_weight=0.5,
        baselines_path=baselines_path,
    )

    print(f"=== 3-epoch run with modality_balanced_loss=True ===", flush=True)
    hist = trainer.fit(epochs=3, verbose=True)

    # Train loss must have decreased.
    assert hist.train_loss[0] > hist.train_loss[-1], hist.train_loss
    # Val MAE must be finite.
    assert all(torch.isfinite(torch.tensor(v)) for v in hist.val_mae), hist.val_mae

    # Meta records the new flags.
    meta = json.loads((trainer.run_path / "meta.json").read_text())
    assert meta["modality_balanced_loss"] is True, meta
    assert meta["modality_balanced_weight"] == 0.5, meta
    assert meta["modality_dropout"] == 0.4, meta

    # Subsets.json must exist after fit
    assert (trainer.run_path / "subsets.json").exists()
    subs = json.loads((trainer.run_path / "subsets.json").read_text())
    assert "all" in subs and "only:wifi" in subs

    print(f"\n  best val_mae (3 epoch) = {hist.best_val_mae:.3f}m")
    print(f"  run_dir = {trainer.run_path}")
    print(f"\nPASS - Action 3 smoke")


if __name__ == "__main__":
    main()
