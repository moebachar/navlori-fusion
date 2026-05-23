"""Smoke test for Action 2: displacement targets + odom column trim.

Verifies:
  1. ``OdomCNN`` default in_features is now 5; the old default (7) still
     works for back-compat callers that explicitly request it.
  2. ``FusionDataset.get_targets`` returns (target, valid) for both
     ``"position"`` and ``"displacement"`` modes, with the same N as
     ``_targets`` and the right shapes.
  3. Displacement targets are zero+invalid for the first ~lookback_s of
     each path, non-zero+valid for later samples in the same path.
  4. An ``EncoderTrainer`` with ``target_mode="displacement"`` trains an
     IMU encoder for 3 epochs and writes ``meta.json`` with the new fields.

Run: ``.venv/Scripts/python.exe scripts/_smoke_action2.py``
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
from src.pipeline.encoders import IMUCNN, OdomCNN  # noqa: E402
from src.pipeline.training.trainer import EncoderTrainer  # noqa: E402


def _t_odom_default():
    print("=== Phase 1: OdomCNN dim sanity ===")
    enc = OdomCNN()
    assert enc.in_features == 5, enc.in_features
    x = torch.randn(4, 16, 5)
    z = enc(x)
    assert z.shape == (4, 128), z.shape
    # Back-compat: explicit 7 still constructs correctly.
    enc7 = OdomCNN(in_features=7)
    z = enc7(torch.randn(4, 16, 7))
    assert z.shape == (4, 128)
    print("  OdomCNN(default).in_features =", enc.in_features, " (was 7, now 5)")
    print("  PASS")


def _t_get_targets():
    print("\n=== Phase 2: FusionDataset.get_targets ===")
    dm = FusionDataModule.from_paths(modalities=["imu"], batch_size=64)
    dm.setup()
    ds = dm.train_ds

    # Position mode = identity
    pos, val = ds.get_targets("position")
    assert torch.allclose(pos, ds._targets), "position mode must equal _targets"
    assert val.all(), "position mode must be all-valid"

    # Displacement mode
    delta, val = ds.get_targets("displacement", lookback_s=1.0)
    assert delta.shape == ds._targets.shape, delta.shape
    assert val.shape == (len(ds._targets),)
    invalid = (~val).sum().item()
    print(f"  N samples              = {len(ds._targets)}")
    print(f"  invalid (early-in-path)= {invalid}")
    print(f"  delta range             = "
          f"[{delta[val].min():.3f}, {delta[val].max():.3f}] m per axis")
    nonzero = delta[val].abs().sum(dim=1).gt(0).float().mean().item()
    print(f"  fraction nonzero        = {nonzero:.3f}")
    # Loose checks: most valid samples must have non-zero displacement
    assert invalid > 0, "expected SOME early-in-path samples to be invalid"
    assert nonzero > 0.9, f"most valid samples should have nonzero delta, got {nonzero}"
    print("  PASS")
    return dm


def _t_train_displacement(dm):
    print("\n=== Phase 3: EncoderTrainer with target_mode='displacement' ===")
    enc = IMUCNN()
    trainer = EncoderTrainer(
        enc, modality="imu", dm=dm,
        lr=1e-3, patience=10,
        target_mode="displacement", target_lookback_s=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    hist = trainer.fit(epochs=3, verbose=False)
    meta = json.loads((trainer.run_path / "meta.json").read_text())
    assert meta["target_mode"] == "displacement", meta
    assert meta["target_lookback_s"] == 1.0, meta
    print(f"  3-epoch val_mae (delta-position) = {hist.best_val_mae:.4f} m")
    print(f"  run_dir = {trainer.run_path}")
    assert hist.train_loss[0] > hist.train_loss[-1], \
        "train loss must decrease over 3 epochs"
    print("  PASS")


def main():
    _t_odom_default()
    dm = _t_get_targets()
    _t_train_displacement(dm)
    print("\nALL PASS — Action 2 smoke")


if __name__ == "__main__":
    main()
