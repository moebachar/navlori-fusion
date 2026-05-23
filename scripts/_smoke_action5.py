"""Smoke test for Action 5: pretrained Stage A loading.

Strategy: train a tiny IMU encoder for 1 epoch with EncoderTrainer (saves
encoder.pt), then build a fusion model with pretrained_paths={'imu': ...}
and verify (a) the encoder weights in the fusion model match what was
saved, (b) builder errors helpfully when paths are bogus.

Run: ``.venv/Scripts/python.exe scripts/_smoke_action5.py``
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data.datamodule import FusionDataModule  # noqa: E402
from src.pipeline.encoders import IMUCNN  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule,
    build_encoders,
    load_config,
    pretrained_paths_from_cfg,
)
from src.pipeline.training.trainer import EncoderTrainer  # noqa: E402


def _phase1_load_pretrained():
    print("=== Phase 1: train IMU 1 epoch -> reload in fusion builder ===")
    dm = FusionDataModule.from_paths(modalities=["imu"], batch_size=128)
    dm.setup()
    enc = IMUCNN(in_features=9, embed_dim=128)
    tr = EncoderTrainer(enc, modality="imu", dm=dm, lr=1e-3, patience=5,
                        target_mode="displacement", target_lookback_s=1.0)
    tr.fit(epochs=1, verbose=False)
    pretrained = tr.run_path / "encoder.pt"
    assert pretrained.exists(), pretrained

    # Snapshot a sample weight before reloading
    sd_disk = torch.load(pretrained, weights_only=True, map_location="cpu")
    sample_key = next(iter(sd_disk))
    sample_weight = sd_disk[sample_key].clone()

    cfg = load_config("simulation")
    cfg.dataset.modalities = ["imu"]  # only what we have a checkpoint for
    dm2 = build_datamodule(cfg)
    encoders, _ = build_encoders(cfg, dm2, pretrained_paths={"imu": pretrained})

    loaded = encoders["imu"].state_dict()[sample_key]
    assert torch.allclose(loaded.cpu(), sample_weight), \
        f"reloaded weight differs from saved at key {sample_key}"
    print(f"  IMU encoder loaded from {pretrained}")
    print(f"  match on key='{sample_key}' (mean={sample_weight.mean():.6f})")
    print("  PASS")


def _phase2_pretrained_paths_from_cfg():
    print("\n=== Phase 2: pretrained_paths_from_cfg ===")
    cfg = load_config("simulation")
    # Block defaults all-null -> empty dict
    p = pretrained_paths_from_cfg(cfg)
    assert p == {}, p
    # Set one path manually
    cfg.stage_a.pretrained.imu = "runs/_does_not_exist.pt"
    p = pretrained_paths_from_cfg(cfg)
    assert "imu" in p, p
    assert isinstance(p["imu"], Path)
    print("  cfg roundtrip OK")
    print("  PASS")


def _phase3_strict_load_errors_helpfully():
    print("\n=== Phase 3: strict load surfaces shape mismatch ===")
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["imu"]
    dm = build_datamodule(cfg)
    # A bogus state dict (zero tensors with wrong key) must raise.
    bad = Path(tempfile.gettempdir()) / "bogus.pt"
    torch.save({"not_a_real_key": torch.zeros(1)}, bad)
    try:
        build_encoders(cfg, dm, pretrained_paths={"imu": bad})
    except RuntimeError as e:
        print(f"  strict-load raised RuntimeError as expected: {type(e).__name__}")
        print("  PASS")
        return
    raise AssertionError("strict load should have raised")


def main():
    _phase1_load_pretrained()
    _phase2_pretrained_paths_from_cfg()
    _phase3_strict_load_errors_helpfully()
    print("\nALL PASS - Action 5 smoke")


if __name__ == "__main__":
    main()
