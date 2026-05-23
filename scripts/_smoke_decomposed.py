"""Smoke test for the decomposed readout (Proposal 1).

Phase 1 — shapes / NaN safety:
  * forward returns (B,2); return_parts gives p_abs/delta/gate.
  * finite outputs when modalities are dropped, including ALL dropped
    (CLS rescue must hold for both anchor and motion queries).
  * forward_attribution returns gate + motion_frac.
Phase 2 — can it learn:
  * 5-epoch train on simulation; train loss decreases; post-fit
    diagnostics + attribution (with gate/motion_frac) run.

Run: .venv/Scripts/python.exe scripts/_smoke_decomposed.py
"""
from __future__ import annotations

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

MODS = ["imu", "odom", "wifi"]


def _encoders(n_aps=117):
    return {
        "imu": IMUCNN(in_features=9, embed_dim=128),
        "odom": OdomCNN(in_features=5, embed_dim=128),
        "wifi": Anchor2Vec(n_aps=n_aps, embed_dim=128),
    }


def phase1():
    print("=== Phase 1: decomposed shapes / NaN ===")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionTransformer(_encoders(), embed_dim=128, depth=2, n_heads=4,
                              readout="decomposed",
                              absolute_modalities={"wifi"}).to(dev)
    B, K = 8, 4
    inp = {
        "imu": torch.randn(B, K, 32, 9, device=dev),
        "odom": torch.randn(B, K, 16, 5, device=dev),
        "wifi": torch.randn(B, K, 1, 117, device=dev),
    }
    avail = {m: torch.ones(B, K, dtype=torch.bool, device=dev) for m in MODS}
    dt = {m: torch.zeros(B, K, device=dev) for m in MODS}
    qdt = torch.zeros(B, device=dev)

    out = model(inp, avail, dt, query_dt=qdt)
    assert out.shape == (B, 2), out.shape
    assert torch.isfinite(out).all()

    pred, parts = model(inp, avail, dt, query_dt=qdt, return_parts=True)
    assert parts["p_abs"].shape == (B, 2)
    assert parts["delta"].shape == (B, 2)
    assert parts["gate"].shape == (B, 1)
    assert torch.allclose(pred, parts["p_abs"] + parts["gate"] * parts["delta"], atol=1e-5)
    print(f"  pred {tuple(out.shape)}  gate mean={parts['gate'].mean():.3f}  OK")

    # Drop WiFi (the only absolute modality) for half the batch — anchor
    # query then sees only CLS for those rows. Must stay finite.
    avail["wifi"][:B // 2] = False
    assert torch.isfinite(model(inp, avail, dt, query_dt=qdt)).all(), "NaN when wifi dropped"

    # Drop everything — both queries see only CLS.
    allgone = {m: torch.zeros(B, K, dtype=torch.bool, device=dev) for m in MODS}
    assert torch.isfinite(model(inp, allgone, dt, query_dt=qdt)).all(), "NaN all-dropped"

    # Attribution
    _, attr = model.forward_attribution(inp, avail, dt, query_dt=qdt)
    assert "gate" in attr and "motion_frac" in attr
    assert attr["attn"].shape == (B, 3)
    print(f"  attribution gate mean={attr['gate'].mean():.3f}  "
          f"motion_frac mean={attr['motion_frac'].mean():.3f}  OK")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params/1e6:.2f}M")
    print("  PASS\n")


def phase2():
    print("=== Phase 2: 5-epoch decomposed train on sim ===")
    dm = FusionDataModule.from_paths(modalities=MODS, batch_size=128)
    dm.setup()
    model = FusionTransformer(_encoders(dm.train_ds.num_wifi_aps),
                              embed_dim=128, depth=2, n_heads=4,
                              readout="decomposed", absolute_modalities={"wifi"})
    tr = FusionTrainer(model, dm, MODS, lr=1e-3, patience=10,
                       n_instants=4, instant_stride=9,
                       modality_dropout=0.4, modality_balanced_loss=True,
                       aux_abs_weight=0.5)
    hist = tr.fit(epochs=5, verbose=True)
    assert hist.train_loss[0] > hist.train_loss[-1], "train loss must drop"
    # Attribution with gate/motion_frac
    counts = {}
    for r in dm.val_ds._gt_rows:
        counts[r["path_id"]] = counts.get(r["path_id"], 0) + 1
    pid = max(counts, key=counts.get)
    recs = tr.log_attribution("val", path_id=pid, max_samples=40, verbose=False)
    assert "gate" in recs[0] and "motion_frac" in recs[0], recs[0].keys()
    import numpy as np
    mf = float(np.mean([r["motion_frac"] for r in recs]))
    g = float(np.mean([r["gate"] for r in recs]))
    print(f"\n  val attribution: mean gate={g:.3f}  mean motion_frac={mf:.3f}")
    print("  PASS\n")


if __name__ == "__main__":
    phase1()
    phase2()
    print("ALL PASS - decomposed smoke")
