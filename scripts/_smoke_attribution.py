"""Smoke test for per-prediction modality attribution.

Trains a tiny query-readout fusion model for a few epochs on simulation,
then exercises FusionTrainer.log_attribution on one val path. Verifies:
  1. forward_attribution returns attn that sums (with CLS) to ~1 per row.
  2. log_attribution writes a JSON and the records carry attn_pct + avail.
  3. The 'cls' readout path raises a clear error (attribution undefined).

Run: .venv/Scripts/python.exe scripts/_smoke_attribution.py
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

MODS = ["imu", "odom", "wifi"]


def _build():
    dm = FusionDataModule.from_paths(modalities=MODS, batch_size=128)
    dm.setup()
    enc = {
        "imu": IMUCNN(in_features=9, embed_dim=128),
        "odom": OdomCNN(in_features=5, embed_dim=128),
        "wifi": Anchor2Vec(n_aps=dm.train_ds.num_wifi_aps, embed_dim=128),
    }
    return dm, enc


def main():
    dm, enc = _build()
    model = FusionTransformer(enc, embed_dim=128, depth=2, n_heads=4,
                              readout="query")
    tr = FusionTrainer(model, dm, MODS, lr=1e-3, patience=10,
                       n_instants=4, instant_stride=9)
    tr.fit(epochs=4, verbose=False)

    # Pick the longest val path.
    counts: dict[int, int] = {}
    for r in dm.val_ds._gt_rows:
        counts[r["path_id"]] = counts.get(r["path_id"], 0) + 1
    pid = max(counts, key=counts.get)
    print(f"=== attribution on val path_{pid:02d} ({counts[pid]} samples) ===")

    recs = tr.log_attribution("val", path_id=pid, max_samples=60, verbose=True)

    # 1. Attn + CLS sum to ~1.
    for r in recs[:20]:
        total = sum(r["attn_pct"].values()) + r["attn_cls_pct"]
        assert abs(total - 100.0) < 1.0, (total, r)
    print("\n  attn+CLS sums to 100% per row: OK")

    # 2. JSON written.
    out = tr.run_path / f"attribution_val_path{pid:02d}.json"
    assert out.exists(), out
    data = json.loads(out.read_text())
    assert len(data) == len(recs)
    assert "attn_pct" in data[0] and "avail_frac" in data[0]
    print(f"  JSON written: {out} ({len(data)} records)")

    # 3. CLS readout raises.
    model_cls = FusionTransformer(enc, embed_dim=128, depth=2, n_heads=4,
                                  readout="cls")
    B = 4
    inp = {m: torch.randn(B, 1, *( (32, 9) if m == 'imu' else (16, 5) if m == 'odom' else (1, dm.train_ds.num_wifi_aps))) for m in MODS}
    av = {m: torch.ones(B, 1, dtype=torch.bool) for m in MODS}
    dt = {m: torch.zeros(B, 1) for m in MODS}
    try:
        model_cls.forward_attribution(inp, av, dt)
    except ValueError as e:
        print(f"  cls readout correctly raised: {type(e).__name__}")
    else:
        raise AssertionError("cls readout should have raised")

    print("\nPASS - attribution smoke")


if __name__ == "__main__":
    main()
