"""PROBE 6 — what is the trained model actually doing?

Loads a saved fusion model (default: the query-readout IPIN bake-off run),
predicts on val, and tests for centroid collapse:
  * spread of predictions vs spread of GT (std in x,y)
  * mean prediction distance to the TRAIN centroid
  * per-path val MAE (is the aggregate hiding bimodal behavior?)
  * correlation of prediction with GT (is it tracking at all?)

Run: .venv/Scripts/python.exe scripts/inspect_06_model_behavior.py [run_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "fusion_20260521_103027"
DATASET = "ipin2024_floor-2"


def main():
    import json
    meta = json.loads((RUN / "meta.json").read_text())
    print(f"run={RUN.name}  readout={meta.get('readout')}  "
          f"modalities={meta['modalities']}  best from history")

    cfg = load_config(DATASET)
    cfg.model.readout = meta.get("readout", "query")
    dm = build_datamodule(cfg)
    enc, _ = build_encoders(cfg, dm)
    model = build_model(cfg, enc)
    sd = torch.load(RUN / "model.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(sd)
    trainer = build_trainer(cfg, model, dm)

    pred, tgt = trainer.predict("val")
    pred, tgt = pred.numpy(), tgt.numpy()
    train_cen = dm.train_ds._targets.numpy().mean(0)

    print(f"\n  val MAE = {np.linalg.norm(pred - tgt, axis=1).mean():.2f} m")
    print(f"  GT   std (x,y) = ({tgt[:,0].std():.1f}, {tgt[:,1].std():.1f}) m")
    print(f"  PRED std (x,y) = ({pred[:,0].std():.1f}, {pred[:,1].std():.1f}) m")
    shrink = 1 - (pred.std(0).mean() / tgt.std(0).mean())
    print(f"  >>> prediction spread shrinkage vs GT = {shrink*100:.0f}%  "
          f"(high = collapsing toward a point)")
    print(f"  mean pred -> train-centroid dist = "
          f"{np.linalg.norm(pred - train_cen, axis=1).mean():.2f} m")
    print(f"  GT      -> train-centroid dist = "
          f"{np.linalg.norm(tgt - train_cen, axis=1).mean():.2f} m")
    # tracking: correlation of pred vs gt per axis
    cx = np.corrcoef(pred[:, 0], tgt[:, 0])[0, 1]
    cy = np.corrcoef(pred[:, 1], tgt[:, 1])[0, 1]
    print(f"  pred-vs-GT correlation: x={cx:.2f}  y={cy:.2f}  "
          f"(1=tracks, 0=ignores GT)")

    # per-path
    pids = np.array([r["path_id"] for r in dm.val_ds._gt_rows])
    print("\n  per-path val MAE:")
    for p in np.unique(pids):
        m = pids == p
        e = np.linalg.norm(pred[m] - tgt[m], axis=1).mean()
        print(f"    path_{p:02d}: {e:.2f} m   (n={m.sum()})")


if __name__ == "__main__":
    main()
