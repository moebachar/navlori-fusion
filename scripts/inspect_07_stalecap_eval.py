"""M2 eval — split val error by WiFi availability under the staleness cap.

Loads a trained wifi-only model run, predicts on IPIN val, and reports MAE
on (a) samples with fresh WiFi (available) vs (b) samples whose WiFi was
capped-out (no fresh fix). The cap is validated if the FRESH subset is good
(cleaner encoder) — the capped subset is honestly prior-level by design.

Run: .venv/Scripts/python.exe scripts/inspect_07_stalecap_eval.py <run_dir>
"""
from __future__ import annotations

import json
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

RUN = Path(sys.argv[1])
DATASET = "ipin2024_floor-2"


def main():
    meta = json.loads((RUN / "meta.json").read_text())
    cfg = load_config(DATASET)
    cfg.dataset.modalities = meta["modalities"]
    cfg.model.readout = meta.get("readout", "query")
    dm = build_datamodule(cfg)
    enc, _ = build_encoders(cfg, dm)
    model = build_model(cfg, enc)
    model.load_state_dict(torch.load(RUN / "model.pt", weights_only=True, map_location="cpu"))
    trainer = build_trainer(cfg, model, dm)

    pred, tgt = trainer.predict("val")
    pred, tgt = pred.numpy(), tgt.numpy()
    err = np.linalg.norm(pred - tgt, axis=1)

    Xv = dm.val_ds.get_tensors("wifi")[0]
    avail = (Xv.flatten(1).abs().sum(1) > 0).numpy()

    print(f"run={RUN.name}  modalities={meta['modalities']}  readout={meta.get('readout')}")
    print(f"  overall val MAE        = {err.mean():.2f} m  (n={len(err)})")
    print(f"  FRESH-wifi subset MAE  = {err[avail].mean():.2f} m  "
          f"(n={avail.sum()}, {avail.mean()*100:.0f}%)")
    print(f"  capped-out subset MAE  = {err[~avail].mean():.2f} m  "
          f"(n={(~avail).sum()}, {(~avail).mean()*100:.0f}%)")
    cen = dm.train_ds._targets.numpy().mean(0)
    cen_err = np.linalg.norm(tgt[~avail] - cen, axis=1).mean()
    print(f"  (capped-out vs centroid baseline = {cen_err:.2f} m — these are "
          f"genuinely WiFi-less samples)")


if __name__ == "__main__":
    main()
