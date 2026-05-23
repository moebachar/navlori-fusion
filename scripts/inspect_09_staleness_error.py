"""Diagnose where the remaining error lives: error vs WiFi staleness.

For a trained model on IPIN val, bin per-sample error by how stale the most
recent real WiFi scan is. If error GROWS with staleness, the model inherits
stale-WiFi error (motion is NOT bridging the gaps). If error is FLAT, motion
is dead-reckoning between scans (the goal).

Run: .venv/Scripts/python.exe scripts/inspect_09_staleness_error.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)

RUN = Path(sys.argv[1])
DATASET = "ipin2024_floor-2"


def per_sample_staleness(dm):
    """For each val sample, seconds since the most recent real WiFi scan."""
    ds = dm.val_ds
    root = dm.data_dir
    # per path: scan times
    stale = np.full(len(ds._gt_rows), np.nan)
    by_path = {}
    for i, r in enumerate(ds._gt_rows):
        by_path.setdefault(r["path_id"], []).append((i, r["time"]))
    for pid, items in by_path.items():
        wf = root / f"path_{pid:02d}" / "wifi.csv"
        if not wf.exists():
            continue
        wt = np.sort(pd.read_csv(wf)["sim_time"].values)
        for i, t in items:
            j = np.searchsorted(wt, t, side="right") - 1
            stale[i] = (t - wt[j]) if j >= 0 else np.nan
    return stale


def main():
    meta = json.loads((RUN / "meta.json").read_text())
    cfg = load_config(DATASET)
    cfg.dataset.modalities = meta["modalities"]
    cfg.model.readout = meta.get("readout", "query")
    dm = build_datamodule(cfg)
    enc, _ = build_encoders(cfg, dm)
    model = build_model(cfg, enc)
    model.load_state_dict(torch.load(RUN / "model.pt", weights_only=True, map_location="cpu"))
    tr = build_trainer(cfg, model, dm)
    pred, tgt = tr.predict("val")
    err = np.linalg.norm(pred.numpy() - tgt.numpy(), axis=1)
    stale = per_sample_staleness(dm)

    m = np.isfinite(stale)
    err, stale = err[m], stale[m]
    print(f"run={RUN.name} readout={meta.get('readout')} overall val MAE={err.mean():.2f}m")
    print(f"  error binned by WiFi staleness:")
    bins = [(0, 1), (1, 3), (3, 6), (6, 15), (15, 1e9)]
    for lo, hi in bins:
        b = (stale >= lo) & (stale < hi)
        if b.sum() == 0:
            continue
        lab = f"{lo}-{hi if hi < 1e9 else 'inf'}s"
        print(f"    stale {lab:>9s}: MAE={err[b].mean():6.2f}m  "
              f"(n={b.sum():5d}, {b.mean()*100:4.1f}%)")
    # correlation staleness<->error
    c = np.corrcoef(stale, err)[0, 1]
    print(f"  corr(staleness, error) = {c:+.2f}  "
          f"(>0 => inheriting stale WiFi; ~0 => motion bridges)")


if __name__ == "__main__":
    main()
