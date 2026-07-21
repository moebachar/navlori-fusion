"""Idea 1: WiFi-as-positional-encoding for IMU.

ONE stream, one transformer. The WiFi scan is encoded once into a "place
embedding" and APPENDED to every IMU sample of the window. The whole network
sees a length-T sequence of place-conditioned IMU vectors and predicts (x, y).

No set transformer, no modality tokens, no time encoding. WiFi is not a token;
it is a *context* that modulates the IMU stream.

Tested on Webots simulation_2mod, K=1 single-instant, seed 42, 30 epochs.
Baseline reference: M1+M2 learned_continuous test = 0.448 +/- 0.044 m (K=4).
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_trainer, load_config,
)

import argparse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class PlaceConditionedIMU(nn.Module):
    """Single stream: place-conditioned IMU sequence -> (x, y)."""

    def __init__(self, n_aps: int, imu_dim: int = 9, T: int = 32,
                 place_dim: int = 64, d_model: int = 128,
                 nhead: int = 4, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.place = nn.Sequential(
            nn.Linear(n_aps, 128), nn.GELU(),
            nn.Linear(128, place_dim), nn.LayerNorm(place_dim),
        )
        self.imu_proj = nn.Linear(imu_dim, d_model - place_dim)
        self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 2),
        )

    def forward(self, wifi, imu):
        # wifi: (B, n_aps); imu: (B, T, imu_dim)
        place = self.place(wifi)                                # (B, P)
        imu_z = self.imu_proj(imu)                              # (B, T, D-P)
        place_b = place.unsqueeze(1).expand(-1, imu.size(1), -1)
        z = torch.cat([imu_z, place_b], dim=-1) + self.pos      # (B, T, D)
        z = self.trunk(z)
        z = self.norm(z.mean(dim=1))                            # pool over T
        return self.head(z)


class FusionTrainerAdapter(nn.Module):
    """Adapt PlaceConditionedIMU to the FusionTrainer's interface.

    FusionTrainer expects: model(inputs, avail, dt, query_dt) with
    inputs[mod] of shape (B, K, *window). For K=1 we collapse the
    instant dimension. avail / dt / query_dt are ignored (no async
    modality dropout for this experiment).
    """
    readout = "cls"  # disable aux losses in the trainer

    def __init__(self, n_aps: int, imu_dim: int = 9):
        super().__init__()
        self.modalities = ["wifi", "imu"]
        self.inner = PlaceConditionedIMU(n_aps=n_aps, imu_dim=imu_dim)

    def forward(self, inputs, avail, dt, query_dt=None, **kwargs):
        w = inputs["wifi"]                                       # (B,1,1,n_aps)
        i = inputs["imu"]                                        # (B,1,T,F)
        B = w.shape[0]
        w = w.reshape(B, -1)                                     # (B, n_aps)
        i = i.reshape(B, i.shape[-2], i.shape[-1])               # (B, T, F)
        return self.inner(w, i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="simulation_2mod")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.dataset)
    cfg.temporal.n_instants = 1                                  # K=1
    cfg.train.modality_balanced_loss = False
    cfg.train.modality_dropout = 0.0                             # single instant; no dropout
    cfg.train.instant_dropout = 0.0

    dm = build_datamodule(cfg)
    n_aps = int(dm.train_ds.feature_dims["wifi"])
    imu_dim = int(dm.train_ds.feature_dims["imu"])
    print(f"[idea1] dataset={args.dataset} K=1 n_aps={n_aps} imu_dim={imu_dim}",
          flush=True)

    model = FusionTrainerAdapter(n_aps=n_aps, imu_dim=imu_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[idea1] params={n_params/1e6:.3f} M", flush=True)

    run_root = REPO / "runs" / "experiments" / f"idea1_{args.dataset}_s{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg, model, dm, extra_inputs={},
                             run_dir=str(run_root))

    t0 = time.time()
    trainer.fit(epochs=args.epochs)
    elapsed = time.time() - t0

    preds_v, tgts_v = trainer.predict("val")
    val_mae = float((preds_v - tgts_v).norm(dim=1).mean())
    test_mae = float("nan")
    if "test" in trainer.splits:
        preds_t, tgts_t = trainer.predict("test")
        test_mae = float((preds_t - tgts_t).norm(dim=1).mean())

    print(f"\n[idea1] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f}  test={test_mae:.3f}  ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
