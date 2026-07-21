"""Idea 2: Neural Gaussian-splat place posterior.

Each WiFi AP is a learnable 2D Gaussian splat (mu_i, log_sigma_i) in (x, y) space.
An RSSI scan is turned into a soft mixture over those splats, which gives a
Gaussian *place posterior* (mu_w, sigma_w) explicitly. The IMU window predicts
a corrective displacement; a small MLP fuses the 5-vector
[mu_x, mu_y, sigma_x, sigma_y, IMU correction magnitude] into the final (x, y).

Hybrid neural + analytic: the per-AP Gaussian parameters are learned; the
mixing is analytic Bayes; the network is only the per-AP table + a small
MLP head + the IMU corrector.

Tested on Webots simulation_2mod, K=1 single-instant, seed 42, 30 epochs.
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


class IMUCorrector(nn.Module):
    """Small 1D-CNN over the IMU window -> 2D corrective displacement."""

    def __init__(self, imu_dim: int = 9, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(imu_dim, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, imu):                                       # (B, T, F)
        z = self.conv(imu.transpose(1, 2)).mean(dim=-1)           # (B, hidden)
        return self.head(z)


class GaussianSplatPlace(nn.Module):
    """RSSI scan -> mixture-of-Gaussians place posterior.

    Each of n_aps APs has a learned (mu_i in R^2, log_sigma_i in R^2). Soft
    mixture weights w come from a learned linear of the RSSI scan. The
    posterior moments are:
        mu_w     = sum_i w_i * mu_i
        sigma_w^2 = sum_i w_i * (sigma_i^2 + mu_i^2) - mu_w^2  (mixture variance)
    """

    def __init__(self, n_aps: int, init_scale: float = 5.0):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n_aps, 2) * init_scale)
        self.log_sigma = nn.Parameter(torch.zeros(n_aps, 2))
        self.score = nn.Linear(n_aps, n_aps)                      # learned scan -> AP weights

    def forward(self, scan):                                      # (B, n_aps)
        w = torch.softmax(self.score(scan), dim=-1)               # (B, n_aps)
        sigma2 = torch.exp(2.0 * self.log_sigma)                  # (n_aps, 2)
        mu_w = w @ self.mu                                        # (B, 2)
        second = w @ (sigma2 + self.mu.pow(2))                    # (B, 2)
        sigma_w2 = (second - mu_w.pow(2)).clamp(min=1e-6)         # (B, 2)
        return mu_w, sigma_w2.sqrt()                              # (B, 2), (B, 2)


class SplatFusion(nn.Module):
    """Gaussian-splat WiFi + IMU corrector -> (x, y)."""

    def __init__(self, n_aps: int, imu_dim: int = 9, init_scale: float = 5.0):
        super().__init__()
        self.place = GaussianSplatPlace(n_aps, init_scale=init_scale)
        self.imu = IMUCorrector(imu_dim=imu_dim)
        # Fuse [mu, sigma, imu_correction, imu_correction_magnitude] -> (x, y)
        self.fuse = nn.Sequential(
            nn.Linear(2 + 2 + 2 + 1, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, wifi, imu):
        mu_w, sigma_w = self.place(wifi)                          # (B, 2), (B, 2)
        delta = self.imu(imu)                                     # (B, 2)
        mag = delta.norm(dim=-1, keepdim=True)                    # (B, 1)
        z = torch.cat([mu_w, sigma_w, delta, mag], dim=-1)        # (B, 7)
        return self.fuse(z)


class FusionTrainerAdapter(nn.Module):
    readout = "cls"

    def __init__(self, n_aps: int, imu_dim: int = 9, init_scale: float = 5.0):
        super().__init__()
        self.modalities = ["wifi", "imu"]
        self.inner = SplatFusion(n_aps=n_aps, imu_dim=imu_dim, init_scale=init_scale)

    def forward(self, inputs, avail, dt, query_dt=None, **kwargs):
        w = inputs["wifi"]                                        # (B,1,1,n_aps)
        i = inputs["imu"]                                         # (B,1,T,F)
        B = w.shape[0]
        w = w.reshape(B, -1)
        i = i.reshape(B, i.shape[-2], i.shape[-1])
        return self.inner(w, i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="simulation_2mod")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.dataset)
    cfg.temporal.n_instants = 1
    cfg.train.modality_balanced_loss = False
    cfg.train.modality_dropout = 0.0
    cfg.train.instant_dropout = 0.0

    dm = build_datamodule(cfg)
    n_aps = int(dm.train_ds.feature_dims["wifi"])
    imu_dim = int(dm.train_ds.feature_dims["imu"])
    y_train = dm.train_ds._targets
    init_scale = float(y_train.abs().mean().item())
    print(f"[idea2] dataset={args.dataset} K=1 n_aps={n_aps} imu_dim={imu_dim} "
          f"init_scale={init_scale:.2f}", flush=True)

    model = FusionTrainerAdapter(n_aps=n_aps, imu_dim=imu_dim, init_scale=init_scale)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[idea2] params={n_params/1e6:.3f} M", flush=True)

    run_root = REPO / "runs" / "experiments" / f"idea2_{args.dataset}_s{args.seed}"
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

    print(f"\n[idea2] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f}  test={test_mae:.3f}  ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
