"""Lead 2 — RSSI MixStyle session augmentation.

Augment WiFi scans during training with MixStyle: blend per-scan (mu, sigma)
statistics with another random scan's stats from the same batch, plus mild
jitter and AP-dropout. Pushes the WiFi encoder to be invariant to per-scan
calibration shifts.

Zero architectural change — wraps the FusionTransformer with a stochastic
input pre-processor.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class RSSIMixStyle(nn.Module):
    """In-batch MixStyle on WiFi scans + mild jitter + AP-dropout."""

    def __init__(self, p: float = 0.5, alpha: float = 0.1,
                 jitter_db: float = 6.0, drop_p: float = 0.10):
        super().__init__()
        self.p = p
        self.jitter = jitter_db
        self.drop_p = drop_p
        self.beta = Beta(torch.tensor([alpha]), torch.tensor([alpha]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        B = x.shape[0]
        last = x.shape[-1]
        flat = x.reshape(B, -1, last)                              # (B, *prod, n_aps)
        perm = torch.randperm(B, device=x.device)
        mask = (flat > 0.005).float()
        cnt = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mu = (flat * mask).sum(dim=-1, keepdim=True) / cnt
        sigma = (((flat - mu) ** 2 * mask).sum(dim=-1, keepdim=True) / cnt + 1e-4).sqrt()
        mu_j = mu[perm]
        sigma_j = sigma[perm]
        lam = self.beta.sample((B,)).to(x.device).view(B, 1, 1)
        mu_b = lam * mu + (1 - lam) * mu_j
        sigma_b = lam * sigma + (1 - lam) * sigma_j
        out = sigma_b * (flat - mu) / sigma + mu_b
        jit = (torch.rand_like(out) * 2 - 1) * (self.jitter / 100.0)
        out = (out + jit * mask).clamp(0.0, 1.0)
        drop = (torch.rand(B, 1, last, device=x.device) < self.drop_p).float()
        out = out * (1 - drop)
        return out.reshape(x.shape)


class MixStyleFusionWrapper(nn.Module):
    """Wraps a FusionTransformer; mixes only the WiFi input during training."""

    def __init__(self, model, mixer):
        super().__init__()
        self.model = model
        self.mixer = mixer
        self.modalities = model.modalities
        self.readout = getattr(model, "readout", "cls")

    def forward(self, inputs, avail, dt, query_dt=None, **kw):
        if self.training and "wifi" in inputs:
            inputs = {**inputs, "wifi": self.mixer(inputs["wifi"])}
        return self.model(inputs, avail, dt, query_dt=query_dt, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msiln_site1_b1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.dataset)
    cfg.temporal.n_instants = 4
    cfg.train.modality_balanced_loss = False

    dm = build_datamodule(cfg)
    encoders, vision = build_encoders(cfg, dm)
    base_model = build_model(cfg, encoders)
    extra = {}

    mixer = RSSIMixStyle(p=0.5, alpha=0.1, jitter_db=6.0, drop_p=0.10)
    model = MixStyleFusionWrapper(base_model, mixer)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lead2] dataset={args.dataset} params={n_params/1e6:.3f} M", flush=True)

    run_root = REPO / "runs" / "experiments" / f"lead2_mix_{args.dataset}_s{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, model, dm, extra_inputs=extra,
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

    print(f"\n[lead2] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f} test={test_mae:.3f} ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
