"""Lead 4 — Open-set retrieval gate / trust feature.

Compute a per-scan 'WiFi trust' scalar = sigmoid(-mahalanobis distance to
training-set mean per AP). Inject the trust feature into the WiFi encoder's
output token. The fusion transformer can then learn to lean on IMU when the
WiFi trust is low.

Targets the gate=0.03 failure mode of the decomposed-readout experiment by
giving the model an EXPLICIT input signal for 'WiFi is OOD now'.
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

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class TrustGatedWiFi(nn.Module):
    """Wrap any WiFi encoder; adds a trust-derived residual to its output."""

    def __init__(self, base, n_aps: int, embed_dim: int = 128):
        super().__init__()
        self.base = base
        self.n_aps = n_aps
        self.embed_dim = embed_dim
        self.register_buffer("mu", torch.zeros(n_aps))
        self.register_buffer("var", torch.ones(n_aps))
        self.trust_mlp = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def fit_stats(self, scans: torch.Tensor) -> None:
        # scans: (N, *window) - flatten leading dims into (N', n_aps)
        x = scans.reshape(-1, scans.shape[-1]).float()
        self.mu = x.mean(0).to(self.mu.device)
        self.var = (x.var(0) + 1e-3).to(self.var.device)

    def _trust(self, x: torch.Tensor) -> torch.Tensor:
        m = (x > 0.005).float()
        n = m.sum(-1).clamp_min(5.0)
        d = ((x - self.mu) ** 2 / self.var * m).sum(-1) / n
        return torch.sigmoid(-d).unsqueeze(-1)

    def forward(self, x):
        z = self.base(x)
        x2 = x.squeeze(1) if x.ndim == 3 else x
        s = self._trust(x2)
        return z + self.trust_mlp(s)


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
    base = encoders["wifi"]
    n_aps = int(dm.train_ds.feature_dims["wifi"])
    gated = TrustGatedWiFi(base, n_aps=n_aps, embed_dim=base.embed_dim)
    # Fit stats from training scans directly via dataset tensors.
    scans, *_ = dm.train_ds.get_tensors("wifi")
    gated.fit_stats(scans)
    if torch.cuda.is_available():
        gated = gated.cuda()
    encoders["wifi"] = gated

    model = build_model(cfg, encoders)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lead4] dataset={args.dataset} params={n_params/1e6:.3f} M", flush=True)

    run_root = REPO / "runs" / "experiments" / f"lead4_trust_{args.dataset}_s{args.seed}"
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

    print(f"\n[lead4] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f} test={test_mae:.3f} ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
