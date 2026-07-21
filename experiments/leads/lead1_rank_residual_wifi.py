"""Lead 1 — Rank-Residual BSSID Set Encoder.

Per-scan rank + residual features are SESSION-INVARIANT under +k dBm shifts.
We subclass WiFiSetTransformer to inject rank-normalised position + residual
RSSI (after per-scan mean subtraction) as extra per-BSSID features.

Hypothesis: the dominant MSILN cross-session error is per-session RSSI
calibration drift; rank statistics are immune to that.
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
from src.pipeline.encoders.wifi_set import WiFiSetTransformer  # noqa: E402


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


class WiFiSetRR(WiFiSetTransformer):
    """WiFiSetTransformer with rank + residual extra per-AP features."""

    def __init__(self, *a, use_rank: bool = True, use_residual: bool = True,
                  **kw):
        super().__init__(*a, **kw)
        self.use_rank = use_rank
        self.use_residual = use_residual
        bdim = self.bssid_embed.embedding_dim
        in_dim = bdim + 1 + int(use_rank) + int(use_residual)
        self.token_proj = nn.Sequential(
            nn.Linear(in_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(1)
        obs = x > self.epsilon
        keys = obs.float() * 10.0 + x
        _, sidx = keys.sort(dim=1, descending=True, stable=True)
        keep_n = max(1, min(int(obs.sum(1).max().item()),
                              self.max_observed_per_scan))
        kidx = sidx[:, :keep_n]
        obs_rssi = x.gather(1, kidx)
        pad = obs_rssi <= self.epsilon
        n_obs = (~pad).sum(1, keepdim=True).clamp(min=1).float()
        rank = (torch.arange(keep_n, device=x.device, dtype=x.dtype)
                .unsqueeze(0).expand_as(obs_rssi))
        rank_norm = (rank / n_obs).masked_fill(pad, 0.0)
        mean_r = ((obs_rssi * (~pad).float()).sum(1, keepdim=True) / n_obs)
        resid = (obs_rssi - mean_r).masked_fill(pad, 0.0)
        feats = [self.bssid_embed(kidx), obs_rssi.unsqueeze(-1)]
        if self.use_rank:
            feats.append(rank_norm.unsqueeze(-1))
        if self.use_residual:
            feats.append(resid.unsqueeze(-1))
        tok = self.token_proj(torch.cat(feats, dim=-1))
        cls = self.cls_token.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        cls_pad = torch.zeros(tok.size(0), 1, dtype=torch.bool,
                                device=x.device)
        kpm = torch.cat([cls_pad, pad], dim=1)
        out = self.encoder(tok, src_key_padding_mask=kpm)
        return self.out_norm(out[:, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msiln_site1_b1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.dataset)
    cfg.temporal.n_instants = 4                       # paper config (override YAML default 8)
    cfg.train.modality_balanced_loss = False

    dm = build_datamodule(cfg)
    encoders, vision = build_encoders(cfg, dm)
    if not isinstance(encoders["wifi"], WiFiSetTransformer):
        print(f"[lead1] WARNING: base wifi encoder is {type(encoders['wifi']).__name__}, "
              f"not WiFiSetTransformer — falling back to no-op (lead is MSILN-only).",
              flush=True)
        sys.exit(0)
    base = encoders["wifi"]
    encoders["wifi"] = WiFiSetRR(
        n_aps=base.n_aps,
        embed_dim=base.embed_dim,
        bssid_dim=base.bssid_embed.embedding_dim,
        max_observed_per_scan=base.max_observed_per_scan,
        epsilon=base.epsilon,
    )
    n_params = sum(p.numel() for p in encoders["wifi"].parameters())
    print(f"[lead1] WiFiSetRR n_aps={base.n_aps} params={n_params/1e6:.3f} M",
          flush=True)

    model = build_model(cfg, encoders)
    extra = {}
    run_root = REPO / "runs" / "experiments" / f"lead1_rr_{args.dataset}_s{args.seed}"
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

    print(f"\n[lead1] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f} test={test_mae:.3f} ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
