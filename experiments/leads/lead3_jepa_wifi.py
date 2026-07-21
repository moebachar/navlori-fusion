"""Lead 3 — Joint-Embedding (JEPA / DINO-style) WiFi pretraining.

Phase 1 — pretrain the WiFi encoder for ~5 epochs on UNLABELLED scans
(train + val + test, no GT leakage since we use only RSSI). Student / EMA
teacher receive different RSSI-masking views; student predicts teacher's
embedding via a small predictor. No reconstruction, no pair construction,
no contrastive negatives.

Phase 2 — fine-tune the full fusion model normally with the pretrained
WiFi encoder weights.

Includes a collapse sentinel: aborts pretrain if teacher CLS batch stdev
< 0.05.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _mask_view(x: torch.Tensor, ratio: float, eps: float = 0.005) -> torch.Tensor:
    """Drop a `ratio` fraction of OBSERVED APs (RSSI > eps); leave unobserved alone."""
    if x.ndim >= 2:
        flat = x.reshape(x.shape[0], -1)
    obs = flat > eps
    rand = torch.rand_like(flat).masked_fill(~obs, -1.0)
    n_keep_per = (obs.sum(1).float() * (1 - ratio)).long().clamp(min=4)
    k_max = int(n_keep_per.max().item())
    _, keep = rand.topk(k_max, dim=1)
    m = torch.zeros_like(flat, dtype=torch.bool)
    m.scatter_(1, keep, True)
    out = flat.masked_fill(~m & obs, 0.0)
    return out.reshape(x.shape)


class JEPAPretrainer(nn.Module):
    def __init__(self, base: nn.Module, embed_dim: int):
        super().__init__()
        self.student = base
        self.teacher = copy.deepcopy(base)
        for p in self.teacher.parameters():
            p.requires_grad = False
        D = embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(D, 2 * D), nn.GELU(),
            nn.LayerNorm(2 * D), nn.Linear(2 * D, D),
        )

    @torch.no_grad()
    def ema_update(self, m: float = 0.996) -> None:
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vs = _mask_view(x, ratio=0.30)
        vt = _mask_view(x, ratio=0.50)
        zs = self.predictor(self.student(vs))
        with torch.no_grad():
            zt = self.teacher(vt)
        zs = F.normalize(zs, dim=-1)
        zt = F.normalize(zt, dim=-1)
        loss = (2.0 - 2.0 * (zs * zt).sum(-1)).mean()
        return loss, zt


def pretrain_jepa(wifi_encoder: nn.Module, scans: torch.Tensor,
                   embed_dim: int, epochs: int = 5, batch_size: int = 256,
                   device: str = "cuda") -> nn.Module:
    print(f"[jepa] pretrain: scans={tuple(scans.shape)} epochs={epochs}", flush=True)
    jepa = JEPAPretrainer(wifi_encoder, embed_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(
        list(jepa.student.parameters()) + list(jepa.predictor.parameters()),
        lr=1.5e-4, weight_decay=1e-4,
    )
    N = scans.shape[0]
    scans_dev = scans.to(device)
    for ep in range(epochs):
        idx = torch.randperm(N, device=device)
        losses = []
        for s in range(0, N, batch_size):
            chunk = idx[s:s + batch_size]
            x = scans_dev[chunk]
            loss, zt = jepa(x)
            opt.zero_grad(); loss.backward(); opt.step()
            jepa.ema_update(0.996)
            losses.append(loss.item())
        # Collapse sentinel: teacher CLS std across batch.
        with torch.no_grad():
            zt = jepa.teacher(scans_dev[:min(512, N)])
            std = zt.std(dim=0).mean().item()
        mean_loss = float(np.mean(losses))
        print(f"[jepa] ep={ep + 1}/{epochs} loss={mean_loss:.4f} t_std={std:.3f}",
              flush=True)
        if std < 0.05:
            print("[jepa] COLLAPSE detected, abort pretrain", flush=True)
            break
    return jepa.student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msiln_site1_b1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--pre", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.dataset)
    cfg.temporal.n_instants = 4
    cfg.train.modality_balanced_loss = False

    dm = build_datamodule(cfg)
    encoders, vision = build_encoders(cfg, dm)
    base = encoders["wifi"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Gather UNLABELLED scans from train + val + test (RSSI only, no GT leak).
    parts = [dm.train_ds.get_tensors("wifi")[0]]
    if getattr(dm, "val_ds", None) is not None:
        parts.append(dm.val_ds.get_tensors("wifi")[0])
    if getattr(dm, "test_ds", None) is not None:
        parts.append(dm.test_ds.get_tensors("wifi")[0])
    scans = torch.cat([p.reshape(-1, p.shape[-1]) for p in parts], dim=0)
    print(f"[lead3] pretrain corpus: {scans.shape[0]} scans, dim={scans.shape[-1]}",
          flush=True)

    base = base.to(device)
    pretrained = pretrain_jepa(base, scans, embed_dim=base.embed_dim,
                                  epochs=args.pre, batch_size=256, device=device)
    encoders["wifi"] = pretrained

    model = build_model(cfg, encoders)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lead3] dataset={args.dataset} params={n_params/1e6:.3f} M",
          flush=True)

    run_root = REPO / "runs" / "experiments" / f"lead3_jepa_{args.dataset}_s{args.seed}"
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

    print(f"\n[lead3] RESULT: dataset={args.dataset} seed={args.seed} "
          f"val={val_mae:.3f} test={test_mae:.3f} ({elapsed/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
