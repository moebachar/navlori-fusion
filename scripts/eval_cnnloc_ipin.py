"""CNNLoc on IPIN floor -2 — Phase B WiFi-only baseline (same data as our fusion).

Uses our FusionDataModule to load IPIN's WiFi (raw, no whitening, no PCA)
exactly as the fusion does — so the comparison vs our fusion is on IDENTICAL
data + protocol. CNNLoc architecture: SAE -> 1D-CNN -> position regressor
(no floor cascade — IPIN is single-floor, so cascade is moot here, and our
CNNLoc reduces to exactly the published architecture for this case).

Run: .venv/Scripts/python.exe scripts/eval_cnnloc_ipin.py [--epochs 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import build_datamodule, load_config  # noqa: E402


class SAE(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU())
        self.dec = nn.Sequential(
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, 256), nn.ELU(),
            nn.Linear(256, n_in))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


class CNNLoc(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.sae = SAE(n_in)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 99, 22, padding=11), nn.ELU(),
            nn.Conv1d(99, 66, 22, padding=11), nn.ELU(),
            nn.Conv1d(66, 33, 22, padding=11), nn.ELU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(33 * 8, 128), nn.ELU(), nn.Linear(128, 2))

    def forward(self, x):
        z, _ = self.sae(x)
        c = self.cnn(z.unsqueeze(1))
        return self.head(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ae-epochs", type=int, default=20)
    ap.add_argument("--dataset", default="ipin2024_floor-2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.dataset)
    cfg.dataset.modalities = ["wifi"]
    dm = build_datamodule(cfg)
    Xtr_full, Ytr = dm.train_ds.get_tensors("wifi")   # (N, 1, n_aps)
    Xva_full, Yva = dm.val_ds.get_tensors("wifi")
    Xtr = Xtr_full.squeeze(1).to(dev)                  # (N, n_aps)
    Xva = Xva_full.squeeze(1).to(dev)
    Ytr = Ytr.to(dev); Yva = Yva.to(dev)
    n_in = Xtr.shape[1]
    print(f"{args.dataset}: train {len(Xtr)} val {len(Xva)} APs {n_in}", flush=True)

    # Filter out wifi-absent samples (zero vectors) for training the WiFi-only
    # baseline (they carry no signal; would learn nothing). Eval on full val.
    keep_tr = (Xtr.abs().sum(1) > 0)
    Xt, Yt = Xtr[keep_tr], Ytr[keep_tr]
    print(f"  wifi-available train: {len(Xt)} / {len(Xtr)}", flush=True)

    model = CNNLoc(n_in).to(dev)
    # SAE pretrain
    ae_opt = torch.optim.Adam(model.sae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for ep in range(args.ae_epochs):
        model.sae.train()
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(max(1, len(Xt) // 256)):
            idx = perm[s * 256:(s + 1) * 256]
            _, rec = model.sae(Xt[idx])
            loss = mse(rec, Xt[idx])
            ae_opt.zero_grad(); loss.backward(); ae_opt.step()
    print(f"  SAE pretrained", flush=True)

    # Train (target = centered position, so it's local meters)
    mu = Yt.mean(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = max(1, len(Xt) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=args.epochs,
                                                steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)
    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(steps):
            idx = perm[s * 256:(s + 1) * 256]
            loss = crit(model(Xt[idx]), Yt[idx] - mu)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            # eval on FULL val (including wifi-absent — they fall back to model's prior)
            mae = torch.linalg.norm(model(Xva) - (Yva - mu), dim=1).mean().item()
        best = min(best, mae)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val mae={mae:.3f}m  (best {best:.3f})", flush=True)

    print(f"\n  >>> CNNLoc on {args.dataset} val: {best:.3f} m mean Euclidean")
    print(f"      (wifi-only baseline on the same data as fusion)")


if __name__ == "__main__":
    main()
