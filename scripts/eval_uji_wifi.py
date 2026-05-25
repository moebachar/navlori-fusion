"""Anchor2Vec on UJIIndoorLoc — WiFi-leg SOTA comparison.

Runs OUR WiFi encoder (Anchor2Vec) as a static RSSI->position regressor on
the standard UJIIndoorLoc benchmark, using its fixed train/validation split,
and reports mean Euclidean error in meters. Reference to beat/match:
eAaT+ / Anchor2Vec published ~8.16 m mean positioning error.

WiFi encoding mirrors the pipeline's M1 "raw" mode (no whitening): the
not-detected sentinel (100) -> no-signal, detected RSSI [-104,0] -> [~0,1]
via a single fixed affine. Targets are (LONGITUDE, LATITUDE) centered by the
train mean (local meters); mean Euclidean error is centering-invariant.

Run: .venv/Scripts/python.exe scripts/eval_uji_wifi.py [--epochs 120]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.encoders import Anchor2Vec  # noqa: E402

DATA = ROOT / "data" / "uji_indoorloc"


def load_split(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    rssi = df[waps].values.astype(np.float32)
    # not-detected sentinel 100 -> -100 (no signal); raw affine (x+100)/100.
    rssi = np.where(rssi == 100, -100.0, rssi)
    rssi = np.clip(rssi, -100.0, 0.0)
    feat = (rssi + 100.0) / 100.0                       # no-signal=0, strong~1
    xy = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    return feat, xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchors", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = load_split(DATA / "trainingData.csv")
    Xva, Yva = load_split(DATA / "validationData.csv")
    n_aps = Xtr.shape[1]
    mu = Ytr.mean(0)                                     # center targets (local m)
    Ytr_c, Yva_c = Ytr - mu, Yva - mu
    print(f"UJIIndoorLoc: train {len(Xtr)}  val {len(Xva)}  APs {n_aps}", flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev).unsqueeze(1)   # (N,1,n_aps)
    Ytr_t = torch.tensor(Ytr_c, device=dev)
    Xva_t = torch.tensor(Xva, device=dev).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=dev)

    enc = Anchor2Vec(n_aps=n_aps, embed_dim=128, n_anchors=args.anchors).to(dev)
    head = nn.Linear(128, 2).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)

    best = float("inf")
    for ep in range(args.epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * 256:(s + 1) * 256]
            pred = head(enc(Xtr_t[idx]))
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            pv = head(enc(Xva_t))
            mae = torch.linalg.norm(pv - Yva_t, dim=1).mean().item()
        best = min(best, mae)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val mean-euclidean = {mae:.3f} m  (best {best:.3f})",
                  flush=True)

    print(f"\n  >>> Anchor2Vec on UJIIndoorLoc val: {best:.3f} m mean Euclidean error")
    print(f"      reference eAaT+/Anchor2Vec published ~8.16 m")
    verdict = "MATCHES/BEATS" if best <= 8.5 else "ABOVE"
    print(f"      [{verdict} the published WiFi-only baseline]")


if __name__ == "__main__":
    main()
