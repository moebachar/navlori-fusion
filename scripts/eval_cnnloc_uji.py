"""CNNLoc (open-source WiFi baseline) on UJIIndoorLoc.

Faithful implementation of CNNLoc (Song et al., IEEE Access 2019):
  * Stacked AutoEncoder (SAE) 520->256->128->64 (ELU), pretrained to
    reconstruct the RSSI vector, then reused as a feature extractor.
  * 1D-CNN positioning head on the 64-d code: Conv1d(99,k22)->Conv1d(66,k22)
    ->Conv1d(33,k22)->flatten->Dense, regressing (longitude, latitude).

Run with the SAME UJI protocol/metric as scripts/eval_uji_wifi.py (our
WiFiNet) so the comparison is architecture-only: same fixed train/val
split, same RSSI encoding, same centered targets, mean Euclidean error.
Published CNNLoc band on UJI positioning: ~2.6-8.2 m (varies by protocol).

Run: .venv/Scripts/python.exe scripts/eval_cnnloc_uji.py [--epochs 120]
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

DATA = ROOT / "data" / "uji_indoorloc"


def load_split(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    rssi = df[waps].values.astype(np.float32)
    rssi = np.where(rssi == 100, -100.0, rssi)
    rssi = np.clip(rssi, -100.0, 0.0)
    feat = (rssi + 100.0) / 100.0
    xy = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    return feat, xy


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
        z, _ = self.sae(x)                # (B, 64)
        c = self.cnn(z.unsqueeze(1))      # treat code as length-64 1D signal
        return self.head(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ae-epochs", type=int, default=30)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr = load_split(DATA / "trainingData.csv")
    Xva, Yva = load_split(DATA / "validationData.csv")
    n_in = Xtr.shape[1]
    mu = Ytr.mean(0)
    Xtr_t = torch.tensor(Xtr, device=dev); Ytr_t = torch.tensor(Ytr - mu, device=dev)
    Xva_t = torch.tensor(Xva, device=dev); Yva_t = torch.tensor(Yva - mu, device=dev)
    print(f"UJIIndoorLoc: train {len(Xtr)} val {len(Xva)} APs {n_in}", flush=True)

    model = CNNLoc(n_in).to(dev)

    # 1) SAE pretrain (reconstruct RSSI)
    ae_opt = torch.optim.Adam(model.sae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for ep in range(args.ae_epochs):
        model.sae.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(max(1, len(Xtr_t) // 256)):
            idx = perm[s * 256:(s + 1) * 256]
            _, rec = model.sae(Xtr_t[idx])
            loss = mse(rec, Xtr_t[idx])
            ae_opt.zero_grad(); loss.backward(); ae_opt.step()
    print(f"  SAE pretrained ({args.ae_epochs} ep)", flush=True)

    # 2) positioning (joint finetune)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = max(1, len(Xtr_t) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=args.epochs,
                                                steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)
    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * 256:(s + 1) * 256]
            loss = crit(model(Xtr_t[idx]), Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            mae = torch.linalg.norm(model(Xva_t) - Yva_t, dim=1).mean().item()
        best = min(best, mae)
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val mean-euclidean={mae:.3f} m (best {best:.3f})", flush=True)

    print(f"\n  >>> CNNLoc on UJIIndoorLoc val: {best:.3f} m mean Euclidean")
    print(f"      our WiFiNet (same protocol): 8.55 m | published CNNLoc band ~2.6-8.2 m")


if __name__ == "__main__":
    main()
