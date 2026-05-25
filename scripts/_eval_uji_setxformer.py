"""WiFiSetTransformer on UJIIndoorLoc — WiFi-leg SOTA comparison (iter-scoped).

Mirrors ``scripts/eval_uji_wifi.py`` but swaps ``Anchor2Vec`` →
``WiFiSetTransformer``. Same dataloader, same target centering, same
metric (mean Euclidean over UJI's official ``validationData.csv``), same
Huber regression head — only the encoder differs so the comparison is
apples-to-apples.

Pre-flight (run on first launch): forward+backward memory budget check
on a synthetic batch at the target shape (B=128, n_aps=520). Bails
out if peak GPU exceeds the 6 GB budget on the 8 GB Quadro/GTX-1080.

Underscore prefix marks this as iteration-scoped (per PLAN_01,
promoted later only if the encoder wins the audit).

Run: ``.venv/Scripts/python.exe scripts/_eval_uji_setxformer.py [--epochs N]``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.encoders import WiFiSetTransformer  # noqa: E402

DATA = ROOT / "data" / "uji_indoorloc"


def load_split(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    rssi = df[waps].values.astype(np.float32)
    rssi = np.where(rssi == 100, -100.0, rssi)
    rssi = np.clip(rssi, -100.0, 0.0)
    feat = (rssi + 100.0) / 100.0  # no-signal=0, strong~1 (same as eval_uji_wifi.py)
    xy = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    return feat, xy


def memory_budget_check(n_aps: int, batch: int = 128, embed_dim: int = 128) -> float:
    """Forward+backward on a synthetic batch at the target UJI shape.
    Returns peak GPU MB. Bails out if > 6000 MB.
    """
    if not torch.cuda.is_available():
        print("memory check skipped (no CUDA)", flush=True)
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    enc = WiFiSetTransformer(n_aps=n_aps, embed_dim=embed_dim).cuda()
    head = nn.Linear(embed_dim, 2).cuda()
    # Realistic occupancy: 4% of 520 ≈ 21 observed APs / row, like UJI val.
    x = torch.rand(batch, 1, n_aps, device="cuda")
    mask_keep = (torch.rand(batch, n_aps, device="cuda") < 0.04)
    x = x.squeeze(1) * mask_keep.float()
    x = x.unsqueeze(1)
    y = torch.randn(batch, 2, device="cuda")
    pred = head(enc(x))
    loss = nn.functional.huber_loss(pred, y)
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  memory check: B={batch} n_aps={n_aps} -> peak {peak_mb:.1f} MB", flush=True)
    del enc, head, x, y, pred, loss
    torch.cuda.empty_cache()
    if peak_mb > 6000:
        raise RuntimeError(f"peak GPU {peak_mb:.0f} MB exceeds 6 GB budget")
    return peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = load_split(DATA / "trainingData.csv")
    Xva, Yva = load_split(DATA / "validationData.csv")
    n_aps = Xtr.shape[1]
    mu = Ytr.mean(0)
    Ytr_c, Yva_c = Ytr - mu, Yva - mu
    print(f"UJIIndoorLoc: train {len(Xtr)}  val {len(Xva)}  APs {n_aps}", flush=True)

    # Pre-flight memory budget check at target shape.
    print("\n[memory budget check]", flush=True)
    peak_mb = memory_budget_check(n_aps=n_aps, batch=args.batch)
    print(f"  passed (peak {peak_mb:.1f} MB < 6000 MB budget)", flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev).unsqueeze(1)
    Ytr_t = torch.tensor(Ytr_c, device=dev)
    Xva_t = torch.tensor(Xva, device=dev).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=dev)

    enc = WiFiSetTransformer(n_aps=n_aps, embed_dim=128).to(dev)
    head = nn.Linear(128, 2).to(dev)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  params: {n_params/1e6:.2f} M", flush=True)

    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)

    best = float("inf")
    best_epoch = -1
    bad = 0
    t_start = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        ep_loss = 0.0
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            pred = head(enc(Xtr_t[idx]))
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss += float(loss.detach())
        ep_loss /= max(1, steps)
        enc.eval(); head.eval()
        with torch.no_grad():
            pv = head(enc(Xva_t))
            mae = torch.linalg.norm(pv - Yva_t, dim=1).mean().item()
        if mae < best - 1e-3:
            best = mae
            best_epoch = ep
            bad = 0
        else:
            bad += 1
        if ep <= 5 or ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss={ep_loss:.4f}  val mean-euclidean={mae:.3f} m  "
                  f"(best {best:.3f} @ ep {best_epoch})", flush=True)
        if bad >= args.patience:
            print(f"  early-stop at epoch {ep} (patience {args.patience})", flush=True)
            break

    elapsed = time.time() - t_start
    print(f"\n  >>> WiFiSetTransformer on UJIIndoorLoc val: {best:.3f} m mean Euclidean (best epoch {best_epoch})")
    print(f"      Anchor2Vec ref (docs): 8.55 m")
    print(f"      wlan_localization (docs): 13.92 m global / 12.99 m cascade-oracle")
    print(f"      params: {n_params/1e6:.2f} M  | elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
