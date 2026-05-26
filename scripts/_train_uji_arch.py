"""PLAN_24 — Anchor2Vec + CNN1D/LSTM-attn aggregator (K=1 degenerate) on UJI.

Pipeline:
  per-scan UJI WiFi vector (520 APs)
    -> Anchor2Vec encoder -> (B, 128)
    -> reshape (B, 1, 128) — K=1 sequence
    -> CNN1D or LSTM-attn aggregator -> (B, 1, 128)
    -> squeeze K -> (B, 128)
    -> Linear(128, 2) -> (longitude, latitude) in centered local meters

At K=1, the K-axis aggregators degenerate to embedding-level transforms
(conv kernel=3 with padding over length-1 input; BiLSTM cell over a
single time step). The expected outcome is α7: both architectures land
within ~5 % of the bare Anchor2Vec baseline (RESULT_01: 8.69 m val
mean Euclidean error).

Run: ``.venv/Scripts/python.exe scripts/_train_uji_arch.py --arch cnn1d``
"""
from __future__ import annotations

import argparse
import json
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

from src.pipeline.encoders import Anchor2Vec  # noqa: E402
from src.pipeline.fusion.bakeoff import _MaskedBiLSTM, _PlainCNN1D  # noqa: E402

DATA = ROOT / "data" / "uji_indoorloc"
OUT_DIR_DEFAULT = ROOT / "runs" / "overnight" / "run2_iter_24"


def load_split(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    rssi = df[waps].values.astype(np.float32)
    rssi = np.where(rssi == 100, -100.0, rssi)
    rssi = np.clip(rssi, -100.0, 0.0)
    feat = (rssi + 100.0) / 100.0
    xy = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    return feat, xy


class UjiCNN1D(nn.Module):
    """Anchor2Vec + aggregator (degenerate at K=1) + Linear(2)."""

    def __init__(self, agg_kind: str, n_aps: int, embed_dim: int = 128,
                 n_anchors: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = Anchor2Vec(n_aps=n_aps, embed_dim=embed_dim,
                               n_anchors=n_anchors)
        if agg_kind == "cnn1d":
            self.aggregator = _PlainCNN1D(embed_dim=embed_dim, dropout=dropout)
        elif agg_kind == "lstm_attn":
            self.aggregator = _MaskedBiLSTM(embed_dim=embed_dim,
                                             hidden_dim=embed_dim)
        else:
            raise ValueError(f"unknown agg_kind {agg_kind}")
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: (B, 1, n_aps) — Anchor2Vec expects this shape and returns (B, D).
        z = self.enc(x)                          # (B, D)
        B = z.shape[0]
        z = z.unsqueeze(1)                       # (B, 1, D) — K=1 sequence
        pad = torch.zeros(B, 1, dtype=torch.bool, device=z.device)
        z = self.aggregator(z, pad)              # (B, 1, D)
        z = z.squeeze(1)                         # (B, D)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnn1d", "lstm_attn"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = load_split(DATA / "trainingData.csv")
    Xva, Yva = load_split(DATA / "validationData.csv")
    n_aps = Xtr.shape[1]
    mu = Ytr.mean(0)
    Ytr_c, Yva_c = Ytr - mu, Yva - mu
    print(f"=== arch={args.arch}  UJIIndoorLoc K=1 M=1 degenerate ===", flush=True)
    print(f"  train {len(Xtr)}  val {len(Xva)}  APs {n_aps}", flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev).unsqueeze(1)
    Ytr_t = torch.tensor(Ytr_c, device=dev)
    Xva_t = torch.tensor(Xva, device=dev).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=dev)

    model = UjiCNN1D(args.arch, n_aps=n_aps, n_anchors=args.anchors).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.3f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)

    best = float("inf"); best_ep = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            pred = model(Xtr_t[idx])
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            mae = torch.linalg.norm(pv - Yva_t, dim=1).mean().item()
        if mae < best:
            best = mae; best_ep = ep
        if ep == 0 or ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  val mean-euclidean = {mae:.3f} m  (best {best:.3f})",
                  flush=True)
    elapsed = time.time() - t0

    # Final val distribution
    model.eval()
    with torch.no_grad():
        pv = model(Xva_t).cpu().numpy()
    errs = np.sqrt(((pv - Yva_c) ** 2).sum(1))
    dist = {
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "p25": float(np.percentile(errs, 25)),
        "p75": float(np.percentile(errs, 75)),
        "p90": float(np.percentile(errs, 90)),
        "max": float(errs.max()),
        "n": int(len(errs)),
    }

    out = {
        "arch": args.arch,
        "n_params": int(n_params),
        "training": {"epochs": args.epochs, "elapsed_s": float(elapsed),
                      "batch": args.batch, "lr": args.lr},
        "best": {"val_mean_euclidean": float(best), "epoch": int(best_ep)},
        "final_val_distribution": dist,
        "anchor2vec_baseline_RESULT_01": 8.69,
        "wlanloc_sota_RESULT_01": 15.17,
    }
    out_path = OUT_DIR / f"{args.arch}_uji.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  >>> {args.arch} on UJI val: best {best:.3f} m  (epoch {best_ep})", flush=True)
    print(f"      final val distribution: mean={dist['mean']:.3f}  median={dist['median']:.3f}  "
          f"p90={dist['p90']:.3f}  max={dist['max']:.3f}", flush=True)
    print(f"      vs Anchor2Vec 8.69 m: delta {(best-8.69)/8.69*100:+.1f} %", flush=True)
    print(f"      vs wlanloc SOTA 15.17 m: delta {(best-15.17)/15.17*100:+.1f} %", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
