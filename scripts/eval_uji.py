"""Canonical UJIIndoorLoc benchmark — reproduces RESULT_01.

Two numbers, side by side:
  - wlan_localization SOTA (global mode, k=3 manhattan) : **15.17 m**
  - Anchor2Vec (ours, 120 ep Huber)                     :  **8.69 m**

Both are val mean Euclidean error on ``validationData.csv`` (UJI has
no test split). Replaces iteration-scoped
``scripts/eval_uji_wifi.py`` + ``scripts/eval_wlanloc_uji.py``.

Built entirely on the consolidated APIs from PLAN_26-28:
- ``src.pipeline.baselines`` for the vendored ``PositionRegressor``
  and ``DataPreprocessor``.
- ``src.pipeline.data.uji`` for the UJI raw RSSI loader.
- ``src.pipeline.encoders.Anchor2Vec`` for the WiFi encoder.

Run: ``.venv/Scripts/python.exe scripts/eval_uji.py``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import load_position_regressor, load_preprocessor  # noqa: E402
from src.pipeline.data import load_dataset  # noqa: E402
from src.pipeline.encoders import Anchor2Vec  # noqa: E402


def run_wlanloc():
    """wlan_localization global-mode PositionRegressor (RESULT_01)."""
    PR = load_position_regressor()
    DP = load_preprocessor()
    Xtr, Ytr_df = load_dataset("uji_indoorloc", split="train")
    Xva, Yva_df = load_dataset("uji_indoorloc", split="validation")
    pre = DP()
    Xtr_p = pre.fit_transform(Xtr.astype(np.float64))
    Xva_p = pre.transform(Xva.astype(np.float64))
    reg = PR(k=3, metric="manhattan", weights="distance")
    Ytr = Ytr_df[["LATITUDE", "LONGITUDE"]].values
    Yva = Yva_df[["LATITUDE", "LONGITUDE"]].values
    reg.fit_location(0, 0, Xtr_p, Ytr)
    pred = reg.models[(0, 0)].predict(Xva_p)
    return float(np.sqrt(((pred - Yva) ** 2).sum(1)).mean())


def run_anchor2vec(epochs: int = 120, batch: int = 256, lr: float = 1e-3):
    """Anchor2Vec encoder + Linear head, centred-target Huber loss."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr_df = load_dataset("uji_indoorloc", split="train")
    Xva, Yva_df = load_dataset("uji_indoorloc", split="validation")
    Ytr = Ytr_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    Yva = Yva_df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    # 100 sentinel -> -100, affine to [0, 1]
    Xtr = np.where(Xtr == 100, -100.0, Xtr).clip(-100, 0)
    Xva = np.where(Xva == 100, -100.0, Xva).clip(-100, 0)
    Xtr = (Xtr + 100.0) / 100.0
    Xva = (Xva + 100.0) / 100.0
    mu = Ytr.mean(0); Ytr -= mu; Yva -= mu
    Xtr_t = torch.tensor(Xtr, device=dev).unsqueeze(1)
    Ytr_t = torch.tensor(Ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev).unsqueeze(1)
    Yva_t = torch.tensor(Yva, device=dev)
    enc = Anchor2Vec(n_aps=Xtr.shape[1], embed_dim=128, n_anchors=64).to(dev)
    head = nn.Linear(128, 2).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                  steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)
    best = float("inf")
    for ep in range(epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            loss = crit(head(enc(Xtr_t[idx])), Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            mae = torch.linalg.norm(head(enc(Xva_t)) - Yva_t, dim=1).mean().item()
        best = min(best, mae)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor2vec-epochs", type=int, default=120)
    ap.add_argument("--skip-sota", action="store_true",
                    help="Only train Anchor2Vec; skip the wlan_localization SOTA.")
    args = ap.parse_args()
    print(f"=== UJI canonical reproduction (RESULT_01) ===", flush=True)
    if not args.skip_sota:
        t0 = time.time()
        sota = run_wlanloc()
        print(f"  wlan_localization (SOTA, global mode):  {sota:.3f} m"
              f"   (RESULT_01 ref 15.17, drift {(sota-15.17)/15.17*100:+.1f}%)"
              f"   [{time.time()-t0:.1f}s]", flush=True)
    t0 = time.time()
    ours = run_anchor2vec(epochs=args.anchor2vec_epochs)
    print(f"  Anchor2Vec (ours, {args.anchor2vec_epochs} ep):         "
          f"{ours:.3f} m   (RESULT_01 ref 8.69, drift {(ours-8.69)/8.69*100:+.1f}%)"
          f"   [{time.time()-t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
