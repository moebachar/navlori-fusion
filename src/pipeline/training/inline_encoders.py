"""Inline per-encoder training helpers for the publication-grade
walkthrough notebook (PLAN_32).

Each helper trains a single encoder + linear head on the canonical
benchmark + returns the trained components. Used by `notebooks/
run2_walkthrough.ipynb` when ``FAST_MODE=False`` to demonstrate
clone-and-reproduce — the same training recipe as the offline
`scripts/eval_*.py` runners (RESULT_01/04/07/08).

API for each ``train_*`` helper::

    encoder, head, history = train_anchor2vec(Xtr, Ytr, Xva, Yva, ...)

``encoder`` is the trained ``nn.Module`` from ``src.pipeline.encoders``;
``head`` is a ``nn.Linear(embed_dim, 2)``; ``history`` is a dict with
``train_loss``, ``val_mae`` lists.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.encoders import Anchor2Vec


def _ensure_tensor(arr, dtype=torch.float32):
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype)
    return torch.tensor(np.asarray(arr), dtype=dtype)


def anchor2vec_predict(enc: Anchor2Vec, head: nn.Linear,
                        X: np.ndarray | torch.Tensor,
                        batch: int = 1024,
                        device: str | None = None) -> np.ndarray:
    """Predict (x, y) for a batch of UJI scans through Anchor2Vec + head."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    enc.eval(); head.eval()
    X_t = _ensure_tensor(X).to(device)
    if X_t.ndim == 2:
        X_t = X_t.unsqueeze(1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            chunk = X_t[i:i + batch]
            preds.append(head(enc(chunk)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def anchor2vec_val_mae(enc: Anchor2Vec, head: nn.Linear,
                        Xva, Yva, mu) -> float:
    """Compute mean Euclidean error in original (un-centered) target frame."""
    pred = anchor2vec_predict(enc, head, Xva)
    Yva_arr = np.asarray(Yva)
    mu_arr = np.asarray(mu)
    return float(np.linalg.norm((pred + mu_arr) - (Yva_arr + mu_arr), axis=1).mean())


def train_anchor2vec(
    Xtr, Ytr, Xva, Yva,
    n_anchors: int = 64,
    embed_dim: int = 128,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 1.0,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[Anchor2Vec, nn.Linear, dict]:
    """Inline Anchor2Vec training for the UJI per-leg WiFi audit.

    Replicates the RESULT_01 recipe. Returns the best-val checkpoint
    (encoder + head + history). ~3 minutes on Quadro P4000 at the
    canonical 120 epochs + 256 batch.

    Inputs
    ------
    Xtr, Xva : (N, n_aps) RSSI arrays, already preprocessed (NaN/100
        sentinel handled, affine to [0, 1]).
    Ytr, Yva : (N, 2) target arrays (longitude, latitude). Will be
        centered by the train mean for training; final val MAE is
        computed in the centered frame (Euclidean distance is
        centering-invariant).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr_arr = np.asarray(Xtr).astype(np.float32)
    Xva_arr = np.asarray(Xva).astype(np.float32)
    Ytr_arr = np.asarray(Ytr).astype(np.float32)
    Yva_arr = np.asarray(Yva).astype(np.float32)
    mu = Ytr_arr.mean(0)
    Ytr_c = Ytr_arr - mu
    Yva_c = Yva_arr - mu

    Xtr_t = torch.tensor(Xtr_arr, device=device).unsqueeze(1)  # (N, 1, n_aps)
    Ytr_t = torch.tensor(Ytr_c, device=device)
    Xva_t = torch.tensor(Xva_arr, device=device).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=device)

    enc = Anchor2Vec(n_aps=Xtr_arr.shape[1], embed_dim=embed_dim,
                      n_anchors=n_anchors).to(device)
    head = nn.Linear(embed_dim, 2).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=weight_decay)
    steps = max(1, len(Xtr_t) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=huber_delta)

    history = {"train_loss": [], "val_mae": []}
    best_mae = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        ep_loss = 0.0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            loss = crit(head(enc(Xtr_t[idx])), Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss += loss.item()
        history["train_loss"].append(ep_loss / max(steps, 1))

        enc.eval(); head.eval()
        with torch.no_grad():
            pv = head(enc(Xva_t))
            mae = float(torch.linalg.norm(pv - Yva_t, dim=1).mean())
        history["val_mae"].append(mae)
        if mae < best_mae:
            best_mae = mae
            best_state = (
                {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
        if verbose and (ep == 0 or ep % 30 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}/{epochs}  train={history['train_loss'][-1]:.4f}  "
                  f"val_mae={mae:.3f}  (best {best_mae:.3f})", flush=True)

    elapsed = time.time() - t0
    if best_state is not None:
        enc.load_state_dict(best_state[0]); enc.to(device)
        head.load_state_dict(best_state[1]); head.to(device)
    history["best_val_mae"] = best_mae
    history["elapsed_s"] = elapsed
    history["target_mu"] = mu.tolist()
    if verbose:
        print(f"  done in {elapsed:.0f}s; best val mean Euclidean = {best_mae:.3f} m",
              flush=True)
    return enc, head, history


__all__ = ["train_anchor2vec", "anchor2vec_predict", "anchor2vec_val_mae"]
