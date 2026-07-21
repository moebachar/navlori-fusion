"""6-metric encoder harness on UJIIndoorLoc — WiFiNet vs WiFiSetTransformer.

Iter-scoped helper for PLAN_01 Step 5. Trains each encoder on UJI (same
dataloader / target centering / Huber head as ``eval_uji_wifi.py`` and
``_eval_uji_setxformer.py``), then runs the project's 6-metric harness
(linear probe / kNN probe / alignment & uniformity / effective
dimensionality / temporal smoothness / trustworthiness) on the val
embeddings.

Per-sample distribution (median, p25, p75, p90, max) of the regression
head's euclidean errors is also reported — UJI is per-scan with no
trajectory structure, so "per-path" doesn't apply but per-sample does.

Run: ``.venv/Scripts/python.exe scripts/_eval_uji_6metric.py``
Output: ``runs/overnight/run2_iter_01/uji_6metric.json``
"""
from __future__ import annotations

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

from src.pipeline.encoders import WiFiNet, WiFiSetTransformer  # noqa: E402
from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    alignment_uniformity,
    effective_dimensionality,
    knn_probe,
    linear_probe,
    temporal_smoothness,
    trustworthiness,
)

DATA = ROOT / "data" / "uji_indoorloc"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_01"


def load_split(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    rssi = df[waps].values.astype(np.float32)
    rssi = np.where(rssi == 100, -100.0, rssi)
    rssi = np.clip(rssi, -100.0, 0.0)
    feat = (rssi + 100.0) / 100.0
    xy = df[["LONGITUDE", "LATITUDE"]].values.astype(np.float32)
    return feat, xy


def train_encoder(encoder: nn.Module, Xtr_t, Ytr_t, Xva_t, Yva_t,
                  epochs: int, lr: float, batch: int, name: str, dev: str):
    head = nn.Linear(128, 2).to(dev)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr_t) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=1.0)

    best = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        encoder.train(); head.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            pred = head(encoder(Xtr_t[idx]))
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            pv = head(encoder(Xva_t))
            mae = torch.linalg.norm(pv - Yva_t, dim=1).mean().item()
        if mae < best:
            best = mae
            best_state = (
                {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            )
        if ep <= 1 or ep % 20 == 0 or ep == epochs - 1:
            print(f"  [{name}] epoch {ep:3d}  val Euclid={mae:.3f} m  (best {best:.3f})", flush=True)
    elapsed = time.time() - t0
    print(f"  [{name}] done in {elapsed:.1f}s  best val Euclid {best:.3f} m", flush=True)
    if best_state is not None:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    return head, best, elapsed


def per_sample_distribution(head, encoder, Xva_t, Yva_t):
    encoder.eval(); head.eval()
    with torch.no_grad():
        pv = head(encoder(Xva_t))
        errs = torch.linalg.norm(pv - Yva_t, dim=1).cpu().numpy()
    return {
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "p25": float(np.percentile(errs, 25)),
        "p75": float(np.percentile(errs, 75)),
        "p90": float(np.percentile(errs, 90)),
        "max": float(errs.max()),
        "n": int(len(errs)),
    }


@torch.no_grad()
def embed_all(encoder, X_t, dev, batch=512):
    encoder.eval()
    zs = []
    for i in range(0, len(X_t), batch):
        zs.append(encoder(X_t[i:i + batch]).cpu().numpy())
    return np.concatenate(zs, axis=0)


def latency_ms(encoder, n_aps, dev, runs=200, batch=1):
    encoder.eval()
    x = torch.zeros(batch, 1, n_aps, device=dev)
    # warmup
    for _ in range(20):
        _ = encoder(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = encoder(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000.0


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Xtr, Ytr = load_split(DATA / "trainingData.csv")
    Xva, Yva = load_split(DATA / "validationData.csv")
    n_aps = Xtr.shape[1]
    mu = Ytr.mean(0)
    Ytr_c, Yva_c = Ytr - mu, Yva - mu
    print(f"UJIIndoorLoc: train {len(Xtr)}  val {len(Xva)}  APs {n_aps}", flush=True)

    Xtr_t = torch.tensor(Xtr, device=dev).unsqueeze(1)
    Ytr_t = torch.tensor(Ytr_c, device=dev)
    Xva_t = torch.tensor(Xva, device=dev).unsqueeze(1)
    Yva_t = torch.tensor(Yva_c, device=dev)
    Ytr_np = Ytr_c
    Yva_np = Yva_c

    results = {}

    for name, cls, kwargs, epochs, lr, batch in [
        ("WiFiNet", WiFiNet,
         dict(n_aps=n_aps, embed_dim=128, n_anchors=64), 120, 1e-3, 256),
        ("WiFiSetTransformer", WiFiSetTransformer,
         dict(n_aps=n_aps, embed_dim=128), 90, 1e-3, 128),
    ]:
        print(f"\n[{name}] training ({epochs} epochs)", flush=True)
        enc = cls(**kwargs).to(dev)
        params_total_pre = n_params(enc)
        head, best_mae, train_s = train_encoder(
            enc, Xtr_t, Ytr_t, Xva_t, Yva_t,
            epochs=epochs, lr=lr, batch=batch, name=name, dev=dev,
        )
        params_total = n_params(enc) + n_params(head)
        per_samp = per_sample_distribution(head, enc, Xva_t, Yva_t)

        # Embeddings (train + val).
        z_train = embed_all(enc, Xtr_t, dev)
        z_val = embed_all(enc, Xva_t, dev)

        print(f"  [{name}] extracting 6-metric harness", flush=True)
        lp = linear_probe(z_train, Ytr_np, z_val, Yva_np, epochs=200, lr=1e-2, device="cpu")
        kp = knn_probe(z_train, Ytr_np, z_val, Yva_np, k=5)
        au = alignment_uniformity(z_val, Yva_np, distance_threshold=1.0, max_samples=1000)
        ed = effective_dimensionality(z_val)
        ts = temporal_smoothness(z_val, Yva_np)
        tw = trustworthiness(Xva, z_val, k=10)

        lat_ms = latency_ms(enc, n_aps, dev)

        results[name] = {
            "train_val_euclid_m": best_mae,
            "params": int(params_total),
            "params_enc": int(params_total_pre),
            "train_time_s": train_s,
            "latency_ms_per_sample_b1": float(lat_ms),
            "per_sample_euclid": per_samp,
            "linear_probe": lp,
            "knn_probe": kp,
            "alignment_uniformity": au,
            "effective_dimensionality": ed,
            "temporal_smoothness": ts,
            "trustworthiness": tw,
        }
        del enc, head, z_train, z_val
        if dev == "cuda":
            torch.cuda.empty_cache()

    out_path = OUT_DIR / "uji_6metric.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Print a quick comparison table.
    print(f"\n{'metric':<35} {'WiFiNet':>15} {'WiFiSetTransformer':>22}")
    a = results["WiFiNet"]
    s = results["WiFiSetTransformer"]
    def row(label, av, sv, fmt="{:.3f}"):
        print(f"  {label:<35} {fmt.format(av):>15} {fmt.format(sv):>22}")
    row("train val Euclid (m)", a["train_val_euclid_m"], s["train_val_euclid_m"])
    row("per-sample p50 (m)", a["per_sample_euclid"]["median"], s["per_sample_euclid"]["median"])
    row("per-sample p90 (m)", a["per_sample_euclid"]["p90"], s["per_sample_euclid"]["p90"])
    row("params (M)", a["params"]/1e6, s["params"]/1e6)
    row("latency b1 (ms)", a["latency_ms_per_sample_b1"], s["latency_ms_per_sample_b1"])
    row("linear-probe Euclid", a["linear_probe"]["mean_euclidean"], s["linear_probe"]["mean_euclidean"])
    row("kNN-probe Euclid", a["knn_probe"]["mean_euclidean"], s["knn_probe"]["mean_euclidean"])
    row("alignment (lower=better)", a["alignment_uniformity"]["alignment"], s["alignment_uniformity"]["alignment"])
    row("uniformity (lower=better)", a["alignment_uniformity"]["uniformity"], s["alignment_uniformity"]["uniformity"])
    row("eff-dim PR", a["effective_dimensionality"]["participation_ratio"], s["effective_dimensionality"]["participation_ratio"], "{:.2f}")
    row("eff-dim 95% var", a["effective_dimensionality"]["dims_95"], s["effective_dimensionality"]["dims_95"], "{:d}")
    row("temporal corr (note: UJI no time)", a["temporal_smoothness"]["correlation"], s["temporal_smoothness"]["correlation"])
    row("trustworthiness (higher=better)", a["trustworthiness"]["trustworthiness"], s["trustworthiness"]["trustworthiness"])


if __name__ == "__main__":
    main()
