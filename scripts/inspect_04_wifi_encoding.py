"""PROBE 4 — which WiFi encoding step destroys the signal?

Scan-level train->val kNN (position = GT at scan time, k=5 distance-weighted)
under four encodings of the SAME scans:
  A. covis      — distance over co-visible APs only (proper fingerprinting)
  B. fill-100   — NaN->-100, raw 166-dim Euclidean
  C. fill-100 + zscore (per-AP, train stats), no PCA
  D. fill-100 + PCA-128 (train-fit) + zscore   <- what the pipeline does

If A << D, the pipeline's feature engineering is the culprit (not the WiFi).

Pure data. Run: .venv/Scripts/python.exe scripts/inspect_04_wifi_encoding.py [dataset]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

from scripts.inspect_03_transfer import covis_dist, load_scans  # noqa: E402
from src.pipeline.fusion.builder import load_config  # noqa: E402


def knn_covis(tr_r, tr_xy, va_r, va_xy, k=5):
    preds = []
    for i in range(len(va_r)):
        d = covis_dist(va_r[i], tr_r)
        if not np.isfinite(d).any():
            preds.append(tr_xy.mean(0)); continue
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] + 1e-6)
        preds.append((tr_xy[order] * w[:, None]).sum(0) / w.sum())
    preds = np.array(preds)
    return np.linalg.norm(preds - va_xy, axis=1).mean()


def knn_vec(tr_x, tr_xy, va_x, va_xy, k=5):
    m = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(tr_x, tr_xy)
    return np.linalg.norm(m.predict(va_x) - va_xy, axis=1).mean()


def analyze(name):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    cfg = load_config(name)
    d = cfg.dataset
    root = ROOT / str(d.root) / d.collection_dir
    tr_r, tr_xy = load_scans(root, list(d.split.train_paths))
    va_r, va_xy = load_scans(root, list(d.split.val_paths))
    if tr_r is None or va_r is None:
        print("  insufficient"); return

    # A. covis
    a = knn_covis(tr_r, tr_xy, va_r, va_xy)

    # B. fill -100
    trB = np.nan_to_num(tr_r, nan=-100.0)
    vaB = np.nan_to_num(va_r, nan=-100.0)
    b = knn_vec(trB, tr_xy, vaB, va_xy)

    # C. fill -100 + zscore (train stats)
    mu, sd = trB.mean(0), trB.std(0) + 1e-8
    c = knn_vec((trB - mu) / sd, tr_xy, (vaB - mu) / sd, va_xy)

    # D. fill -100 + PCA-128 + zscore (pipeline)
    ncomp = min(128, trB.shape[1], trB.shape[0])
    pca = PCA(n_components=ncomp).fit(trB)
    trP, vaP = pca.transform(trB), pca.transform(vaB)
    mu2, sd2 = trP.mean(0), trP.std(0) + 1e-8
    dd = knn_vec((trP - mu2) / sd2, tr_xy, (vaP - mu2) / sd2, va_xy)

    # E. fill -100 + PCA, NO zscore (isolates whitening vs rotation)
    e = knn_vec(trP, tr_xy, vaP, va_xy)

    print(f"  train->val kNN spatial MAE (k=5):")
    print(f"    A. covis (proper fingerprint)        {a:6.2f} m")
    print(f"    B. fill-100 raw euclidean            {b:6.2f} m")
    print(f"    C. fill-100 + zscore                 {c:6.2f} m")
    print(f"    E. fill-100 + PCA{ncomp} (no zscore){e:6.2f} m")
    print(f"    D. fill-100 + PCA{ncomp} + zscore  {dd:6.2f} m   <- pipeline")
    print(f"  >>> best simple encoding {min(a,b,c,e):.2f} m vs pipeline {dd:.2f} m "
          f"= {dd/min(a,b,c,e):.1f}x worse")


def main():
    for ds in (sys.argv[1:] or ["simulation", "ipin2024_floor-2"]):
        try:
            analyze(ds)
        except Exception as e:
            print(f"  ERROR {ds}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
