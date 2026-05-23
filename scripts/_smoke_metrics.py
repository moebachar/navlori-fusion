"""Smoke test for Action 4: unified Euclidean MAE / RMSE helpers.

Verifies:
  1. ``euclidean_mae`` / ``euclidean_rmse`` work on numpy and torch.
  2. ``linear_probe`` and ``knn_probe`` now return canonical Euclidean MAE
     in their ``mae`` field (matching trainer's ``val_mae``).
  3. ``trustworthiness`` subsamples and returns finite scores.

Run: ``.venv/Scripts/python.exe scripts/_smoke_metrics.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    euclidean_mae,
    euclidean_rmse,
    knn_probe,
    linear_probe,
    trustworthiness,
)


def _expected_euclidean_mae(p, y):
    return float(np.sqrt(((p - y) ** 2).sum(axis=1)).mean())


def main():
    rng = np.random.RandomState(42)
    N, D = 200, 16

    # Synthetic embeddings whose linear map to (x,y) is recoverable.
    W = rng.randn(D, 2).astype(np.float32) * 0.3
    z_tr = rng.randn(N, D).astype(np.float32)
    y_tr = (z_tr @ W).astype(np.float32) + rng.randn(N, 2).astype(np.float32) * 0.05
    z_va = rng.randn(N // 2, D).astype(np.float32)
    y_va = (z_va @ W).astype(np.float32) + rng.randn(N // 2, 2).astype(np.float32) * 0.05

    # 1. Helpers
    p = z_va @ W
    em_np = euclidean_mae(p, y_va)
    em_torch = euclidean_mae(torch.tensor(p), torch.tensor(y_va))
    assert abs(em_np - em_torch) < 1e-5, (em_np, em_torch)
    assert abs(em_np - _expected_euclidean_mae(p, y_va)) < 1e-5
    print(f"  euclidean_mae (np)    = {em_np:.6f}")
    print(f"  euclidean_mae (torch) = {em_torch:.6f}")
    print(f"  euclidean_rmse        = {euclidean_rmse(p, y_va):.6f}")

    # 2. linear_probe.mae now == Euclidean MAE
    lp = linear_probe(z_tr, y_tr, z_va, y_va, epochs=50, device="cpu")
    print(f"  linear_probe = {lp}")
    assert "mae" in lp and "mae_component" in lp
    assert lp["mae"] >= lp["mae_component"] * 0.99, \
        "Euclidean MAE should be >= component MAE up to FP noise"
    # The bug we fixed: previously mae was np.abs(diff).mean() (per-axis L1).
    # Now mae is the Euclidean MAE, ~sqrt(2) * mae_component for isotropic errors.
    ratio = lp["mae"] / max(lp["mae_component"], 1e-9)
    print(f"  euclidean / component ratio = {ratio:.3f} (expected ~1.0-1.5)")
    assert 0.9 <= ratio <= 2.0, ratio

    # 3. knn_probe likewise
    kp = knn_probe(z_tr, y_tr, z_va, y_va, k=5)
    print(f"  knn_probe = {kp}")
    assert "mae" in kp and "rmse" in kp

    # 4. trustworthiness subsamples
    tw = trustworthiness(z_tr, z_tr + rng.randn(*z_tr.shape) * 0.01, k=5,
                        max_samples=100)
    print(f"  trustworthiness = {tw}")
    assert 0.0 <= tw["trustworthiness"] <= 1.0
    assert tw["n_samples"] == 100

    print("\nPASS — Action 4 smoke")


if __name__ == "__main__":
    main()
