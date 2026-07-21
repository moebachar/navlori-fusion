"""K-sweep at test time on MSILN site1/B1.

Re-evaluates the trained MSILN FusionTransformer (trained at K=4) with
K in {1, 2, 4, 8} to show how test-time MAE depends on the recent-instants
window size. The set-transformer is permutation-invariant in tokens, so
the architecture handles any K without retraining; only the datamodule
temporal window is rebuilt by ``load_trained``.

Writes:
    revision/k_period_d3/k_sweep_msiln.json
    revision/k_period_d3/k_sweep_msiln.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

# Repo root on sys.path so ``src...`` imports resolve regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.training import load_trained  # noqa: E402


CKPT = "runs/main_table/msiln_site1_b1/transformer"
DATASET = "msiln_site1_b1"
ARCH = "transformer"
K_GRID = [1, 2, 4, 8]

OUT_DIR = REPO_ROOT / "revision" / "k_period_d3"
OUT_JSON = OUT_DIR / "k_sweep_msiln.json"
OUT_MD = OUT_DIR / "k_sweep_msiln.md"


def euclidean_mae(preds: torch.Tensor, tgts: torch.Tensor) -> float:
    """Mean Euclidean error in metres (matches FusionTrainer convention)."""
    return float(torch.linalg.norm(preds - tgts, dim=1).mean().item())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Switch cwd to repo root so load_trained's relative checkpoint path
    # ("runs/main_table/...") resolves identically to the notebook setup.
    import os
    os.chdir(REPO_ROOT)

    print(f"[k_sweep] checkpoint={CKPT}", flush=True)
    print(f"[k_sweep] dataset={DATASET}  arch={ARCH}", flush=True)
    print(f"[k_sweep] K grid={K_GRID}", flush=True)

    test_mae_m: list[float] = []
    n_samples: list[int] = []
    elapsed_sec: list[float] = []

    for K in K_GRID:
        print(f"\n[k_sweep] === K = {K} ===", flush=True)
        t0 = time.time()
        tr = load_trained(CKPT, arch=ARCH, dataset=DATASET, K=int(K))
        n_test = int(tr.n["test"])
        print(f"[k_sweep] K={K}  n_test={n_test}  modalities={list(tr.modalities)}",
              flush=True)

        preds, tgts = tr.predict("test")
        mae = euclidean_mae(preds, tgts)
        dt = time.time() - t0

        test_mae_m.append(mae)
        n_samples.append(int(preds.shape[0]))
        elapsed_sec.append(dt)
        print(f"[k_sweep] K={K}  test_MAE={mae:.4f} m  n={preds.shape[0]}"
              f"  ({dt:.1f}s)", flush=True)

        # Free GPU memory between Ks (datamodule rebuilds for each).
        del tr, preds, tgts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "checkpoint": CKPT,
        "dataset": DATASET,
        "arch": ARCH,
        "trained_K": 4,
        "K": list(map(int, K_GRID)),
        "test_mae_m": test_mae_m,
        "n_samples": n_samples,
        "elapsed_sec": elapsed_sec,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[k_sweep] wrote {OUT_JSON}", flush=True)

    # Markdown report.
    lines: list[str] = []
    lines.append("# MSILN site1/B1 - test-time K sweep")
    lines.append("")
    lines.append(f"Checkpoint: `{CKPT}` (trained at K=4)")
    lines.append(f"Dataset: `{DATASET}`  -  arch: `{ARCH}`")
    lines.append("")
    lines.append("| K | test MAE (m) | n samples |")
    lines.append("|---|--------------|-----------|")
    for k, m, n in zip(K_GRID, test_mae_m, n_samples):
        lines.append(f"| {k} | {m:.3f} | {n} |")
    lines.append("")
    headline_parts = " | ".join(
        f"K={k} {m:.2f}" for k, m in zip(K_GRID, test_mae_m)
    )
    lines.append(f"Headline: MSILN test MAE: {headline_parts} (trained at K=4).")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"[k_sweep] wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
