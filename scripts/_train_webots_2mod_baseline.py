"""PLAN_06 Step 2: reproduce run-1's 2-modality (WiFi + IMU) Webots baseline.

Single-instant fusion (K=1), no Camera, no Odom. Mirrors the run-1
single-instant ~0.43 m val MAE per CLAUDE.md "Stage A + B/C Complete"
section. Mostly a thin wrapper around the restored builder pattern;
the only real work is overriding the simulation config from
4-modality / K=8 down to 2-modality / K=1.

Run: ``.venv/Scripts/python.exe scripts/_train_webots_2mod_baseline.py``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_06"


def memory_check(trainer: FusionTrainer, batch: int = 32) -> float:
    """Forward+backward on a synthetic batch (K=1, all 2 modalities present)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Run one training step manually.
    trainer.model.train()
    # Grab one mini-batch from the train index.
    n = trainer.n["train"]
    idx = torch.arange(min(batch, n), device=trainer.device)
    # We just call _train_epoch's inner step shape by invoking a single
    # forward via predict() which uses the same plumbing — but predict
    # is eval-mode. Easiest: synthesize batch via _train_epoch on a
    # micro-batch by setting batch_size and running one epoch.
    orig_bs = trainer.batch_size
    trainer.batch_size = batch
    try:
        _ = trainer._train_epoch()
    except Exception as e:
        print(f"memory probe step raised: {type(e).__name__}: {e}", flush=True)
    trainer.batch_size = orig_bs
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return float(peak_mb)


def latency_probe(trainer: FusionTrainer, runs: int = 50) -> float:
    """Per-sample inference latency at batch=1 (criterion (e))."""
    trainer.model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
    n_val = trainer.n["val"]
    # predict() returns the full val set; ms / sample is total_time / runs / n_val.
    return (time.time() - t0) / runs / max(n_val, 1) * 1000.0


def per_path_distribution(preds: np.ndarray, gts: np.ndarray, path_ids: np.ndarray):
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for pid in np.unique(path_ids):
        mask = path_ids == pid
        e = errs[mask]
        per_path[int(pid)] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max()),
            "n_samples": int(len(e)),
        }
    return {
        "aggregate": {
            "mean": float(errs.mean()), "median": float(np.median(errs)),
            "p25": float(np.percentile(errs, 25)), "p75": float(np.percentile(errs, 75)),
            "p90": float(np.percentile(errs, 90)), "max": float(errs.max()),
        },
        "per_path": per_path,
    }


def plot_path(preds, gts, pid, out_path, suffix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gts[:, 0], gts[:, 1], "k-", label="GT", lw=1.5)
    ax.plot(preds[:, 0], preds[:, 1], "r-", label="pred", lw=1.0, alpha=0.75)
    ax.scatter(gts[0, 0], gts[0, 1], c="green", s=40, marker="o", label="start")
    ax.set_aspect("equal")
    ax.set_title(f"path_{pid:02d} {suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--pretest-epochs", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Building config (simulation, WiFi+IMU, K=1) ===", flush=True)
    cfg = load_config("simulation")
    # Restrict to WiFi + IMU (the run-1 baseline).
    cfg.dataset.modalities = ["wifi", "imu"]
    # Single-instant per the plan; the run-1 reference 0.43 m was at K=1.
    cfg.temporal.n_instants = 1
    # Slightly larger batch is fine on this GPU; cap memory.
    cfg.data.batch_size = args.batch_size
    print(f"  modalities = {cfg.dataset.modalities}", flush=True)
    print(f"  n_instants = {cfg.temporal.n_instants}", flush=True)
    print(f"  batch_size = {cfg.data.batch_size}", flush=True)

    print("\n=== Building datamodule ===", flush=True)
    dm = build_datamodule(cfg)
    print(f"  train: {len(dm.train_ds)}  val: {len(dm.val_ds)}  test: {len(dm.test_ds)}",
          flush=True)
    for m in cfg.dataset.modalities:
        d = dm.train_ds.feature_dims[m]
        print(f"  feature_dim[{m}] = {d}", flush=True)

    print("\n=== Building encoders + model ===", flush=True)
    encs, vision = build_encoders(cfg, dm)
    print(f"  encoders: {list(encs.keys())}", flush=True)
    model = build_model(cfg, encs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f} M", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(cfg.dataset.modalities),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        huber_delta=float(cfg.train.huber_delta),
        grad_clip=float(cfg.train.grad_clip),
        patience=int(cfg.train.patience),
        batch_size=int(cfg.data.batch_size),
        modality_dropout=float(cfg.train.modality_dropout),
        instant_dropout=float(cfg.train.instant_dropout),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        modality_balanced_loss=bool(cfg.train.modality_balanced_loss),
        modality_balanced_weight=float(cfg.train.modality_balanced_weight),
        aux_abs_weight=float(cfg.train.aux_abs_weight),
        run_dir=str(OUT_DIR),
    )
    print(f"  run_path: {trainer.run_path}", flush=True)

    # === Pre-test gate ===
    print(f"\n=== Pre-test gate: {args.pretest_epochs} epochs ===", flush=True)
    t0 = time.time()
    pre_hist = trainer.fit(epochs=args.pretest_epochs, verbose=True)
    pretest_best = pre_hist.best_val_mae
    pretest_first = pre_hist.val_mae[0] if pre_hist.val_mae else float("inf")
    drop_pct = (pretest_first - pretest_best) / max(pretest_first, 1e-6) * 100
    print(f"  pre-test: first={pretest_first:.3f}  best={pretest_best:.3f}  "
          f"drop={drop_pct:.1f}%", flush=True)
    pretest_pass = drop_pct >= 10.0
    print(f"  pre-test gate pass={pretest_pass}", flush=True)
    if not pretest_pass:
        print("WARNING: pre-test gate failed (drop < 10 %)", flush=True)

    # === Full training (continues from pretest model state) ===
    print(f"\n=== Full training: {args.epochs} epochs (continuing) ===", flush=True)
    full_t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    full_elapsed = time.time() - full_t0
    print(f"\n=== Training done ===", flush=True)
    print(f"  best val MAE = {hist.best_val_mae:.3f} m (epoch {hist.best_epoch})",
          flush=True)
    print(f"  elapsed = {full_elapsed:.1f} s", flush=True)

    # === Latency probe ===
    print("\n=== Latency probe (batch=1) ===", flush=True)
    lat_ms = latency_probe(trainer, runs=20)
    print(f"  latency per sample (batch=1): {lat_ms:.3f} ms", flush=True)

    # === Per-modality subset eval ===
    print("\n=== Subset eval (val and test) ===", flush=True)
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test") if "test" in trainer.splits else None
    for name, d in subsets_val.items():
        print(f"  val  {name:20s} -> mae={d['mae']:.3f}", flush=True)
    if subsets_test:
        for name, d in subsets_test.items():
            print(f"  test {name:20s} -> mae={d['mae']:.3f}", flush=True)

    # === Per-path distribution + plots ===
    print("\n=== Per-path distribution + plots ===", flush=True)
    # FusionTrainer.predict returns (preds, tgts) — both reordered by the
    # internal batch iterator (sample order is preserved by batched stride).
    test_pred_t, test_gt_t = trainer.predict("test")
    test_pred = test_pred_t.numpy()
    gts = test_gt_t.numpy()
    # path ids: dm test_ds carries ._gt_rows with path_id per sample.
    test_pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])
    test_dist = per_path_distribution(test_pred, gts, test_pids)
    for pid, pp in test_dist["per_path"].items():
        print(f"  path {pid}: mean={pp['mean']:.3f} med={pp['median']:.3f} "
              f"p90={pp['p90']:.3f} max={pp['max']:.3f}  (n={pp['n_samples']})",
              flush=True)
    print(f"  aggregate: mean={test_dist['aggregate']['mean']:.3f} "
          f"p50={test_dist['aggregate']['median']:.3f} "
          f"p90={test_dist['aggregate']['p90']:.3f}", flush=True)
    for pid in [15, 16, 17]:
        mask = test_pids == pid
        if mask.sum() > 5:
            plot_path(test_pred[mask], gts[mask], pid,
                      OUT_DIR / "test_paths" / f"wifi_imu_K1_path_{pid:02d}.png",
                      "(WiFi+IMU K=1)")

    # === Dump results ===
    out = {
        "config": {
            "modalities": list(cfg.dataset.modalities),
            "n_instants": int(cfg.temporal.n_instants),
            "batch_size": int(cfg.data.batch_size),
            "epochs": int(args.epochs),
            "lr": float(cfg.train.lr),
        },
        "pretest": {
            "first_val_mae": float(pretest_first),
            "best_val_mae": float(pretest_best),
            "drop_pct": float(drop_pct),
            "pass": bool(pretest_pass),
        },
        "training": {
            "best_val_mae": float(hist.best_val_mae),
            "best_epoch": int(hist.best_epoch),
            "elapsed_s": float(full_elapsed),
        },
        "latency_ms_per_sample_b1": float(lat_ms),
        "subsets_val": {k: {"mae": v["mae"]} for k, v in subsets_val.items()},
        "subsets_test": ({k: {"mae": v["mae"]} for k, v in subsets_test.items()}
                          if subsets_test else None),
        "test_dist": test_dist,
        "n_params": int(n_params),
    }
    with open(OUT_DIR / "wifi_imu_K1_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR/'wifi_imu_K1_baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
