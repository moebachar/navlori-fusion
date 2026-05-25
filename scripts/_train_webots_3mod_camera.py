"""PLAN_09 — add Camera (DPVOMotion-P-A) to the WiFi+IMU K=1 fusion.

Mirrors `scripts/_train_webots_2mod_baseline.py` but with
`cfg.dataset.modalities = ['wifi', 'imu', 'camera']`. The fusion
builder constructs `DPVOMotionEncoder`, the runner calls
`extract_vision_tokens(dm, vision_encoder, device)` to cache the
per-pair patch tokens once (the trunk is frozen — only the
`_MotionHead` trains in the fusion model), then trains FusionTrainer
with `extra_inputs={'camera': {'train': T, 'val': T, 'test': T}}`.

Run: ``.venv/Scripts/python.exe scripts/_train_webots_3mod_camera.py``
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
    build_datamodule, build_encoders, build_model, extract_vision_tokens,
    load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_09"


def per_path_distribution(preds, gts, pid):
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for p in np.unique(pid):
        m = pid == p
        e = errs[m]
        per_path[int(p)] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max()),
            "n": int(len(e)),
        }
    return {
        "aggregate": {
            "mean": float(errs.mean()), "median": float(np.median(errs)),
            "p25": float(np.percentile(errs, 25)), "p75": float(np.percentile(errs, 75)),
            "p90": float(np.percentile(errs, 90)), "max": float(errs.max()),
        },
        "per_path": per_path,
    }


def per_traj_smoothness(preds, gts, pid):
    per_path = {}
    for p in np.unique(pid):
        m = pid == p
        if m.sum() < 5:
            continue
        pp = preds[m]
        gg = gts[m]
        dp = np.linalg.norm(np.diff(pp, axis=0), axis=1)
        dg = np.linalg.norm(np.diff(gg, axis=0), axis=1)
        if dp.std() < 1e-9 or dg.std() < 1e-9:
            per_path[int(p)] = 0.0
        else:
            per_path[int(p)] = float(np.corrcoef(dp, dg)[0, 1])
    rs = list(per_path.values())
    return {"per_path": per_path,
             "median_r": float(np.median(rs)) if rs else 0.0,
             "min_r": float(min(rs)) if rs else 0.0,
             "max_r": float(max(rs)) if rs else 0.0}


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
    print(f"=== build config (simulation, WiFi+IMU+Camera, K=1) ===", flush=True)
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera"]
    cfg.temporal.n_instants = 1
    cfg.data.batch_size = args.batch_size
    print(f"  modalities = {list(cfg.dataset.modalities)}", flush=True)
    print(f"  n_instants = {cfg.temporal.n_instants}", flush=True)
    print(f"  batch_size = {cfg.data.batch_size}", flush=True)

    print("\n=== build datamodule ===", flush=True)
    dm = build_datamodule(cfg)
    print(f"  train: {len(dm.train_ds)}  val: {len(dm.val_ds)}  test: {len(dm.test_ds)}",
          flush=True)

    print("\n=== build encoders + vision tokens ===", flush=True)
    encs, vision = build_encoders(cfg, dm)
    print(f"  encoders: {list(encs.keys())}", flush=True)
    print(f"  vision encoder: {type(vision).__name__ if vision is not None else None}",
          flush=True)
    t0 = time.time()
    extra = extract_vision_tokens(dm, vision, device="cuda")
    print(f"  vision token extraction: {time.time()-t0:.1f}s; "
          f"train/val/test shapes: "
          f"{extra['camera']['train'].shape if 'train' in extra['camera'] else 'n/a'} / "
          f"{extra['camera']['val'].shape if 'val' in extra['camera'] else 'n/a'} / "
          f"{extra['camera']['test'].shape if 'test' in extra['camera'] else 'n/a'}",
          flush=True)

    print("\n=== build fusion model ===", flush=True)
    model = build_model(cfg, encs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f} M", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(cfg.dataset.modalities),
        extra_inputs=extra,
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

    # === memory budget check via one trainer step ===
    print(f"\n=== memory budget probe (one train epoch on small batch) ===", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        orig_bs = trainer.batch_size
        trainer.batch_size = 32
        try:
            _ = trainer._train_epoch()
        except Exception as e:
            print(f"  probe step raised: {type(e).__name__}: {e}", flush=True)
        trainer.batch_size = orig_bs
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
    else:
        peak_mb = 0.0
    print(f"  peak GPU = {peak_mb:.1f} MB", flush=True)
    if peak_mb > 6000:
        raise RuntimeError(f"peak {peak_mb} MB exceeds 6 GB budget")

    # === pre-test gate ===
    print(f"\n=== pre-test gate: {args.pretest_epochs} epochs ===", flush=True)
    pre_hist = trainer.fit(epochs=args.pretest_epochs, verbose=True)
    pretest_first = pre_hist.val_mae[0] if pre_hist.val_mae else float("inf")
    pretest_best = pre_hist.best_val_mae
    drop_pct = (pretest_first - pretest_best) / max(pretest_first, 1e-6) * 100
    print(f"  first={pretest_first:.3f}  best={pretest_best:.3f}  drop={drop_pct:.1f}%",
          flush=True)
    pretest_pass = drop_pct >= 10.0

    # === full training (continues) ===
    print(f"\n=== full training: {args.epochs} epochs ===", flush=True)
    full_t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    elapsed = time.time() - full_t0
    print(f"\n  best val MAE = {hist.best_val_mae:.3f} (epoch {hist.best_epoch})  "
          f"elapsed {elapsed:.1f}s", flush=True)

    # === latency probe (predict on val) ===
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
    n_val = trainer.n["val"]
    lat_ms = (time.time() - t0) / 10 / max(n_val, 1) * 1000.0
    print(f"\nlatency b=1 (via predict batched): {lat_ms:.4f} ms / sample", flush=True)

    # === subset eval (val + test) ===
    print("\n=== subset eval ===", flush=True)
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    for n, d in subsets_val.items():
        print(f"  val  {n:20s} -> mae={d['mae']:.3f}", flush=True)
    for n, d in subsets_test.items():
        print(f"  test {n:20s} -> mae={d['mae']:.3f}", flush=True)

    # === per-path distribution + smoothness on test ===
    pred_t, gt_t = trainer.predict("test")
    pred = pred_t.numpy(); gt = gt_t.numpy()
    pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
    test_dist = per_path_distribution(pred, gt, pids)
    test_smooth = per_traj_smoothness(pred, gt, pids)
    print(f"\n=== test per-path distribution ===", flush=True)
    for p, pp in test_dist["per_path"].items():
        print(f"  path {p}: mean={pp['mean']:.3f} med={pp['median']:.3f} "
              f"p90={pp['p90']:.3f} max={pp['max']:.3f}", flush=True)
    print(f"  aggregate: mean={test_dist['aggregate']['mean']:.3f} "
          f"p50={test_dist['aggregate']['median']:.3f} "
          f"p90={test_dist['aggregate']['p90']:.3f}", flush=True)
    print(f"  smoothness per path: {test_smooth['per_path']}", flush=True)
    print(f"  smoothness median r = {test_smooth['median_r']:.3f}", flush=True)
    for p in [15, 16, 17]:
        mask = pids == p
        if mask.sum() > 5:
            plot_path(pred[mask], gt[mask], p,
                      OUT_DIR / "test_paths" / f"3mod_path_{p:02d}.png",
                      "(WiFi+IMU+Camera K=1)")

    # === dump JSON ===
    out = {
        "config": {"modalities": list(cfg.dataset.modalities),
                    "n_instants": int(cfg.temporal.n_instants),
                    "batch_size": int(cfg.data.batch_size),
                    "epochs": int(args.epochs)},
        "memory_budget_mb": float(peak_mb),
        "pretest": {"first_val_mae": float(pretest_first),
                     "best_val_mae": float(pretest_best),
                     "drop_pct": float(drop_pct),
                     "pass": bool(pretest_pass)},
        "training": {"best_val_mae": float(hist.best_val_mae),
                      "best_epoch": int(hist.best_epoch),
                      "elapsed_s": float(elapsed)},
        "latency_ms_per_sample_b1": float(lat_ms),
        "subsets_val": {k: {"mae": float(v["mae"])} for k, v in subsets_val.items()},
        "subsets_test": {k: {"mae": float(v["mae"])} for k, v in subsets_test.items()},
        "test_dist": test_dist,
        "test_smoothness": test_smooth,
        "n_params": int(n_params),
    }
    with open(OUT_DIR / "wifi_imu_camera_K1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'wifi_imu_camera_K1.json'}", flush=True)


if __name__ == "__main__":
    main()
