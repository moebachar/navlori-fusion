"""PLAN_17 — train any bake-off architecture at the K=4 4-mod B=128
full-data config. Same protocol as RESULT_13's wrapper, parameterised
by `--arch {incumbent, cnn1d, lstm_attn, tcn}`.

Run: ``.venv/Scripts/python.exe scripts/_train_webots_4mod_arch.py --arch cnn1d``
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

from src.pipeline.fusion.bakeoff import CANDIDATES  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, extract_vision_tokens, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_17"


def per_path_distribution(preds, gts, pid):
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for p in np.unique(pid):
        m = pid == p; e = errs[m]
        per_path[int(p)] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max()), "n": int(len(e)),
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
        if m.sum() < 5: continue
        pp, gg = preds[m], gts[m]
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
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gts[:, 0], gts[:, 1], "k-", label="GT", lw=1.5)
    ax.plot(preds[:, 0], preds[:, 1], "r-", label="pred", lw=1.0, alpha=0.75)
    ax.scatter(gts[0, 0], gts[0, 1], c="green", s=40, marker="o", label="start")
    ax.set_aspect("equal"); ax.set_title(f"path_{pid:02d} {suffix}")
    ax.legend(); ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(CANDIDATES.keys()))
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--pretest-epochs", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== arch={args.arch}  full-data K=4 4-mod B=128 ===", flush=True)
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera", "odom"]
    cfg.temporal.n_instants = 4
    cfg.data.batch_size = args.batch_size

    dm = build_datamodule(cfg)
    print(f"  train: {len(dm.train_ds)}  val: {len(dm.val_ds)}  test: {len(dm.test_ds)}",
          flush=True)

    encs, vision = build_encoders(cfg, dm)
    extra = extract_vision_tokens(dm, vision, device="cuda")

    incumbent_kwargs = dict(
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.n_heads),
        ff_mult=int(cfg.model.ff_mult),
        dropout=float(cfg.model.dropout),
        use_time=bool(cfg.model.use_time),
        readout=str(cfg.model.readout),
        absolute_modalities=list(cfg.model.get("absolute_modalities", None) or ["wifi"]),
    )

    torch.manual_seed(42)
    model = CANDIDATES[args.arch](incumbent_kwargs, encs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f} M", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities),
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
        run_dir=str(OUT_DIR / args.arch),
    )

    # memory probe via one train epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # pre-test
    print(f"\n=== pre-test gate: {args.pretest_epochs} epochs ===", flush=True)
    pre_hist = trainer.fit(epochs=args.pretest_epochs, verbose=True)
    pretest_first = pre_hist.val_mae[0] if pre_hist.val_mae else float("inf")
    pretest_best = pre_hist.best_val_mae
    drop_pct = (pretest_first - pretest_best) / max(pretest_first, 1e-6) * 100
    print(f"  first={pretest_first:.3f}  best={pretest_best:.3f}  drop={drop_pct:.1f}%",
          flush=True)

    # full training
    print(f"\n=== full training: {args.epochs} epochs ===", flush=True)
    t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  best val MAE = {hist.best_val_mae:.3f} (epoch {hist.best_epoch})  "
          f"elapsed {elapsed:.1f}s", flush=True)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  peak GPU: {peak_mb:.1f} MB", flush=True)

    # latency
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(3): _ = trainer.predict("val")
        if trainer.device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5): _ = trainer.predict("val")
        if trainer.device == "cuda": torch.cuda.synchronize()
    lat_ms = (time.time() - t0) / 5 / max(trainer.n["val"], 1) * 1000.0

    # subset eval
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    for n, d in subsets_val.items():
        print(f"  val  {n:30s} -> mae={d['mae']:.3f}", flush=True)
    for n, d in subsets_test.items():
        print(f"  test {n:30s} -> mae={d['mae']:.3f}", flush=True)

    # per-path test + smoothness
    pred_t, gt_t = trainer.predict("test")
    pred = pred_t.numpy(); gt = gt_t.numpy()
    pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
    test_dist = per_path_distribution(pred, gt, pids)
    test_smooth = per_traj_smoothness(pred, gt, pids)
    print(f"\n  per-path: {test_dist['per_path']}", flush=True)
    print(f"  smoothness median r = {test_smooth['median_r']:.3f}", flush=True)
    for p in [15, 16, 17]:
        mask = pids == p
        if mask.sum() > 5:
            plot_path(pred[mask], gt[mask], p,
                      OUT_DIR / "test_paths" / f"{args.arch}_path_{p:02d}.png",
                      f"({args.arch} K=4 4-mod)")

    out = {
        "arch": args.arch,
        "config": {"modalities": list(model.modalities),
                    "n_instants": int(cfg.temporal.n_instants),
                    "batch_size": int(cfg.data.batch_size),
                    "epochs": int(args.epochs)},
        "pretest": {"first_val_mae": float(pretest_first),
                     "best_val_mae": float(pretest_best),
                     "drop_pct": float(drop_pct)},
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
    with open(OUT_DIR / f"{args.arch}_full.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / f'{args.arch}_full.json'}", flush=True)


if __name__ == "__main__":
    main()
