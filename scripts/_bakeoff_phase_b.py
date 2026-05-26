"""PLAN_16 bake-off runner — train 4 fusion aggregators on a 10 %
Webots subset (paths [1, 3] = ~2 of 11 train paths; same val/test
splits as RESULT_14).

Each candidate inherits FusionTransformer's encoders + CLS +
PositionQuery readout, so the differentiator is the K-M token
aggregator alone.

Run: ``.venv/Scripts/python.exe scripts/_bakeoff_phase_b.py``
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
    build_datamodule, build_encoders, extract_vision_tokens, load_config,
)
from src.pipeline.fusion.bakeoff import CANDIDATES  # noqa: E402
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_16"

# 10 % subset: 2 train paths out of 11 to keep S/N for pre-test.
SUBSET_TRAIN_PATHS = [1, 3]


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
             "median_r": float(np.median(rs)) if rs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== bake-off: K=4 4-mod B=128, 30 epochs on 2-path subset ===",
          flush=True)
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera", "odom"]
    cfg.temporal.n_instants = 4
    cfg.data.batch_size = args.batch_size
    cfg.dataset.split.train_paths = SUBSET_TRAIN_PATHS
    print(f"  train paths (subset): {list(cfg.dataset.split.train_paths)}", flush=True)
    print(f"  val paths: {list(cfg.dataset.split.val_paths)}", flush=True)
    print(f"  test paths: {list(cfg.dataset.split.test_paths)}", flush=True)

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

    results = {}
    for name, builder in CANDIDATES.items():
        print(f"\n--- candidate: {name} ---", flush=True)
        torch.manual_seed(42)
        model = builder(incumbent_kwargs, encs)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params/1e6:.2f} M", flush=True)

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
            run_dir=str(OUT_DIR / name),
        )

        t0 = time.time()
        hist = trainer.fit(epochs=args.epochs, verbose=False)
        elapsed = time.time() - t0
        print(f"  best val MAE = {hist.best_val_mae:.3f}  (epoch {hist.best_epoch})  "
              f"{elapsed:.0f} s", flush=True)

        # subset eval (3 informative rows)
        subsets_val = trainer.evaluate_all_subsets("val")
        subsets_test = trainer.evaluate_all_subsets("test")
        only_wifi_test = float(subsets_test["wifi"]["mae"])
        wifi_imu_camera_test = float(subsets_test.get("wifi+imu+camera",
                                                       subsets_test["wifi"])["mae"])
        full_test = float(subsets_test["wifi+imu+camera+odom"]["mae"]
                          if "wifi+imu+camera+odom" in subsets_test
                          else subsets_test[list(subsets_test.keys())[-1]]["mae"])
        full_val = float(subsets_val["wifi+imu+camera+odom"]["mae"]
                          if "wifi+imu+camera+odom" in subsets_val
                          else subsets_val[list(subsets_val.keys())[-1]]["mae"])
        print(f"  test only:wifi = {only_wifi_test:.3f}  "
              f"wifi+imu+camera = {wifi_imu_camera_test:.3f}  "
              f"full = {full_test:.3f}", flush=True)

        # smoothness
        pred_t, gt_t = trainer.predict("test")
        pred = pred_t.numpy(); gt = gt_t.numpy()
        pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
        smooth = per_traj_smoothness(pred, gt, pids)
        print(f"  smoothness median r = {smooth['median_r']:.3f}", flush=True)

        results[name] = {
            "params": int(n_params),
            "val_mae": float(hist.best_val_mae),
            "test_mae": full_test,
            "best_epoch": int(hist.best_epoch),
            "elapsed_s": float(elapsed),
            "subsets": {
                "only_wifi_test": only_wifi_test,
                "wifi_imu_camera_test": wifi_imu_camera_test,
                "full_test": full_test,
                "full_val": full_val,
            },
            "smoothness_median_r": float(smooth["median_r"]),
            "smoothness_per_path": smooth["per_path"],
        }
        del model, trainer
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'candidate':<14} {'params':>10} {'val':>8} {'test':>8} "
          f"{'only:wifi':>10} {'wic':>8} {'r':>8} {'wall (s)':>10}",
          flush=True)
    for name, r in results.items():
        print(f"  {name:<14} {r['params']/1e6:>7.2f}M  "
              f"{r['val_mae']:>7.3f}  {r['test_mae']:>7.3f}  "
              f"{r['subsets']['only_wifi_test']:>9.3f}  "
              f"{r['subsets']['wifi_imu_camera_test']:>7.3f}  "
              f"{r['smoothness_median_r']:>+7.3f}  "
              f"{r['elapsed_s']:>9.0f}", flush=True)

    with open(OUT_DIR / "bakeoff_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'bakeoff_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
