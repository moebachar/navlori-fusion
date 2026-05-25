"""PLAN_10 — add Odom 1.5-modality to the WiFi+IMU+Camera K=1 fusion.

Per RESULT_04's recommendation (iii): feed BOTH the OdomCNN-P-B
learned embedding AND a raw integrated `(odom_x, odom_y)` column
(for smoothness). Two modality slots in the fusion model:

- ``odom`` — `OdomCNN(in_features=5)` on the 5-feature windowed
  P-B input (Δ-features on theta_deg/etc, raw on velocities + wheel
  speeds). Already wired via the existing builder + dataset.
- ``odom_raw`` — per-sample `(Δx, Δy)` = wheel-odometry position at
  the anchor time, minus the path's t=0 position, in the path's local
  frame. Single instant, 2-d input. Encoded by a tiny
  2 → 64 → 128 MLP.

The `odom_raw` features are pre-computed from each path's
`odometry.csv` once, then served to the FusionTrainer via
``extra_inputs={'odom_raw': {'train': T, 'val': T, 'test': T}}``,
the same path the Camera modality uses for its frozen-trunk tokens.

Run: ``.venv/Scripts/python.exe scripts/_train_webots_4mod_odom1p5.py``
"""
from __future__ import annotations

import argparse
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

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, extract_vision_tokens,
    load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_10"


class OdomRawEncoder(nn.Module):
    """Tiny encoder for the 1.5-modality raw odom path: 2 -> 64 -> 128."""

    def __init__(self, in_dim: int = 2, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 2) — single-instant flattened by the fusion's encode_tokens
        # to (B*K, 1, 2) where K=1, so we squeeze.
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        return self.net(x)


def build_odom_raw_features(dm) -> dict[str, torch.Tensor]:
    """Per-sample (Δx, Δy) wheel-odometry displacement from each path's t=0.

    For each sample at (path, sim_time): interpolate the path's
    odometry.csv to get (odom_x, odom_y) at that time, then subtract
    the first odometry row's (odom_x, odom_y) of the same path. The
    result is "how far we've travelled from the path's start, per the
    wheel odometry" — drifts cumulatively but is locally smooth
    (RESULT_04's r=0.999 signal).
    """
    out = {}
    for split, ds in [("train", dm.train_ds), ("val", dm.val_ds), ("test", dm.test_ds)]:
        if ds is None:
            continue
        # Cache odom CSV per path on first access.
        odom_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        feats = np.zeros((len(ds._gt_rows), 2), dtype=np.float32)
        for i, r in enumerate(ds._gt_rows):
            pid = int(r["path_id"])
            t = float(r["time"])
            if pid not in odom_cache:
                df = pd.read_csv(Path(r["path_dir"]) / "odometry.csv")
                odom_cache[pid] = (
                    df["sim_time"].values.astype(np.float64),
                    df["odom_x"].values.astype(np.float32),
                    df["odom_y"].values.astype(np.float32),
                )
            ts, xs, ys = odom_cache[pid]
            x_at_t = float(np.interp(t, ts, xs))
            y_at_t = float(np.interp(t, ts, ys))
            feats[i, 0] = x_at_t - xs[0]
            feats[i, 1] = y_at_t - ys[0]
        # Add a window=1 axis so the trainer treats this as a 1-step window.
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(1)  # (N, 1, 2)
        out[split] = feats_t
        print(f"  odom_raw {split}: {feats_t.shape}  "
              f"range=[{feats[:, 0].min():.2f}, {feats[:, 0].max():.2f}] x "
              f"[{feats[:, 1].min():.2f}, {feats[:, 1].max():.2f}]", flush=True)
    return out


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
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--pretest-epochs", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== build config (simulation, 4-mod + odom_raw, K=1) ===", flush=True)
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera", "odom"]
    cfg.temporal.n_instants = 1
    cfg.data.batch_size = args.batch_size
    print(f"  base modalities (from datamodule): {list(cfg.dataset.modalities)}",
          flush=True)

    dm = build_datamodule(cfg)
    print(f"  train: {len(dm.train_ds)}  val: {len(dm.val_ds)}  test: {len(dm.test_ds)}",
          flush=True)

    print("\n=== build encoders + vision tokens + odom_raw cache ===", flush=True)
    encs, vision = build_encoders(cfg, dm)
    # Add the 1.5-modality raw odom encoder.
    encs["odom_raw"] = OdomRawEncoder(in_dim=2, embed_dim=cfg.model.embed_dim)
    print(f"  encoders (with odom_raw): {list(encs.keys())}", flush=True)

    extra = extract_vision_tokens(dm, vision, device="cuda")
    print(f"  vision tokens: train {extra['camera']['train'].shape}, "
          f"val {extra['camera']['val'].shape}, test {extra['camera']['test'].shape}",
          flush=True)

    extra["odom_raw"] = build_odom_raw_features(dm)

    print("\n=== build fusion model ===", flush=True)
    # The model receives the encoders dict in order — the modalities list given to
    # FusionTrainer must match this order exactly.
    model = build_model(cfg, encs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f} M  (modalities = {model.modalities})",
          flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm,
        modalities=list(model.modalities),  # = wifi, imu, camera, odom, odom_raw
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

    # === memory probe ===
    print("\n=== memory budget probe ===", flush=True)
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

    # === full training ===
    print(f"\n=== full training: {args.epochs} epochs ===", flush=True)
    full_t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    elapsed = time.time() - full_t0
    print(f"\n  best val MAE = {hist.best_val_mae:.3f} (epoch {hist.best_epoch})  "
          f"elapsed {elapsed:.1f}s", flush=True)

    # === latency probe ===
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(5): _ = trainer.predict("val")
        if trainer.device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): _ = trainer.predict("val")
        if trainer.device == "cuda": torch.cuda.synchronize()
    lat_ms = (time.time() - t0) / 10 / max(trainer.n["val"], 1) * 1000.0
    print(f"\nlatency b=1: {lat_ms:.4f} ms/sample", flush=True)

    # === subset eval ===
    print("\n=== subset eval ===", flush=True)
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    for n, d in subsets_val.items():
        print(f"  val  {n:30s} -> mae={d['mae']:.3f}", flush=True)
    for n, d in subsets_test.items():
        print(f"  test {n:30s} -> mae={d['mae']:.3f}", flush=True)

    # === per-path test ===
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
                      OUT_DIR / "test_paths" / f"4mod_path_{p:02d}.png",
                      "(WiFi+IMU+Camera+Odom1.5 K=1)")

    out = {
        "config": {"modalities": list(model.modalities),
                    "n_instants": int(cfg.temporal.n_instants),
                    "batch_size": int(cfg.data.batch_size),
                    "epochs": int(args.epochs)},
        "memory_budget_mb": float(peak_mb),
        "pretest": {"first_val_mae": float(pretest_first),
                     "best_val_mae": float(pretest_best),
                     "drop_pct": float(drop_pct),
                     "pass": bool(drop_pct >= 10.0)},
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
    with open(OUT_DIR / "wifi_imu_camera_odom1p5_K1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'wifi_imu_camera_odom1p5_K1.json'}", flush=True)


if __name__ == "__main__":
    main()
