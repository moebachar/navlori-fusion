"""Post-hoc per-path distribution + plots + JSON dump for the
WiFi+IMU K=1 baseline trained by `_train_webots_2mod_baseline.py`.

The training run wrote `model.pt` and `history.json` to its run_path
but the wrapper's per-path code failed due to a tuple-unpack bug.
This script reloads the saved model and finishes the per-path step
without retraining.

Usage: ``.venv/Scripts/python.exe scripts/_postprocess_2mod_baseline.py``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_06"


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
    # Find the most recent fusion run dir.
    runs = sorted(OUT_DIR.glob("fusion_*"))
    if not runs:
        raise SystemExit("No fusion_* run dir found in runs/overnight/run2_iter_06/")
    run_path = runs[-1]
    print(f"loading run from {run_path}", flush=True)

    # Rebuild the exact same config + trainer the training script used.
    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu"]
    cfg.temporal.n_instants = 1
    cfg.data.batch_size = 128

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    model = build_model(cfg, encs)
    # Load the saved best-val model.
    state = torch.load(run_path / "model.pt", weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    print("model loaded.", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(cfg.dataset.modalities),
        lr=float(cfg.train.lr),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        batch_size=int(cfg.data.batch_size),
        run_dir=str(OUT_DIR / "postproc_skip"),
    )

    # === Per-path distribution + plots for val and test ===
    out = {"split": {}}
    for split in ("val", "test"):
        if split not in trainer.splits:
            continue
        pred_t, gt_t = trainer.predict(split)
        pred = pred_t.numpy()
        gt = gt_t.numpy()
        ds = dm.val_ds if split == "val" else dm.test_ds
        pids = np.array([r["path_id"] for r in ds._gt_rows])
        # Note: predict iterates batches in order, so pids align with the
        # sample order of pred/gt.
        if len(pids) != len(pred):
            print(f"WARN: {split} pids({len(pids)}) != pred({len(pred)}); "
                  "taking first len(pred) pids", flush=True)
            pids = pids[:len(pred)]
        dist = per_path_distribution(pred, gt, pids)
        print(f"\n=== {split} per-path distribution ===", flush=True)
        for p, pp in dist["per_path"].items():
            print(f"  path {p}: mean={pp['mean']:.3f} med={pp['median']:.3f} "
                  f"p90={pp['p90']:.3f} max={pp['max']:.3f}  (n={pp['n_samples']})",
                  flush=True)
        print(f"  aggregate: mean={dist['aggregate']['mean']:.3f} "
              f"p50={dist['aggregate']['median']:.3f} "
              f"p90={dist['aggregate']['p90']:.3f}", flush=True)
        out["split"][split] = dist
        if split == "test":
            for pid in [15, 16, 17]:
                mask = pids == pid
                if mask.sum() > 5:
                    plot_path(pred[mask], gt[mask], pid,
                              OUT_DIR / "test_paths" / f"wifi_imu_K1_path_{pid:02d}.png",
                              "(WiFi+IMU K=1)")

    # === Subset eval ===
    out["subsets_val"] = {k: {"mae": float(v["mae"])}
                          for k, v in trainer.evaluate_all_subsets("val").items()}
    if "test" in trainer.splits:
        out["subsets_test"] = {k: {"mae": float(v["mae"])}
                                for k, v in trainer.evaluate_all_subsets("test").items()}

    # === Latency probe ===
    trainer.model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = trainer.predict("val")
        if trainer.device == "cuda":
            torch.cuda.synchronize()
    n_val = trainer.n["val"]
    out["latency_ms_per_sample_b1"] = (time.time() - t0) / 20 / max(n_val, 1) * 1000.0
    print(f"\nlatency b=1: {out['latency_ms_per_sample_b1']:.4f} ms / sample", flush=True)

    # === Load history.json for best_val_mae and best_epoch ===
    with open(run_path / "history.json") as f:
        hist = json.load(f)
    out["training"] = {
        "best_val_mae": float(hist["best_val_mae"]),
        "best_epoch": int(hist["best_epoch"]),
        "elapsed_s": float(hist.get("elapsed_sec", 0.0)),
        "run_path": str(run_path.relative_to(ROOT)),
    }

    with open(OUT_DIR / "wifi_imu_K1_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR/'wifi_imu_K1_baseline.json'}", flush=True)


if __name__ == "__main__":
    main()
