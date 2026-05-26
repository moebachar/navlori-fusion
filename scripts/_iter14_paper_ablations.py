"""PLAN_14 — Phase B winner full ablations (loads RESULT_13 checkpoint).

Step 0A: load the RESULT_13 K=4 4-mod B=128 model and re-run:
  - 8-lag WiFi staleness sweep for paper-figure resolution
  - b=1 + b=32 latency probes
  - full subset eval sanity-check
  - per-trajectory smoothness sanity-check

Writes a JSON summary at
``runs/overnight/run2_iter_14/winner_ablations.json``.

Run: ``.venv/Scripts/python.exe scripts/_iter14_paper_ablations.py``
"""
from __future__ import annotations

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

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_14"
ITER13_DIRS = sorted((ROOT / "runs" / "overnight" / "run2_iter_13").glob("fusion_*"))


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


def plot_staleness(lags, mae, out_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    secs = [l * 0.9 for l in lags]
    ax.plot(secs, mae, "o-", color="C0", lw=2, markersize=7)
    ax.set_xlabel("WiFi staleness (s)")
    ax.set_ylabel("test MAE (m)")
    ax.set_title("WiFi staleness sweep: K=4 + 4-mod + B=128 (RESULT_14 winner)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="C3 gate 0.5 m")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def staleness_probe(trainer, lags):
    out = {}
    trainer.model.eval()
    orig = trainer.X["test"]["wifi"]
    for lag in lags:
        if lag == 0:
            trainer.X["test"]["wifi"] = orig
        else:
            shifted = orig.clone()
            shifted[lag:] = orig[:-lag]
            trainer.X["test"]["wifi"] = shifted
        with torch.no_grad():
            pred_t, gt_t = trainer.predict("test")
            mae = float(torch.linalg.norm(pred_t - gt_t, dim=1).mean())
        out[lag] = mae
        print(f"  WiFi lag={lag:3d} instants ({lag * 0.9:5.1f} s): test MAE = {mae:.3f} m",
              flush=True)
    trainer.X["test"]["wifi"] = orig
    return out


def latency_probe(trainer, n_warmup=20, n_runs=100, batch=1):
    trainer.model.eval()
    n_val = trainer.n["val"]
    if batch >= n_val:
        batch = max(1, n_val // 2)
    with torch.no_grad():
        # use first-`batch` samples as a steady probe.
        idx = torch.arange(batch, device=trainer.device)
        for _ in range(n_warmup):
            inputs, avail, dt, ya, yi = trainer._batch("val", idx, drop=False)
            y, q = trainer._resolve_query(idx, dt, ya, yi, randomize=False)
            _ = trainer.model(inputs, avail, dt, query_dt=q)
        if trainer.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            inputs, avail, dt, ya, yi = trainer._batch("val", idx, drop=False)
            y, q = trainer._resolve_query(idx, dt, ya, yi, randomize=False)
            _ = trainer.model(inputs, avail, dt, query_dt=q)
        if trainer.device == "cuda":
            torch.cuda.synchronize()
    total = time.time() - t0
    per_sample_ms = total / n_runs / batch * 1000.0
    per_batch_ms = total / n_runs * 1000.0
    return {"batch": batch, "n_runs": n_runs,
             "ms_per_sample": per_sample_ms,
             "ms_per_batch": per_batch_ms}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ITER13_DIRS:
        raise SystemExit("No iter_13 fusion run dir found")
    run_path = ITER13_DIRS[-1]
    print(f"loading RESULT_13 checkpoint from {run_path}", flush=True)

    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera", "odom"]
    cfg.temporal.n_instants = 4
    cfg.data.batch_size = 128

    dm = build_datamodule(cfg)
    print(f"  train: {len(dm.train_ds)}  val: {len(dm.val_ds)}  test: {len(dm.test_ds)}",
          flush=True)

    encs, vision = build_encoders(cfg, dm)
    extra = extract_vision_tokens(dm, vision, device="cuda")
    model = build_model(cfg, encs)
    state = torch.load(run_path / "model.pt", weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    print(f"  model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f} M params)",
          flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities),
        extra_inputs=extra,
        lr=float(cfg.train.lr),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        batch_size=128,
        run_dir=str(OUT_DIR / "postproc_skip"),
    )

    # Sanity: reproduce RESULT_13 val/test MAE.
    pred_v, gt_v = trainer.predict("val")
    pred_t, gt_t = trainer.predict("test")
    val_mae = float(torch.linalg.norm(pred_v - gt_v, dim=1).mean())
    test_mae = float(torch.linalg.norm(pred_t - gt_t, dim=1).mean())
    print(f"sanity reproduction: val {val_mae:.3f} m  test {test_mae:.3f} m", flush=True)

    out = {
        "checkpoint": str(run_path.relative_to(ROOT)),
        "sanity": {"val_mae": val_mae, "test_mae": test_mae},
    }

    # === full subset eval ===
    print("\n=== subset eval ===", flush=True)
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    out["subsets_val"] = {k: {"mae": float(v["mae"])} for k, v in subsets_val.items()}
    out["subsets_test"] = {k: {"mae": float(v["mae"])} for k, v in subsets_test.items()}
    for k, v in subsets_test.items():
        print(f"  test {k:30s} -> mae={v['mae']:.3f}", flush=True)

    # === extended staleness sweep ===
    print("\n=== staleness sweep (8 lags) ===", flush=True)
    lags = [0, 1, 3, 5, 10, 15, 20, 30]
    staleness = staleness_probe(trainer, lags)
    out["staleness"] = staleness
    plot_staleness(lags, [staleness[l] for l in lags],
                    OUT_DIR / "staleness_curve.png")

    # === per-path test + smoothness ===
    pred = pred_t.numpy(); gt = gt_t.numpy()
    pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
    out["test_dist"] = per_path_distribution(pred, gt, pids)
    out["test_smoothness"] = per_traj_smoothness(pred, gt, pids)
    print(f"\n=== smoothness median r = {out['test_smoothness']['median_r']:.3f} ===",
          flush=True)

    # === latency probes ===
    print("\n=== latency b=1 + b=32 ===", flush=True)
    out["latency_b1"] = latency_probe(trainer, batch=1, n_runs=100)
    out["latency_b32"] = latency_probe(trainer, batch=32, n_runs=50)
    print(f"  b=1:  {out['latency_b1']['ms_per_sample']:.4f} ms/sample  "
          f"({out['latency_b1']['ms_per_batch']:.2f} ms/batch)", flush=True)
    print(f"  b=32: {out['latency_b32']['ms_per_sample']:.4f} ms/sample  "
          f"({out['latency_b32']['ms_per_batch']:.2f} ms/batch)", flush=True)

    out_path = OUT_DIR / "winner_ablations.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
