"""PLAN_18 — ablation suite on RESULT_17 winners (CNN1D + LSTM-attn).

Loads checkpoints from ``runs/overnight/run2_iter_17/<arch>/fusion_*/model.pt``
(built via the bakeoff CANDIDATES registry) and runs:
  - Sanity reproduction (val + test MAE)
  - Full 16-row subset eval (cached -> re-load)
  - 8-lag (CNN1D) / 4-lag (LSTM-attn) WiFi staleness sweep with plot
  - Per-path distribution + per-trajectory smoothness with plots
  - Latency probe b=1 (100 trials) + b=32 (50 trials)

Run: ``.venv/Scripts/python.exe scripts/_iter18_cnn1d_ablations.py --arch cnn1d``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.bakeoff import CANDIDATES  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, extract_vision_tokens, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_18"


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


def plot_staleness(lags, mae, out_path, title):
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
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="C3 gate 0.5 m")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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


def linear_slope(lags_s, mae):
    x = np.asarray(lags_s, dtype=float)
    y = np.asarray(mae, dtype=float)
    if len(x) < 2: return {"slope_m_per_s": 0.0, "r2": 0.0}
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = m * x + c
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope_m_per_s": float(m), "intercept": float(c),
             "r2": 1.0 - ss_res / max(ss_tot, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(CANDIDATES.keys()))
    ap.add_argument("--lags", default="full",
                    help="'full' = [0,1,3,5,10,15,20,30] or 'short' = [0,5,15,30]")
    ap.add_argument("--paths", default="15,16,17",
                    help="comma-separated test path IDs to plot")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== arch={args.arch}  PLAN_18 ablations ===", flush=True)

    # Look in iter_17 (CNN1D/LSTM-attn) or iter_21 (MoTTransformer) per arch.
    candidate_dirs = []
    for iter_n in ("run2_iter_17", "run2_iter_21"):
        candidate_dirs.extend(
            sorted((ROOT / "runs" / "overnight" / iter_n / args.arch).glob("fusion_*"))
        )
    if not candidate_dirs:
        raise SystemExit(f"No checkpoint dir found for {args.arch}")
    run_path = candidate_dirs[-1]
    print(f"loading {args.arch} checkpoint from {run_path}", flush=True)

    cfg = load_config("simulation")
    cfg.dataset.modalities = ["wifi", "imu", "camera", "odom"]
    cfg.temporal.n_instants = 4
    cfg.data.batch_size = 128

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
    state = torch.load(run_path / "model.pt", weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model loaded ({n_params/1e6:.2f} M params)", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities),
        extra_inputs=extra,
        lr=float(cfg.train.lr),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        batch_size=128,
        run_dir=str(OUT_DIR / f"{args.arch}_postproc_skip"),
    )

    pred_v, gt_v = trainer.predict("val")
    pred_t, gt_t = trainer.predict("test")
    val_mae = float(torch.linalg.norm(pred_v - gt_v, dim=1).mean())
    test_mae = float(torch.linalg.norm(pred_t - gt_t, dim=1).mean())
    print(f"sanity: val {val_mae:.3f}  test {test_mae:.3f}", flush=True)

    out = {
        "arch": args.arch,
        "checkpoint": str(run_path.relative_to(ROOT)),
        "n_params": int(n_params),
        "sanity": {"val_mae": val_mae, "test_mae": test_mae},
    }

    # === subset eval (re-run to be consistent) ===
    print("\n=== subset eval ===", flush=True)
    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    out["subsets_val"] = {k: {"mae": float(v["mae"])} for k, v in subsets_val.items()}
    out["subsets_test"] = {k: {"mae": float(v["mae"])} for k, v in subsets_test.items()}
    for k, v in subsets_test.items():
        print(f"  test {k:30s} -> mae={v['mae']:.3f}", flush=True)

    # === staleness sweep ===
    if args.lags == "full":
        lags = [0, 1, 3, 5, 10, 15, 20, 30]
    else:
        lags = [0, 5, 15, 30]
    print(f"\n=== staleness sweep ({len(lags)} lags) ===", flush=True)
    staleness = staleness_probe(trainer, lags)
    out["staleness"] = staleness
    slope = linear_slope([l * 0.9 for l in lags], [staleness[l] for l in lags])
    out["staleness_slope"] = slope
    print(f"  linear fit: slope={slope['slope_m_per_s']:.4f} m/s, R^2={slope['r2']:.3f}",
          flush=True)
    plot_staleness(lags, [staleness[l] for l in lags],
                    OUT_DIR / f"{args.arch}_staleness.png",
                    f"WiFi staleness: {args.arch} K=4 4-mod B=128")

    # === per-path + smoothness + plots ===
    pred = pred_t.numpy(); gt = gt_t.numpy()
    pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
    out["test_dist"] = per_path_distribution(pred, gt, pids)
    out["test_smoothness"] = per_traj_smoothness(pred, gt, pids)
    print(f"\n=== smoothness median r = {out['test_smoothness']['median_r']:.3f} ===",
          flush=True)
    for p_str in args.paths.split(","):
        p = int(p_str)
        m = pids == p
        if m.sum() > 5:
            plot_path(pred[m], gt[m], p,
                      OUT_DIR / "test_paths" / f"{args.arch}_path_{p:02d}.png",
                      f"({args.arch} K=4 4-mod)")

    # === latency probes ===
    print("\n=== latency b=1 + b=32 ===", flush=True)
    out["latency_b1"] = latency_probe(trainer, batch=1, n_runs=100)
    out["latency_b32"] = latency_probe(trainer, batch=32, n_runs=50)
    print(f"  b=1:  {out['latency_b1']['ms_per_sample']:.4f} ms/sample  "
          f"({out['latency_b1']['ms_per_batch']:.2f} ms/batch)", flush=True)
    print(f"  b=32: {out['latency_b32']['ms_per_sample']:.4f} ms/sample  "
          f"({out['latency_b32']['ms_per_batch']:.2f} ms/batch)", flush=True)

    out_path = OUT_DIR / f"{args.arch}_ablations.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
