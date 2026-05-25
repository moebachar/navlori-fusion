"""PLAN_03: train FusionTransformer baseline on msiln_site1_b1.

One-shot wrapper that does:
- smoke phase 2 inline (overfit 16 samples to confirm capacity);
- 90-epoch training via the standard builder pipeline (no src/ changes);
- best-val eval: per-sample + per-path + per-waypoint + subset table;
- inference latency probe (per-sample + batch 32);
- per-trajectory plots for the 5 test paths.

All outputs land under `runs/fusion_msiln_b1_<ts>/` (gitignored).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule,
    build_encoders,
    build_model,
    build_trainer,
    load_config,
)


def _print(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


# ─── smoke: overfit 16 samples ──────────────────────────────────────────────

def smoke_overfit(cfg, dm, device: str) -> dict:
    _print("\n=== Smoke phase 2: overfit a 16-sample batch ===")
    encoders, _ = build_encoders(cfg, dm)
    model = build_model(cfg, encoders).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    crit = torch.nn.HuberLoss(delta=0.5)

    B = 16
    modalities = list(cfg.dataset.modalities)
    X = {m: dm.train_ds.get_tensors(m)[0][:B].unsqueeze(1).to(device)
         for m in modalities}
    y = dm.train_ds._targets[:B].to(device)
    avail = {m: torch.ones(B, 1, dtype=torch.bool, device=device) for m in modalities}
    dt = {m: torch.zeros(B, 1, device=device) for m in modalities}

    model.train()
    losses, maes = [], []
    for step in range(500):
        pred = model(X, avail, dt)
        loss = crit(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 499:
            mae = torch.linalg.norm(pred.detach() - y, dim=1).mean().item()
            _print(f"  step {step:3d}  loss={loss.item():.5f}  mae={mae:.3f}m")
            losses.append(loss.item()); maes.append(mae)
    drop_pct = 100.0 * (1.0 - losses[-1] / max(losses[0], 1e-9))
    final_mae = maes[-1]
    # Capacity gate: model must be able to overfit small batch (>=80% loss drop).
    # We don't gate on absolute MAE because msiln coords are not zero-centered
    # (range x~50-280, y~80-230); absolute scale is dataset-dependent.
    _print(f"  loss drop {drop_pct:.1f}% over 500 steps (gate: >= 80%)")
    return {"loss_drop_pct": drop_pct, "final_mae": final_mae,
            "pass": drop_pct >= 80.0}


# ─── eval helpers ───────────────────────────────────────────────────────────

def per_sample_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    err = np.linalg.norm(pred - y, axis=1)
    return {
        "n": int(len(err)),
        "mae": float(err.mean()),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max": float(err.max()),
    }


def per_path_distribution(pred: np.ndarray, y: np.ndarray, pids: np.ndarray) -> dict:
    err = np.linalg.norm(pred - y, axis=1)
    per_path = []
    for pid in np.unique(pids):
        m = pids == pid
        per_path.append({"path_id": int(pid),
                         "mae": float(err[m].mean()),
                         "n": int(m.sum())})
    arr = np.array([p["mae"] for p in per_path])
    return {
        "n_paths": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "per_path": per_path,
    }


def per_waypoint_mae(pred: np.ndarray, y: np.ndarray, ds) -> tuple[float, int]:
    """Evaluate at original surveyor-clicked waypoints only.

    Walks each path's source .txt via vendored io_f and matches anchor
    timestamps to the dataset's sim_time within 50 ms.
    """
    spec = importlib.util.spec_from_file_location(
        "msiln20_io_f", Path(r"C:\Users\FabLab\AppData\Local\Temp\msiln20") / "io_f.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    read_data_file = mod.read_data_file

    pids = np.array([r["path_id"] for r in ds._gt_rows])
    times = ds._timestamps.cpu().numpy()
    is_wp = np.zeros(len(pids), dtype=bool)

    for pid in np.unique(pids):
        idx_pid = np.where(pids == pid)[0]
        path_dir = Path(ds._gt_rows[int(idx_pid[0])]["path_dir"])
        meta = json.loads((path_dir / "metadata.json").read_text())
        src = (Path(r"C:\Users\FabLab\AppData\Local\Temp\msiln20") / "data"
               / meta["site"] / meta["floor"] / "path_data_files" / meta["source_file"])
        if not src.exists():
            continue
        d = read_data_file(str(src))
        wp = d.waypoint
        if len(wp) == 0:
            continue
        t_lo_ms = float(wp[0, 0])
        wp_sim_s = (wp[:, 0].astype(np.float64) - t_lo_ms) / 1000.0
        path_times = times[idx_pid]
        for wts in wp_sim_s:
            j = int(np.argmin(np.abs(path_times - wts)))
            if abs(path_times[j] - wts) < 0.05:
                is_wp[idx_pid[j]] = True
    err = np.linalg.norm(pred - y, axis=1)
    n_wp = int(is_wp.sum())
    if n_wp == 0:
        return float("nan"), 0
    return float(err[is_wp].mean()), n_wp


# ─── latency probe ──────────────────────────────────────────────────────────

def measure_latency(trainer, device: str) -> dict:
    """Median forward latency at batch=1 and batch=32 (CUDA-synchronised)."""
    modalities = trainer.modalities
    dm = trainer.dm

    def _make_batch(B):
        X = {m: dm.val_ds.get_tensors(m)[0][:B].to(device) for m in modalities}
        # Trainer prepends instant axis (1, K) — match the (B, 1, win, feat) shape.
        X = {m: x.unsqueeze(1) for m, x in X.items()}
        avail = {m: torch.ones(B, 1, dtype=torch.bool, device=device) for m in modalities}
        dt = {m: torch.zeros(B, 1, device=device) for m in modalities}
        return X, avail, dt

    out = {}
    trainer.model.eval()
    with torch.no_grad():
        for B in (1, 32):
            X, avail, dt = _make_batch(B)
            # warmup
            for _ in range(10):
                _ = trainer.model(X, avail, dt)
            if device == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = trainer.model(X, avail, dt)
                if device == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
            ts = np.array(ts)
            per_sample = float(np.median(ts)) / B
            out[f"batch{B}"] = {
                "total_ms_median": float(np.median(ts)),
                "total_ms_p90":    float(np.percentile(ts, 90)),
                "per_sample_ms":   per_sample,
            }
    return out


# ─── per-trajectory plots ───────────────────────────────────────────────────

def trajectory_metrics_and_plots(trainer, pred_np: np.ndarray, y_np: np.ndarray,
                                 ds, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = np.array([r["path_id"] for r in ds._gt_rows])
    times = ds._timestamps.cpu().numpy()
    rows = []
    for pid in np.unique(pids):
        m = pids == pid
        idx = np.where(m)[0]
        order = idx[np.argsort(times[idx])]
        t = times[order]
        p = pred_np[order]
        g = y_np[order]
        err = np.linalg.norm(p - g, axis=1)
        final_drift = float(np.linalg.norm(p[-1] - g[-1]))
        mean_pred_step = float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean()) if len(p) > 1 else 0.0
        mean_gt_step = float(np.linalg.norm(np.diff(g, axis=0), axis=1).mean()) if len(g) > 1 else 1e-9
        smoothness = mean_pred_step / max(mean_gt_step, 1e-9)
        rows.append({
            "path_id": int(pid), "n": int(len(t)),
            "duration_s": round(float(t[-1] - t[0]), 1),
            "mae": float(err.mean()), "final_drift_m": final_drift,
            "smoothness_ratio": smoothness,
        })
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(g[:, 0], g[:, 1], "k-", lw=2, label="GT", alpha=0.6)
        axes[0].plot(p[:, 0], p[:, 1], "r-", lw=1, label="pred", alpha=0.8)
        axes[0].plot(g[0, 0], g[0, 1], "go", ms=10, label="start")
        axes[0].plot(g[-1, 0], g[-1, 1], "rs", ms=10, label="end (GT)")
        axes[0].plot(p[-1, 0], p[-1, 1], "rD", ms=8, label="end (pred)")
        axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
        axes[0].set_aspect("equal"); axes[0].legend(fontsize=8)
        axes[0].set_title(f"path_{pid:02d}  trajectory  (MAE={err.mean():.2f}m)")

        axes[1].plot(t, err, "k-")
        axes[1].axhline(float(err.mean()), color="r", ls="--", label=f"mean {err.mean():.2f}m")
        axes[1].set_xlabel("sim_time (s)"); axes[1].set_ylabel("Euclidean error (m)")
        axes[1].legend(fontsize=8); axes[1].set_title("per-sample error over time")
        fig.tight_layout()
        fig.savefig(out_dir / f"path_{pid:02d}.png", dpi=100)
        plt.close(fig)
    return rows


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msiln_site1_b1")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="Override cfg.model.embed_dim (default: from config)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override cfg.data.batch_size (drop on OOM)")
    ap.add_argument("--wifi-pca", type=int, default=None,
                    help="Override cfg.dataset.preprocessing.wifi_pca")
    ap.add_argument("--modalities", default=None,
                    help="Comma-separated subset to enable (e.g. 'wifi' for wifi-only)")
    ap.add_argument("--wifi-encoder", default=None,
                    choices=["anchor2vec", "set_transformer"],
                    help="Override cfg.dataset.wifi_encoder_type")
    ap.add_argument("--patience", type=int, default=None,
                    help="Override cfg.train.patience")
    ap.add_argument("--n-instants", type=int, default=None,
                    help="Override cfg.temporal.n_instants (K)")
    ap.add_argument("--run-label", default="",
                    help="Suffix tag appended to summary JSON name; identifies the probe.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _print(f"=== train+eval on {args.dataset}  (device={device}, label={args.run_label or 'default'}) ===")
    cfg = load_config(args.dataset)

    # ── apply CLI overrides BEFORE building anything ─────────────────────────
    if args.embed_dim is not None:
        _print(f"  override model.embed_dim {cfg.model.embed_dim} -> {args.embed_dim}")
        cfg.model.embed_dim = int(args.embed_dim)
    if args.batch_size is not None:
        _print(f"  override data.batch_size {cfg.data.batch_size} -> {args.batch_size}")
        cfg.data.batch_size = int(args.batch_size)
    if args.wifi_pca is not None:
        pre = cfg.dataset.get("preprocessing", {}) or {}
        _print(f"  override dataset.preprocessing.wifi_pca "
               f"{pre.get('wifi_pca', None)} -> {args.wifi_pca}")
        if "preprocessing" not in cfg.dataset:
            cfg.dataset.preprocessing = {}
        cfg.dataset.preprocessing.wifi_pca = int(args.wifi_pca)
    if args.modalities is not None:
        new_mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
        _print(f"  override dataset.modalities {list(cfg.dataset.modalities)} -> {new_mods}")
        cfg.dataset.modalities = new_mods
    if args.wifi_encoder is not None:
        _print(f"  override dataset.wifi_encoder_type "
               f"{cfg.dataset.get('wifi_encoder_type', 'anchor2vec')} -> {args.wifi_encoder}")
        cfg.dataset.wifi_encoder_type = args.wifi_encoder
    if args.patience is not None:
        _print(f"  override train.patience {cfg.train.patience} -> {args.patience}")
        cfg.train.patience = int(args.patience)
    if args.n_instants is not None:
        _print(f"  override temporal.n_instants {cfg.temporal.n_instants} -> {args.n_instants}")
        cfg.temporal.n_instants = int(args.n_instants)

    _print(f"  config: depth={cfg.model.depth}  heads={cfg.model.n_heads}  "
           f"embed_dim={cfg.model.embed_dim}  readout={cfg.model.readout}  "
           f"K={cfg.temporal.n_instants}  stride={cfg.temporal.instant_stride}  "
           f"batch={cfg.data.batch_size}")

    dm = build_datamodule(cfg)
    _print(dm.summary())

    summary: dict = {"dataset": args.dataset, "started_at": datetime.now().isoformat()}

    # --- smoke phase 2
    if not args.skip_smoke:
        smoke = smoke_overfit(cfg, dm, device)
        summary["smoke_phase2"] = smoke
        if not smoke["pass"]:
            _print("[abort] smoke phase 2 FAILED — overfit gate not met")
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = ROOT / "runs" / f"fusion_msiln_b1_{ts_str}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            return 2

    # --- fresh model for full training (smoke trained a separate instance)
    _print("\n=== Phase 3: full training ===")
    encoders, _ = build_encoders(cfg, dm)
    model = build_model(cfg, encoders)
    trainer = build_trainer(cfg, model, dm)
    _print(f"  trainer.run_path = {trainer.run_path}")
    t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    train_wall_s = time.time() - t0
    summary["training"] = {
        "epochs": args.epochs,
        "best_val_mae": float(hist.best_val_mae),
        "best_epoch": int(hist.best_epoch),
        "wall_s": round(train_wall_s, 1),
        "run_path": str(trainer.run_path),
    }
    _print(f"\nTraining done: best_val_mae={hist.best_val_mae:.3f}m @ epoch {hist.best_epoch}; "
           f"wall={train_wall_s/60:.1f} min")

    # --- subset eval
    _print("\n=== Subset eval ===")
    subsets = {}
    for split in ("val", "test"):
        try:
            subsets[split] = trainer.evaluate_all_subsets(split)
        except Exception as e:  # noqa: BLE001
            _print(f"  [{split}] evaluate_all_subsets failed: {e}")
            subsets[split] = {}
        for label, m in subsets[split].items():
            _print(f"  {split:5s}  {label:20s}  MAE={m.get('mae', float('nan')):.3f}m")
    summary["subsets"] = subsets

    # --- per-sample / per-path / per-waypoint
    _print("\n=== Per-sample / per-path / per-waypoint ===")
    eval_block = {}
    for split in ("val", "test"):
        ds = getattr(dm, f"{split}_ds")
        pred, y = trainer.predict(split)
        pred_np, y_np = pred.numpy(), y.numpy()
        pids = np.array([r["path_id"] for r in ds._gt_rows])
        ps = per_sample_stats(pred_np, y_np)
        pp = per_path_distribution(pred_np, y_np, pids)
        wp_mae, n_wp = per_waypoint_mae(pred_np, y_np, ds)
        eval_block[split] = {
            "per_sample": ps, "per_path": pp,
            "per_waypoint_mae": wp_mae, "n_waypoints": n_wp,
        }
        _print(f"  {split:5s}  per_sample MAE={ps['mae']:.3f}m  "
               f"per_path med={pp['median']:.3f}m  p90={pp['p90']:.3f}m  "
               f"per_waypoint MAE={wp_mae:.3f}m (n={n_wp})")
    summary["eval"] = eval_block

    # --- latency
    _print("\n=== Latency probe ===")
    lat = measure_latency(trainer, device)
    for k, v in lat.items():
        _print(f"  {k}: median={v['total_ms_median']:.2f}ms  per-sample={v['per_sample_ms']:.3f}ms")
    summary["latency"] = lat

    # --- per-trajectory plots (test split only — 5 paths)
    _print("\n=== Per-trajectory plots (test) ===")
    pred_test, y_test = trainer.predict("test")
    plots_dir = Path(trainer.run_path) / "test_paths"
    rows = trajectory_metrics_and_plots(trainer, pred_test.numpy(), y_test.numpy(),
                                        dm.test_ds, plots_dir)
    smooth_arr = np.array([r["smoothness_ratio"] for r in rows])
    smooth_med = float(np.median(smooth_arr)) if len(smooth_arr) else float("nan")
    summary["test_trajectories"] = rows
    summary["test_smoothness_median"] = smooth_med
    for r in rows:
        _print(f"  path_{r['path_id']:02d}  MAE={r['mae']:.2f}m  "
               f"final_drift={r['final_drift_m']:.2f}m  smooth={r['smoothness_ratio']:.2f}")
    _print(f"  smoothness median across 5 paths = {smooth_med:.2f}")

    # --- save summary
    summary["run_label"] = args.run_label
    summary["embed_dim"] = int(cfg.model.embed_dim)
    summary["batch_size"] = int(cfg.data.batch_size)
    summary["modalities_active"] = list(cfg.dataset.modalities)
    out_name = f"summary_{args.run_label or 'default'}.json"
    out = Path(trainer.run_path) / out_name
    out.write_text(json.dumps(summary, indent=2))
    _print(f"\nWrote {out}")
    _print(f"Run dir: {trainer.run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
