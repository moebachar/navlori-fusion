"""OdomCNN audit on Webots Tiago sim (PLAN_04 Steps 1-5).

Canonical CLAUDE.md split:
  train = paths [1, 3-12]   (11 paths)
  val   = paths [2, 13, 14]
  test  = paths [15, 16, 17]

Odometry schema (per `data/async_collection/path_01/odometry.csv` header):
  sim_time, odom_x, odom_y, odom_theta_deg, odom_linear_vel,
  odom_angular_vel, wheel_left_vel, wheel_right_vel
→ 7 feature columns.

Variant I-A (the "trivial integration baseline" floor): use `odom_x`,
`odom_y` directly (Webots controller already integrates), with a
per-path origin shift aligning the first odom row to the first GT row.

Three runs:
  * P-A — raw 7 features (per-feature train mean/std normalisation).
  * P-B — Δ-features (replace odom_x, odom_y, odom_theta_deg with their
          first differences; keep velocities + wheel speeds as-is). The
          encoder is forced to learn from local motion only.
  * P-A-window32 — same as P-A but window=32 instead of 16 (capacity /
          window probe per the plan's Step 4 option (b)).

Reports per-path distribution, per-trajectory smoothness, 6-metric
harness, and per-trajectory plots for paths 15/16/17.

Run: ``.venv/Scripts/python.exe scripts/_eval_webots_odom.py``
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

from src.pipeline.encoders import OdomCNN  # noqa: E402
from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    alignment_uniformity,
    effective_dimensionality,
    knn_probe,
    linear_probe,
    temporal_smoothness,
    trustworthiness,
)

DATA = ROOT / "data" / "async_collection"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_04"

TRAIN_PATHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]

ODOM_COLS = ["odom_x", "odom_y", "odom_theta_deg",
             "odom_linear_vel", "odom_angular_vel",
             "wheel_left_vel", "wheel_right_vel"]


def load_path(idx: int):
    pdir = DATA / f"path_{idx:02d}"
    odo = pd.read_csv(pdir / "odometry.csv")
    gt = pd.read_csv(pdir / "ground_truth.csv")
    if len(odo) < 20 or len(gt) < 5:
        return None
    return {"odo": odo, "gt": gt, "path_id": idx}


# ---------------------------------------------------------------------------
# Trivial integration baseline (Step 1)
# ---------------------------------------------------------------------------


def trivial_baseline_path(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each odom sample, compute (pred_xy = odom_x,y shifted to first GT)
    and (gt_xy = GT position interp'd at the odom sim_time).

    Returns (pred (N,2), gt (N,2), times (N,)).
    """
    odo = c["odo"]
    gt = c["gt"]
    # Origin shift: first odom row is at the first GT position.
    t0 = float(odo["sim_time"].iloc[0])
    gt_x0 = float(np.interp(t0, gt["sim_time"].values, gt["gt_x"].values))
    gt_y0 = float(np.interp(t0, gt["sim_time"].values, gt["gt_y"].values))
    odom_x0 = float(odo["odom_x"].iloc[0])
    odom_y0 = float(odo["odom_y"].iloc[0])
    pred_x = odo["odom_x"].values - odom_x0 + gt_x0
    pred_y = odo["odom_y"].values - odom_y0 + gt_y0
    pred = np.stack([pred_x, pred_y], axis=1).astype(np.float32)
    t = odo["sim_time"].values.astype(np.float32)
    gt_at_t = np.stack([
        np.interp(t, gt["sim_time"].values, gt["gt_x"].values),
        np.interp(t, gt["sim_time"].values, gt["gt_y"].values),
    ], axis=1).astype(np.float32)
    return pred, gt_at_t, t


def trivial_baseline_split(paths: list[int]):
    all_pred, all_gt, all_pid = [], [], []
    for pid in paths:
        c = load_path(pid)
        if c is None:
            continue
        pred, gt, _t = trivial_baseline_path(c)
        all_pred.append(pred)
        all_gt.append(gt)
        all_pid.append(np.full(len(pred), pid, dtype=np.int64))
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    pid = np.concatenate(all_pid)
    return pred, gt, pid


# ---------------------------------------------------------------------------
# Windowed dataset (Step 2 / 4)
# ---------------------------------------------------------------------------


def build_windows(paths: list[int], window: int, mode: str = "P-A",
                   stride: int = 1):
    """Per-path windowed inputs + position targets.

    mode == "P-A": features = ODOM_COLS as-is.
    mode == "P-B": replace odom_x, odom_y, odom_theta_deg with their
                  first-difference; keep velocities + wheel speeds.
    """
    Xs, Ys, pids, times = [], [], [], []
    for pid in paths:
        c = load_path(pid)
        if c is None:
            continue
        odo = c["odo"]
        gt = c["gt"]
        feat = odo[ODOM_COLS].values.astype(np.float32)
        t = odo["sim_time"].values.astype(np.float32)
        if mode == "P-B":
            # First-difference of the first three columns; row 0 = zeros.
            df = np.diff(feat[:, :3], axis=0)
            df = np.vstack([np.zeros((1, 3), dtype=np.float32), df])
            feat = np.concatenate([df, feat[:, 3:]], axis=1)
        N = len(feat)
        if N < window + stride:
            continue
        ends = np.arange(window, N, stride)
        for e in ends:
            x = feat[e - window:e]
            t_end = t[e - 1]
            gx = float(np.interp(t_end, gt["sim_time"].values, gt["gt_x"].values))
            gy = float(np.interp(t_end, gt["sim_time"].values, gt["gt_y"].values))
            Xs.append(x)
            Ys.append([gx, gy])
            pids.append(pid)
            times.append(t_end)
    return {
        "X": np.stack(Xs).astype(np.float32),
        "Y": np.array(Ys, dtype=np.float32),
        "pid": np.array(pids, dtype=np.int64),
        "t": np.array(times, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Trainer + per-path distribution
# ---------------------------------------------------------------------------


class OdomCNNWithHead(nn.Module):
    def __init__(self, in_features=7, embed_dim=128, channels=(16, 32, 64)):
        super().__init__()
        self.encoder = OdomCNN(in_features=in_features, embed_dim=embed_dim,
                                channels=channels)
        self.pos = nn.Linear(embed_dim, 2)

    def forward(self, x):  # (B, window, in_features) → (B, 2)
        return self.pos(self.encoder(x))

    def embed(self, x):
        return self.encoder(x)


def memory_budget_check(model_factory, batch=64, window=16, in_features=7):
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = model_factory().cuda()
    x = torch.randn(batch, window, in_features, device="cuda")
    y = torch.randn(batch, 2, device="cuda")
    pred = m(x)
    loss = nn.functional.huber_loss(pred, y)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    del m, x, y, pred, loss
    torch.cuda.empty_cache()
    return peak


def latency_ms(model, window, in_features, dev, runs=200):
    model.eval()
    x = torch.zeros(1, window, in_features, device=dev)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000.0


def train_model(model, Xtr, Ytr, Xva, Yva, epochs, batch, lr, dev, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    Xtr_t = torch.tensor(Xtr, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    Yva_t = torch.tensor(Yva, device=dev)
    best = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            pred = model(Xtr_t[idx])
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t)
            mae = float(torch.linalg.norm(pred_va - Yva_t, dim=1).mean())
        if mae < best:
            best = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep <= 1 or ep % 5 == 0 or ep == epochs - 1:
            print(f"  [{name}] ep {ep:3d}  val MAE={mae:.3f} m  (best {best:.3f})", flush=True)
    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, elapsed


def per_path_distribution(preds: np.ndarray, gts: np.ndarray, pid: np.ndarray):
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for p in np.unique(pid):
        mask = pid == p
        e = errs[mask]
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


def per_traj_smoothness(preds, gts, pid):
    per_path = {}
    for p in np.unique(pid):
        mask = pid == p
        pp = preds[mask]
        gg = gts[mask]
        if len(pp) < 5:
            continue
        dp = np.linalg.norm(np.diff(pp, axis=0), axis=1)
        dg = np.linalg.norm(np.diff(gg, axis=0), axis=1)
        if dp.std() < 1e-9 or dg.std() < 1e-9:
            per_path[int(p)] = 0.0
        else:
            per_path[int(p)] = float(np.corrcoef(dp, dg)[0, 1])
    rs = list(per_path.values())
    return {"per_path": per_path, "median_r": float(np.median(rs)) if rs else 0.0}


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)

    # Echo schema (Step 0b)
    sample = pd.read_csv(DATA / "path_01" / "odometry.csv", nrows=1)
    print("odometry columns:", list(sample.columns), flush=True)

    # =============== STEP 1: Trivial integration baseline ===============
    print("\n=========== STEP 1: trivial integration baseline ===========",
          flush=True)
    tr_pred, tr_gt, tr_pid = trivial_baseline_split(TRAIN_PATHS)
    va_pred, va_gt, va_pid = trivial_baseline_split(VAL_PATHS)
    te_pred, te_gt, te_pid = trivial_baseline_split(TEST_PATHS)
    tr_dist = per_path_distribution(tr_pred, tr_gt, tr_pid)
    va_dist = per_path_distribution(va_pred, va_gt, va_pid)
    te_dist = per_path_distribution(te_pred, te_gt, te_pid)
    te_smooth = per_traj_smoothness(te_pred, te_gt, te_pid)
    print(f"  train: agg mean MAE = {tr_dist['aggregate']['mean']:.3f} m",
          flush=True)
    print(f"  val:   agg mean MAE = {va_dist['aggregate']['mean']:.3f} m  "
          f"p50={va_dist['aggregate']['median']:.3f}  "
          f"p90={va_dist['aggregate']['p90']:.3f}", flush=True)
    print(f"  test:  agg mean MAE = {te_dist['aggregate']['mean']:.3f} m  "
          f"p50={te_dist['aggregate']['median']:.3f}  "
          f"p90={te_dist['aggregate']['p90']:.3f}", flush=True)
    gap_pct = (te_dist['aggregate']['mean'] - va_dist['aggregate']['mean']
               ) / max(va_dist['aggregate']['mean'], 1e-6) * 100.0
    print(f"  test-val gap = {gap_pct:+.1f}%", flush=True)
    # Per-traj plots for trivial baseline.
    for p in TEST_PATHS:
        mask = te_pid == p
        if mask.sum() > 5:
            plot_path(te_pred[mask], te_gt[mask], p,
                      OUT_DIR / "test_paths" / f"trivial_path_{p:02d}.png",
                      "(trivial integration)")

    results = {
        "trivial_baseline": {
            "val_mae": va_dist["aggregate"]["mean"],
            "test_mae": te_dist["aggregate"]["mean"],
            "test_val_gap_pct": gap_pct,
            "val_dist": va_dist, "test_dist": te_dist,
            "test_smoothness": te_smooth,
        },
        "runs": {},
    }

    # =============== STEPS 2/4: OdomCNN runs (P-A, P-B, P-A-win32) ===============
    print("\n=========== STEP 2: pre-test gate (P-A, 10 % train, 5 ep) ===========",
          flush=True)
    # Pre-test: use only the first train path as the 10 % subset.
    pretest_paths = TRAIN_PATHS[:max(1, len(TRAIN_PATHS) // 10)]
    pre_data = build_windows(pretest_paths, window=16, mode="P-A")
    val_pa = build_windows(VAL_PATHS, window=16, mode="P-A")
    if len(pre_data["X"]) > 0 and len(val_pa["X"]) > 0:
        # Per-feature mu/sd from pre-test data.
        mu_pre = pre_data["X"].reshape(-1, pre_data["X"].shape[2]).mean(0)
        sd_pre = pre_data["X"].reshape(-1, pre_data["X"].shape[2]).std(0) + 1e-6
        Xp = (pre_data["X"] - mu_pre) / sd_pre
        Xv = (val_pa["X"] - mu_pre) / sd_pre
        mu_y = pre_data["Y"].mean(0)
        Yp = pre_data["Y"] - mu_y
        Yv = val_pa["Y"] - mu_y
        pre_model = OdomCNNWithHead(7, 128).to(dev)
        pre_mae, _ = train_model(pre_model, Xp, Yp, Xv, Yv, epochs=5,
                                  batch=args.batch, lr=args.lr, dev=dev,
                                  name="pretest-P-A")
        del pre_model
        torch.cuda.empty_cache()
        results["pretest"] = {"val_mae_5ep": float(pre_mae)}
        print(f"  pre-test val MAE = {pre_mae:.3f}", flush=True)

    print("\n========== memory budget check (B=64, window=16, 7 ch) ==========",
          flush=True)
    mem_pa = memory_budget_check(lambda: OdomCNNWithHead(7, 128), 64, 16, 7)
    mem_pb = memory_budget_check(lambda: OdomCNNWithHead(7, 128), 64, 16, 7)
    mem_w32 = memory_budget_check(lambda: OdomCNNWithHead(7, 128), 64, 32, 7)
    print(f"  OdomCNN P-A: peak {mem_pa:.1f} MB", flush=True)
    print(f"  OdomCNN P-B: peak {mem_pb:.1f} MB", flush=True)
    print(f"  OdomCNN P-A win32: peak {mem_w32:.1f} MB", flush=True)
    if max(mem_pa, mem_pb, mem_w32) > 6000:
        raise RuntimeError("memory budget exceeded")
    results["memory_budget_mb"] = {"P-A": mem_pa, "P-B": mem_pb,
                                   "P-A-window32": mem_w32}

    # Full runs.
    run_specs = [
        ("P-A", 16, "P-A"),
        ("P-B", 16, "P-B"),
        ("P-A-window32", 32, "P-A"),
    ]
    for label, window, mode in run_specs:
        print(f"\n=========== STEP {2 if 'window' not in label else 4}: {label} "
              f"(window={window}, mode={mode}) ===========", flush=True)
        tr = build_windows(TRAIN_PATHS, window=window, mode=mode)
        va = build_windows(VAL_PATHS, window=window, mode=mode)
        te = build_windows(TEST_PATHS, window=window, mode=mode)
        # Per-feature normalisation from train.
        mu_x = tr["X"].reshape(-1, tr["X"].shape[2]).mean(0)
        sd_x = tr["X"].reshape(-1, tr["X"].shape[2]).std(0) + 1e-6
        Xtr = (tr["X"] - mu_x) / sd_x
        Xva = (va["X"] - mu_x) / sd_x
        Xte = (te["X"] - mu_x) / sd_x
        mu_y = tr["Y"].mean(0)
        Ytr = tr["Y"] - mu_y
        Yva = va["Y"] - mu_y
        Yte = te["Y"] - mu_y
        print(f"  train windows: {len(Xtr)}  val: {len(Xva)}  test: {len(Xte)}",
              flush=True)
        model = OdomCNNWithHead(7, 128).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        best_va, train_s = train_model(model, Xtr, Ytr, Xva, Yva,
                                        epochs=args.epochs, batch=args.batch,
                                        lr=args.lr, dev=dev, name=label)
        # Final eval.
        model.eval()
        with torch.no_grad():
            pred_va = model(torch.tensor(Xva, device=dev)).cpu().numpy() + mu_y
            pred_te = model(torch.tensor(Xte, device=dev)).cpu().numpy() + mu_y
            emb_tr = model.embed(torch.tensor(Xtr, device=dev)).cpu().numpy()
            emb_va = model.embed(torch.tensor(Xva, device=dev)).cpu().numpy()
        val_dist = per_path_distribution(pred_va, va["Y"], va["pid"])
        test_dist = per_path_distribution(pred_te, te["Y"], te["pid"])
        test_smooth = per_traj_smoothness(pred_te, te["Y"], te["pid"])
        gap = (test_dist["aggregate"]["mean"] - val_dist["aggregate"]["mean"]
               ) / max(val_dist["aggregate"]["mean"], 1e-6) * 100.0
        lat = latency_ms(model, window, 7, dev)
        print(f"  >> val MAE = {val_dist['aggregate']['mean']:.3f}  "
              f"test MAE = {test_dist['aggregate']['mean']:.3f}  "
              f"gap = {gap:+.1f}%", flush=True)
        # 6-metric harness.
        try:
            lp = linear_probe(emb_tr, Ytr, emb_va, Yva, epochs=200, lr=1e-2, device="cpu")
        except Exception as e:
            lp = {"error": str(e)}
        try:
            kp = knn_probe(emb_tr, Ytr, emb_va, Yva, k=5)
        except Exception as e:
            kp = {"error": str(e)}
        try:
            au = alignment_uniformity(emb_va, Yva, distance_threshold=1.0,
                                       max_samples=1000)
        except Exception as e:
            au = {"error": str(e)}
        try:
            ed = effective_dimensionality(emb_va)
        except Exception as e:
            ed = {"error": str(e)}
        try:
            order = np.lexsort((va["t"], va["pid"]))
            ts = temporal_smoothness(emb_va[order], Yva[order])
        except Exception as e:
            ts = {"error": str(e)}
        try:
            tw = trustworthiness(Xva.reshape(len(Xva), -1), emb_va, k=10)
        except Exception as e:
            tw = {"error": str(e)}
        # Per-trajectory plots.
        plot_paths = {}
        for p in np.unique(te["pid"]):
            mask = te["pid"] == p
            out_png = OUT_DIR / "test_paths" / f"{label}_path_{p:02d}.png"
            plot_path(pred_te[mask], te["Y"][mask], int(p), out_png,
                      f"({label})")
            plot_paths[int(p)] = str(out_png.relative_to(ROOT))

        results["runs"][label] = {
            "window": window, "mode": mode, "params": int(n_params),
            "best_val_mae": float(best_va), "train_time_s": float(train_s),
            "val_mae": float(val_dist["aggregate"]["mean"]),
            "test_mae": float(test_dist["aggregate"]["mean"]),
            "test_val_gap_pct": float(gap),
            "val_dist": val_dist, "test_dist": test_dist,
            "test_smoothness": test_smooth,
            "latency_ms_per_window_b1": float(lat),
            "linear_probe": lp, "knn_probe": kp,
            "alignment_uniformity": au, "effective_dimensionality": ed,
            "temporal_smoothness": ts, "trustworthiness": tw,
            "plot_paths": plot_paths,
        }
        del model
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "webots_odom.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Summary
    print(f"\n{'run':<18} {'val MAE':>10} {'test MAE':>10} {'gap %':>8} "
          f"{'test p90':>10} {'params':>10} {'lin-probe':>10}")
    print(f"  {'trivial integration':<18} "
          f"{results['trivial_baseline']['val_mae']:>9.3f}  "
          f"{results['trivial_baseline']['test_mae']:>9.3f}  "
          f"{results['trivial_baseline']['test_val_gap_pct']:>+7.1f}  "
          f"{results['trivial_baseline']['test_dist']['aggregate']['p90']:>9.3f}  "
          f"{'-':>10}  {'-':>10}")
    for label, r in results["runs"].items():
        print(f"  {label:<18} "
              f"{r['val_mae']:>9.3f}  {r['test_mae']:>9.3f}  "
              f"{r['test_val_gap_pct']:>+7.1f}  "
              f"{r['test_dist']['aggregate']['p90']:>9.3f}  "
              f"{r['params']/1e3:>9.1f}k  "
              f"{r['linear_probe'].get('mean_euclidean', float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
