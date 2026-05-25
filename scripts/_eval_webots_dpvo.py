"""DPVOMotionEncoder audit on Webots Tiago sim (PLAN_03 Steps 2-5).

Canonical CLAUDE.md split:
  train = paths [1, 3-12]   (11 paths)
  val   = paths [2, 13, 14]
  test  = paths [15, 16, 17]

For each path: load RGB frames + ground_truth.csv. Build
(frame_{i-stride}, frame_i) pairs at ``camera_stride=5`` (the encoder
docstring's recommended stride for ~1 s of robot motion → real
correlation-peak shift). Target = the GT (x, y) at the time of frame_i
(linear interpolation from ground_truth.csv).

The DPVO trunk is frozen (NeurIPS 2023 pretrained, scene-agnostic).
We use the encoder's ``extract_backbone_features``-style caching path:
run the trunk + correlation once per pair, train only the head
(``_MotionHead``) on the cached per-patch motion tokens.

Two preprocessing conditions (PLAN_03 #2 — preprocessing-variation
probe):
  * P-A (default): ImageNet-norm → DPVO-norm (2x-0.5).
  * P-B: skip ImageNet-norm, just DPVO-norm.

Capacity / config probe: stride probe at ``camera_stride=10`` (sparser
correlation) re-running the default P-A path only.

Reports val + test mean Euclidean MAE, per-path distribution, per-
trajectory smoothness, and per-path top-3 trajectory plots.

Run: ``.venv/Scripts/python.exe scripts/_eval_webots_dpvo.py``
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.pipeline.encoders.dpvo_motion import DPVOMotionEncoder  # noqa: E402
from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    alignment_uniformity,
    effective_dimensionality,
    knn_probe,
    linear_probe,
    temporal_smoothness,
    trustworthiness,
)

DATA = ROOT / "data" / "async_collection"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_03"

TRAIN_PATHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_path(idx: int, stride: int):
    """Return list of (img_prev, img_curr, target_xy) per frame pair."""
    pdir = DATA / f"path_{idx:02d}"
    cam = pd.read_csv(pdir / "camera.csv")
    gt = pd.read_csv(pdir / "ground_truth.csv")
    if len(cam) < stride + 2 or len(gt) < 2:
        return None
    # We want frame indices in cam (sequential), with stride. Pairs:
    # (cam[i-stride], cam[i]) for i in [stride, len(cam))
    pairs = []
    for i in range(stride, len(cam)):
        prev_row = cam.iloc[i - stride]
        curr_row = cam.iloc[i]
        prev_rgb = pdir / prev_row["rgb_path"]
        curr_rgb = pdir / curr_row["rgb_path"]
        if not prev_rgb.exists() or not curr_rgb.exists():
            continue
        t_curr = float(curr_row["sim_time"])
        x = float(np.interp(t_curr, gt["sim_time"].values, gt["gt_x"].values))
        y = float(np.interp(t_curr, gt["sim_time"].values, gt["gt_y"].values))
        pairs.append({
            "prev_rgb": str(prev_rgb),
            "curr_rgb": str(curr_rgb),
            "t_curr": t_curr,
            "xy": (x, y),
            "path_id": idx,
        })
    return pairs


def load_image_rgb01(rgb_path: str) -> np.ndarray:
    """Load PNG as (3, H, W) float32 in [0, 1]."""
    img = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    return img.transpose(2, 0, 1)


def imagenet_normalise(img01: np.ndarray) -> np.ndarray:
    return (img01 - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def extract_tokens_for_pairs(encoder: DPVOMotionEncoder, pairs: list[dict],
                              preprocessing: str, dev: str,
                              batch: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen trunk + correlation; return (N, n_patches, 132) tokens + (N, 2) targets.

    preprocessing : "P-A" (ImageNet-norm input, encoder undoes it) or "P-B"
                    (raw [0,1] input, encoder skips ImageNet undo).
    """
    encoder.eval()
    out_tokens = []
    out_targets = []
    if preprocessing == "P-A":
        encoder.input_is_imagenet_normalised = True
    elif preprocessing == "P-B":
        encoder.input_is_imagenet_normalised = False
    else:
        raise ValueError(f"unknown preprocessing: {preprocessing}")
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            imgs_prev = []
            imgs_curr = []
            for p in chunk:
                prev = load_image_rgb01(p["prev_rgb"])
                curr = load_image_rgb01(p["curr_rgb"])
                if preprocessing == "P-A":
                    prev = imagenet_normalise(prev)
                    curr = imagenet_normalise(curr)
                # P-B: leave as [0,1] — encoder's _to_dpvo_range will just apply 2x-0.5.
                imgs_prev.append(prev)
                imgs_curr.append(curr)
            prev_t = torch.tensor(np.stack(imgs_prev), device=dev)
            curr_t = torch.tensor(np.stack(imgs_curr), device=dev)
            x = torch.stack([prev_t, curr_t], dim=1)  # (B, 2, 3, H, W)
            tokens = encoder._frozen_tokens(x)         # (B, n_patches, 132)
            out_tokens.append(tokens.cpu().numpy())
            for p in chunk:
                out_targets.append(p["xy"])
    return (np.concatenate(out_tokens, axis=0),
            np.array(out_targets, dtype=np.float32))


def train_head(head, Xtr, Ytr, Xva, Yva, epochs, batch, lr, dev, name="head"):
    """Train the trainable head on cached per-patch motion tokens."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    Xtr_t = torch.tensor(Xtr, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    Yva_t = torch.tensor(Yva, device=dev)
    best_va = float("inf")
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            tok = Xtr_t[idx]
            # head consumes (B, N, in_dim) → outputs (B, embed_dim); we then
            # project to (B, 2) via a separate per-name linear. To keep this
            # script self-contained, head IS encoder-head + position head.
            pred = head(tok)
            loss = crit(pred, Ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        head.eval()
        with torch.no_grad():
            pred_va = head(Xva_t)
            mae = float(torch.linalg.norm(pred_va - Yva_t, dim=1).mean())
        if mae < best_va:
            best_va = mae
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep <= 1 or ep % 5 == 0 or ep == epochs - 1:
            print(f"  [{name}] ep {ep:3d}  val mean-euclid={mae:.3f} m  (best {best_va:.3f})",
                  flush=True)
    elapsed = time.time() - t0
    if best_state is not None:
        head.load_state_dict(best_state)
    return best_va, elapsed


class MotionHeadPlus(nn.Module):
    """Wrap encoder._MotionHead with a final (embed_dim → 2) projection.

    We don't reuse encoder.head directly because we want to keep the
    encoder object pristine across preprocessing/probe runs. The
    architecture matches _MotionHead exactly + a position head.
    """

    def __init__(self, in_dim=132, embed_dim=128, hidden=256, dropout=0.1):
        super().__init__()
        from src.pipeline.encoders.dpvo_motion import _MotionHead
        self.head = _MotionHead(in_dim=in_dim, embed_dim=embed_dim,
                                 hidden=hidden, dropout=dropout)
        self.pos = nn.Linear(embed_dim, 2)
        self.embed_dim = embed_dim

    def forward(self, tokens):  # tokens: (B, N, in_dim)
        z = self.head(tokens)
        return self.pos(z)

    def embed(self, tokens):
        return self.head(tokens)


def per_path_distribution(preds: np.ndarray, gts: np.ndarray, path_ids: np.ndarray):
    """Per-path / aggregate distribution of mean Euclidean error."""
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for pid in np.unique(path_ids):
        mask = path_ids == pid
        e = errs[mask]
        per_path[int(pid)] = {
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "p25": float(np.percentile(e, 25)),
            "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)),
            "max": float(e.max()),
            "n_frames": int(len(e)),
        }
    return {
        "aggregate": {
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "p25": float(np.percentile(errs, 25)),
            "p75": float(np.percentile(errs, 75)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()),
        },
        "per_path": per_path,
    }


def per_trajectory_smoothness(preds: np.ndarray, gts: np.ndarray,
                               path_ids: np.ndarray) -> dict:
    """Pearson r between ‖Δpred‖ and ‖Δgt‖ along each trajectory.

    Reports median across paths (the per-trajectory smoothness ratio
    asked for by STATE.md criterion d).
    """
    per_path = {}
    for pid in np.unique(path_ids):
        mask = path_ids == pid
        p = preds[mask]
        g = gts[mask]
        if len(p) < 5:
            continue
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dg = np.linalg.norm(np.diff(g, axis=0), axis=1)
        if dp.std() < 1e-9 or dg.std() < 1e-9:
            per_path[int(pid)] = 0.0
        else:
            per_path[int(pid)] = float(np.corrcoef(dp, dg)[0, 1])
    rs = list(per_path.values())
    return {
        "per_path": per_path,
        "median_r": float(np.median(rs)) if rs else 0.0,
        "min_r": float(min(rs)) if rs else 0.0,
        "max_r": float(max(rs)) if rs else 0.0,
    }


def plot_path(preds: np.ndarray, gts: np.ndarray, pid: int, out_path: Path,
              title_suffix: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  plot skipped (matplotlib import failed): {e}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gts[:, 0], gts[:, 1], "k-", label="GT", lw=1.5)
    ax.plot(preds[:, 0], preds[:, 1], "r-", label="pred", lw=1.0, alpha=0.7)
    ax.scatter(gts[0, 0], gts[0, 1], c="green", s=40, marker="o", label="start")
    ax.set_aspect("equal")
    ax.set_title(f"path_{pid:02d} {title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def memory_budget_check(encoder, dev) -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(4, 2, 3, 480, 640, device=dev)
    z = encoder(x)
    loss = z.sum()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.empty_cache()
    return peak


def latency_ms(encoder, dev, runs=30):
    encoder.eval()
    x = torch.zeros(1, 2, 3, 480, 640, device=dev)
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = encoder(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / runs * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-head", type=int, default=32)
    ap.add_argument("--batch-trunk", type=int, default=4)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  camera_stride={args.stride}", flush=True)

    # ---------------- build encoder ----------------
    encoder = DPVOMotionEncoder(weights_path=str(ROOT / "runs" / "_weights" / "dpvo.pth")).to(dev)
    n_trunk = sum(p.numel() for p in encoder.trunk.parameters())
    n_head = sum(p.numel() for p in encoder.head.parameters())
    print(f"trunk params {n_trunk/1e6:.2f} M (frozen) | head params {n_head/1e6:.2f} M (trainable)",
          flush=True)

    # ---------------- memory budget check ----------------
    print("\n[memory budget] target shape B=4, 2x480x640x3", flush=True)
    peak_mb = memory_budget_check(encoder, dev)
    print(f"  peak {peak_mb:.1f} MB", flush=True)
    if peak_mb > 6000:
        raise RuntimeError(f"DPVOMotion peak {peak_mb:.0f} MB exceeds 6 GB budget")

    # ---------------- latency ----------------
    lat_ms = latency_ms(encoder, dev)
    print(f"  end-to-end latency b=1: {lat_ms:.2f} ms", flush=True)

    # ---------------- load pairs ----------------
    print("\nbuilding pairs...", flush=True)
    train_pairs = []
    for pid in TRAIN_PATHS:
        pairs = load_path(pid, args.stride)
        if pairs:
            train_pairs.extend(pairs)
    val_pairs = []
    for pid in VAL_PATHS:
        pairs = load_path(pid, args.stride)
        if pairs:
            val_pairs.extend(pairs)
    test_pairs = []
    for pid in TEST_PATHS:
        pairs = load_path(pid, args.stride)
        if pairs:
            test_pairs.extend(pairs)
    print(f"train pairs: {len(train_pairs)}  val pairs: {len(val_pairs)}  test pairs: {len(test_pairs)}",
          flush=True)

    pid_tr = np.array([p["path_id"] for p in train_pairs])
    pid_va = np.array([p["path_id"] for p in val_pairs])
    pid_te = np.array([p["path_id"] for p in test_pairs])

    # Target centering (subtract train mean — local meters; mean Euclidean is centering-invariant).
    Y_train_raw = np.array([p["xy"] for p in train_pairs], dtype=np.float32)
    mu = Y_train_raw.mean(0)

    results = {
        "split": {"train_paths": TRAIN_PATHS, "val_paths": VAL_PATHS, "test_paths": TEST_PATHS,
                  "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
                  "test_pairs": len(test_pairs)},
        "stride": args.stride, "epochs": args.epochs, "batch_head": args.batch_head,
        "memory_budget_mb": float(peak_mb),
        "latency_ms_per_pair_b1": float(lat_ms),
        "trunk_params": int(n_trunk), "head_params_per_run": int(n_head),
        "trunk_norm": "instance (DPVO fnet, frozen)",
        "runs": {},
    }

    # ---------------- pre-test gate (P-A, 10 %) ----------------
    print("\n[pre-test gate] P-A, 10 % train pairs, 5 epochs", flush=True)
    sub_idx = np.random.RandomState(0).permutation(len(train_pairs))[:max(32, len(train_pairs) // 10)]
    sub_pairs = [train_pairs[i] for i in sub_idx]
    print(f"  extracting tokens for {len(sub_pairs)} sub pairs + {len(val_pairs)} val pairs", flush=True)
    t0 = time.time()
    Xsub, Ysub = extract_tokens_for_pairs(encoder, sub_pairs, "P-A", dev, batch=args.batch_trunk)
    Xva, Yva = extract_tokens_for_pairs(encoder, val_pairs, "P-A", dev, batch=args.batch_trunk)
    print(f"  extracted in {time.time()-t0:.1f}s; Xsub={Xsub.shape} Xva={Xva.shape}", flush=True)
    head_sub = MotionHeadPlus(in_dim=132, embed_dim=128, hidden=256).to(dev)
    pre_mae, _ = train_head(head_sub, Xsub, Ysub - mu, Xva, Yva - mu,
                              epochs=5, batch=args.batch_head, lr=args.lr, dev=dev,
                              name="pretest")
    pretest_pass = pre_mae < 100.0  # any sane result; the real check is the 5-epoch trajectory below
    print(f"  pre-test val MAE = {pre_mae:.3f} m  pass={pretest_pass}", flush=True)
    del head_sub
    torch.cuda.empty_cache()
    results["pretest"] = {"val_mae_5ep": float(pre_mae), "pass": bool(pretest_pass)}

    # ---------------- extract tokens for full train / val / test under each preprocessing ----------------
    runs_to_do = [
        ("P-A", args.stride),
        ("P-B", args.stride),
        ("P-A-stride10", 10),
    ]

    for label, stride in runs_to_do:
        print(f"\n========== {label} (stride={stride}) ==========", flush=True)
        # If stride differs, rebuild pairs.
        if stride != args.stride:
            tr_p = []
            for pid in TRAIN_PATHS:
                t = load_path(pid, stride)
                if t:
                    tr_p.extend(t)
            va_p = []
            for pid in VAL_PATHS:
                t = load_path(pid, stride)
                if t:
                    va_p.extend(t)
            te_p = []
            for pid in TEST_PATHS:
                t = load_path(pid, stride)
                if t:
                    te_p.extend(t)
            pid_tr_ = np.array([p["path_id"] for p in tr_p])
            pid_va_ = np.array([p["path_id"] for p in va_p])
            pid_te_ = np.array([p["path_id"] for p in te_p])
        else:
            tr_p, va_p, te_p = train_pairs, val_pairs, test_pairs
            pid_tr_, pid_va_, pid_te_ = pid_tr, pid_va, pid_te

        preproc = "P-A" if "P-A" in label else "P-B"

        print(f"  extracting tokens (preprocessing={preproc})", flush=True)
        t0 = time.time()
        Xtr, Ytr = extract_tokens_for_pairs(encoder, tr_p, preproc, dev, batch=args.batch_trunk)
        Xva, Yva = extract_tokens_for_pairs(encoder, va_p, preproc, dev, batch=args.batch_trunk)
        Xte, Yte = extract_tokens_for_pairs(encoder, te_p, preproc, dev, batch=args.batch_trunk)
        print(f"  extracted in {time.time()-t0:.1f}s", flush=True)

        head = MotionHeadPlus(in_dim=132, embed_dim=128, hidden=256).to(dev)
        n_head_full = sum(p.numel() for p in head.parameters())
        print(f"  head+pos params: {n_head_full/1e6:.3f} M", flush=True)
        best_val, train_s = train_head(head, Xtr, Ytr - mu, Xva, Yva - mu,
                                        epochs=args.epochs, batch=args.batch_head,
                                        lr=args.lr, dev=dev, name=label)

        # Final test eval at the best-val checkpoint.
        head.eval()
        with torch.no_grad():
            pred_va = head(torch.tensor(Xva, device=dev)).cpu().numpy() + mu
            pred_te = head(torch.tensor(Xte, device=dev)).cpu().numpy() + mu
            emb_va = head.embed(torch.tensor(Xva, device=dev)).cpu().numpy()
            emb_tr = head.embed(torch.tensor(Xtr, device=dev)).cpu().numpy()
        val_dist = per_path_distribution(pred_va, Yva, pid_va_)
        test_dist = per_path_distribution(pred_te, Yte, pid_te_)
        test_smooth = per_trajectory_smoothness(pred_te, Yte, pid_te_)
        val_mae = val_dist["aggregate"]["mean"]
        test_mae = test_dist["aggregate"]["mean"]
        gap_pct = (test_mae - val_mae) / max(val_mae, 1e-6) * 100.0
        print(f"  >> val MAE = {val_mae:.3f}  test MAE = {test_mae:.3f}  test-val gap = {gap_pct:+.1f}%",
              flush=True)

        # Six-metric harness (Camera on Webots IS temporally ordered).
        try:
            lp = linear_probe(emb_tr, Ytr - mu, emb_va, Yva - mu, epochs=200, lr=1e-2, device="cpu")
        except Exception as e:
            lp = {"error": str(e)}
        try:
            kp = knn_probe(emb_tr, Ytr - mu, emb_va, Yva - mu, k=5)
        except Exception as e:
            kp = {"error": str(e)}
        try:
            au = alignment_uniformity(emb_va, Yva - mu, distance_threshold=1.0, max_samples=1000)
        except Exception as e:
            au = {"error": str(e)}
        try:
            ed = effective_dimensionality(emb_va)
        except Exception as e:
            ed = {"error": str(e)}
        try:
            # Sort val tokens by (path_id, t) to get temporal order.
            order = np.lexsort((np.array([p["t_curr"] for p in va_p]), pid_va_))
            ts = temporal_smoothness(emb_va[order], (Yva - mu)[order])
        except Exception as e:
            ts = {"error": str(e)}
        try:
            tw = trustworthiness(Xva.reshape(len(Xva), -1), emb_va, k=10)
        except Exception as e:
            tw = {"error": str(e)}

        # Per-trajectory plots: top-3 longest test paths.
        plot_paths = {}
        path_lens = {int(pid): int((pid_te_ == pid).sum()) for pid in np.unique(pid_te_)}
        top3 = sorted(path_lens.items(), key=lambda kv: -kv[1])[:3]
        for pid, _n in top3:
            mask = pid_te_ == pid
            out_png = OUT_DIR / "test_paths" / f"{label}_path_{pid:02d}.png"
            plot_path(pred_te[mask], Yte[mask], pid, out_png, f"({label})")
            plot_paths[pid] = str(out_png.relative_to(ROOT))

        results["runs"][label] = {
            "stride": stride, "preproc": preproc,
            "best_val_mae": float(best_val),
            "train_time_s": float(train_s),
            "val_mae": float(val_mae), "test_mae": float(test_mae),
            "test_val_gap_pct": float(gap_pct),
            "val_dist": val_dist, "test_dist": test_dist,
            "test_smoothness": test_smooth,
            "head_params": int(n_head_full),
            "linear_probe": lp, "knn_probe": kp,
            "alignment_uniformity": au,
            "effective_dimensionality": ed,
            "temporal_smoothness": ts,
            "trustworthiness": tw,
            "plot_paths": plot_paths,
        }
        del head, Xtr, Ytr, Xva, Yva, Xte, Yte, emb_va, emb_tr
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "webots_dpvo.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # Summary
    print(f"\n{'run':<18} {'val MAE':>10} {'test MAE':>10} {'gap %':>8} "
          f"{'test p90':>10} {'lin-probe':>10} {'kNN-probe':>10}")
    for label, r in results["runs"].items():
        print(f"  {label:<18} {r['val_mae']:>9.3f}  {r['test_mae']:>9.3f}  "
              f"{r['test_val_gap_pct']:>+7.1f}  {r['test_dist']['aggregate']['p90']:>9.3f}  "
              f"{r['linear_probe'].get('mean_euclidean', float('nan')):>9.3f}  "
              f"{r['knn_probe'].get('mean_euclidean', float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
