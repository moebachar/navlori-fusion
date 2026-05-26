"""PLAN_23 — CNN1D / LSTM-attn aggregator over K=4 IMUCNN sub-window tokens on
canonical RoNIN unseen-subjects.

Pipeline:
  raw 200-step IMU window
    -> chunk into K=4 contiguous 50-step sub-windows
    -> IMUCNN(50) -> 4 tokens of D=128
    -> CNN1D / LSTM-attn aggregator (bakeoff candidate) -> (B, 4, 128)
    -> mean-pool over K -> (B, 128)
    -> Linear(128, 2) -> velocity (vx, vy)

Same training protocol as the existing IMUCNN canonical eval
(`_eval_imucnn_ronin_canonical.py`): 20 epochs, AdamW + OneCycleLR,
Huber(δ=0.5), B=128, lr=1e-3. Inference integrates per-step velocity
into a trajectory and reports ATE/RTE via RoNIN's `compute_ate_rte`.

Run: ``.venv/Scripts/python.exe scripts/_train_ronin_canonical_arch.py --arch cnn1d``
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

# Runtime shim for RoNIN's data_glob_speed (uses np.int).
np.int = int  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RONIN_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\ronin\source")
if str(RONIN_SRC) not in sys.path:
    sys.path.insert(0, str(RONIN_SRC))

from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset  # noqa: E402
from metric import compute_ate_rte  # noqa: E402

from src.pipeline.encoders import IMUCNN  # noqa: E402
from src.pipeline.fusion.bakeoff import _MaskedBiLSTM, _PlainCNN1D  # noqa: E402

LISTS = Path(r"C:\Users\FabLab\AppData\Local\Temp\ronin\lists")
TRAIN_DIR = ROOT / "data" / "ronin_frdr" / "train"
TEST_DIR = ROOT / "data" / "ronin_frdr" / "unseen"
OUT_DIR_DEFAULT = ROOT / "runs" / "overnight" / "run2_iter_23"

K = 4         # number of sub-window instants
WINDOW = 200  # canonical RoNIN window
SUB = WINDOW // K  # 50 steps per sub-window


def _filter_existing(seqs, root):
    return [s for s in seqs if (root / s).is_dir()]


def _umeyama_align(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s; dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return ((s * (R @ src.T)).T + t).astype(np.float32)


class RoninCNN1D(nn.Module):
    """IMUCNN per K=4 sub-window + CNN1D aggregator + mean-pool + linear head."""

    def __init__(self, agg_kind: str, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.imucnn = IMUCNN(in_features=6, embed_dim=embed_dim)
        if agg_kind == "cnn1d":
            self.aggregator = _PlainCNN1D(embed_dim=embed_dim, dropout=dropout)
        elif agg_kind == "lstm_attn":
            self.aggregator = _MaskedBiLSTM(embed_dim=embed_dim,
                                             hidden_dim=embed_dim)
        else:
            raise ValueError(f"unknown agg_kind {agg_kind}")
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: (B, 200, 6) -> chunk into K=4 sub-windows of 50.
        # IMUCNN expects (B, window, in_features); it transposes internally.
        B = x.shape[0]
        x = x.view(B, K, SUB, 6)              # (B, K, SUB, 6)
        x = x.reshape(B * K, SUB, 6)          # (B*K, SUB, 6)
        tokens = self.imucnn(x)               # (B*K, D)
        tokens = tokens.view(B, K, -1)        # (B, K, D)
        # Aggregator wants (B, S, D) + key_padding_mask (B, S) — all False here.
        pad = torch.zeros(B, K, dtype=torch.bool, device=tokens.device)
        agg = self.aggregator(tokens, pad)    # (B, K, D)
        pooled = agg.mean(dim=1)              # (B, D)
        return self.head(pooled)              # (B, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnn1d", "lstm_attn"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== arch={args.arch} RoNIN canonical IMU-only K={K} ===", flush=True)

    canonical_train = (LISTS / "list_train.txt").read_text().split()
    canonical_test = (LISTS / "list_test_unseen.txt").read_text().split()
    train_list = _filter_existing(canonical_train, TRAIN_DIR)
    test_list = _filter_existing(canonical_test, TEST_DIR)
    print(f"train: {len(train_list)}/{len(canonical_train)}  "
          f"unseen test: {len(test_list)}/{len(canonical_test)}", flush=True)

    train_ds = StridedSequenceDataset(
        GlobSpeedSequence, str(TRAIN_DIR), train_list, None,
        args.step, args.window, random_shift=args.step // 2, shuffle=False)
    print(f"train windows: {len(train_ds)}  feat={train_ds.feature_dim}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=False)

    model = RoninCNN1D(args.arch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch} canonical model params: {n_params/1e6:.3f} M",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(train_ds) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    torch.cuda.empty_cache()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for feat, targ, _, _ in loader:
            # RoNIN returns feat as (B, 6, window) per its loader; we want
            # (B, window, 6) and then split inside the model.
            x = feat.transpose(1, 2).contiguous().to(dev)  # (B, window, 6)
            y = targ.to(dev)
            pred = model(x)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if ep <= 1 or ep % 2 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  vel huber={tot/max(n,1):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    print(f"training done in {train_s:.1f}s", flush=True)

    # === Per-test-sequence ATE ===
    model.eval()
    pred_per_min = 200 * 60
    rows = []
    with torch.no_grad():
        for sname in test_list:
            seq = GlobSpeedSequence(str(TEST_DIR / sname),
                                     interval=args.window,
                                     max_ori_error=20.0, grv_only=True)
            feat = seq.features
            ts = seq.ts
            gt = seq.gt_pos[:, :2]
            if len(feat) < args.window + 5:
                continue
            stride_inf = args.step
            ends = np.arange(args.window, len(feat), stride_inf)
            vel_chunks = []
            BS = 512
            for i in range(0, len(ends), BS):
                ebatch = ends[i:i + BS]
                wins = np.stack([feat[e - args.window:e] for e in ebatch]).astype(np.float32)
                xw = torch.tensor(wins, device=dev)  # (BS, window, 6)
                vel_chunks.append(model(xw).cpu().numpy())
                del xw
            vel = np.concatenate(vel_chunks, axis=0)

            traj = np.zeros((len(ends), 2), np.float32)
            cur = gt[args.window].copy()
            traj[0] = cur
            for i in range(1, len(ends)):
                dt = ts[ends[i]] - ts[ends[i - 1]]
                cur = cur + vel[i - 1] * dt
                traj[i] = cur
            gtm = gt[ends]

            ate_ronin, rte_ronin = compute_ate_rte(traj, gtm, pred_per_min)
            raw_simple = float(np.sqrt(((traj - gtm) ** 2).sum(1).mean()))
            traj_u = _umeyama_align(traj, gtm)
            umeyama = float(np.sqrt(((traj_u - gtm) ** 2).sum(1).mean()))

            rows.append({
                "seq": sname,
                "ate_ronin": float(ate_ronin),
                "raw_simple": float(raw_simple),
                "umeyama": float(umeyama),
                "rte_ronin": float(rte_ronin),
                "n_windows": int(len(ends)),
            })
            print(f"  {sname:15s}  ate={ate_ronin:6.3f}  raw={raw_simple:6.3f}  "
                  f"umey={umeyama:6.3f}  rte={rte_ronin:6.3f}  ({len(ends)} win)",
                  flush=True)

    summary = {}
    for k in ("ate_ronin", "raw_simple", "umeyama", "rte_ronin"):
        vals = np.array([r[k] for r in rows if not np.isnan(r[k])])
        if len(vals) == 0:
            summary[k] = {"mean": float("nan"), "n": 0}
            continue
        summary[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n": int(len(vals)),
        }
    out = {
        "arch": args.arch,
        "n_params": int(n_params),
        "training": {"epochs": args.epochs, "elapsed_s": float(train_s),
                     "window": args.window, "step": args.step,
                     "batch": args.batch, "lr": args.lr, "K": K, "SUB": SUB},
        "summary": summary,
        "per_seq": rows,
    }
    out_path = OUT_DIR / f"{args.arch}_ronin_canonical.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    print(f"\n{args.arch} on RoNIN canonical unseen ({len(rows)} seqs):")
    for k in ("ate_ronin", "raw_simple", "umeyama", "rte_ronin"):
        s = summary[k]
        if "mean" in s and not np.isnan(s.get("mean", float("nan"))):
            print(f"  {k:12s}  mean={s['mean']:6.3f}  median={s['median']:6.3f}  "
                  f"p90={s['p90']:6.3f}  max={s['max']:6.3f}", flush=True)
        else:
            print(f"  {k:12s}  no valid values", flush=True)


if __name__ == "__main__":
    main()
