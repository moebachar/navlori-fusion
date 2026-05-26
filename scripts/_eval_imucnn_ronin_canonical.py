"""PLAN_07 Step 2 — IMUCNN on canonical RoNIN unseen-subjects.

Same flow as `scripts/eval_ronin_ate_fixed.py` (run-1) but targeting
the FRDR-extracted unseen-subjects set + adding (a) RoNIN's own
`compute_ate_rte` metric for apples-to-apples with PLAN_07 Step 1,
and (b) Umeyama-aligned ATE per amended-rubric correction #3.

Train list filtered to the canonical `list_train.txt` ∩ what we
actually extracted from `data/ronin_frdr/train`. Test list = canonical
`list_test_unseen.txt` (all 32 sequences present).

Run: ``.venv/Scripts/python.exe scripts/_eval_imucnn_ronin_canonical.py``
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

from src.pipeline.baselines import (  # noqa: E402
    GlobSpeedSequence, StridedSequenceDataset, compute_ate_rte,
    RONIN_LISTS,
)
from src.pipeline.encoders import IMUCNN  # noqa: E402

LISTS = RONIN_LISTS
TRAIN_DIR = ROOT / "data" / "ronin_frdr" / "train"
TEST_DIR = ROOT / "data" / "ronin_frdr" / "unseen"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_07"


def _filter_existing(seqs: list[str], root: Path) -> list[str]:
    return [s for s in seqs if (root / s).is_dir()]


def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Sim(3) Umeyama alignment src -> dst (with optimal scale).
    Returns the aligned src trajectory.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === lists ===
    canonical_train = (LISTS / "list_train.txt").read_text().split()
    canonical_test = (LISTS / "list_test_unseen.txt").read_text().split()
    train_list = _filter_existing(canonical_train, TRAIN_DIR)
    test_list = _filter_existing(canonical_test, TEST_DIR)
    print(f"canonical train: {len(canonical_train)}  available: {len(train_list)}",
          flush=True)
    print(f"canonical unseen: {len(canonical_test)}  available: {len(test_list)}",
          flush=True)

    # === train set via RoNIN's StridedSequenceDataset ===
    train_ds = StridedSequenceDataset(
        GlobSpeedSequence, str(TRAIN_DIR), train_list, None,
        args.step, args.window, random_shift=args.step // 2, shuffle=False)
    print(f"train samples: {len(train_ds)}  feat={train_ds.feature_dim}  "
          f"window={args.window}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=False)

    enc = IMUCNN(in_features=6, embed_dim=128).to(dev)
    head = nn.Linear(128, 2).to(dev)
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"IMUCNN+head params: {n_params/1e6:.3f} M", flush=True)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(train_ds) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    torch.cuda.empty_cache()

    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        tot, n = 0.0, 0
        for feat, targ, _, _ in loader:
            x = feat.transpose(1, 2).contiguous().to(dev)
            y = targ.to(dev)
            pred = head(enc(x))
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if ep <= 1 or ep % 2 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  vel huber={tot/max(n,1):.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    train_s = time.time() - t0
    print(f"training done in {train_s:.1f}s", flush=True)

    # === Per-test-sequence ATE ===
    enc.eval(); head.eval()
    pred_per_min = 200 * 60  # same as RoNIN's ronin_resnet.py default
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
                xw = torch.tensor(wins, device=dev)
                vel_chunks.append(head(enc(xw)).cpu().numpy())
                del xw
            vel = np.concatenate(vel_chunks, axis=0)  # (N, 2)

            # Reconstruct trajectory via forward-Euler integration starting
            # from GT[0]. The RoNIN convention `recon_traj_with_preds`
            # integrates from gt[0]; ours starts from gt[args.window] but
            # then RoNIN's `compute_ate_rte` re-anchors via gt_start so the
            # two are comparable.
            traj = np.zeros((len(ends), 2), np.float32)
            cur = gt[args.window].copy()
            traj[0] = cur
            for i in range(1, len(ends)):
                dt = ts[ends[i]] - ts[ends[i - 1]]
                cur = cur + vel[i - 1] * dt
                traj[i] = cur
            gtm = gt[ends]

            # Metrics:
            #   * raw_ronin: RoNIN's own ATE (anchored at gt[0], no rotation/scale).
            #   * raw_simple: ours, mean Euclidean.
            #   * umeyama: Sim(3) Umeyama alignment with scale.
            #   * rte: RoNIN's RTE (1-min sliding window).
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
            print(f"  {sname:15s}  ate_ronin={ate_ronin:6.3f}  "
                  f"raw_simple={raw_simple:6.3f}  umeyama={umeyama:6.3f}  "
                  f"rte={rte_ronin:6.3f}  ({len(ends)} win)", flush=True)

    # === Aggregate ===
    if not rows:
        raise SystemExit("No sequences evaluated.")
    summary = {}
    for k in ("ate_ronin", "raw_simple", "umeyama", "rte_ronin"):
        vals = np.array([r[k] for r in rows])
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
        "split": {"train_used": len(train_list),
                  "train_canonical": len(canonical_train),
                  "test_used": len(test_list),
                  "test_canonical": len(canonical_test)},
        "training": {"epochs": args.epochs, "elapsed_s": float(train_s),
                     "params": int(n_params), "batch": args.batch,
                     "window": args.window, "step": args.step, "lr": args.lr},
        "summary": summary,
        "per_seq": rows,
    }
    out_path = OUT_DIR / "imucnn_canonical.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    print(f"\nIMUCNN on RoNIN canonical unseen (32 seqs):")
    for k in ("ate_ronin", "raw_simple", "umeyama", "rte_ronin"):
        s = summary[k]
        print(f"  {k:12s}  mean={s['mean']:6.3f}  median={s['median']:6.3f}  "
              f"p90={s['p90']:6.3f}  max={s['max']:6.3f}", flush=True)


if __name__ == "__main__":
    main()
