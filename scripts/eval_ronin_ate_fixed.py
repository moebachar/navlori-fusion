"""Our IMUCNN on RoNIN unseen, with RoNIN's OFFICIAL preprocessing.

Fixes the 29-52 m disaster from `scripts/eval_ronin_ate.py`: that script did a
naive yaw-only rotation with no IMU calibration and no gravity-aware
stabilization, leaking gravity into horizontal accel and integrating it twice
into tens of meters of drift.

This version uses RoNIN's own `GlobSpeedSequence` data loader (their
open-source preprocessing, unmodified) to obtain:
  * 6-channel world-frame features (gyro + accel, calibrated, rotated via the
    FULL device-orientation quaternion into the Tango world frame).
  * 2D world-frame velocity targets (Δposition / Δt over their window).
Then trains **our light IMUCNN** on top of that input (same encoder used by
the fusion model), integrates per-step velocity into a trajectory, computes
ATE per sequence on the unseen-subjects test set. Reference: RoNIN ResNet 5.14m.

Demand #3 (use baseline open source unmodified): we IMPORT RoNIN's loader; we
do not duplicate or reimplement it. A runtime numpy shim (np.int = int) is
applied here, not in their files, so their source stays pristine.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import GlobSpeedSequence, StridedSequenceDataset  # noqa: E402
from src.pipeline.encoders import IMUCNN  # noqa: E402


def _seq_paths(lists_dir: Path, name: str) -> list[str]:
    return [ln.strip() for ln in (lists_dir / name).read_text().splitlines() if ln.strip()]


def _ate(pred, gt):
    return float(np.sqrt(((pred - gt) ** 2).sum(1).mean()))


def _ate_aligned(pred, gt):
    pc, gc = pred - pred.mean(0), gt - gt.mean(0)
    H = pc.T @ gc
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = Vt.T @ U.T
    return _ate(pc @ Rm.T + gt.mean(0), gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = str(ROOT / "data" / "FRDR_dataset_538_download_259_202604270443" / "Data")
    lists = ROOT / "runs" / "ronin_adapted_lists"
    train_list = _seq_paths(lists, "list_train.txt")
    test_list = _seq_paths(lists, "list_test_unseen.txt")
    print(f"train seqs: {len(train_list)}  test (unseen) seqs: {len(test_list)}", flush=True)

    # === build train set via RoNIN's StridedSequenceDataset ===
    train_ds = StridedSequenceDataset(
        GlobSpeedSequence, root_dir, train_list, None,
        args.step, args.window, random_shift=args.step // 2, shuffle=False)
    print(f"train velocity samples: {len(train_ds)}  feat={train_ds.feature_dim}  "
          f"window={args.window}", flush=True)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True,
                                          num_workers=0, drop_last=True,
                                          pin_memory=False)

    # === our light IMUCNN (5-channel default — but RoNIN feeds 6; use 6) ===
    enc = IMUCNN(in_features=6, embed_dim=128).to(dev)
    head = nn.Linear(128, 2).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    steps = max(1, len(train_ds) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)
    torch.cuda.empty_cache()

    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        tot = 0.0; n = 0
        for feat, targ, _, _ in loader:
            # feat: (B, 6, window) -> need (B, window, 6) for IMUCNN
            x = feat.transpose(1, 2).contiguous().to(dev)
            y = targ.to(dev)
            pred = head(enc(x))
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if ep % 2 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  vel huber={tot/max(n,1):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    # === per-test-sequence ATE ===
    enc.eval(); head.eval()
    ates, ates_al, names = [], [], []
    with torch.no_grad():
        for sname in test_list:
            seq = GlobSpeedSequence(str(Path(root_dir) / sname),
                                     interval=args.window,
                                     max_ori_error=20.0, grv_only=False)
            feat = seq.features                       # (T, 6) world-frame
            ts = seq.ts                                # (T,)
            gt = seq.gt_pos[:, :2]                     # (T, 2)
            if len(feat) < args.window + 5:
                continue
            # build windows ending at every 10th sample (200 Hz / 10 = 20 Hz
            # trajectory points — matches RoNIN's step_size for fair comparison)
            stride = 10
            ends = np.arange(args.window, len(feat), stride)
            vel_chunks = []
            BS = 512
            for i in range(0, len(ends), BS):
                ebatch = ends[i:i + BS]
                wins = np.stack([feat[e - args.window:e] for e in ebatch]).astype(np.float32)
                xw = torch.tensor(wins, device=dev)
                vel_chunks.append(head(enc(xw)).cpu().numpy())
                del xw
            vel = np.concatenate(vel_chunks, axis=0)   # (N, 2) m/s
            # integrate from gt[0]; trajectory at times ts[ends]
            traj = np.zeros((len(ends), 2), np.float32)
            cur = gt[args.window].copy()
            traj[0] = cur
            for i in range(1, len(ends)):
                dt = ts[ends[i]] - ts[ends[i - 1]]
                cur = cur + vel[i - 1] * dt
                traj[i] = cur
            gtm = gt[ends]
            a = _ate(traj, gtm); aa = _ate_aligned(traj, gtm)
            ates.append(a); ates_al.append(aa); names.append(sname.split('/')[-1])
    ates = np.array(ates); ates_al = np.array(ates_al)
    print(f"\n>>> Our light IMUCNN on RoNIN unseen ATE (RoNIN preprocessing):")
    print(f"    raw      mean={ates.mean():.2f} m  median={np.median(ates):.2f}  max={ates.max():.2f}  (n={len(ates)})")
    print(f"    aligned  mean={ates_al.mean():.2f} m  median={np.median(ates_al):.2f}")
    print(f"    reference RoNIN ResNet (full ResNet18 on same data) 5.93 m (our reproduction) / 5.14 m (paper)")


if __name__ == "__main__":
    main()
