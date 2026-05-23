"""RoNIN ResNet1D on IPIN floor -2 — Phase B IMU-only baseline (same data as fusion).

Imports RoNIN's `model_resnet1d.ResNet1D` (their official architecture from
github.com/Sachini/ronin), trains it on IPIN IMU windows -> 2D velocity, then
integrates to a trajectory and reports per-sample MAE vs GT positions on
IPIN val. Uses the SAME IPIN data + per-sample MAE as our fusion runs, so
the comparison is controlled.

Input: 6-channel IMU window (gyro_xyz + accel_xyz body frame), 50 samples
(~2s at 25Hz IPIN rate; RoNIN used 200@200Hz which is the same temporal span).

Run: .venv/Scripts/python.exe scripts/eval_ronin_ipin.py [--epochs 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# import RoNIN architecture from their cloned repo
RONIN_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\ronin\source")
if str(RONIN_SRC) not in sys.path:
    sys.path.insert(0, str(RONIN_SRC))

from model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D  # noqa: E402

from src.pipeline.fusion.builder import load_config  # noqa: E402

WIN = 50            # 2 s at IPIN ~25 Hz
LOOKBACK = 1.0      # velocity target = displacement over the last 1s


def build_samples(dataset, paths):
    """For each GT sample in `paths`, build (imu_window 50x6, velocity 2)."""
    X, V, ts, pids = [], [], [], []
    cfg = load_config(dataset)
    root = ROOT / str(cfg.dataset.root) / cfg.dataset.collection_dir
    for p in paths:
        pdir = root / f"path_{p:02d}"
        if not (pdir / "imu.csv").exists() or not (pdir / "ground_truth.csv").exists():
            continue
        imu = pd.read_csv(pdir / "imu.csv")
        gt = pd.read_csv(pdir / "ground_truth.csv")
        if len(imu) < WIN + 5 or len(gt) < 3:
            continue
        cols = ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
        imu_arr = imu[cols].values.astype(np.float32)
        imu_t = imu["sim_time"].values
        gt_t = gt["sim_time"].values
        gt_xy = gt[["gt_x", "gt_y"]].values.astype(np.float32)
        # For each GT sample, find IMU window ending at its time, and velocity
        # = (gt[t] - gt[t-lookback]) / lookback.
        for k, t in enumerate(gt_t):
            j = int(np.searchsorted(imu_t, t, side="right") - 1)
            if j < WIN - 1:
                continue
            t0 = t - LOOKBACK
            k0 = int(np.searchsorted(gt_t, t0, side="right") - 1)
            if k0 < 0:
                continue
            v = (gt_xy[k] - gt_xy[k0]) / max(t - gt_t[k0], 1e-6)
            X.append(imu_arr[j - WIN + 1:j + 1])
            V.append(v)
            ts.append(t)
            pids.append(p)
    return (np.array(X, np.float32), np.array(V, np.float32),
            np.array(ts), np.array(pids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset", default="ipin2024_floor-2")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.dataset)
    Xtr, Vtr, _, _ = build_samples(args.dataset, list(cfg.dataset.split.train_paths))
    Xva, Vva, Tva, Pva = build_samples(args.dataset, list(cfg.dataset.split.val_paths))
    print(f"{args.dataset}: train windows {len(Xtr)}  val windows {len(Xva)}", flush=True)

    # Normalize input (per-channel) using train stats — matches what RoNIN's loader does internally.
    mu = Xtr.reshape(-1, 6).mean(0); sd = Xtr.reshape(-1, 6).std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    # ResNet1D expects (B, C, T)
    Xt = torch.tensor(Xtr_n.transpose(0, 2, 1), device=dev)
    Vt = torch.tensor(Vtr, device=dev)
    Xv = torch.tensor(Xva_n.transpose(0, 2, 1), device=dev)
    Vv = torch.tensor(Vva, device=dev)

    # Match RoNIN ResNet18 config (their ronin_resnet.py:25-26) but adapt
    # in_dim to our 50-sample window: in_dim = window // 32 + 1 = 2.
    fc_cfg = {"fc_dim": 512, "in_dim": WIN // 32 + 1, "dropout": 0.5,
              "trans_planes": 128}
    net = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64,
                   output_block=FCOutputModule, kernel_size=3, **fc_cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = max(1, len(Xt) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=steps, pct_start=0.3)
    crit = nn.HuberLoss(delta=0.5)

    best_v = float("inf"); best_state = None
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(steps):
            idx = perm[s * 128:(s + 1) * 128]
            loss = crit(net(Xt[idx]), Vt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            vh = float(crit(net(Xv), Vv).item())
        if vh < best_v:
            best_v = vh
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  val huber={vh:.4f}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state); net.to(dev)
    net.eval()

    # --- Integrate per-path velocity into trajectory, compute MAE per sample ---
    # Pull val GT in order per path; predict velocity at each window-aligned step,
    # integrate from the GT origin of each path.
    cfg2 = load_config(args.dataset)
    root = ROOT / str(cfg2.dataset.root) / cfg2.dataset.collection_dir
    errs_all = []
    per_path = {}
    with torch.no_grad():
        # group val windows by path
        for pid in np.unique(Pva):
            mask = Pva == pid
            order = np.argsort(Tva[mask])
            idx = np.where(mask)[0][order]
            xw = torch.tensor(Xva_n[idx].transpose(0, 2, 1), device=dev,
                              dtype=torch.float32)
            vel = net(xw).cpu().numpy()
            tw = Tva[idx]
            # GT for this path (for origin anchoring and per-sample error)
            gt = pd.read_csv(root / f"path_{pid:02d}" / "ground_truth.csv")
            gt_t = gt["sim_time"].values
            gt_xy = gt[["gt_x", "gt_y"]].values.astype(np.float32)
            # Integrate from gt[0]
            pos = np.zeros((len(tw), 2), np.float32)
            cur = gt_xy[0].copy()
            prev_t = gt_t[0]
            for i in range(len(tw)):
                cur = cur + vel[i] * (tw[i] - prev_t)
                pos[i] = cur
                prev_t = tw[i]
            # Match GT positions at each prediction time (nearest)
            gt_match = []
            for t in tw:
                k = int(np.argmin(np.abs(gt_t - t)))
                gt_match.append(gt_xy[k])
            gt_match = np.array(gt_match)
            err = np.linalg.norm(pos - gt_match, axis=1)
            errs_all.append(err)
            per_path[int(pid)] = float(err.mean())

    errs_all = np.concatenate(errs_all)
    print(f"\n  >>> RoNIN ResNet on {args.dataset} val (IMU-only baseline):")
    print(f"      mean per-sample MAE = {errs_all.mean():.2f} m  (n={len(errs_all)})")
    print(f"      per-path: " + ", ".join(f"p{p:02d}={e:.1f}m" for p, e in sorted(per_path.items())))


if __name__ == "__main__":
    main()
