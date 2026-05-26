"""PLAN_19 Step 1b — RoNIN ResNet1D on IMUWiFine floor 4 IMU (NEW measurement).

Clone of `scripts/eval_ronin_ipin.py` (RESULT_02 template), retargeted at
the IMUWiFine config. Trains ResNet1D from scratch on IMUWiFine train,
reports val + test per-sample MAE (m) and per-path distribution.

IMU WIN matches the IMUWiFine dataset config (`imu: 32` = ~1 s at the
post-downsample rate). 6-channel input: gyro_xyz + accel_xyz.

Test paths in IMUWiFine "carry only WiFi + ground truth (no IMU)" per
the config comment. If a test path's imu.csv is missing-or-empty the
loader skips it and that path is excluded from the IMU SOTA test
number; the RESULT documents this asymmetry.

Run: ``.venv/Scripts/python.exe scripts/_eval_ronin_imuwifine.py``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import BasicBlock1D, FCOutputModule, ResNet1D  # noqa: E402
from src.pipeline.fusion.builder import load_config  # noqa: E402

OUT_DIR_DEFAULT = ROOT / "runs" / "overnight" / "run2_iter_19"

WIN = 32           # imu window = 32 samples (~1 s after downsample)
LOOKBACK = 1.0     # velocity target = displacement over the last 1 s


def build_samples(dataset, paths):
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
        if not set(cols).issubset(imu.columns):
            continue
        imu_arr = imu[cols].values.astype(np.float32)
        imu_t = imu["sim_time"].values
        gt_t = gt["sim_time"].values
        gt_xy = gt[["gt_x", "gt_y"]].values.astype(np.float32)
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


def integrate_per_path(net, Xs_n, ts, pids, mu, sd, root, dev):
    errs_all = []
    per_path = {}
    with torch.no_grad():
        for pid in np.unique(pids):
            mask = pids == pid
            order = np.argsort(ts[mask])
            idx = np.where(mask)[0][order]
            xw = torch.tensor(Xs_n[idx].transpose(0, 2, 1), device=dev,
                              dtype=torch.float32)
            vel = net(xw).cpu().numpy()
            tw = ts[idx]
            gt = pd.read_csv(root / f"path_{pid:02d}" / "ground_truth.csv")
            gt_t = gt["sim_time"].values
            gt_xy = gt[["gt_x", "gt_y"]].values.astype(np.float32)
            pos = np.zeros((len(tw), 2), np.float32)
            cur = gt_xy[0].copy()
            prev_t = gt_t[0]
            for i in range(len(tw)):
                cur = cur + vel[i] * (tw[i] - prev_t)
                pos[i] = cur
                prev_t = tw[i]
            gt_match = []
            for t in tw:
                k = int(np.argmin(np.abs(gt_t - t)))
                gt_match.append(gt_xy[k])
            gt_match = np.array(gt_match)
            err = np.linalg.norm(pos - gt_match, axis=1)
            errs_all.append(err)
            per_path[int(pid)] = float(err.mean())
    if not errs_all:
        return np.array([]), per_path
    return np.concatenate(errs_all), per_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset", default="imuwifine")
    ap.add_argument("--out-dir", default=None,
                    help="Override OUT_DIR (default: runs/overnight/run2_iter_19)")
    args = ap.parse_args()
    OUT_DIR = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.dataset)
    root = ROOT / str(cfg.dataset.root) / cfg.dataset.collection_dir
    Xtr, Vtr, _, _ = build_samples(args.dataset, list(cfg.dataset.split.train_paths))
    Xva, Vva, Tva, Pva = build_samples(args.dataset, list(cfg.dataset.split.val_paths))
    Xte, Vte, Tte, Pte = build_samples(args.dataset, list(cfg.dataset.split.test_paths))
    print(f"{args.dataset}: train w {len(Xtr)}  val w {len(Xva)}  test w {len(Xte)}",
          flush=True)

    mu = Xtr.reshape(-1, 6).mean(0); sd = Xtr.reshape(-1, 6).std(0) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xva_n = (Xva - mu) / sd
    Xte_n = (Xte - mu) / sd if len(Xte) else Xte
    Xt = torch.tensor(Xtr_n.transpose(0, 2, 1), device=dev)
    Vt = torch.tensor(Vtr, device=dev)
    Xv = torch.tensor(Xva_n.transpose(0, 2, 1), device=dev)
    Vv = torch.tensor(Vva, device=dev)

    # Derive actual conv output length empirically by probing the truncated
    # network without the FCOutputModule (ResNet1D's conv stem + 4 stride-2
    # blocks). For WIN=32 this gives in_dim=1, for WIN=50 in_dim=2, etc.
    with torch.no_grad():
        probe_in = torch.zeros(2, 6, WIN, device=dev)
        dummy_fc = {"fc_dim": 512, "in_dim": 1, "dropout": 0.5, "trans_planes": 128}
        dummy = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64,
                         output_block=FCOutputModule, kernel_size=3, **dummy_fc).to(dev)
        dummy.eval()
        x = dummy.input_block(probe_in)
        for r in dummy.residual_groups:
            x = r(x)
        conv_out_T = int(x.shape[-1])
        del dummy
    print(f"  conv output sequence length: {conv_out_T} (WIN={WIN})", flush=True)
    fc_cfg = {"fc_dim": 512, "in_dim": conv_out_T, "dropout": 0.5,
              "trans_planes": 128}
    net = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64,
                   output_block=FCOutputModule, kernel_size=3, **fc_cfg).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  ResNet1D params: {n_params/1e6:.2f} M", flush=True)

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

    val_errs, val_pp = integrate_per_path(net, Xva_n, Tva, Pva, mu, sd, root, dev)
    print(f"\n  VAL: per-sample MAE = {val_errs.mean():.2f} m  "
          f"(n={len(val_errs)} windows, n_paths={len(val_pp)})", flush=True)

    test_summary = {}
    if len(Xte):
        test_errs, test_pp = integrate_per_path(net, Xte_n, Tte, Pte, mu, sd, root, dev)
        if len(test_errs):
            print(f"  TEST: per-sample MAE = {test_errs.mean():.2f} m  "
                  f"(n={len(test_errs)} windows, n_paths={len(test_pp)})", flush=True)
            test_summary = {
                "n_windows": int(len(test_errs)),
                "n_paths": int(len(test_pp)),
                "mean_mae": float(test_errs.mean()),
                "median_mae": float(np.median(test_errs)),
                "p90_mae": float(np.percentile(test_errs, 90)),
                "max_mae": float(test_errs.max()),
                "per_path": test_pp,
            }
    else:
        print("  TEST: no IMU windows available (test paths lack imu.csv).", flush=True)

    out = {
        "method": f"RoNIN ResNet1D (trained from scratch on {args.dataset} train)",
        "n_params": int(n_params),
        "epochs": int(args.epochs),
        "win": WIN,
        "val": {
            "n_windows": int(len(val_errs)),
            "n_paths": int(len(val_pp)),
            "mean_mae": float(val_errs.mean()),
            "median_mae": float(np.median(val_errs)),
            "p90_mae": float(np.percentile(val_errs, 90)),
            "max_mae": float(val_errs.max()),
            "per_path": val_pp,
        },
        "test": test_summary,
        "best_val_huber": best_v,
    }
    out_fn = f"ronin_{args.dataset}.json"
    with open(OUT_DIR / out_fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / out_fn}", flush=True)


if __name__ == "__main__":
    main()
