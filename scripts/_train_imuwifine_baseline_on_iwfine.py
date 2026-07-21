"""One-shot: train IMUWiFine baseline LSTM on the IMUWiFine fl.4 dataset
(its own training domain) and save per-path test predictions for the notebook.

This gives a 3rd method for the IMUWiFine fl.4 showcase bar chart (alongside
wlanloc + Ours). Reuses the IMUWiFineModel class but with an IMUWiFine fl.4
data loader instead of the MSILN one.
"""
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.pipeline.baselines.imuwifine import IMUWiFineModel, _normalize_imu_inplace
from src.pipeline.baselines.imuwifine import _MsilnWindowDataset

ROOT = Path(__file__).resolve().parent
IWF_ROOT = ROOT / "data" / "imuwifine_floor4"
IWF_TRAIN = list(range(0, 40))
IWF_VAL   = list(range(40, 60))
IWF_TEST  = list(range(60, 80))


def _normalize_wifi(rssi, no_signal_dbm=-100.0):
    x = rssi.astype(np.float32) + abs(no_signal_dbm)
    x /= abs(no_signal_dbm); x = np.power(x, math.e)
    return x


def _interp(t_query, t_known, vals):
    out = np.empty((len(t_query), vals.shape[1]), dtype=np.float32)
    for k in range(vals.shape[1]):
        out[:, k] = np.interp(t_query, t_known, vals[:, k])
    return out


def _wifi_snapshot_at(t_query, wifi_df, rssi_cols, no_signal=-100.0):
    n_aps = len(rssi_cols)
    out = np.full((len(t_query), n_aps), no_signal, dtype=np.float32)
    if wifi_df.empty:
        return out
    wifi_t = wifi_df["sim_time"].values.astype(np.float64)
    wifi_rssi = wifi_df[rssi_cols].values.astype(np.float32)
    wifi_rssi = np.where(np.isnan(wifi_rssi), no_signal, wifi_rssi)
    idx = np.searchsorted(wifi_t, t_query, side="right") - 1
    valid = idx >= 0
    out[valid] = wifi_rssi[idx[valid]]
    return out


def load_iwf_paths(path_ids, target_hz=10.0):
    """Load IMUWiFine fl.4 paths in IMUWiFineModel input format.

    Returns (paths_list, rssi_cols). IMUWiFine fl.4 test paths have no IMU on
    disk — for those, we zero-fill the IMU stream and warp the GT time grid
    to match the WiFi cadence.
    """
    probe = pd.read_csv(IWF_ROOT / "path_00" / "wifi.csv", nrows=1)
    rssi_cols = [c for c in probe.columns if c.startswith("wifi_rssi_")]

    paths = []
    for pid in path_ids:
        pdir = IWF_ROOT / f"path_{pid:02d}"
        if not pdir.is_dir():
            continue
        wifi = pd.read_csv(pdir / "wifi.csv")
        gt = pd.read_csv(pdir / "ground_truth.csv")
        imu_path = pdir / "imu.csv"
        has_imu = imu_path.exists() and imu_path.stat().st_size > 200

        # time grid
        t_start = max(float(gt["sim_time"].min()), float(wifi["sim_time"].min()) if not wifi.empty else 0)
        t_end = min(float(gt["sim_time"].max()), float(wifi["sim_time"].max()) if not wifi.empty else 0)
        if t_end - t_start < 1.0:
            continue
        step = 1.0 / target_hz
        t_grid = np.arange(t_start, t_end, step, dtype=np.float64)
        if len(t_grid) < 2:
            continue

        # IMU resample (or zero-fill)
        if has_imu:
            imu = pd.read_csv(imu_path)
            imu_t = imu["sim_time"].values.astype(np.float64)
            imu_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
            for c in imu_cols:
                if c not in imu.columns: imu[c] = 0.0
            imu_vals = imu[imu_cols].values.astype(np.float32)
            imu_resamp = _interp(t_grid, imu_t, imu_vals)
        else:
            imu_resamp = np.zeros((len(t_grid), 6), dtype=np.float32)

        # GT
        gt_t = gt["sim_time"].values.astype(np.float64)
        gt_vals = gt[["gt_x", "gt_y"]].values.astype(np.float32)
        gt_resamp = _interp(t_grid, gt_t, gt_vals)

        # WiFi snapshot
        for c in rssi_cols:
            if c not in wifi.columns: wifi[c] = np.nan
        wifi_resamp = _wifi_snapshot_at(t_grid, wifi, rssi_cols)

        paths.append({
            "path_id": int(pid),
            "t": t_grid.astype(np.float32),
            "wifi": wifi_resamp,
            "imu":  imu_resamp,
            "gt":   gt_resamp,
        })
    return paths, rssi_cols


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)

    print("Loading IMUWiFine fl.4 train/val/test (target_hz=10)...", flush=True)
    train_paths, rssi_cols = load_iwf_paths(IWF_TRAIN, target_hz=10.0)
    val_paths,   _         = load_iwf_paths(IWF_VAL,   target_hz=10.0)
    test_paths,  _         = load_iwf_paths(IWF_TEST,  target_hz=10.0)
    print(f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} paths", flush=True)

    train_ds = _MsilnWindowDataset(train_paths, window=30, stride=10)
    val_ds   = _MsilnWindowDataset(val_paths,   window=30, stride=30)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}", flush=True)

    n_aps = len(rssi_cols)
    input_dim = n_aps + 6
    model = IMUWiFineModel(input_dim=input_dim, hidden_dim=256,
                            output_dim=2, n_layers=4, dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params / 1e6:.2f} M  input_dim={input_dim}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    history = {"train_loss": [], "val_mae": []}
    best_val = float("inf"); best_state = None
    t0 = time.time()
    BATCH = 8
    for ep in range(40):
        model.train()
        perm = np.random.permutation(len(train_ds))
        ep_loss, n_seen = 0.0, 0
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            tuples = [train_ds[i] for i in idx]
            x = torch.stack([t[0] for t in tuples]).to(device)
            y = torch.stack([t[1] for t in tuples]).to(device)
            pred = model(x)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx); n_seen += len(idx)
        sched.step()
        history["train_loss"].append(ep_loss / max(n_seen, 1))

        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(len(val_ds)):
                x, y = val_ds[i]
                pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
                errs.append(np.linalg.norm(pred - y.numpy(), axis=1))
        val_mae = float(np.concatenate(errs).mean()) if errs else float("nan")
        history["val_mae"].append(val_mae)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep <= 2 or ep % 5 == 0 or ep == 39:
            print(f"ep {ep:3d}/40  loss={ep_loss / max(n_seen, 1):.4f}  val_mae={val_mae:.3f}m  "
                  f"(best {best_val:.3f}m, {time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    print(f"\nbest val MAE: {best_val:.3f} m  total {time.time() - t0:.0f}s", flush=True)

    # Per-path predictions on test
    print("\nPer-path test predictions:", flush=True)
    per_path = {}
    model.eval()
    for p in test_paths:
        pid = p["path_id"]
        ds_one = _MsilnWindowDataset([p], window=30, stride=30)
        preds, gts = [], []
        with torch.no_grad():
            for i in range(len(ds_one)):
                x, y = ds_one[i]
                pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0).numpy()
                preds.append(pred); gts.append(y.numpy())
        if not preds: continue
        pred = np.concatenate(preds, 0); gt = np.concatenate(gts, 0)
        mae = float(np.linalg.norm(pred - gt, axis=1).mean())
        per_path[pid] = {"mae": mae, "n": int(len(gt)),
                          "pred": pred.tolist(), "gt": gt.tolist()}
        print(f"  path_{pid}: MAE {mae:.3f}m  n={len(gt)}", flush=True)

    # Save
    save_dir = ROOT / "runs" / "main_table" / "imuwifine" / "imuwifine_baseline"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"input_dim": input_dim, "hidden_dim": 256, "output_dim": 2,
                    "n_layers": 4, "dropout": 0.2},
        "history": history, "best_val_mae": best_val,
    }, save_dir / "model.pt")
    (save_dir / "per_path_test.json").write_text(json.dumps(per_path, indent=2))
    print(f"\nSaved model.pt + per_path_test.json under {save_dir}")


if __name__ == "__main__":
    main()
