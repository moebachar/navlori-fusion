"""IMUWiFine fusion baseline — clean-room reimplementation (PLAN_40).

Based on Nurpeiissov, Kuzdeuov, Assylkhanov, Khassanov & Varol, "End-to-End
Sequential Indoor Localization Using Smartphone Inertial Sensors and WiFi",
IEEE/SICE SII 2022 (DOI 10.1109/SII52469.2022.9708854). The reference
implementation at https://github.com/IS2AI/IMUWiFine has no license file,
so we re-implement the published architecture from the paper text here
(Demand #3 compliant: no vendored source imported).

Architecture (as described in the paper):
    input  = concat(WiFi RSSI vector, IMU vector)
    stack  = 4 × (Linear + ReLU)  with hidden_dim
    seq    = 4-layer LSTM (batch_first, dropout)
    head   = Linear(hidden_dim, output_dim)

Faithful reproduction policy (user directive 2026-06-04):
- No target normalization beyond what the reference dataset.py does
  (i.e. NONE — model predicts raw position in metres).
- Reference recipe: AdamW lr=1e-3, MSE loss, no LR warmup, dropout 0.2.

MSILN adaptations (the only changes vs reference, all due to MSILN ≠
IMUWiFine raw format):
- input_dim: 229 (paper, 220 WiFi + 9 IMU incl magn) → 1425 (MSILN, 1419
  WiFi + 6 IMU; MSILN site1/B1 has no magnetometer).
- hidden_dim: paper's hidden==input==229 is too large at MSILN's 1425
  (≈30M params with 4-layer LSTM). We use hidden_dim=256, the smallest
  reasonable substitute. This is an MSILN-tractability adaptation, not
  a fix.
- output_dim: 3 (paper) → 2 (MSILN is 2D).
- Window size: 300 samples @ 100 Hz (paper) → 30 samples @ 10 Hz (MSILN).
  Both correspond to 3 s contexts.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ._msiln_loader import (
    TRAIN_PATHS, VAL_PATHS, TEST_PATHS,
    load_ap_vocab, load_msiln_paths_for_imuwifine,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IMUWiFineModel(nn.Module):
    """LSTM-fusion of WiFi + IMU → continuous (x, y) regression.

    Clean-room reimplementation of Nurpeiissov et al. 2022. Architecture
    spec is verbatim from the paper; only the hidden_dim downsample is
    an MSILN-adaptation (see module docstring).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 output_dim: int = 2, n_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # 4× Linear+ReLU stack (paper: linear_in1..4 with hidden_dim==input_dim)
        # We downsample to hidden_dim first to keep LSTM tractable on MSILN.
        self.linear_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                              num_layers=n_layers, dropout=dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, T, output_dim)."""
        x = self.dropout_layer(x)
        x = self.linear_in(x)
        seq_out, _ = self.lstm(x)
        return self.head(seq_out)


# ---------------------------------------------------------------------------
# Data preparation (resampled MSILN → fixed-window batches)
# ---------------------------------------------------------------------------

def _normalize_wifi(rssi: np.ndarray, no_signal_dbm: float = -100.0) -> np.ndarray:
    """Paper's normalize_wifi: shift then divide then ^e.

        x += 90;  x /= 90;  x **= e

    Adapted for MSILN's -100 not-detected sentinel: shift by abs(no_signal).
    """
    x = rssi.astype(np.float32) + abs(no_signal_dbm)  # -> [0, 100]
    x /= abs(no_signal_dbm)                            # -> [0, 1]
    x = np.power(x, math.e)
    return x


def _normalize_imu_inplace(imu: np.ndarray) -> np.ndarray:
    """Min-max normalize each IMU dim to [0, 1]. Stable across MSILN scale."""
    imu = imu.astype(np.float32).copy()
    for k in range(imu.shape[1]):
        v = imu[:, k]
        mn, mx = float(v.min()), float(v.max())
        if mx > mn:
            imu[:, k] = (v - mn) / (mx - mn)
    return imu


@dataclass
class _MsilnWindowDataset:
    """In-memory window slicer over the resampled MSILN paths.

    Matches the IMUWiFine reference dataset.py exactly: raw position
    targets in metres, no normalization. The model predicts raw (x, y).
    """
    paths: list[dict]
    window: int
    stride: int

    def __post_init__(self):
        self.windows = []
        for p in self.paths:
            T = len(p["t"])
            for s in range(0, T - self.window + 1, self.stride):
                self.windows.append((p, s))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        p, s = self.windows[idx]
        e = s + self.window
        wifi = _normalize_wifi(p["wifi"][s:e])
        imu  = _normalize_imu_inplace(p["imu"][s:e])
        x = np.concatenate([wifi, imu], axis=1)
        y = p["gt"][s:e]  # raw (x, y) in metres — matches the reference recipe
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float())


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def train_imuwifine_msiln(
    *,
    epochs: int = 60,
    batch_size: int = 8,
    window: int = 30,
    stride: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    dropout: float = 0.2,
    lr: float = 1e-3,
    target_hz: float = 10.0,
    seed: int = 42,
    device: str | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[IMUWiFineModel, dict, dict]:
    """Train the IMUWiFine LSTM-fusion baseline on MSILN site1/B1 cross-session.

    Returns ``(model, history, summary)`` where ``summary`` includes per-split
    MAE in meters. Saves checkpoint to ``save_dir/model.pt`` if provided.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    ap_vocab = load_ap_vocab()
    n_aps = len(ap_vocab)

    if verbose:
        print(f"[IMUWiFine] Loading MSILN train/val/test (target_hz={target_hz})...", flush=True)
    train_paths, _ = load_msiln_paths_for_imuwifine(TRAIN_PATHS, ap_vocab, target_hz=target_hz)
    val_paths,   _ = load_msiln_paths_for_imuwifine(VAL_PATHS,   ap_vocab, target_hz=target_hz)
    test_paths,  _ = load_msiln_paths_for_imuwifine(TEST_PATHS,  ap_vocab, target_hz=target_hz)

    if verbose:
        print(f"[IMUWiFine] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} paths",
               flush=True)

    train_ds = _MsilnWindowDataset(train_paths, window=window, stride=stride)
    val_ds   = _MsilnWindowDataset(val_paths,   window=window, stride=window)
    if verbose:
        print(f"[IMUWiFine] train windows={len(train_ds)} val windows={len(val_ds)}",
               flush=True)

    input_dim = n_aps + 6  # WiFi + IMU(6, no magn)
    model = IMUWiFineModel(input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=2, n_layers=n_layers, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[IMUWiFine] params={n_params / 1e6:.2f} M  input_dim={input_dim}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train_loss": [], "val_mae": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        # Training pass — raw position targets, original IMUWiFine recipe
        model.train()
        perm = np.random.permutation(len(train_ds))
        ep_loss = 0.0
        n_seen = 0
        for s in range(0, len(perm), batch_size):
            batch_idx = perm[s:s + batch_size]
            tuples = [train_ds[i] for i in batch_idx]
            x = torch.stack([t[0] for t in tuples]).to(device)
            y = torch.stack([t[1] for t in tuples]).to(device)
            pred = model(x)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
        sched.step()
        history["train_loss"].append(ep_loss / max(n_seen, 1))

        # Val MAE in meters
        model.eval()
        val_err = []
        with torch.no_grad():
            for i in range(len(val_ds)):
                x, y = val_ds[i]
                x = x.unsqueeze(0).to(device)
                pred = model(x).cpu().squeeze(0).numpy()
                err = np.linalg.norm(pred - y.numpy(), axis=1)
                val_err.append(err)
        val_mae = float(np.concatenate(val_err).mean()) if val_err else float("nan")
        history["val_mae"].append(val_mae)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep <= 2 or ep % 10 == 0 or ep == epochs - 1):
            elapsed = time.time() - t0
            print(f"[IMUWiFine] ep {ep:3d}/{epochs}  train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_mae={val_mae:.3f}m  (best {best_val:.3f}m, {elapsed:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    if verbose:
        print(f"[IMUWiFine] Training done in {elapsed:.0f}s. Best val MAE: {best_val:.3f} m", flush=True)

    # Final eval on all splits — Euclidean error in meters
    summary = {"best_val_mae": best_val, "elapsed_s": elapsed,
                "n_params": n_params, "input_dim": input_dim,
                "n_train_paths": len(train_paths), "n_val_paths": len(val_paths),
                "n_test_paths": len(test_paths)}
    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        ds = _MsilnWindowDataset(paths, window=window, stride=window)
        errs = []
        with torch.no_grad():
            for i in range(len(ds)):
                x, y = ds[i]
                x = x.unsqueeze(0).to(device)
                pred = model(x).cpu().squeeze(0).numpy()
                errs.append(np.linalg.norm(pred - y.numpy(), axis=1))
        if errs:
            errs = np.concatenate(errs)
            summary[f"{split_name}_mae"] = float(errs.mean())
            summary[f"{split_name}_n"] = int(len(errs))
        else:
            summary[f"{split_name}_mae"] = float("nan")
            summary[f"{split_name}_n"] = 0

    if verbose:
        print(f"[IMUWiFine] Final: train_mae={summary.get('train_mae', float('nan')):.3f}m  "
              f"val_mae={summary['val_mae']:.3f}m  test_mae={summary['test_mae']:.3f}m",
               flush=True)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": input_dim, "hidden_dim": hidden_dim,
                "output_dim": 2, "n_layers": n_layers, "dropout": dropout,
                "window": window, "stride": stride, "target_hz": target_hz,
            },
            "history": history,
            "summary": summary,
        }
        torch.save(ckpt, save_dir / "model.pt")
        if verbose:
            print(f"[IMUWiFine] Saved {save_dir / 'model.pt'}", flush=True)

    return model, history, summary


@torch.no_grad()
def predict_imuwifine_msiln(model: IMUWiFineModel, *, split: str = "test",
                             window: int = 30, target_hz: float = 10.0,
                             device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run IMUWiFine on the requested MSILN split; return ``(pred, gt)`` arrays.

    Aggregates per-window predictions and matched GT into flat arrays
    of shape ``(N_samples, 2)`` for plotting / metric computation.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths_map = {"train": TRAIN_PATHS, "val": VAL_PATHS, "test": TEST_PATHS}
    paths, _ = load_msiln_paths_for_imuwifine(paths_map[split], target_hz=target_hz)
    ds = _MsilnWindowDataset(paths, window=window, stride=window)
    model.eval().to(device)
    preds, gts = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        pred = model(x).cpu().squeeze(0).numpy()
        preds.append(pred); gts.append(y.numpy())
    if not preds:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)


def load_imuwifine_msiln(ckpt_path: str | Path) -> tuple[IMUWiFineModel, dict, dict]:
    """Load a saved IMUWiFine-MSILN checkpoint.

    Returns ``(model, history, summary)``. The ``summary`` dict carries
    ``target_mu`` / ``target_std`` so eval can un-normalize predictions.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = IMUWiFineModel(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"], n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["history"], ckpt["summary"]


__all__ = [
    "IMUWiFineModel",
    "train_imuwifine_msiln",
    "load_imuwifine_msiln",
    "predict_imuwifine_msiln",
]
