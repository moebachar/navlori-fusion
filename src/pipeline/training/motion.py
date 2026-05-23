"""Motion-encoder training, dead-reckoning and profiling.

A motion encoder (``DPVOMotionEncoder``) maps a frame *pair* to a
world-frame displacement ``(dx, dy)``. This module turns that into a
usable, dataset-agnostic pipeline:

* :func:`build_motion_pairs` — from any ``async_collection`` path with
  camera frames, form one ``(frame[t-stride], frame[t])`` sample per
  camera frame, target = GT displacement over that span (GT interpolated
  to camera timestamps).
* :func:`cache_tokens` — run the encoder's frozen path once, cache the
  ``(N, n_patches, 132)`` per-patch motion tokens (everything before the
  trainable head is frozen / parameter-free).
* :class:`DeltaRegressor` — the trainable head: ``encoder.head`` +
  a linear ``(embed_dim -> 2)`` displacement layer.
* :func:`train_delta_head` — standardised-target Huber regression.
* :func:`deadreckon_path` — walk a path's camera frames in
  non-overlapping ``stride`` segments, accumulating predicted
  displacements from a single GT anchor (the online use-case).
* :func:`profile_deltas` — bias / scale / direction diagnostics.

Why world-frame ``(dx, dy)`` and not ego-motion: a world-frame delta is
re-derived per frame, so dead-reckoning has *no heading-integration
drift* — only a translation random-walk. Ego-motion would compound
heading error. Target needs only GT positions, no ``theta``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data — frame pairs
# ---------------------------------------------------------------------------

@dataclass
class MotionPair:
    """One training sample: a camera frame pair + its GT world displacement."""
    path_id: int
    i_prev: int                 # camera-frame index of the earlier frame
    i_curr: int                 # camera-frame index of the later frame
    rgb_prev: str               # rel path under the path dir
    rgb_curr: str
    delta: tuple[float, float]  # world-frame (dx, dy) over the span
    t_prev: float               # sim_time of the earlier frame
    t_curr: float


def build_motion_pairs(path_dir: str | Path, stride: int = 5) -> list[MotionPair]:
    """Form ``(frame[t-stride], frame[t])`` pairs for one path.

    Target displacement = ``GT(t) - GT(t-stride)``, with GT ``(x, y)``
    linearly interpolated from ``ground_truth.csv`` onto the camera
    timestamps (GT ~10 Hz, camera ~5 Hz — interpolation error is sub-cm).
    One pair per camera frame ``t >= stride``.
    """
    path_dir = Path(path_dir)
    cam = pd.read_csv(path_dir / "camera.csv")
    gt = pd.read_csv(path_dir / "ground_truth.csv")
    if len(cam) <= stride or len(gt) == 0:
        return []

    cam_t = cam["sim_time"].to_numpy(dtype=np.float64)
    gt_t = gt["sim_time"].to_numpy(dtype=np.float64)
    gx = np.interp(cam_t, gt_t, gt["gt_x"].to_numpy(dtype=np.float64))
    gy = np.interp(cam_t, gt_t, gt["gt_y"].to_numpy(dtype=np.float64))
    pid = int(gt["path_id"].iloc[0]) if "path_id" in gt.columns else -1

    pairs: list[MotionPair] = []
    for t in range(stride, len(cam)):
        pairs.append(MotionPair(
            path_id=pid,
            i_prev=t - stride, i_curr=t,
            rgb_prev=str(cam.iloc[t - stride]["rgb_path"]),
            rgb_curr=str(cam.iloc[t]["rgb_path"]),
            delta=(float(gx[t] - gx[t - stride]),
                   float(gy[t] - gy[t - stride])),
            t_prev=float(cam_t[t - stride]), t_curr=float(cam_t[t]),
        ))
    return pairs


def gt_at_camera_frames(path_dir: str | Path) -> np.ndarray:
    """GT ``(x, y)`` interpolated onto every camera-frame timestamp → (N, 2)."""
    path_dir = Path(path_dir)
    cam = pd.read_csv(path_dir / "camera.csv")
    gt = pd.read_csv(path_dir / "ground_truth.csv")
    cam_t = cam["sim_time"].to_numpy(dtype=np.float64)
    gt_t = gt["sim_time"].to_numpy(dtype=np.float64)
    gx = np.interp(cam_t, gt_t, gt["gt_x"].to_numpy(dtype=np.float64))
    gy = np.interp(cam_t, gt_t, gt["gt_y"].to_numpy(dtype=np.float64))
    return np.stack([gx, gy], axis=1).astype(np.float32)


def _load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def _imagenet_norm(rgb01: np.ndarray) -> torch.Tensor:
    """(H,W,3) float[0,1] → (3,H,W) ImageNet-normalised tensor."""
    x = (rgb01 - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).float()


def _pair_tensor(path_dir: Path, pair: MotionPair) -> torch.Tensor:
    """Load one MotionPair as a (2, 3, H, W) ImageNet-normalised tensor."""
    a = _imagenet_norm(_load_rgb(path_dir / pair.rgb_prev))
    b = _imagenet_norm(_load_rgb(path_dir / pair.rgb_curr))
    return torch.stack([a, b], dim=0)


# ---------------------------------------------------------------------------
# Frozen-token caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_tokens(
    encoder: nn.Module,
    path_dirs: list[Path],
    stride: int,
    device: str = "cuda",
    batch_size: int = 8,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[MotionPair]]:
    """Run the encoder's frozen path over all pairs; cache per-patch tokens.

    Returns
    -------
    tokens : ``(N, n_patches, 132)`` frozen motion tokens.
    deltas : ``(N, 2)`` world-frame GT displacement targets.
    pairs  : the ``MotionPair`` list, aligned with the first two tensors.
    """
    encoder.eval().to(device)
    all_tokens: list[torch.Tensor] = []
    all_pairs: list[MotionPair] = []

    for path_dir in path_dirs:
        path_dir = Path(path_dir)
        pairs = build_motion_pairs(path_dir, stride)
        if not pairs:
            continue
        t0 = time.time()
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            x = torch.stack([_pair_tensor(path_dir, p) for p in chunk])
            tok = encoder._frozen_tokens(x.to(device))      # (B, N, 132)
            all_tokens.append(tok.cpu())
            all_pairs.extend(chunk)
        if verbose:
            print(f"[cache] {path_dir.name}: {len(pairs)} pairs "
                  f"({time.time() - t0:.1f}s)")

    tokens = torch.cat(all_tokens, dim=0)
    deltas = torch.tensor([p.delta for p in all_pairs], dtype=torch.float32)
    return tokens, deltas, all_pairs


# ---------------------------------------------------------------------------
# Trainable head
# ---------------------------------------------------------------------------

class DeltaRegressor(nn.Module):
    """Trainable motion head: per-patch tokens → world-frame ``(dx, dy)``.

    Reuses the encoder's ``_MotionHead`` (per-patch proj → attentive pool
    → MLP → ``embed_dim``) and adds a linear displacement layer. Targets
    are standardised (zero-mean / unit-std per axis) for stable training;
    :meth:`forward` returns *un-standardised* metres.
    """

    def __init__(self, motion_head: nn.Module, embed_dim: int = 128):
        super().__init__()
        self.head = motion_head
        self.delta = nn.Linear(embed_dim, 2)
        # Target standardisation (filled by train_delta_head).
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.head(tokens)                       # (B, embed_dim)
        norm = self.delta(z)                        # standardised (dx, dy)
        return norm * self.y_std + self.y_mean      # metres


@dataclass
class MotionHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)   # metres, ‖Δ‖ error
    best_epoch: int = 0
    best_val_mae: float = float("inf")
    elapsed_sec: float = 0.0


def train_delta_head(
    regressor: DeltaRegressor,
    tokens_tr: torch.Tensor, deltas_tr: torch.Tensor,
    tokens_va: torch.Tensor, deltas_va: torch.Tensor,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    huber_delta: float = 0.1,
    device: str = "cuda",
    patience: int = 20,
    verbose: bool = True,
) -> MotionHistory:
    """Train the delta head on cached tokens with standardised targets."""
    regressor.to(device)

    # Standardise targets on the training split; store in the regressor so
    # forward() returns metres.
    y_mean = deltas_tr.mean(dim=0)
    y_std = deltas_tr.std(dim=0).clamp_min(1e-4)
    regressor.y_mean.copy_(y_mean.to(device))
    regressor.y_std.copy_(y_std.to(device))

    def _norm_targets(d: torch.Tensor) -> torch.Tensor:
        return (d - y_mean) / y_std

    tr = torch.utils.data.TensorDataset(tokens_tr, _norm_targets(deltas_tr))
    va = torch.utils.data.TensorDataset(tokens_va, _norm_targets(deltas_va))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va, batch_size=batch_size)

    opt = torch.optim.AdamW(regressor.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=max(1, len(tr_loader)))
    crit = nn.HuberLoss(delta=huber_delta)

    hist = MotionHistory()
    best_state = None
    bad = 0
    t0 = time.time()

    for ep in range(epochs):
        regressor.train()
        tl = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred_norm = regressor.delta(regressor.head(xb))
            loss = crit(pred_norm, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(xb)
        tl /= len(tr)

        regressor.eval()
        vl = 0.0
        mae = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_norm = regressor.delta(regressor.head(xb))
                vl += crit(pred_norm, yb).item() * len(xb)
                # MAE in metres
                pred_m = pred_norm * regressor.y_std + regressor.y_mean
                gt_m = yb * regressor.y_std + regressor.y_mean
                mae += (pred_m - gt_m).norm(dim=1).sum().item()
        vl /= len(va)
        mae /= len(va)

        hist.train_loss.append(tl)
        hist.val_loss.append(vl)
        hist.val_mae.append(mae)
        if mae < hist.best_val_mae:
            hist.best_val_mae = mae
            hist.best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in regressor.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and (ep % 10 == 0 or ep == epochs - 1 or bad == 0):
            print(f"  ep {ep:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}  "
                  f"val_MAE={mae:.3f}m {'*' if bad == 0 else ''}")
        if bad >= patience:
            if verbose:
                print(f"  early stop @ {ep} (best {hist.best_val_mae:.3f}m)")
            break

    if best_state is not None:
        regressor.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    hist.elapsed_sec = time.time() - t0
    if verbose:
        print(f"  done in {hist.elapsed_sec:.1f}s — best val MAE "
              f"{hist.best_val_mae:.3f}m @ epoch {hist.best_epoch}")
    return hist


# ---------------------------------------------------------------------------
# Dead-reckoning (the online use-case)
# ---------------------------------------------------------------------------

@torch.no_grad()
def deadreckon_path(
    encoder: nn.Module,
    regressor: DeltaRegressor,
    path_dir: str | Path,
    stride: int,
    device: str = "cuda",
) -> dict:
    """Reconstruct a full trajectory from ONE GT anchor + predicted deltas.

    Walks the path's camera frames in **non-overlapping** ``stride``
    segments ``[0, s], [s, 2s], ...`` so the predicted displacements tile
    the path exactly (no double-counting). ``p[0]`` is the GT position at
    camera frame 0; every later position is dead-reckoned. This is exactly
    the online inference scenario.

    Returns a dict with ``pred`` / ``gt`` ``(K, 2)`` arrays at the segment
    endpoints, plus per-step and final-drift errors.
    """
    encoder.eval().to(device)
    regressor.eval().to(device)
    path_dir = Path(path_dir)

    pairs = build_motion_pairs(path_dir, stride)
    segs = sorted((p for p in pairs if p.i_prev % stride == 0),
                  key=lambda p: p.i_prev)
    gt_xy = gt_at_camera_frames(path_dir)                # (N_frames, 2)

    pos = gt_xy[0].astype(np.float64).copy()             # single GT anchor
    pred = [pos.copy()]
    gt = [gt_xy[0].astype(np.float64)]
    frame_idx = [0]

    for seg in segs:
        x = _pair_tensor(path_dir, seg).unsqueeze(0).to(device)
        tok = encoder._frozen_tokens(x)                  # (1, N, 132)
        d = regressor(tok)[0].cpu().numpy().astype(np.float64)
        pos = pos + d
        pred.append(pos.copy())
        gt.append(gt_xy[seg.i_curr].astype(np.float64))
        frame_idx.append(seg.i_curr)

    pred = np.stack(pred)
    gt = np.stack(gt)
    err = np.linalg.norm(pred - gt, axis=1)
    return {
        "pred": pred, "gt": gt, "frame_idx": np.array(frame_idx),
        "err": err,
        "mean_err": float(err.mean()),
        "final_drift": float(err[-1]),
        "path_len": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def track_patches(
    encoder: nn.Module,
    frame_paths: list[str | Path],
    n_track: int = 16,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Track ``n_track`` points across a frame sequence via the encoder's
    correlation — for the patch-correspondence visualisation.

    Points are seeded on a grid in frame 0, then chained: each step takes
    a point's descriptor from the *previous* frame, correlates it against
    the current frame inside the encoder's local search window, and uses
    the windowed soft-argmax peak as the new location.

    Returns
    -------
    coords : ``(T, n_track, 2)`` tracked locations in **image pixels**.
    sharp  : ``(T-1, n_track)`` per-step match sharpness (frame 0 has none).
    """
    import torch.nn.functional as F

    encoder.eval().to(device)
    imgs = torch.stack([_imagenet_norm(_load_rgb(Path(p))) for p in frame_paths])
    fmaps = encoder._trunk_features(
        encoder._to_dpvo_range(imgs.to(device)))             # (T, C, h, w)
    T, _C, h, w = fmaps.shape

    # Seed points on an interior grid (feature-map coords).
    g = max(1, int(round(n_track ** 0.5)))
    ys = torch.linspace(h * 0.2, h * 0.8, g, device=device)
    xs = torch.linspace(w * 0.2, w * 0.8, g, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cur = torch.stack([gx.flatten(), gy.flatten()], dim=-1)[:n_track]  # (K,2)
    cur = cur.unsqueeze(0)                                   # (1, K, 2)

    coords = [cur.squeeze(0).cpu().numpy().copy()]
    sharps = []
    sr = getattr(encoder, "search_radius", 32)
    for t in range(1, T):
        desc = encoder._bilinear_sample(fmaps[t - 1:t], cur)        # (1,K,C)
        desc_n = F.normalize(desc, dim=-1)
        fmap_n = F.normalize(fmaps[t:t + 1], dim=1)                 # (1,C,h,w)
        corr = torch.einsum("bnc,bchw->bnhw", desc_n, fmap_n)
        corr = encoder._mask_to_local_window(corr, cur, sr)
        nxt, sharp = encoder._windowed_soft_argmax(corr, radius=3)  # (1,K,2)
        coords.append(nxt.squeeze(0).cpu().numpy().copy())
        sharps.append(sharp.squeeze(0).cpu().numpy().copy())
        cur = nxt

    res = encoder.OUTPUT_STRIDE                              # feature px → image px
    return np.stack(coords) * res, np.stack(sharps)


def profile_deltas(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-segment delta diagnostics — the root-cause panel.

    * ``mae`` — mean ‖pred − gt‖ (metres).
    * ``bias`` — mean (pred − gt) per axis; non-zero ⇒ systematic offset.
    * ``scale`` — mean‖pred‖ / mean‖gt‖; <1 ⇒ shrinkage (regression to
      the mean), the classic dead-reckoning killer.
    * ``dir_err_deg`` — median angle between pred and gt, over segments
      with real motion (‖gt‖ > 2 cm).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    err = np.linalg.norm(pred - gt, axis=1)
    pn = np.linalg.norm(pred, axis=1)
    gn = np.linalg.norm(gt, axis=1)

    moving = gn > 0.02
    if moving.any():
        cos = ((pred[moving] * gt[moving]).sum(1)
               / (pn[moving] * gn[moving] + 1e-9)).clip(-1, 1)
        dir_err = float(np.degrees(np.arccos(cos)).mean())
    else:
        dir_err = float("nan")

    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": (pred - gt).mean(axis=0).tolist(),
        "scale": float(pn.mean() / (gn.mean() + 1e-9)),
        "dir_err_deg": dir_err,
        "n": int(len(pred)),
    }
