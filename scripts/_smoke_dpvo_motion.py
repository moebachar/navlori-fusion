"""DPVO motion-encoder dev harness — iterative build/profile.

Phase 1 (this file, --phase 1): synthetic-shift sanity for the windowed
soft-argmax. Shift a real frame by a known pixel offset, run the encoder's
frozen path, and check the recovered per-patch flow matches the shift.

Later phases hook in here as they're built.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.encoders.dpvo_motion import DPVOMotionEncoder   # noqa: E402
from src.pipeline.training.motion import (                        # noqa: E402
    build_motion_pairs, cache_tokens, DeltaRegressor,
    train_delta_head, deadreckon_path, profile_deltas,
)

DATA_ROOT = ROOT / "data/async_collection"
WEIGHTS = ROOT / "runs/_weights/dpvo.pth"
STRIDE = 5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _imagenet_norm(rgb01: np.ndarray) -> torch.Tensor:
    """(H,W,3) float[0,1] -> (3,H,W) ImageNet-normalised tensor."""
    x = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).float()


def _flow_of(enc, frame_a: np.ndarray, frame_b: np.ndarray, device) -> tuple:
    """Run the frozen path on a frame pair; return (flow Nx2, sharp N)."""
    pair = torch.stack([_imagenet_norm(frame_a), _imagenet_norm(frame_b)], dim=0)
    pair = pair.unsqueeze(0).to(device)                      # (1, 2, 3, H, W)
    with torch.no_grad():
        tokens = enc._frozen_tokens(pair)                    # (1, N, 132)
    return (tokens[0, :, 128:130].cpu().numpy(),
            tokens[0, :, 131].cpu().numpy())


def phase1_synthetic_shift() -> bool:
    """Verify the encoder recovers known image shifts.

    Two checks:
      (a) identity — frame_b == frame_a → flow must be ~0;
      (b) clean crop shifts — both frames are crops of the SAME larger
          image at different offsets (no np.roll wrap seam to lock onto).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[p1] device={device}")

    enc = DPVOMotionEncoder(weights_path=ROOT / "runs/_weights/dpvo.pth").to(device)
    enc.eval()

    import pandas as pd
    path_dir = ROOT / "data/async_collection/path_02"
    cam_csv = pd.read_csv(path_dir / "camera.csv")
    frame_path = path_dir / cam_csv.iloc[100]["rgb_path"]    # RGB, not depth
    img = np.array(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
    H, W, _ = img.shape
    print(f"[p1] frame {frame_path.name}  ({W}x{H})")

    ok_all = True

    def _judge(label, flow, sharp, exp):
        """Judge recovery on the CONFIDENT half of patches.

        Uniform-region patches are genuinely ambiguous (no localisable
        texture) — the encoder's attentive pool is meant to discount them
        via the `sharp` feature, so the pass criterion is whether the
        high-sharpness patches recover the true shift.
        """
        nonlocal ok_all
        hi = sharp >= np.median(sharp)                       # confident half
        med = np.median(flow[hi], axis=0)
        err = float(np.linalg.norm(med - exp))
        near = float(np.mean(np.linalg.norm(flow[hi] - exp, axis=1) < 2.0))
        status = "OK " if err < 1.5 else "BAD"
        ok_all &= err < 1.5
        print(f"[p1] {label}  expect={exp.round(2)}  hi-sharp median={med.round(2)}  "
              f"err={err:.2f}  near={near:.0%}  "
              f"sharp(hi)~{sharp[hi].mean():.3f}  [{status}]")

    # (a) identity check
    flow, sharp = _flow_of(enc, img, img, device)
    _judge("identity        ", flow, sharp, np.array([0.0, 0.0]))

    # (b) clean crop shifts — crop CW x CH from base offset, no wrap.
    CW, CH = 448, 320
    ox, oy = 96, 80                                          # base crop offset
    for sx, sy in [(40, 0), (0, 24), (32, 16), (-28, 12), (-16, -20)]:
        a = img[oy:oy + CH, ox:ox + CW]
        b = img[oy + sy:oy + sy + CH, ox + sx:ox + sx + CW]
        flow, sharp = _flow_of(enc, a, b, device)
        # frame_b content is frame_a shifted by (-sx,-sy): a feature at
        # column c in crop-a sits at c-sx in crop-b → flow ≈ -(sx,sy)/4.
        exp = np.array([-sx / 4.0, -sy / 4.0])
        _judge(f"crop d=({sx:+3d},{sy:+3d})", flow, sharp, exp)

    print(f"[p1] {'PASS' if ok_all else 'FAIL'} — windowed soft-argmax "
          f"{'recovers' if ok_all else 'does NOT recover'} known shifts")
    return ok_all


def phase2_pairs_and_cache() -> bool:
    """Build pairs + cache frozen tokens for one path; sanity-check stats."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DPVOMotionEncoder(weights_path=WEIGHTS).to(device).eval()

    pdir = DATA_ROOT / "path_02"
    pairs = build_motion_pairs(pdir, STRIDE)
    deltas = np.array([p.delta for p in pairs])
    print(f"[p2] path_02: {len(pairs)} pairs (stride={STRIDE})")
    print(f"[p2] delta mean={deltas.mean(0).round(3)}  std={deltas.std(0).round(3)}  "
          f"|d| mean={np.linalg.norm(deltas,axis=1).mean():.3f}m  "
          f"max={np.linalg.norm(deltas,axis=1).max():.3f}m")

    tokens, dts, _ = cache_tokens(enc, [pdir], STRIDE, device, verbose=False)
    flow = tokens[:, :, 128:130]                      # (N, P, 2) feature px
    sharp = tokens[:, :, 131]
    print(f"[p2] tokens {tuple(tokens.shape)}  "
          f"flow |.| mean={flow.norm(dim=-1).mean():.2f}  "
          f"sharp mean={sharp.mean():.3f}")
    ok = (tokens.shape[0] == len(pairs) and tokens.shape[2] == 132
          and torch.isfinite(tokens).all())
    print(f"[p2] {'PASS' if ok else 'FAIL'} — pair building + token cache")
    return bool(ok)


def phase3_overfit_one_path() -> bool:
    """Overfit a single path (train==val): loss must collapse, drift tiny."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DPVOMotionEncoder(weights_path=WEIGHTS).to(device).eval()
    pdir = DATA_ROOT / "path_01"

    tokens, deltas, _ = cache_tokens(enc, [pdir], STRIDE, device, verbose=False)
    print(f"[p3] path_01: {tokens.shape[0]} pairs")

    reg = DeltaRegressor(enc.head, embed_dim=enc.embed_dim)
    hist = train_delta_head(
        reg, tokens, deltas, tokens, deltas,        # train == val (overfit)
        epochs=80, lr=2e-3, patience=80, device=device, verbose=False,
    )
    print(f"[p3] overfit: train_loss {hist.train_loss[0]:.4f} -> "
          f"{hist.train_loss[-1]:.4f}   best val_MAE={hist.best_val_mae:.3f}m")

    dr = deadreckon_path(enc, reg, pdir, STRIDE, device)
    print(f"[p3] dead-reckon path_01: mean_err={dr['mean_err']:.3f}m  "
          f"final_drift={dr['final_drift']:.3f}m  path_len={dr['path_len']:.1f}m")

    # Overfitting one path should drive Δ-MAE well below the motion scale.
    dmae = hist.best_val_mae
    scale = np.linalg.norm(deltas.numpy(), axis=1).mean()
    ok = dmae < 0.4 * scale
    print(f"[p3] {'PASS' if ok else 'FAIL'} — Δ-MAE {dmae:.3f}m vs "
          f"motion scale {scale:.3f}m (need < {0.4*scale:.3f})")
    return bool(ok)


TRAIN_PATHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VAL_PATHS = [2, 13, 14]


def phase4_full_train_and_profile() -> bool:
    """Train on the real split, profile Δ predictions + dead-reckon val."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DPVOMotionEncoder(weights_path=WEIGHTS).to(device).eval()

    tr_dirs = [DATA_ROOT / f"path_{p:02d}" for p in TRAIN_PATHS]
    va_dirs = [DATA_ROOT / f"path_{p:02d}" for p in VAL_PATHS]

    print("[p4] caching train tokens ...")
    tok_tr, dlt_tr, _ = cache_tokens(enc, tr_dirs, STRIDE, device, verbose=False)
    print("[p4] caching val tokens ...")
    tok_va, dlt_va, pairs_va = cache_tokens(enc, va_dirs, STRIDE, device, verbose=False)
    print(f"[p4] train={tok_tr.shape[0]} pairs   val={tok_va.shape[0]} pairs")

    reg = DeltaRegressor(enc.head, embed_dim=enc.embed_dim)
    hist = train_delta_head(reg, tok_tr, dlt_tr, tok_va, dlt_va,
                            epochs=150, lr=1e-3, patience=30, device=device)

    # --- Δ-prediction profiling on val ---
    reg.eval()
    with torch.no_grad():
        pred_va = reg(tok_va.to(device)).cpu().numpy()
    prof = profile_deltas(pred_va, dlt_va.numpy())
    print(f"[p4] Δ-profile  MAE={prof['mae']:.3f}m  RMSE={prof['rmse']:.3f}m  "
          f"bias={np.round(prof['bias'],3).tolist()}  "
          f"scale={prof['scale']:.3f}  dir_err={prof['dir_err_deg']:.1f}deg")

    # --- dead-reckon each val path from a single GT anchor ---
    drifts = []
    for p in VAL_PATHS:
        dr = deadreckon_path(enc, reg, DATA_ROOT / f"path_{p:02d}", STRIDE, device)
        rel = dr["final_drift"] / max(dr["path_len"], 1e-6)
        drifts.append(dr["mean_err"])
        print(f"[p4] dead-reckon path_{p:02d}: mean_err={dr['mean_err']:.2f}m  "
              f"final_drift={dr['final_drift']:.2f}m  "
              f"path_len={dr['path_len']:.1f}m  ({rel:.1%} of path)")

    mean_drift = float(np.mean(drifts))
    # "Working" target for this phase: mean trajectory error under ~2 m.
    ok = mean_drift < 2.0
    print(f"[p4] {'PASS' if ok else 'NEEDS WORK'} — mean val trajectory "
          f"error {mean_drift:.2f}m")
    return bool(ok)


def phase5_per_path_root_cause() -> bool:
    """Train once, then dissect each val path: Δ-error, flow-clipping,
    and whether drift is just the expected random walk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DPVOMotionEncoder(weights_path=WEIGHTS).to(device).eval()
    sr = enc.search_radius

    tr_dirs = [DATA_ROOT / f"path_{p:02d}" for p in TRAIN_PATHS]
    tok_tr, dlt_tr, _ = cache_tokens(enc, tr_dirs, STRIDE, device, verbose=False)
    va_dirs = [DATA_ROOT / f"path_{p:02d}" for p in VAL_PATHS]
    tok_va, dlt_va, _ = cache_tokens(enc, va_dirs, STRIDE, device, verbose=False)

    reg = DeltaRegressor(enc.head, embed_dim=enc.embed_dim)
    train_delta_head(reg, tok_tr, dlt_tr, tok_va, dlt_va,
                     epochs=150, lr=1e-3, patience=30, device=device,
                     verbose=False)
    reg.eval()

    print(f"[p5] search_radius={sr} cells  stride={STRIDE}")
    for p in VAL_PATHS:
        pdir = DATA_ROOT / f"path_{p:02d}"
        tok, dlt, _ = cache_tokens(enc, [pdir], STRIDE, device, verbose=False)
        with torch.no_grad():
            pred = reg(tok.to(device)).cpu().numpy()
        prof = profile_deltas(pred, dlt.numpy())

        # Flow clipping: fraction of patches whose |flow| is within 1 cell
        # of the search-radius cap → their true match may be cut off.
        flow_mag = tok[:, :, 128:130].norm(dim=-1)            # (N, P)
        clipped = float((flow_mag > sr - 1.0).float().mean())

        dr = deadreckon_path(enc, reg, pdir, STRIDE, device)
        n_seg = len(dr["pred"]) - 1
        rw = prof["mae"] * (n_seg ** 0.5)                     # random-walk predict
        print(f"[p5] path_{p:02d}: Δ-MAE={prof['mae']:.3f}m  "
              f"dir={prof['dir_err_deg']:.1f}deg  clip={clipped:.1%}  "
              f"| {n_seg} segs  mean_err={dr['mean_err']:.2f}m  "
              f"drift={dr['final_drift']:.2f}m  (rand-walk≈{rw:.2f}m)")

    print("[p5] done — inspect Δ-MAE vs clip% to see if drift is "
          "irreducible random walk or a fixable cause")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()

    phases = {1: phase1_synthetic_shift,
              2: phase2_pairs_and_cache,
              3: phase3_overfit_one_path,
              4: phase4_full_train_and_profile,
              5: phase5_per_path_root_cause}
    if args.phase not in phases:
        raise SystemExit(f"phase {args.phase} not implemented yet")
    ok = phases[args.phase]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
