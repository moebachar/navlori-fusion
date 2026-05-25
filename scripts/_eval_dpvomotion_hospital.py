"""PLAN_08 Step 3 — DPVOMotionEncoder on TartanAir hospital P000.

The encoder ships frozen Webots-trained backbone + correlation; the
position head from RESULT_03 was trained on Webots and is NOT saved
to a domain-portable checkpoint. To produce a comparable ATE on
hospital_P000 we:

1. Run the **frozen** trunk + correlation on all hospital pairs to
   get per-pair motion tokens ``(N, 64, 132)``.
2. Train a tiny linear head **on hospital_P000 itself** (first 80 %
   of frames as train, last 20 % as test) to map per-pair tokens →
   per-pair (Δx, Δy) GT motion (NED, taken from `pose_left.txt`).
3. Integrate predicted Δ-motion forward from pose[0] → trajectory →
   Umeyama-aligned ATE on the last-20 % test slice.

This is NOT Mode α (out-of-domain transfer test). The plan's Mode α
needed a Webots-trained head checkpoint we don't have. **Mode 3-prime**
(this script's mode): in-domain head, frozen encoder, tests whether
the FROZEN backbone+correlation provides usable motion information
on TartanAir-style RGB. This is the **transferability of the trunk**,
not the head — the right question for a frozen pretrained
backbone.

Run: ``.venv/Scripts/python.exe scripts/_eval_dpvomotion_hospital.py``
"""
from __future__ import annotations

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

from PIL import Image  # noqa: E402

from src.pipeline.encoders.dpvo_motion import DPVOMotionEncoder  # noqa: E402

SEQ_ROOT = ROOT / "data" / "tartanair_hospital" / "P000"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_08"


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_imagenet(rgb_path: str, target_hw=(480, 640)) -> np.ndarray:
    img = Image.open(rgb_path).convert("RGB").resize((target_hw[1], target_hw[0]))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a.transpose(2, 0, 1)  # (3, H, W)
    a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return a


def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return ((s * (R @ src.T)).T + t).astype(np.float32), float(s)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # === load encoder ===
    print("loading DPVOMotionEncoder (Webots-pretrained trunk)...", flush=True)
    enc = DPVOMotionEncoder(weights_path=str(ROOT / "runs" / "_weights" / "dpvo.pth"))
    enc.to(dev).eval()
    n_trunk = sum(p.numel() for p in enc.trunk.parameters())
    print(f"  trunk params: {n_trunk/1e6:.2f} M (frozen)", flush=True)

    # === load hospital frames + GT poses ===
    pose_file = SEQ_ROOT / "pose_left.txt"
    img_dir = SEQ_ROOT / "image_left"
    poses = np.loadtxt(pose_file)
    img_files = sorted(img_dir.glob("*.png"))
    n_frames = min(len(poses), len(img_files))
    print(f"hospital P000: {n_frames} frames", flush=True)

    # === extract tokens for all pairs (stride=1, since TartanAir is 30 FPS) ===
    n_pairs = n_frames - 1
    print(f"extracting tokens for {n_pairs} pairs...", flush=True)
    t0 = time.time()
    tokens_list = []
    batch = 4
    with torch.no_grad():
        for i in range(0, n_pairs, batch):
            chunk_idx = list(range(i, min(i + batch, n_pairs)))
            prev_imgs = np.stack([load_image_imagenet(str(img_files[j])) for j in chunk_idx])
            curr_imgs = np.stack([load_image_imagenet(str(img_files[j + 1])) for j in chunk_idx])
            prev_t = torch.tensor(prev_imgs, device=dev)
            curr_t = torch.tensor(curr_imgs, device=dev)
            x = torch.stack([prev_t, curr_t], dim=1)  # (B, 2, 3, H, W)
            tok = enc._frozen_tokens(x)  # (B, 64, 132)
            tokens_list.append(tok.cpu().numpy())
            if (i // batch) % 25 == 0:
                print(f"  pair {i}/{n_pairs} ({time.time()-t0:.1f}s)", flush=True)
    tokens = np.concatenate(tokens_list, axis=0)  # (n_pairs, 64, 132)
    extract_s = time.time() - t0
    print(f"extracted {tokens.shape} in {extract_s:.1f}s "
          f"({extract_s*1000/max(1, n_pairs):.1f} ms/pair)", flush=True)

    # === per-pair GT motion (Δx, Δy, Δz) ===
    delta_xyz = poses[1:n_frames, :3] - poses[:n_frames - 1, :3]  # (n_pairs, 3)

    # === train tiny linear head: tokens (mean-pooled over patches) -> (Δx, Δy, Δz) ===
    # Train on first 80 %, test on last 20 %.
    n_train = int(0.8 * n_pairs)
    Xtr = tokens[:n_train].mean(axis=1)        # (n_train, 132)
    Ytr = delta_xyz[:n_train]                  # (n_train, 3)
    Xte = tokens[n_train:].mean(axis=1)
    Yte = delta_xyz[n_train:]

    # Normalise X using train mean/std.
    mu_x = Xtr.mean(0); sd_x = Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - mu_x) / sd_x
    Xte_n = (Xte - mu_x) / sd_x
    mu_y = Ytr.mean(0)
    Ytr_c = Ytr - mu_y
    Yte_c = Yte - mu_y

    # Linear head trained with closed-form least squares.
    A = np.linalg.lstsq(Xtr_n, Ytr_c, rcond=None)[0]  # (132, 3)
    pred_test = Xte_n @ A + mu_y  # (n_test, 3)
    per_pair_err = np.linalg.norm(pred_test - Yte, axis=1)
    delta_mae = float(per_pair_err.mean())
    print(f"\nlinear-head Δ-motion (3-D):", flush=True)
    print(f"  per-pair MAE  = {delta_mae:.5f} m  (mean over {len(pred_test)} test pairs)",
          flush=True)
    print(f"  GT per-pair motion magnitude mean = {np.linalg.norm(Yte, axis=1).mean():.5f} m",
          flush=True)

    # === integrate predicted Δ-motion into trajectory ===
    # Test trajectory starts at GT pose[n_train].
    start_xyz = poses[n_train, :3]
    traj = np.zeros((len(pred_test) + 1, 3), dtype=np.float32)
    traj[0] = start_xyz
    for i in range(len(pred_test)):
        traj[i + 1] = traj[i] + pred_test[i]
    gt_traj = poses[n_train: n_train + len(pred_test) + 1, :3]

    # === Umeyama-aligned ATE on the test trajectory ===
    aligned, scale = _umeyama_align(traj.astype(np.float64), gt_traj.astype(np.float64))
    errs = np.linalg.norm(aligned - gt_traj, axis=1)
    ate_rmse = float(np.sqrt((errs ** 2).mean()))
    ate_mean = float(errs.mean())
    ate_median = float(np.median(errs))
    ate_p90 = float(np.percentile(errs, 90))
    ate_max = float(errs.max())
    print(f"\nDPVOMotion (frozen trunk + linear hospital head) ATE:", flush=True)
    print(f"  Umeyama-aligned RMSE = {ate_rmse:.4f} m", flush=True)
    print(f"  mean   = {ate_mean:.4f} m", flush=True)
    print(f"  median = {ate_median:.4f} m", flush=True)
    print(f"  p90    = {ate_p90:.4f} m", flush=True)
    print(f"  max    = {ate_max:.4f} m", flush=True)
    print(f"  scale  = {scale:.4f}", flush=True)
    print(f"  test pairs = {len(pred_test)} (last 20 % of P000)", flush=True)

    # === Per-pair latency on encoder (b=1) ===
    enc.eval()
    x = torch.zeros(1, 2, 3, 480, 640, device=dev)
    with torch.no_grad():
        for _ in range(5):
            _ = enc._frozen_tokens(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            _ = enc._frozen_tokens(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / 30 * 1000.0
    print(f"\nencoder latency (b=1, frozen tokens only): {latency_ms:.2f} ms/pair", flush=True)

    out = {
        "method": "DPVOMotionEncoder frozen trunk + linear head trained on hospital_P000",
        "dataset": "TartanAir hospital P000 (test split = last 20 %)",
        "n_frames": int(n_frames),
        "n_pairs_total": int(n_pairs),
        "n_pairs_train": int(n_train),
        "n_pairs_test": int(len(pred_test)),
        "extraction_elapsed_s": float(extract_s),
        "extraction_ms_per_pair": float(extract_s * 1000 / max(1, n_pairs)),
        "encoder_latency_ms_b1": float(latency_ms),
        "linear_head_delta_mae_m": float(delta_mae),
        "gt_motion_magnitude_mean_m": float(np.linalg.norm(Yte, axis=1).mean()),
        "ate_umeyama": {
            "rmse_m": ate_rmse, "mean_m": ate_mean, "median_m": ate_median,
            "p90_m": ate_p90, "max_m": ate_max, "scale": float(scale),
        },
    }
    with open(OUT_DIR / "dpvomotion_hospital.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savetxt(str(OUT_DIR / "dpvomotion_hospital_aligned_gt.txt"), gt_traj)
    np.savetxt(str(OUT_DIR / "dpvomotion_hospital_aligned_est.txt"), aligned)
    print(f"\nwrote {OUT_DIR / 'dpvomotion_hospital.json'}", flush=True)


if __name__ == "__main__":
    main()
