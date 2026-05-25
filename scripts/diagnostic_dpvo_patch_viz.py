"""DPVO patch-viz diagnostic.

Decides whether the patch-tracking centre-collapse we see in the notebook
viz is caused by:

    (a) flat correlation maps (the model genuinely can't match patches
        on Webots imagery)             -> algo problem
    (b) peaked correlation maps that soft-argmax averages out into the
        grid centre                     -> softmax-temperature problem
    (c) sharp & sharp correctly recovered, viz mishandles coords
                                        -> notebook viz bug

Run:

    .venv\\Scripts\\python.exe scripts/diagnostic_dpvo_patch_viz.py

Saves: docs/dpvo_correlation_diagnostic.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Make `src` and `external` importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.encoders import DPVOMotionEncoder      # noqa: E402

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
PATH_ID = 2
N_FRAMES = 5
STRIDE = 5
PATCH_PX = 14

DATA_DIR = ROOT / "data" / "async_collection"
WEIGHTS = ROOT / "runs" / "_weights" / "dpvo.pth"
OUT = ROOT / "docs" / "dpvo_correlation_diagnostic.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[diag] loading encoder from {WEIGHTS}")
encoder = DPVOMotionEncoder(weights_path=str(WEIGHTS))
encoder.to(device).eval()

# --------------------------------------------------------------------------
# Load 5 evenly-spaced frames from the chosen path
# --------------------------------------------------------------------------
pdir = DATA_DIR / f"path_{PATH_ID:02d}"
cam_df = pd.read_csv(pdir / "camera.csv")
anchor = len(cam_df) // 2
indices = [anchor - (N_FRAMES - 1 - k) * STRIDE for k in range(N_FRAMES)]
indices = [max(0, min(len(cam_df) - 1, i)) for i in indices]
print(f"[diag] frames: {indices}  (path {PATH_ID})")

_imn = np.array([0.485, 0.456, 0.406])
_ist = np.array([0.229, 0.224, 0.225])
_tf = T.Compose([
    T.Resize((480, 640)), T.ToTensor(),
    T.Normalize(mean=list(_imn), std=list(_ist)),
])


def _load(idx: int) -> torch.Tensor:
    return _tf(Image.open(pdir / cam_df.iloc[idx]["rgb_path"]).convert("RGB"))


def _denorm(t_chw: torch.Tensor) -> np.ndarray:
    a = t_chw.cpu().numpy().transpose(1, 2, 0)
    return np.clip(a * _ist + _imn, 0, 1)


frames = [_load(i) for i in indices]

# --------------------------------------------------------------------------
# Run the frozen trunk
# --------------------------------------------------------------------------
fmaps = []
with torch.no_grad():
    for f in frames:
        x = encoder._to_dpvo_range(f.unsqueeze(0).to(device))
        fmaps.append(encoder._trunk_features(x))   # (1, 128, 120, 160)

B, C, Hf, Wf = fmaps[0].shape
print(f"[diag] feature grid: {Hf}x{Wf} (stride {encoder.OUTPUT_STRIDE})")

# Pick K=4 patches at well-separated locations in frame 0
K = 4
chosen_xy = torch.tensor([
    [Wf * 0.25, Hf * 0.30],   # upper-left
    [Wf * 0.75, Hf * 0.30],   # upper-right
    [Wf * 0.50, Hf * 0.50],   # centre
    [Wf * 0.50, Hf * 0.80],   # lower-centre
], dtype=torch.float32, device=device).unsqueeze(0)   # (1, K, 2)

descriptors = encoder._bilinear_sample(fmaps[0], chosen_xy)   # (1, K, 128)
print(f"[diag] anchor descriptors {descriptors.shape}")

# --------------------------------------------------------------------------
# Correlate against every frame (no chaining — like the original buggy viz),
# also do CHAINED tracking so we can compare both.
# --------------------------------------------------------------------------
def _correlate(d, fm):
    return torch.einsum("bnc,bchw->bnhw", d, fm) / (C ** 0.5)


def _hard_argmax(s):
    B, N, H, W = s.shape
    flat = s.view(B, N, -1)
    idx = flat.argmax(dim=-1)
    y = (idx // W).float()
    x = (idx % W).float()
    return torch.stack([x, y], dim=-1)


def _soft_argmax(s):
    return encoder._soft_argmax_2d(s)[0]


def _peak_strength(s):
    """Peak-to-mean ratio: 1.0 = uniform, >> 1 = highly peaked."""
    flat = s.view(s.shape[0], s.shape[1], -1)
    return (flat.max(dim=-1).values - flat.mean(dim=-1)) / (flat.std(dim=-1) + 1e-6)


# Frame-0-anchored (no chaining):
corr_maps_anchored = []
hard_anchored = []
soft_anchored = []
peaks_anchored = []
with torch.no_grad():
    for fm in fmaps:
        corr = _correlate(descriptors, fm)
        corr_maps_anchored.append(corr.cpu())
        hard_anchored.append(_hard_argmax(corr).cpu())
        soft_anchored.append(_soft_argmax(corr).cpu())
        peaks_anchored.append(_peak_strength(corr).cpu())

# Chained:
corr_maps_chain = []
hard_chain = []
soft_chain = []
peaks_chain = []
curr_d = descriptors
curr_xy = chosen_xy
with torch.no_grad():
    # Frame 0 — no correlation, just the anchor
    corr_maps_chain.append(None)
    hard_chain.append(curr_xy.cpu())
    soft_chain.append(curr_xy.cpu())
    peaks_chain.append(torch.full((1, K), float("nan")))
    for fm in fmaps[1:]:
        corr = _correlate(curr_d, fm)
        corr_maps_chain.append(corr.cpu())
        hxy = _hard_argmax(corr)
        sxy = _soft_argmax(corr)
        hard_chain.append(hxy.cpu())
        soft_chain.append(sxy.cpu())
        peaks_chain.append(_peak_strength(corr).cpu())
        # Refresh descriptors at the predicted coords
        curr_d = encoder._bilinear_sample(fm, sxy.unsqueeze(0) if sxy.dim() == 2 else sxy)
        curr_xy = sxy

# --------------------------------------------------------------------------
# Plot grid:  rows = patches (K=4), cols = frames (N=5)
# --------------------------------------------------------------------------
print("[diag] rendering...")
ST = encoder.OUTPUT_STRIDE
fig, axes = plt.subplots(K, N_FRAMES, figsize=(3.5 * N_FRAMES, 3.5 * K))
patch_colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]

for k in range(K):
    for j in range(N_FRAMES):
        ax = axes[k, j]
        ax.imshow(_denorm(frames[j]))
        ax.set_xticks([]); ax.set_yticks([])

        # Anchor square (frame 0 reference, drawn lightly on every frame)
        if j == 0:
            ax_x, ax_y = chosen_xy[0, k].cpu().numpy() * ST
            ax.add_patch(Rectangle(
                (ax_x - PATCH_PX / 2, ax_y - PATCH_PX / 2),
                PATCH_PX, PATCH_PX, fill=False,
                edgecolor=patch_colors[k], lw=2.5,
            ))
            ax.set_title(f"patch {k} — anchor", color=patch_colors[k], fontsize=10)
            continue

        # Overlay correlation heatmap for THIS patch on this frame
        cm_anch = corr_maps_anchored[j][0, k].numpy()
        # Resize to image res for plotting
        cm_full = np.kron(cm_anch, np.ones((ST, ST)))
        ax.imshow(cm_full, cmap="hot", alpha=0.35,
                  extent=(0, Wf * ST, Hf * ST, 0))

        h_anch = hard_anchored[j][0, k].numpy() * ST
        s_anch = soft_anchored[j][0, k].numpy() * ST
        s_chain = soft_chain[j][0, k].numpy() * ST
        ax.scatter([h_anch[0]], [h_anch[1]], s=80, c="white",
                   marker="x", linewidths=2.5, label="hard argmax (frame-0 anchored)")
        ax.scatter([s_anch[0]], [s_anch[1]], s=80, edgecolors="cyan",
                   facecolors="none", marker="o", linewidths=2.0,
                   label="soft argmax (frame-0 anchored)")
        ax.scatter([s_chain[0]], [s_chain[1]], s=80, edgecolors="lime",
                   facecolors="none", marker="s", linewidths=2.0,
                   label="soft argmax (chained)")
        peak_anch = peaks_anchored[j][0, k].item()
        peak_ch = peaks_chain[j][0, k].item()
        ax.set_title(
            f"frame {indices[j]} | peak σ={peak_anch:.2f} (anch) / "
            f"{peak_ch:.2f} (chain)",
            fontsize=9,
        )
        if k == 0 and j == 1:
            ax.legend(fontsize=7, loc="lower left")

plt.suptitle(
    "DPVO correlation diagnostic — overlay = corr map; "
    "white × = hard argmax; cyan ○ = soft argmax (frame-0 anchored); "
    "lime □ = soft argmax (chained)",
    fontsize=11, y=1.001,
)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"[diag] saved {OUT}")

# --------------------------------------------------------------------------
# Print numeric summary
# --------------------------------------------------------------------------
print()
print(f"{'frame':>6s} | {'patch':>5s} | {'peak σ anch':>11s} | {'peak σ chain':>12s} | "
      f"{'hard (anch)':>20s} | {'soft (anch)':>20s} | {'soft (chain)':>20s}")
for k in range(K):
    for j in range(1, N_FRAMES):
        ph = peaks_anchored[j][0, k].item()
        pc = peaks_chain[j][0, k].item()
        ha = (hard_anchored[j][0, k] * ST).numpy()
        sa = (soft_anchored[j][0, k] * ST).numpy()
        sc = (soft_chain[j][0, k] * ST).numpy()
        print(f"{indices[j]:>6d} | {k:>5d} | {ph:>11.2f} | {pc:>12.2f} | "
              f"({ha[0]:>7.1f},{ha[1]:>7.1f})  | "
              f"({sa[0]:>7.1f},{sa[1]:>7.1f})  | "
              f"({sc[0]:>7.1f},{sc[1]:>7.1f})")

print(f"\nGrid centre (where soft-argmax falls on a flat map): "
      f"({Wf * ST / 2:.0f}, {Hf * ST / 2:.0f}) px")
