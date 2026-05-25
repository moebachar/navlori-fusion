"""Render the patch trajectories produced by upstream DPVO using THE SAME viz
code as our notebook (boxes + ConnectionPatch lines). Saves the comparison
image to docs/dpvo_upstream_patches.png.

Inputs:
    output/dpvo_upstream_patches.pt   (produced by the docker run)

Output:
    docs/dpvo_upstream_patches.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, ConnectionPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "output" / "dpvo_upstream_patches.pt"
DATA = ROOT / "data" / "async_collection" / "path_02"
OUT = ROOT / "docs" / "dpvo_upstream_patches.png"

print(f"[viz] loading {PT}")
data = torch.load(PT, weights_only=True)
positions = data["positions"].numpy()        # (N_frames, M_patches, 2)
frame_paths = [DATA / "camera" / p for p in data["frame_paths"]]
for p in frame_paths:
    if not p.exists():
        raise FileNotFoundError(f"Missing frame: {p}")
N, M, _ = positions.shape
print(f"[viz] N={N} frames, M={M} patches")

# Sub-sample patches for legibility (DPVO defaults to 96/frame, way too many).
K = 16
rng = np.random.default_rng(0)
keep = rng.choice(M, size=min(K, M), replace=False)
positions = positions[:, keep, :]
M = positions.shape[1]

# Same parameters as the notebook viz cell.
PATCH_PX = 14
colors = cm.tab20(np.linspace(0, 1, M))

fig, axes = plt.subplots(1, N, figsize=(3.8 * N, 3.4))
if N == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    img = np.array(Image.open(frame_paths[i]).convert("RGB")) / 255.0
    ax.imshow(img)
    for k in range(M):
        x, y = positions[i, k]
        rect = Rectangle((x - PATCH_PX / 2, y - PATCH_PX / 2),
                         PATCH_PX, PATCH_PX, fill=False,
                         edgecolor=colors[k], lw=1.6, alpha=0.95)
        ax.add_patch(rect)
    ax.set_title(f"frame {data['frame_indices'][i]}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Cross-frame trajectory lines.
for i in range(1, N):
    for k in range(M):
        xa, ya = positions[i - 1, k]
        xb, yb = positions[i, k]
        con = ConnectionPatch(
            xyA=(xa, ya), coordsA=axes[i - 1].transData,
            xyB=(xb, yb), coordsB=axes[i].transData,
            color=colors[k], lw=0.8, alpha=0.7,
        )
        fig.add_artist(con)

plt.suptitle(
    f"Upstream DPVO — {M} patch trajectories on val path 2 (same viz code)",
    fontsize=13,
)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"[viz] saved {OUT}")
