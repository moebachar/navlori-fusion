"""Run upstream DPVO on a Webots sequence and dump per-keyframe patch
reprojections.

Runs INSIDE the docker container (paths assume /work/* layout).
Loads ~30 consecutive camera frames around the same anchor we use in the
notebook viz, feeds them to DPVO, then reprojects every tracked patch into
each keyframe using DPVO's own `pops.transform`.

Output: /work/output/dpvo_upstream_patches.pt
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/work/external/dpvo_upstream")

from dpvo.config import cfg                    # noqa: E402
from dpvo.dpvo import DPVO                     # noqa: E402
from dpvo import projective_ops as pops        # noqa: E402
from dpvo.lietorch import SE3                  # noqa: E402

DATA_DIR = Path("/work/data/async_collection/path_02")
WEIGHTS = "/work/runs/_weights/dpvo.pth"
OUT = Path("/work/output/dpvo_upstream_patches.pt")
INFO = json.load(open(DATA_DIR / "camera_info.json"))

intrinsics = torch.tensor(
    [INFO["fx"], INFO["fy"], INFO["cx"], INFO["cy"]],
    dtype=torch.float32, device="cuda",
)

cam_csv = pd.read_csv(DATA_DIR / "camera.csv")
N_TOTAL = len(cam_csv)

# Match the notebook viz: anchor = mid path; want a few keyframes spread
# across ~6 s. DPVO will keyframe naturally; we cap PATCHES_PER_FRAME to fit
# in the 8 GB Pascal VRAM.
ANCHOR = N_TOTAL // 2
WINDOW = 30                                 # consecutive frames to feed
half = WINDOW // 2
input_idx = list(range(max(0, ANCHOR - half),
                       min(N_TOTAL, ANCHOR - half + WINDOW)))
print(f"[infer] feeding {len(input_idx)} consecutive frames ({input_idx[0]}..{input_idx[-1]}), anchor={ANCHOR}")

cfg.merge_from_file("/work/external/dpvo_upstream/config/default.yaml")
cfg.merge_from_list([
    "PATCHES_PER_FRAME", 48,
    "BUFFER_SIZE", 32,
    "OPTIMIZATION_WINDOW", 8,
    "REMOVAL_WINDOW", 16,
    "MIXED_PRECISION", True,
])
print("[infer] cfg:", cfg)

slam = None
for t, idx in enumerate(input_idx):
    img_path = DATA_DIR / cam_csv.iloc[idx]["rgb_path"]
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")
    img_t = torch.from_numpy(img).permute(2, 0, 1).cuda()
    if slam is None:
        slam = DPVO(cfg, WEIGHTS, ht=img.shape[0], wd=img.shape[1], viz=False)
    if (t % 10) == 0 or t == len(input_idx) - 1:
        print(f"[infer]   frame t={t} ({img_path.name}) "
              f"-> n_kfs={slam.pg.n}  m_patches={slam.pg.m}", flush=True)
    slam(t, img_t, intrinsics)

n = int(slam.pg.n)
m = int(slam.pg.m)
P = slam.P
M = slam.M
print(f"[infer] FINAL: keyframes={n}, patches={m}, P={P}, M={M}")

# tstamps_[:n] holds the 't' indices we passed in (= positions in input_idx)
kf_t = slam.pg.tstamps_[:n].astype(int)              # (n,)
kf_input_idx = [input_idx[t] for t in kf_t]          # csv row index of each kf
print(f"[infer] keyframe t indices: {kf_t.tolist()}")
print(f"[infer] keyframe csv indices: {kf_input_idx}")

# Reproject every patch into every keyframe.
patches = slam.pg.patches_[:n, :, ...]               # (n, M, 3, P, P)
poses = SE3(slam.pg.poses_[:n].view(1, n, 7))
intr = slam.pg.intrinsics_[:n].view(1, n, 4)

ix = slam.pg.ix[:m]
ii = ix.view(-1, 1).expand(-1, n).reshape(-1)        # source frame
jj = torch.arange(n, device="cuda").view(1, -1).expand(m, -1).reshape(-1)  # target
kk = torch.arange(m, device="cuda").view(-1, 1).expand(-1, n).reshape(-1)

patches_flat = patches.view(1, n * M, 3, P, P)
print(f"[infer] reprojecting {len(kk):,} pairs...")
coords = pops.transform(poses, patches_flat, intr, ii, jj, kk)
center = coords[0, :, P // 2, P // 2, :].view(m, n, 2) * slam.RES   # image px

# Pick 5 evenly spaced keyframes for the comparison viz
N_VIS = 5
sel = np.linspace(0, n - 1, N_VIS).round().astype(int).tolist()
sel = sorted(set(sel))
print(f"[infer] selected keyframe indices for viz: {sel} -> "
      f"csv={[kf_input_idx[i] for i in sel]}")

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "frame_paths": [Path(cam_csv.iloc[kf_input_idx[i]]["rgb_path"]).name
                    for i in sel],
    "frame_indices": [kf_input_idx[i] for i in sel],
    "patch_source_keyframe": ix.cpu(),
    "positions": center[:, sel, :].permute(1, 0, 2).cpu(),    # (N_vis, m, 2)
    "n_keyframes_total": n,
}, OUT)
print(f"[infer] wrote {OUT}: positions {tuple(center[:, sel, :].permute(1, 0, 2).shape)}")
