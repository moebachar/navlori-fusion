"""Extract per-camera-frame DPVO ``imap`` pool for every path.

The "hidden state" we care about for the encoder is DPVO's per-frame
context features (``imap`` in ``dpvo.net.Patchifier``) — the 384-D vector
that the GRU update operator consumes. **Crucially, ``imap`` is a pure
function of the current frame**: it's the output of ``inet(image)``
sampled at patch centroids. It does *not* depend on SLAM state.

That means we can skip the full ``DPVO`` state machine (whose growing
patch-graph buffer kept OOM-ing on our 8 GB Pascal) and call
``network.patchify(...)`` directly per frame. Symmetry with online
inference is preserved: when ``DPVOOnlineRunner`` runs full DPVO at test
time, the same ``patchify`` runs underneath and produces the same
distribution of patch features.

Output (per path): ``data/async_collection/path_XX/dpvo_features.pt``::

    {
        "features":  (N_frames, 384) float32,   # mean over patches
        "rgb_paths": list[str],
        "sim_time":  (N_frames,) float64,
        "dim": 384,
        "n_patches": int,
    }

Run inside the docker container. Idempotent: skips paths that already have
a features file. Pass ``--force`` to re-extract.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/work/external/dpvo_upstream")

from dpvo.config import cfg                       # noqa: E402
from dpvo.net import VONet                        # noqa: E402

DATA_ROOT = Path("/work/data/async_collection")
WEIGHTS = "/work/runs/_weights/dpvo.pth"
DPVO_CFG = "/work/external/dpvo_upstream/config/default.yaml"

# Match DPVO's defaults for patchify: 32 patches sampled per frame, biased
# toward high-gradient regions (the same setting DPVO uses online).
PATCHES_PER_FRAME = 32
CENTROID_SEL_STRAT = "GRADIENT_BIAS"
DIM = 384
FORCE = "--force" in sys.argv


def _load_network():
    """Load VONet weights into a fresh network, eval mode, cuda."""
    cfg.merge_from_file(DPVO_CFG)
    from collections import OrderedDict
    state_dict = torch.load(WEIGHTS)
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        if "update.lmbda" in k:
            continue
        cleaned[k.replace("module.", "")] = v
    net = VONet()
    net.load_state_dict(cleaned)
    net.eval().cuda()
    return net


@torch.no_grad()
def extract_path(path_dir: Path, network) -> bool:
    out_path = path_dir / "dpvo_features.pt"
    if out_path.exists() and not FORCE:
        print(f"[skip] {path_dir.name}: features already present")
        return True

    cam_csv_path = path_dir / "camera.csv"
    if not cam_csv_path.exists():
        print(f"[skip] {path_dir.name}: no camera.csv")
        return False

    cam_csv = pd.read_csv(cam_csv_path)
    N = len(cam_csv)
    if N == 0:
        print(f"[skip] {path_dir.name}: empty camera.csv")
        return False

    features = np.zeros((N, DIM), dtype=np.float32)
    t0 = time.time()
    bad = 0

    for t, (_, row) in enumerate(cam_csv.iterrows()):
        rgb_path = path_dir / row["rgb_path"]
        img_bgr = cv2.imread(str(rgb_path))
        if img_bgr is None:
            print(f"[warn] {path_dir.name}: missing {rgb_path}")
            bad += 1
            continue

        # DPVO does: image_uint8 -> (1, 1, 3, H, W) -> 2 * (x / 255) - 0.5
        img = torch.from_numpy(img_bgr).permute(2, 0, 1).cuda()
        img = img[None, None].float()                          # (1, 1, 3, H, W)
        img = 2.0 * (img / 255.0) - 0.5

        try:
            # Patchifier signature: (images, patches_per_image, disps,
            # centroid_sel_strat, return_color). imap output: (B, M, DIM, 1, 1)
            _fmap, _gmap, imap, _patches, _ix = network.patchify(
                img,
                patches_per_image=PATCHES_PER_FRAME,
                centroid_sel_strat=CENTROID_SEL_STRAT,
                return_color=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                bad += 1
                continue
            raise

        # Mean-pool over patches -> (DIM,)
        feat = imap.view(-1, DIM).mean(dim=0).float().cpu().numpy()
        features[t] = feat

        if (t % 80) == 0 or t == N - 1:
            print(f"[ {path_dir.name} ] t={t:4d}/{N}  "
                  f"feat_norm={np.linalg.norm(feat):.2f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    payload = {
        "features": torch.from_numpy(features),
        "rgb_paths": cam_csv["rgb_path"].tolist(),
        "sim_time": torch.from_numpy(cam_csv["sim_time"].values.astype(np.float64)),
        "dim": DIM,
        "n_patches": PATCHES_PER_FRAME,
    }
    torch.save(payload, out_path)
    print(f"[done] {path_dir.name}: wrote {out_path.name} "
          f"({N} frames, {bad} bad, {elapsed:.1f}s, {N/elapsed:.1f} fps)")
    torch.cuda.empty_cache()
    return True


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"Data root not found: {DATA_ROOT}")

    print("[run] loading DPVO VONet weights (patchifier-only pipeline)")
    network = _load_network()
    print(f"[run] DPVO ok — DIM={DIM}, patches/frame={PATCHES_PER_FRAME}, "
          f"centroid={CENTROID_SEL_STRAT}")

    paths = sorted(p for p in DATA_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith("path_"))
    print(f"[run] {len(paths)} candidate paths under {DATA_ROOT}")

    ok = 0
    for p in paths:
        try:
            if extract_path(p, network):
                ok += 1
        except Exception as exc:
            print(f"[fail] {p.name}: {exc}")
    print(f"[run] finished: {ok}/{len(paths)} paths processed")


if __name__ == "__main__":
    main()
