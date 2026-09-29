"""Pre-build the DPVO camera-token cache for the F2 replay dataset at a
reduced (~1 Hz) cadence, so a later 4-modality run reuses it instantly.

Uses extract_vision_tokens_fast: dedup exact-duplicate frame-pairs + only run
DPVO on every Nth unique pair (subsample_stride). Small flush batch keeps GPU
memory low so it can share the GPU with a training run.

Run: .venv/Scripts/python.exe scripts/_cache_camera.py [--stride 5] [--flush 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, extract_vision_tokens_fast, load_config,
)

DATASET = "iln20_5d27099f_F2_replay"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5,
                    help="run DPVO on every Nth unique frame-pair. camera is "
                         "5 Hz, so stride 5 ~= 1 Hz distinct tokens")
    ap.add_argument("--flush", type=int, default=6,
                    help="DPVO forward batch (small = low GPU mem, GPU-share)")
    args = ap.parse_args()

    print(f"=== camera cache: stride {args.stride} (~{5/args.stride:.1f} Hz), "
          f"flush {args.flush} ===", flush=True)
    cfg = load_config(DATASET)          # keeps camera modality
    dm = build_datamodule(cfg)
    _, vision = build_encoders(cfg, dm)
    if vision is None:
        sys.exit("no camera modality?")
    t0 = time.time()
    out = extract_vision_tokens_fast(dm, vision, device="cuda",
                                     subsample_stride=args.stride,
                                     flush_batch=args.flush)
    for split, feats in out["camera"].items():
        print(f"  {split}: {tuple(feats.shape)} cached", flush=True)
    print(f"=== camera cache done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
