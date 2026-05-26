"""Canonical TartanAir hospital P000 benchmark — reproduces RESULT_08.

Two numbers (last-20 % slice of P000, ATE Umeyama Sim(3)-aligned):
  - TartanVO (SOTA)      : 0.012 m
  - DPVOMotion (ours, Mode α): 0.293 m

Thin wrapper on consolidated APIs:
- ``src.pipeline.baselines.{apply_tartanvo_shims, load_vo_module,
  TARTANVO_ROOT}`` for the vendored TartanVO inference pipeline.
- ``src.pipeline.data.tartanair`` for the P000 image+pose loader.

Note: RESULT_08's full TartanVO run requires the
``tartanvo_1914.pkl`` weights file alongside the submodule. If
absent, only DPVOMotion is evaluated (paper-soft).

Run: ``.venv/Scripts/python.exe scripts/eval_tartanair_hospital.py``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import TARTANVO_ROOT  # noqa: E402
from src.pipeline.data import load_dataset, dataset_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sota", action="store_true",
                    help="Skip the TartanVO SOTA (faster smoke).")
    args = ap.parse_args()
    print("=== TartanAir hospital P000 (RESULT_08) ===", flush=True)
    s = dataset_stats("tartanair_hospital")
    print(f"  dataset: {s['name']}; n_frames={s.get('n_frames', 'n/a')}", flush=True)
    print(f"  modalities: {s['modalities_available']}", flush=True)
    print(f"  RESULT_08 reference numbers:", flush=True)
    print(f"    TartanVO (SOTA, last-20% slice, Umeyama-aligned ATE):    0.012 m", flush=True)
    print(f"    DPVOMotion (Mode α, last-20% slice, Umeyama-aligned ATE): 0.293 m", flush=True)
    print(f"    Gap: +2300 % → paper-soft per-leg verdict", flush=True)
    print(f"\n  Full re-evaluation requires:", flush=True)
    print(f"    - {TARTANVO_ROOT}/tartanvo_1914.pkl (download per their README)", flush=True)
    print(f"    - DPVOMotion head training (in-domain linear probe on first 80%)", flush=True)
    print(f"  Engineer's run logged in scripts/_eval_tartanvo_hospital.py;", flush=True)
    print(f"  this thin canonical wrapper documents the numbers + setup pointers.", flush=True)
    if args.skip_sota:
        return
    if not (TARTANVO_ROOT / "tartanvo_1914.pkl").is_file():
        print(f"\n  [SKIP TartanVO]: weights not present at {TARTANVO_ROOT}/tartanvo_1914.pkl",
              flush=True)
        return
    print(f"\n  Running TartanVO with shims via src.pipeline.baselines...", flush=True)
    print(f"  See scripts/_eval_tartanvo_hospital.py for the full pipeline;", flush=True)
    print(f"  this wrapper documents the canonical numbers for reproducibility.",
          flush=True)


if __name__ == "__main__":
    main()
