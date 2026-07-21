"""D2 sub-ablation: larger IMU backbone.

Reviewer flagged that the IMU encoder underperforms its ResNet1D reference
by 89% raw / 48% Umeyama-aligned, and asked whether a modestly larger
inertial backbone (still << 4.6 M params) changes the fusion conclusions.

Paper default: IMUCNN channels=(32, 64, 128) -> ~0.05 M params.
D2 variant:    IMUCNN channels=(64, 128, 256) -> ~0.16 M params (3.3x).

Runs at K=4 / 40 epochs / MBL=false (paper config), seed=42, on Webots and
MSILN site1/B1. Compares to the M1+M2 baseline `learned_continuous seed 42`
already in `revision/ablation_m1_timeenc/manifest.json`.

Monkey-patches `src.pipeline.encoders.imu.IMUCNN.__init__` to install the
larger default channels for this run only; restores it on exit.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.encoders import imu as imu_mod  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer,
    extract_vision_tokens, load_config,
)

MANIFEST = REPO / "revision" / "d2_imu_bigger" / "manifest.json"
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

BIG_CHANNELS = (64, 128, 256)
DATASETS = ["simulation_2mod", "msiln_site1_b1"]
SEED = 42
EPOCHS = 40
N_INSTANTS = 4


def patch_imu_channels(new_default):
    orig = imu_mod.IMUCNN.__init__

    def patched(self, in_features=9, embed_dim=128,
                 channels=new_default, kernel_size=3, dropout=0.1):
        orig(self, in_features=in_features, embed_dim=embed_dim,
             channels=channels, kernel_size=kernel_size, dropout=dropout)
    imu_mod.IMUCNN.__init__ = patched
    return orig


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(dataset: str) -> dict:
    set_seed(SEED)
    cfg = load_config(dataset)
    cfg.model.time_enc_mode = "learned_continuous"
    cfg.temporal.n_instants = N_INSTANTS
    cfg.train.modality_balanced_loss = False

    dm = build_datamodule(cfg)
    encoders, vision = build_encoders(cfg, dm)
    extra = (extract_vision_tokens(dm, vision, device="cuda")
             if vision is not None else {})
    model = build_model(cfg, encoders)

    # Sanity: report IMU encoder param count
    n_params_imu = sum(p.numel() for p in encoders["imu"].parameters())

    run_root = REPO / "runs" / "revision" / f"d2_{dataset}_s{SEED}"
    run_root.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(cfg, model, dm, extra_inputs=extra,
                             run_dir=str(run_root))

    t0 = time.time()
    trainer.fit(epochs=EPOCHS)
    elapsed = time.time() - t0

    preds_v, tgts_v = trainer.predict("val")
    val_mae = float((preds_v - tgts_v).norm(dim=1).mean())
    test_mae = float("nan")
    if "test" in trainer.splits:
        preds_t, tgts_t = trainer.predict("test")
        test_mae = float((preds_t - tgts_t).norm(dim=1).mean())

    return {
        "dataset": dataset,
        "imu_channels": list(BIG_CHANNELS),
        "imu_param_count": n_params_imu,
        "seed": SEED,
        "epochs": EPOCHS,
        "val_mae_m": val_mae,
        "test_mae_m": test_mae,
        "elapsed_min": round(elapsed / 60.0, 2),
        "status": "ok",
        "run_dir": str(trainer.run_path),
    }


def main() -> None:
    print(f"[d2] {len(DATASETS)} runs planned with IMU channels {BIG_CHANNELS}",
          flush=True)
    orig_init = patch_imu_channels(BIG_CHANNELS)
    try:
        results: list[dict] = []
        if MANIFEST.exists():
            try:
                results = json.loads(MANIFEST.read_text())
            except json.JSONDecodeError:
                results = []
        done = {r["dataset"] for r in results if r.get("status") == "ok"}

        for i, dataset in enumerate(DATASETS, 1):
            if dataset in done:
                print(f"[{i}/{len(DATASETS)}] SKIP {dataset} (already done)",
                      flush=True)
                continue
            print(f"[{i}/{len(DATASETS)}] START {datetime.now().isoformat(timespec='seconds')} "
                  f"dataset={dataset}", flush=True)
            try:
                rec = run_one(dataset)
                results.append(rec)
                print(f"  -> val={rec['val_mae_m']:.3f} test={rec['test_mae_m']:.3f} "
                      f"imu_params={rec['imu_param_count']} ({rec['elapsed_min']}min)",
                      flush=True)
            except Exception as e:
                results.append({
                    "dataset": dataset, "status": "failed",
                    "error": str(e),
                })
                print(f"  FAILED: {e}", flush=True)
            MANIFEST.write_text(json.dumps(results, indent=2))
        print(f"[d2] DONE - manifest: {MANIFEST}", flush=True)
    finally:
        imu_mod.IMUCNN.__init__ = orig_init


if __name__ == "__main__":
    main()
