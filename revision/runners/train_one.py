"""Train one fusion model for the M1+M2 ablation.

Args control (dataset, time-encoding mode, seed, modalities). Output is a
trained checkpoint + a tiny summary.json with val_mae / test_mae.

Designed to be looped over by revision/runners/batch_m1_m2.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (
    build_datamodule,
    build_encoders,
    build_model,
    build_trainer,
    extract_vision_tokens,
    load_config,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="data config name, e.g. simulation_2mod, msiln_site1_b1")
    ap.add_argument("--time-enc-mode", required=True,
                    choices=["learned_continuous", "none", "binned", "posindex"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40,
                    help="training epochs (default 40 = paper main_table)")
    ap.add_argument("--n-instants", type=int, default=4,
                    help="K, recent instants per modality (default 4 = paper)")
    ap.add_argument("--mbl", action="store_true",
                    help="enable modality_balanced_loss (default false = paper)")
    ap.add_argument("--modalities", default=None,
                    help="comma-separated modality override (for D1)")
    ap.add_argument("--time-min-period", type=float, default=None)
    ap.add_argument("--time-max-period", type=float, default=None)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    set_global_seed(args.seed)

    cfg = load_config(args.dataset)
    cfg.model.time_enc_mode = args.time_enc_mode
    if args.time_min_period is not None:
        cfg.model.time_min_period = float(args.time_min_period)
    if args.time_max_period is not None:
        cfg.model.time_max_period = float(args.time_max_period)
    if args.modalities is not None:
        cfg.dataset.modalities = [m.strip() for m in args.modalities.split(",")
                                   if m.strip()]
    cfg.temporal.n_instants = int(args.n_instants)
    cfg.train.modality_balanced_loss = bool(args.mbl)
    epochs = int(args.epochs)

    print(f"[run] dataset={args.dataset} time_enc_mode={args.time_enc_mode} "
          f"seed={args.seed} epochs={epochs} mods={list(cfg.dataset.modalities)}",
          flush=True)

    dm = build_datamodule(cfg)
    encoders, vision = build_encoders(cfg, dm)
    extra = (extract_vision_tokens(dm, vision, device="cuda")
             if vision is not None else {})
    model = build_model(cfg, encoders)

    run_root = REPO / "runs" / "revision" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg, model, dm, extra_inputs=extra,
                             run_dir=str(run_root))

    t0 = time.time()
    history = trainer.fit(epochs=epochs)
    elapsed = time.time() - t0

    preds_v, tgts_v = trainer.predict("val")
    val_mae = float((preds_v - tgts_v).norm(dim=1).mean())
    test_mae = float("nan")
    if "test" in trainer.splits:
        preds_t, tgts_t = trainer.predict("test")
        test_mae = float((preds_t - tgts_t).norm(dim=1).mean())

    summary = {
        "dataset": args.dataset,
        "time_enc_mode": args.time_enc_mode,
        "seed": args.seed,
        "epochs": epochs,
        "modalities": list(cfg.dataset.modalities),
        "val_mae_m": val_mae,
        "test_mae_m": test_mae,
        "elapsed_s": elapsed,
        "run_dir": str(trainer.run_path),
    }

    summary_p = run_root / "summary.json"
    summary_p.write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
