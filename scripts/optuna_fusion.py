"""Optuna hyperparameter search for the FusionTransformer.

Searches the space declared in ``configs/stage_c/fusion.yaml`` (``optuna``
block) with a TPE sampler, minimising best validation MAE. Vision DPVO
patch tokens are extracted once up front and shared across all trials.

    python scripts/optuna_fusion.py            # uses config n_trials
    python scripts/optuna_fusion.py --trials 8 # quick run

Results: runs/optuna_fusion/best.json (+ trials.csv).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer,
    extract_vision_tokens, load_config, pretrained_paths_from_cfg,
)


def make_objective(cfg, dm, extra, epochs: int, run_dir: Path,
                   pretrained_paths: dict):
    """Return an Optuna objective closured over the fixed data/vision setup."""
    ss = cfg.optuna.search_space

    def objective(trial: optuna.Trial) -> float:
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        c.model.depth = trial.suggest_int("depth", *map(int, ss.depth))
        c.model.n_heads = trial.suggest_categorical(
            "n_heads", [int(x) for x in ss.n_heads])
        c.model.ff_mult = trial.suggest_int("ff_mult", *map(int, ss.ff_mult))
        c.model.dropout = trial.suggest_float("dropout", *map(float, ss.dropout))
        c.train.lr = trial.suggest_float("lr", *map(float, ss.lr), log=True)
        c.train.modality_dropout = trial.suggest_float(
            "modality_dropout", *map(float, ss.modality_dropout))
        c.train.instant_dropout = trial.suggest_float(
            "instant_dropout", *map(float, ss.instant_dropout))
        c.temporal.n_instants = trial.suggest_int(
            "n_instants", *map(int, ss.n_instants))
        c.temporal.instant_stride = trial.suggest_int(
            "instant_stride", *map(int, ss.instant_stride))

        encoders, _ = build_encoders(c, dm, pretrained_paths=pretrained_paths)
        model = build_model(c, encoders)
        trainer = build_trainer(
            c, model, dm, extra_inputs=extra,
            run_dir=str(run_dir / "trials"))
        # Verbose=False: per-trial logs would spam Optuna's output. The
        # post-fit diagnostics (Action 3) only print when verbose=True; we
        # rely on the best-trial rerun below to print them once.
        hist = trainer.fit(epochs=epochs, verbose=False)
        return hist.best_val_mae

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--dataset", type=str, default=None,
                    help="configs/data/<name>.yaml (default: config's)")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    dataset_name = cfg.dataset.selected
    n_trials = args.trials or int(cfg.optuna.n_trials)
    epochs = int(cfg.optuna.epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Optuna fusion search — {n_trials} trials x {epochs} epochs "
          f"| dataset={dataset_name} "
          f"| modalities={list(cfg.dataset.modalities)} | device={device}",
          flush=True)
    dm = build_datamodule(cfg)
    extra = None
    if "camera" in cfg.dataset.modalities:
        _, vision = build_encoders(cfg, dm)
        print("  extracting DPVO patch tokens (cached)...", flush=True)
        extra = extract_vision_tokens(dm, vision, device)

    # Honor cfg.stage_a.pretrained — each trial loads the same pretrained
    # weights, so the search runs over fusion hyperparameters with Stage A
    # held constant. Empty when no checkpoints are configured.
    pretrained = pretrained_paths_from_cfg(cfg)
    if pretrained:
        print(f"  pretrained encoders: {list(pretrained.keys())}", flush=True)

    # Per-dataset output dir so multiple Optuna runs don't overwrite.
    out = ROOT / "runs" / "optuna_fusion" / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(make_objective(cfg, dm, extra, epochs, out, pretrained),
                   n_trials=n_trials, show_progress_bar=False)

    (out / "best.json").write_text(json.dumps({
        "dataset": dataset_name,
        "best_value_mae": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "epochs_per_trial": epochs,
    }, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)

    print("\n=== Optuna done ===", flush=True)
    print(f"best val MAE : {study.best_value:.4f} m")
    print("best params  :", json.dumps(study.best_params, indent=2))

    # Baseline gate: did the best Optuna trial beat the dataset's best
    # baseline? If not, the architecture has more wrong than just hparams.
    baselines = ROOT / "runs" / "baselines" / dataset_name / "baselines.json"
    if baselines.exists():
        b = json.loads(baselines.read_text())
        v = b.get("splits", {}).get("val", {})
        if v:
            best_name = v.get("best")
            best_mae = v.get("best_mae", float("inf"))
            gap = study.best_value - best_mae
            status = "PASS" if gap < 0 else "FAIL"
            print(f"\n  vs best baseline ({best_name} @ {best_mae:.3f}m): "
                  f"gap {gap:+.3f}m [{status}]")
            if gap >= 0:
                print("  (Optuna found a config that LOSES to the trivial "
                      "baseline. The architecture, not the hparams, is the "
                      "issue.)")
    else:
        print(f"  (no baselines.json for {dataset_name}; "
              f"run scripts/baselines.py first)")

    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
