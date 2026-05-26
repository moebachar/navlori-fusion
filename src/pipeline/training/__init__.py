"""Training utilities for encoder and pipeline training.

Public methods on ``FusionTrainer`` (consolidated in PLAN_28 for the
run-2 walkthrough notebook):
- ``evaluate_all_subsets(split)`` — 2^M-1 subset + full → MAE map.
- ``evaluate_staleness(modality, split)`` — K-axis stale-modality MAE.
- ``compute_per_trajectory_smoothness(split)`` — Pearson r per path
  (criterion (d) gate target r > 0.20).
- ``latency_probe(batch_sizes, n_trials)`` — per-batch latency
  (criterion (e) gate target < 100 ms / sample).
- ``predict(split)`` — return ``(pred_xy, gt_xy)`` tensors.

Module-level ``load_trained(checkpoint_dir, arch, dataset)`` rebuilds
a trained ``FusionTrainer`` from a saved ``model.pt`` directory — the
notebook's primary entry point for evaluating run-2 checkpoints.
"""

from .fusion_trainer import FusionTrainer, load_trained
from .inline_encoders import (
    anchor2vec_predict,
    anchor2vec_val_mae,
    train_anchor2vec,
)
from .trainer import EncoderTrainer

__all__ = [
    "EncoderTrainer", "FusionTrainer", "load_trained",
    "train_anchor2vec", "anchor2vec_predict", "anchor2vec_val_mae",
]
