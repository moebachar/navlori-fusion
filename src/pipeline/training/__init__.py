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
    compute_trivial_integration_floor,
    load_webots_odom_pb,
    train_dpvo_motion_head,
    train_fusion_arch,
    train_imucnn,
    train_odomcnn,
    train_ronin_canonical_arch,
    train_uji_arch,
    train_wifi_net,
    wifi_net_predict,
    wifi_net_val_mae,
)
from .trainer import EncoderTrainer

__all__ = [
    "EncoderTrainer", "FusionTrainer", "load_trained",
    "train_wifi_net", "wifi_net_predict", "wifi_net_val_mae",
    "train_imucnn",
    "train_odomcnn", "load_webots_odom_pb", "compute_trivial_integration_floor",
    "train_dpvo_motion_head",
    "train_fusion_arch",
    "train_uji_arch", "train_ronin_canonical_arch",
]
