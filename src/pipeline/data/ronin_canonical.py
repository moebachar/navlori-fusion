"""RoNIN canonical unseen-subjects — single-modality IMU benchmark.

Data: FRDR dataset 538 (extracted RESULT_07) at ``data/ronin_frdr/``:
- ``train/``: 73 canonical training sequences (69/73 present locally,
  4 missing per RESULT_07 coverage probe).
- ``unseen/``: 32 canonical test sequences (all present).
- ``Pretrained_Models/``: ``ronin_resnet/checkpoint_gsn_latest.pt``
  (reproduces paper's 5.140 m raw ATE; RESULT_07).

200-step window per RoNIN's convention; 6-channel world-frame
IMU input (gyro + accel after device-orientation rotation handled
by ``GlobSpeedSequence``).
"""
from __future__ import annotations

from pathlib import Path

from ._common import not_applicable, path_to


DATASET_NAME = "ronin_canonical"
DATA_DIR = "ronin_frdr"


def load(split: str = "unseen", **kwargs):
    """Load a ``GlobSpeedSequence``-backed dataset.

    ``split`` is one of ``'train'`` or ``'unseen'`` (test). Returns
    a ``StridedSequenceDataset`` keyed on the canonical RoNIN list
    files.
    """
    from src.pipeline.baselines import (
        GlobSpeedSequence, StridedSequenceDataset, load_test_list, load_train_list,
    )
    root = path_to(f"data/{DATA_DIR}")
    if split == "unseen" or split == "test":
        seq_list = load_test_list("list_test_unseen.txt")
        seq_dir = root / "unseen"
    elif split == "train":
        seq_list = load_train_list()
        seq_dir = root / "train"
    elif split == "val":
        seq_list = load_test_list("list_val.txt")
        seq_dir = root / "train"
    else:
        raise ValueError(f"Unknown RoNIN canonical split: {split!r}")
    seq_list = [s for s in seq_list if (seq_dir / s).is_dir()]
    return StridedSequenceDataset(
        GlobSpeedSequence, str(seq_dir), seq_list, None,
        step_size=kwargs.get("step", 10),
        window_size=kwargs.get("window", 200),
        random_shift=kwargs.get("random_shift", 5),
        shuffle=False,
    )


def stats() -> dict:
    from src.pipeline.baselines import RONIN_LISTS, load_test_list
    root = path_to(f"data/{DATA_DIR}")
    train_dir = root / "train"
    unseen_dir = root / "unseen"
    canon_train = load_test_list("list_train.txt")
    canon_unseen = load_test_list("list_test_unseen.txt")
    train_present = [s for s in canon_train if (train_dir / s).is_dir()] if train_dir.is_dir() else []
    unseen_present = [s for s in canon_unseen if (unseen_dir / s).is_dir()] if unseen_dir.is_dir() else []
    pretrained_p = root / "pretrained_resnet" / "ronin_resnet" / "checkpoint_gsn_latest.pt"
    return {
        "name": DATASET_NAME,
        "data_dir": str(root.relative_to(path_to("."))) if root.is_dir() else f"data/{DATA_DIR}",
        "modalities_available": ["imu"],
        "lists_dir": str(RONIN_LISTS.relative_to(path_to("."))) if RONIN_LISTS.exists() else str(RONIN_LISTS),
        "canonical_lists": ["list_train.txt", "list_val.txt", "list_test_unseen.txt", "list_test_seen.txt"],
        "n_train_canonical": len(canon_train),
        "n_train_present_locally": len(train_present),
        "n_unseen_canonical": len(canon_unseen),
        "n_unseen_present_locally": len(unseen_present),
        "pretrained_checkpoint": str(pretrained_p.relative_to(path_to("."))) if pretrained_p.is_file() else f"{pretrained_p} (NOT PRESENT — see RESULT_07 for extraction)",
        "canonical_window": 200,
        "canonical_step": 10,
        "evaluation_metric": "raw ATE (anchored at GT[0]) + Umeyama-aligned ATE + RTE (1-min sliding window) via RoNIN's compute_ate_rte",
        "known_caveats": [
            "Window=200 RoNIN convention; our IMUCNN window is 32 (RESULT_23 chunks 200 into K=4 sub-windows of 50).",
            "ATE/RTE/Umeyama via vendored RoNIN metric.compute_ate_rte (NEVER hand-rolled SVD per amended rubric correction #3).",
            "Pretrained ResNet1D reproduces paper's 5.140 m exactly (RESULT_07).",
            "C2 audit: IMUCNN canonical raw 9.961 m / Umeyama 7.876 m vs ResNet1D 5.14 m — raw +94 % outside 20 % gate (RESULT_07).",
            "CNN1D aggregator over IMUCNN sub-windows: raw 7.59 m / Umeyama 5.95 m — Umeyama gate cleared at +15.7 % (RESULT_23).",
        ],
        "source_result": "RESULT_07 (canonical extraction + SOTA reproduction), RESULT_23 (aggregator over IMUCNN sub-windows)",
    }


def preprocessing_demo(modality: str, n_samples: int = 1) -> dict:
    import numpy as np
    if modality != "imu":
        return not_applicable(modality, DATASET_NAME)
    from src.pipeline.baselines import GlobSpeedSequence, load_test_list
    root = path_to(f"data/{DATA_DIR}/unseen")
    seq_list = load_test_list("list_test_unseen.txt")
    seq_list = [s for s in seq_list if (root / s).is_dir()]
    if not seq_list:
        return not_applicable(modality, DATASET_NAME)
    seq = GlobSpeedSequence(str(root / seq_list[0]), interval=200,
                             max_ori_error=20.0, grv_only=True)
    feat = seq.features[:200]  # one canonical 200-step window
    return {
        "raw": feat,
        "preprocessed": feat,  # GlobSpeedSequence already returns world-frame; "raw" here = post-rotation features
        "description_raw": f"6-ch world-frame IMU after GlobSpeedSequence's device-orientation rotation (200-step window from {seq_list[0]})",
        "description_preprocessed": "z-score per channel using train-set statistics; chunked into K=4 sub-windows of 50 by RESULT_23 aggregator",
        "preprocessing_pipeline": ["device-frame -> world-frame rotation (RoNIN GlobSpeedSequence)", "window 200", "K=4 sub-windows of 50 for aggregator (RESULT_23)"],
    }


__all__ = ["load", "stats", "preprocessing_demo"]
