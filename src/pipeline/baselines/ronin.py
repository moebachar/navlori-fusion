"""IMU SOTA baseline — Sachini/ronin (MIT).

Exports the canonical class definitions and the metric function
required by our wrappers. Adds the RoNIN ``source/`` directory to
``sys.path`` so the upstream's absolute imports work; applies the
``np.int`` compatibility shim before any of their data files load.
"""
from __future__ import annotations

import sys

from ._paths import RONIN_LISTS, RONIN_SRC
from ._shims import apply_np_int_shim

if str(RONIN_SRC) not in sys.path:
    sys.path.insert(0, str(RONIN_SRC))

apply_np_int_shim()

from model_resnet1d import BasicBlock1D, FCOutputModule, ResNet1D  # noqa: E402
from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset  # noqa: E402
from metric import compute_ate_rte  # noqa: E402


def load_test_list(name: str = "list_test_unseen.txt") -> list[str]:
    """Read a sequence-name list file under ``external_methods/ronin/lists/``."""
    return [
        ln.strip()
        for ln in (RONIN_LISTS / name).read_text().splitlines()
        if ln.strip()
    ]


def load_train_list() -> list[str]:
    """Convenience: ``load_test_list('list_train.txt')``."""
    return load_test_list("list_train.txt")


__all__ = [
    "BasicBlock1D", "FCOutputModule", "ResNet1D",
    "GlobSpeedSequence", "StridedSequenceDataset",
    "compute_ate_rte",
    "load_test_list", "load_train_list",
]
