"""Centralised SOTA-baseline loaders.

All wrappers and the run-2 walkthrough notebook should import from
this package so the path / shim machinery lives in one place. The
four baselines wrapped:

- **wlan_localization** (WiFi SOTA): ``load_position_regressor()``,
  ``load_preprocessor()``.
- **ronin** (IMU SOTA): ``ResNet1D`` / ``BasicBlock1D`` /
  ``FCOutputModule`` classes, ``GlobSpeedSequence`` /
  ``StridedSequenceDataset`` loaders, ``compute_ate_rte`` metric,
  ``load_test_list()`` / ``load_train_list()`` helpers.
- **tartanvo** (Camera SOTA): ``apply_tartanvo_shims()``,
  ``load_vo_module()``.
- **dpvo_trunk** (Camera trunk for ``DPVOMotionEncoder``):
  ``load_basic_encoder4()``.

All paths resolve from ``external_methods/<repo>/`` per the
post-run-2 consolidation directive (PLAN_26). The path constants
themselves are exposed via ``_paths`` for callers that need to look
up data lists or weights.
"""
from .wlanloc import load_position_regressor, load_preprocessor
from .ronin import (
    BasicBlock1D, FCOutputModule, ResNet1D,
    GlobSpeedSequence, StridedSequenceDataset,
    compute_ate_rte,
    load_test_list, load_train_list,
)
from .tartanvo import apply_tartanvo_shims, load_vo_module
from .dpvo_trunk import BasicEncoder4, get_basic_encoder4_class, load_basic_encoder4
from ._paths import (
    PROJECT_ROOT, EXTERNAL_METHODS,
    WLANLOC_SRC, RONIN_SRC, RONIN_LISTS,
    TARTANVO_ROOT, DPVO_ROOT,
)
from ._shims import (
    apply_np_int_shim,
    apply_scipy_as_dcm_shim,
    apply_numpy_linalg_submodule_shim,
    apply_cupy_compat_shim,
)

__all__ = [
    # wlanloc
    "load_position_regressor", "load_preprocessor",
    # ronin
    "BasicBlock1D", "FCOutputModule", "ResNet1D",
    "GlobSpeedSequence", "StridedSequenceDataset",
    "compute_ate_rte",
    "load_test_list", "load_train_list",
    # tartanvo
    "apply_tartanvo_shims", "load_vo_module",
    # dpvo trunk
    "BasicEncoder4", "get_basic_encoder4_class", "load_basic_encoder4",
    # paths
    "PROJECT_ROOT", "EXTERNAL_METHODS",
    "WLANLOC_SRC", "RONIN_SRC", "RONIN_LISTS",
    "TARTANVO_ROOT", "DPVO_ROOT",
    # shims
    "apply_np_int_shim", "apply_scipy_as_dcm_shim",
    "apply_numpy_linalg_submodule_shim", "apply_cupy_compat_shim",
]
