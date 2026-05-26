"""Camera SOTA baseline — castacks/tartanvo (python3 branch, MIT).

Applies the 3 shims TartanVO needs (scipy ``as_dcm`` rename, numpy
``linalg.linalg`` deprecated submodule alias, cupy
``compile_with_cache`` removed) and exposes a thin
``load_vo_module()`` returning the ``TartanVO`` class so the caller
can instantiate with their checkpoint path and run inference.
"""
from __future__ import annotations

import sys

from ._paths import TARTANVO_ROOT
from ._shims import (
    apply_cupy_compat_shim,
    apply_numpy_linalg_submodule_shim,
    apply_scipy_as_dcm_shim,
)


def apply_tartanvo_shims() -> None:
    """Apply all 3 shims TartanVO's inference path requires.

    Idempotent. ``np.int`` shim is NOT applied here because TartanVO's
    inference doesn't hit that codepath; callers that need it can call
    ``apply_np_int_shim()`` separately.
    """
    apply_scipy_as_dcm_shim()
    apply_numpy_linalg_submodule_shim()
    apply_cupy_compat_shim()


def _ensure_path() -> None:
    if str(TARTANVO_ROOT) not in sys.path:
        sys.path.insert(0, str(TARTANVO_ROOT))


def load_vo_module():
    """Import TartanVO's ``TartanVO`` class. Caller instantiates with
    a checkpoint path and runs inference."""
    apply_tartanvo_shims()
    _ensure_path()
    from TartanVO import TartanVO  # noqa: E402  type: ignore
    return TartanVO


__all__ = ["apply_tartanvo_shims", "load_vo_module"]
