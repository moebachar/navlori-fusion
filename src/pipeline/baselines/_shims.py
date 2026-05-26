"""Runtime compatibility shims for vendored baselines.

Each function is idempotent — safe to call multiple times. NO vendored
sources are edited; shims monkey-patch our process's loaded modules
only (Demand #3 honoured).
"""
from __future__ import annotations


def apply_np_int_shim() -> None:
    """RoNIN's ``data_glob_speed.py`` uses ``np.int`` (removed in numpy 1.20+)."""
    import numpy as np
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def apply_scipy_as_dcm_shim() -> None:
    """TartanVO's vendored ``transformation.py`` uses ``Rotation.as_dcm``,
    renamed to ``as_matrix`` in scipy 1.4. Same for ``from_dcm``."""
    from scipy.spatial.transform import Rotation
    if not hasattr(Rotation, "as_dcm"):
        Rotation.as_dcm = Rotation.as_matrix  # type: ignore[attr-defined]
    if not hasattr(Rotation, "from_dcm"):
        Rotation.from_dcm = Rotation.from_matrix  # type: ignore[attr-defined]


def apply_numpy_linalg_submodule_shim() -> None:
    """``numpy.linalg.linalg`` was a deprecated nested submodule
    referenced by TartanVO's PWC correlation kernel; restore the alias
    so the import resolves."""
    import numpy.linalg as nplinalg
    if not hasattr(nplinalg, "linalg"):
        nplinalg.linalg = nplinalg  # type: ignore[attr-defined]


def apply_cupy_compat_shim() -> None:
    """``cupy.cuda.compile_with_cache`` was removed in cupy 12+; map to
    a ``RawModule``-wrapped compat class. Used by TartanVO's
    ``Network/PWC/correlation.py`` CUDA kernels."""
    try:
        import cupy
    except ImportError:
        return
    if hasattr(cupy.cuda, "compile_with_cache"):
        return

    class _CompatModule:
        def __init__(self, source):
            self._mod = cupy.RawModule(code=source)

        def get_function(self, name):
            return self._mod.get_function(name)

    cupy.cuda.compile_with_cache = lambda src: _CompatModule(src)  # type: ignore[attr-defined]


__all__ = [
    "apply_np_int_shim",
    "apply_scipy_as_dcm_shim",
    "apply_numpy_linalg_submodule_shim",
    "apply_cupy_compat_shim",
]
