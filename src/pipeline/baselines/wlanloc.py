"""WiFi SOTA baseline — sharan-naribole/wlan_localization (MIT).

Bypass-import their ``PositionRegressor`` and ``DataPreprocessor``
classes directly from the vendored sources. The package's
``__init__.py`` drags in ``imbalanced-learn``/``scikit-learn`` version
conflicts that don't resolve cleanly in our venv, so we use
``importlib.util.spec_from_file_location`` to load the two class files
directly.

Demand #3: no edits to the vendored sources under
``external_methods/wlan_localization/``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types

from ._paths import WLANLOC_SRC


def _stub_wlanloc_logger() -> None:
    """Pre-create stub modules ``wlan_localization.utils.logger`` so
    relative imports inside the source files resolve to a no-op logger
    getter."""
    pkg = types.ModuleType("wlan_localization")
    utils = types.ModuleType("wlan_localization.utils")
    logmod = types.ModuleType("wlan_localization.utils.logger")
    logmod.get_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("wlan_localization", pkg)
    sys.modules.setdefault("wlan_localization.utils", utils)
    sys.modules.setdefault("wlan_localization.utils.logger", logmod)


def _load_pure(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, WLANLOC_SRC / "wlan_localization" / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_position_regressor():
    """Return the ``PositionRegressor`` class."""
    _stub_wlanloc_logger()
    return _load_pure(
        "models/position_regressor.py", "wlan_pos_reg").PositionRegressor


def load_preprocessor():
    """Return the ``DataPreprocessor`` class."""
    _stub_wlanloc_logger()
    return _load_pure(
        "data/preprocessor.py", "wlan_preproc").DataPreprocessor


__all__ = ["load_position_regressor", "load_preprocessor"]
