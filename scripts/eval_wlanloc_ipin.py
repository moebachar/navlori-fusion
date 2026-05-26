"""Open-source WiFi baseline on IPIN floor -2 (Phase B comparison).

Same approach as `scripts/eval_wlanloc_uji.py` but on IPIN. Single floor, so
their cascade reduces to one `PositionRegressor` call (no building or floor
classification needed). Their classes imported pure (no source edits).

This replaces our `scripts/eval_cnnloc_ipin.py` (manual reimplementation,
violated demand #3).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WLAN_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src")


def _load(rel: str, name: str):
    sys.modules.setdefault("wlan_localization", types.ModuleType("wlan_localization"))
    sys.modules.setdefault("wlan_localization.utils", types.ModuleType("wlan_localization.utils"))
    logmod = types.ModuleType("wlan_localization.utils.logger")
    logmod.get_logger = lambda n: logging.getLogger(n)
    sys.modules["wlan_localization.utils.logger"] = logmod
    spec = importlib.util.spec_from_file_location(
        name, WLAN_SRC / "wlan_localization" / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PositionRegressor = _load("models/position_regressor.py", "_pr").PositionRegressor
DataPreprocessor = _load("data/preprocessor.py", "_dp").DataPreprocessor


def load_ipin_wifi():
    from src.pipeline.fusion.builder import build_datamodule, load_config
    cfg = load_config("ipin2024_floor-2")
    cfg.dataset.modalities = ["wifi"]
    # Use raw RSSI (no PCA, no z-score) so the baseline's own preprocessor
    # can do its job. Override the config defaults.
    cfg.dataset.preprocessing.wifi_pca = None
    cfg.dataset.preprocessing.wifi_norm = "raw"
    dm = build_datamodule(cfg)
    Xtr, Ytr = dm.train_ds.get_tensors("wifi")
    Xva, Yva = dm.val_ds.get_tensors("wifi")
    # raw mode returns (rssi+100)/100 in [0,1]. Invert back to original
    # RSSI ([-100,0]) so wlan_localization's preprocessor sees real RSSI.
    Xtr = (Xtr.squeeze(1).numpy() * 100.0 - 100.0).astype(np.float64)
    Xva = (Xva.squeeze(1).numpy() * 100.0 - 100.0).astype(np.float64)
    return Xtr, Ytr.numpy().astype(np.float64), Xva, Yva.numpy().astype(np.float64)


def main():
    Xtr, Ytr, Xva, Yva = load_ipin_wifi()
    print(f"IPIN floor -2 WiFi: train {len(Xtr)} val {len(Xva)}  APs {Xtr.shape[1]}")

    # Their preprocessor expects -104..0 + sentinel 100. Our raw IPIN may have
    # samples that are "wifi absent" (all entries near -100 = no signal). The
    # preprocessor handles missing via the `missing_value` param; default is 100
    # but our absent samples are -100. Use missing_value=-100 instead.
    pre = DataPreprocessor(missing_value=-100.0)
    Xtr_p = pre.fit_transform(Xtr)
    Xva_p = pre.transform(Xva)
    print(f"  preprocessor: {Xtr.shape[1]} -> {Xtr_p.shape[1]} dims")

    reg = PositionRegressor(k=5, metric="manhattan", weights="distance")
    reg.fit_location(0, 0, Xtr_p, Ytr)
    pred = reg.models[(0, 0)].predict(Xva_p)
    mae = float(np.sqrt(((pred - Yva) ** 2).sum(1)).mean())
    print(f"\n>>> wlan_localization on IPIN val (WiFi-only baseline): {mae:.3f} m")


if __name__ == "__main__":
    main()
