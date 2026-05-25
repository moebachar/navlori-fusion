"""Open-source WiFi baseline on UJIIndoorLoc — wlan_localization's PositionRegressor.

Uses sharan-naribole/wlan_localization (open source, MIT, 2.6-8.2 m on UJI
per their README), specifically their `PositionRegressor` class (per-floor
KNN regressor on raw RSSI). We load the class directly from their source
file via `importlib` to bypass their package `__init__` chain (which pulls
in an `imbalanced-learn` dep version incompatible with our scikit-learn).
NO modification to their code — only Python-level loading mechanics.

Two evaluation modes:
  * `cascade-oracle`: their per-(building,floor) regression with KNOWN
    building/floor on val (oracle classification). This tests just the
    position-regression quality, isolated from class-imbalance issues.
  * `global`: one global KNN regressor (no cascade). Apples-to-apples with
    our Anchor2Vec (also pure regression, no floor cascade).
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UJI = ROOT / "data" / "uji_indoorloc"
WLANLOC_SRC = Path(r"C:\Users\FabLab\AppData\Local\Temp\wlan_localization\src")


def _stub_logger():
    base = types.ModuleType("wlan_localization")
    utils = types.ModuleType("wlan_localization.utils")
    logmod = types.ModuleType("wlan_localization.utils.logger")
    import logging
    logmod.get_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("wlan_localization", base)
    sys.modules.setdefault("wlan_localization.utils", utils)
    sys.modules.setdefault("wlan_localization.utils.logger", logmod)


def _load_pure(rel_path: str, mod_name: str):
    """Load one of their source files directly, bypassing the package init
    chain that drags in imblearn (incompatible with our sklearn)."""
    spec = importlib.util.spec_from_file_location(
        mod_name, WLANLOC_SRC / "wlan_localization" / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_position_regressor():
    _stub_logger()
    return _load_pure("models/position_regressor.py", "wlan_pos_reg").PositionRegressor


def _load_preprocessor():
    _stub_logger()
    return _load_pure("data/preprocessor.py", "wlan_preproc").DataPreprocessor


def load_uji(csv):
    df = pd.read_csv(csv)
    waps = [c for c in df.columns if c.startswith("WAP")]
    return (df[waps].values.astype(np.float64),
            df[["LATITUDE", "LONGITUDE", "BUILDINGID", "FLOOR"]].copy())


def main():
    PositionRegressor = _load_position_regressor()
    DataPreprocessor = _load_preprocessor()
    Xtr_raw, Ytr = load_uji(UJI / "trainingData.csv")
    Xva_raw, Yva = load_uji(UJI / "validationData.csv")
    print(f"UJI: train {len(Xtr_raw)} val {len(Xva_raw)} APs {Xtr_raw.shape[1]}")

    # Apply their preprocessor (missing-value handling + Box-Cox + scaling).
    pre = DataPreprocessor()
    Xtr = pre.fit_transform(Xtr_raw)
    Xva = pre.transform(Xva_raw)
    print(f"preprocessor: in dim {Xtr_raw.shape[1]} -> out dim {Xtr.shape[1]}")

    # === mode A: their cascade with ORACLE building+floor ===
    print("\n[cascade-oracle] their PositionRegressor per (building, floor), oracle labels")
    reg = PositionRegressor(k=3, metric="manhattan", weights="distance")
    t0 = time.time()
    # train per (building, floor)
    for (b, f), grp in Ytr.groupby(["BUILDINGID", "FLOOR"]):
        idx = grp.index.values
        reg.fit_location(int(b), int(f),
                          Xtr[idx],
                          Ytr.loc[idx, ["LATITUDE", "LONGITUDE"]].values)
    # predict per (building, floor) using oracle val labels
    preds = np.zeros((len(Xva), 2))
    for (b, f), grp in Yva.groupby(["BUILDINGID", "FLOOR"]):
        idx = grp.index.values
        if (int(b), int(f)) in reg.models:
            preds[idx] = reg.models[(int(b), int(f))].predict(Xva[idx])
    gt = Yva[["LATITUDE", "LONGITUDE"]].values
    err_o = np.sqrt(((preds - gt) ** 2).sum(1))
    err_oracle = float(err_o.mean())
    print(f"  cascade-oracle mean Euclidean = {err_oracle:.3f} m  ({time.time()-t0:.0f}s)")
    print(f"    per-sample p25={np.percentile(err_o,25):.2f} p50={np.percentile(err_o,50):.2f} "
          f"p75={np.percentile(err_o,75):.2f} p90={np.percentile(err_o,90):.2f} max={err_o.max():.2f}")

    # === mode B: one global model (apples-to-apples with Anchor2Vec pure regression) ===
    print("\n[global] one PositionRegressor model, all-buildings/floors (pure regression)")
    reg2 = PositionRegressor(k=3, metric="manhattan", weights="distance")
    t0 = time.time()
    reg2.fit_location(0, 0, Xtr, Ytr[["LATITUDE", "LONGITUDE"]].values)
    preds2 = reg2.models[(0, 0)].predict(Xva)
    err_g = np.sqrt(((preds2 - gt) ** 2).sum(1))
    err_global = float(err_g.mean())
    print(f"  global  mean Euclidean = {err_global:.3f} m  ({time.time()-t0:.0f}s)")
    print(f"    per-sample p25={np.percentile(err_g,25):.2f} p50={np.percentile(err_g,50):.2f} "
          f"p75={np.percentile(err_g,75):.2f} p90={np.percentile(err_g,90):.2f} max={err_g.max():.2f}")

    print(f"\n>>> wlan_localization (open-source) on UJI val:")
    print(f"    cascade-oracle (their full pipeline, known building/floor): {err_oracle:.2f} m")
    print(f"    global         (pure regression, apples-to-apples vs ours): {err_global:.2f} m")
    print(f"    reference: ~5.28 m global / 2.6-8.2 m per-location (their README)")


if __name__ == "__main__":
    main()
