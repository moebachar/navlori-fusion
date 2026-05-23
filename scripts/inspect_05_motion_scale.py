"""PROBE 5 — motion-signal quality + spatial scale context.

(1) Motion learnability: can IMU predict its own ~1 s displacement? Build
    (imu_window -> GT displacement over the window) pairs, fit a kNN on
    train, evaluate displacement MAE on val. Compare to:
      - 'predict zero motion' (= mean |displacement|)
    If kNN << zero-motion, IMU carries usable displacement signal.

(2) Spatial scale: floor extent + the 'predict global centroid' error
    (the true do-nothing floor) so MAE numbers are interpretable.

Pure data + cheap kNN. Run:
  .venv/Scripts/python.exe scripts/inspect_05_motion_scale.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

from src.pipeline.data.datamodule import FusionDataModule  # noqa: E402
from src.pipeline.fusion.builder import load_config  # noqa: E402

DATASETS = ["simulation", "ipin2024_floor-2", "ronin_a000"]


def build_dm(name):
    cfg = load_config(name)
    d = cfg.dataset
    pre = d.get("preprocessing", {}) or {}
    dm = FusionDataModule(
        data_dir=ROOT / str(d.root) / d.collection_dir,
        train_paths=list(d.split.train_paths), val_paths=list(d.split.val_paths),
        test_paths=list(d.split.test_paths) or [list(d.split.train_paths)[0]],
        modalities=list(d.modalities),
        windows=dict(d.windows) if d.get("windows") else None,
        normalize=pre.get("normalize", True), batch_size=128,
        wifi_pca=pre.get("wifi_pca", None))
    dm.setup()
    return dm


def analyze(name):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    dm = build_dm(name)
    tr, va = dm.train_ds, dm.val_ds

    # --- spatial scale + centroid floor ---
    ytr = tr._targets.numpy(); yva = va._targets.numpy()
    ext = [ytr[:, 0].max() - ytr[:, 0].min(), ytr[:, 1].max() - ytr[:, 1].min()]
    cen = ytr.mean(0)
    cen_err = np.linalg.norm(yva - cen, axis=1).mean()
    print(f"  floor extent ~ {ext[0]:.0f} x {ext[1]:.0f} m")
    print(f"  'predict global centroid' val MAE = {cen_err:.2f} m  (do-nothing floor)")

    # --- motion learnability (IMU -> 1s displacement) ---
    if "imu" not in tr.modalities:
        print("  no imu modality"); return
    Xtr = tr.get_tensors("imu")[0]                    # (N, win, feat) normalized
    Xva = va.get_tensors("imu")[0]
    dtr, vtr = tr.get_targets("displacement", 1.0)
    dva, vva = va.get_targets("displacement", 1.0)
    Xtr, dtr = Xtr[vtr].reshape(int(vtr.sum()), -1).numpy(), dtr[vtr].numpy()
    Xva, dva = Xva[vva].reshape(int(vva.sum()), -1).numpy(), dva[vva].numpy()
    if len(Xva) == 0:
        print("  no valid displacement samples"); return

    zero_motion = np.linalg.norm(dva, axis=1).mean()  # predict (0,0)
    knn = KNeighborsRegressor(n_neighbors=10, weights="distance").fit(Xtr, dtr)
    pred = knn.predict(Xva)
    knn_err = np.linalg.norm(pred - dva, axis=1).mean()
    disp_mag = np.linalg.norm(dva, axis=1).mean()
    print(f"  IMU 1s-displacement (mean |disp| = {disp_mag:.3f} m):")
    print(f"    predict-zero-motion MAE = {zero_motion:.3f} m")
    print(f"    kNN(IMU->disp) val MAE  = {knn_err:.3f} m")
    skill = 1 - knn_err / zero_motion
    verdict = ("usable" if skill > 0.2 else "WEAK" if skill > 0.05 else "NONE")
    print(f"    >>> motion skill = {skill*100:.0f}%  [{verdict}]")


def main():
    for ds in (sys.argv[1:] or DATASETS):
        try:
            analyze(ds)
        except Exception as e:
            print(f"  ERROR {ds}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
