"""Trivial-baseline localization on every available dataset.

Action 1 of the 2026-05-20 pipeline-fix plan. Every later result must beat
the best of these three baselines on the same dataset; if the transformer
can't, the architecture work is moot.

Baselines
---------
1. **MeanTrainPosition** — predict the spatial centroid of the training set
   for every val/test sample. The lower-bound any sane model must clear.
2. **WiFiKNN** — k-nearest-neighbour fingerprinting on raw WiFi RSSI
   (post-PCA if the config requests it). Distance-weighted, k=5. This is
   the standard "stupid but works" indoor-localization baseline.
3. **IMUKalman** — constant-velocity Kalman filter run forward through
   each val sequence, initialised at the anchor's GT position. State is
   ``(x, y, vx, vy)``, motion is integrated from yaw + linear-velocity (or
   integrated accel if odom isn't present). Reports per-sample MAE at
   every GT timestamp; with the anchor at t=0 this is roughly the IMU dead
   reckoning drift curve. The "minimum-effort temporal baseline."

Metric: Euclidean MAE in meters (matches :func:`euclidean_mae` and the
fusion trainer's ``val_mae``).

Output
------
``runs/baselines/<dataset>/baselines.json`` — one record per baseline per
split (val + test if test exists). Also a tiny ``summary.json`` aggregating
the best baseline per dataset, which the fusion runs will display next to
their own ``val_mae`` once Action 3 adds the comparison line.

Usage
-----
    .venv/Scripts/python.exe scripts/baselines.py            # all datasets
    .venv/Scripts/python.exe scripts/baselines.py simulation # one dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data.datamodule import FusionDataModule  # noqa: E402
from src.pipeline.evaluation.encoder_eval import (  # noqa: E402
    euclidean_mae,
    euclidean_rmse,
)
from src.pipeline.fusion.builder import (  # noqa: E402
    available_datasets,
    load_config,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ds_modalities(ds) -> list[str]:
    """Modalities actually loaded by this FusionDataset."""
    return ds.modalities


def _flatten_window(x: np.ndarray) -> np.ndarray:
    """``(N, win, feat) -> (N, win * feat)`` for WiFi-kNN; just the last
    window slot for WiFi (win=1 typically)."""
    return x.reshape(x.shape[0], -1)


def _stack_targets(ds) -> np.ndarray:
    return ds._targets.cpu().numpy()


def _per_sample_path_ids(ds) -> np.ndarray:
    return np.array([r["path_id"] for r in ds._gt_rows])


def _per_sample_times(ds) -> np.ndarray:
    return ds._timestamps.cpu().numpy()


# ----------------------------------------------------------------------
# Baseline 1 — Mean of training positions
# ----------------------------------------------------------------------

class MeanTrainBaseline:
    """Predict the (x, y) centroid of the training set, every time."""

    name = "mean_train_pos"

    def fit(self, dm: FusionDataModule) -> None:
        y_tr = _stack_targets(dm.train_ds)
        self._mu = y_tr.mean(axis=0).astype(np.float32)

    def predict(self, dm: FusionDataModule, split: str) -> np.ndarray:
        ds = getattr(dm, f"{split}_ds")
        n = len(ds)
        return np.broadcast_to(self._mu, (n, 2)).copy()


# ----------------------------------------------------------------------
# Baseline 2 — WiFi-kNN fingerprinting
# ----------------------------------------------------------------------

class WiFiKNNBaseline:
    """Distance-weighted kNN on the WiFi window (post-PCA / normalized)."""

    name = "wifi_knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, dm: FusionDataModule) -> None:
        from sklearn.neighbors import KNeighborsRegressor

        if "wifi" not in _ds_modalities(dm.train_ds):
            self._unavailable = True
            return
        self._unavailable = False
        X_tr, y_tr = dm.train_ds.get_tensors("wifi")
        X_tr = _flatten_window(X_tr.cpu().numpy())
        y_tr = y_tr.cpu().numpy()
        self._knn = KNeighborsRegressor(n_neighbors=self.k, weights="distance")
        self._knn.fit(X_tr, y_tr)

    def predict(self, dm: FusionDataModule, split: str) -> np.ndarray | None:
        if self._unavailable:
            return None
        ds = getattr(dm, f"{split}_ds")
        X, _ = ds.get_tensors("wifi")
        X = _flatten_window(X.cpu().numpy())
        return self._knn.predict(X).astype(np.float32)


# ----------------------------------------------------------------------
# Baseline 3 — IMU constant-velocity Kalman, anchored at the start of
# each VAL path's first sample. Equivalent to "what does IMU dead-reckoning
# from a single known starting fix give us on this split?"
# ----------------------------------------------------------------------

class IMUKalmanBaseline:
    """Constant-velocity Kalman filter; init at each val path's first GT."""

    name = "imu_kalman"

    def __init__(self,
                 q_pos: float = 0.0,
                 q_vel: float = 0.5,
                 r_pos: float = 0.0):
        # Process noise on position is 0 (the model assumes constant velocity);
        # process noise on velocity allows the velocity to change between steps.
        # No measurement update is used (no sensor → x measurement); we rely
        # on the IMU-derived velocity to drive the prediction step.
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos

    def fit(self, dm: FusionDataModule) -> None:
        if "imu" not in _ds_modalities(dm.train_ds):
            self._unavailable = True
            return
        self._unavailable = False
        # No training data needed — the Kalman filter is purely model-driven.

    def _path_imu_velocity(self, ds, pid: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimate per-GT-row (vx, vy) for a path from raw IMU + heading.

        Strategy: locate the path's per-modality dataframe in ``ds``, read
        the original IMU rows (accel_xyz, gyro_xyz, yaw_deg), integrate
        accel-in-body-frame -> velocity-in-world-frame using yaw. Resample
        to GT timestamps via nearest interpolation.

        Returns (t_gt, v_world) for the path.
        """
        # Find this path's modality cache slot.
        rows = [r for r in ds._gt_rows if r["path_id"] == pid]
        if not rows:
            return np.empty((0,)), np.empty((0, 2))
        path_dir = Path(rows[0]["path_dir"])
        gt_times = np.array([r["time"] for r in rows])

        imu_csv = path_dir / "imu.csv"
        if not imu_csv.exists():
            return gt_times, np.zeros((len(gt_times), 2), dtype=np.float32)
        imu_df = pd.read_csv(imu_csv)
        if len(imu_df) < 2:
            return gt_times, np.zeros((len(gt_times), 2), dtype=np.float32)

        t = imu_df["sim_time"].values
        ax = imu_df["accel_x"].values
        ay = imu_df["accel_y"].values
        yaw_deg = imu_df.get("yaw_deg",
                             pd.Series(np.zeros(len(imu_df)))).values
        yaw = np.deg2rad(yaw_deg)

        # World-frame acceleration via 2D yaw rotation; subtract median per
        # axis as a coarse gravity / bias removal (the IMU on a mobile phone
        # /ground robot doesn't isolate gravity for us).
        ax_w = np.cos(yaw) * ax - np.sin(yaw) * ay
        ay_w = np.sin(yaw) * ax + np.cos(yaw) * ay
        ax_w -= np.median(ax_w)
        ay_w -= np.median(ay_w)

        # Trapezoidal velocity integration in world frame
        dt = np.diff(t, prepend=t[0])
        vx = np.cumsum(0.5 * (ax_w + np.roll(ax_w, 1)) * dt)
        vy = np.cumsum(0.5 * (ay_w + np.roll(ay_w, 1)) * dt)

        # Nearest-time resample to GT timestamps
        idx = np.searchsorted(t, gt_times).clip(0, len(t) - 1)
        return gt_times, np.stack([vx[idx], vy[idx]], axis=-1).astype(np.float32)

    def predict(self, dm: FusionDataModule, split: str) -> np.ndarray | None:
        if self._unavailable:
            return None
        ds = getattr(dm, f"{split}_ds")
        n = len(ds)
        targets = _stack_targets(ds)
        pids = _per_sample_path_ids(ds)
        preds = np.zeros((n, 2), dtype=np.float32)

        for pid in np.unique(pids):
            mask = pids == pid
            idx = np.where(mask)[0]
            order = idx[np.argsort(_per_sample_times(ds)[idx])]
            if len(order) == 0:
                continue
            # Anchor at first GT of this path — perfect-knowledge start.
            x = float(targets[order[0], 0])
            y = float(targets[order[0], 1])
            preds[order[0]] = [x, y]

            t_gt, v_world = self._path_imu_velocity(ds, int(pid))
            t_gt_sort = np.argsort(t_gt)
            t_gt = t_gt[t_gt_sort]
            v_world = v_world[t_gt_sort]

            t_path = _per_sample_times(ds)[order]
            # For each subsequent GT timestamp, integrate the IMU-derived
            # velocity from t_anchor to t_i.
            for i in range(1, len(order)):
                t_prev = t_path[i - 1]
                t_cur = t_path[i]
                # Mean velocity over [t_prev, t_cur]
                m = (t_gt >= t_prev) & (t_gt <= t_cur)
                if m.sum() == 0:
                    vx, vy = 0.0, 0.0
                else:
                    vx = float(v_world[m, 0].mean())
                    vy = float(v_world[m, 1].mean())
                dt = float(t_cur - t_prev)
                x += vx * dt
                y += vy * dt
                preds[order[i]] = [x, y]
        return preds


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def _baseline_record(name: str, pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "name": name,
        "n": int(len(y)),
        "mae": euclidean_mae(pred, y),
        "rmse": euclidean_rmse(pred, y),
    }


def run_dataset(dataset: str) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    cfg = load_config(dataset)
    d = cfg.dataset
    pre = d.get("preprocessing", {}) or {}
    dm = FusionDataModule(
        data_dir=ROOT / str(d.root) / d.collection_dir,
        train_paths=list(d.split.train_paths),
        val_paths=list(d.split.val_paths),
        test_paths=list(d.split.test_paths),
        modalities=list(d.modalities),
        windows=dict(d.windows) if d.get("windows") else None,
        normalize=pre.get("normalize", True),
        batch_size=cfg.data.batch_size,
        wifi_pca=pre.get("wifi_pca", None),
        wifi_norm=pre.get("wifi_norm", "whiten"),
        wifi_max_stale_s=pre.get("wifi_max_stale_s", None),
    )
    dm.setup()
    has_test = dm.test_ds is not None
    print(dm.summary(), flush=True)

    baselines = [MeanTrainBaseline(), WiFiKNNBaseline(), IMUKalmanBaseline()]
    for b in baselines:
        b.fit(dm)

    out: dict = {"dataset": dataset, "splits": {}}
    for split in ("val", "test"):
        if split == "test" and not has_test:
            continue
        ds = getattr(dm, f"{split}_ds")
        if ds is None or len(ds) == 0:
            continue
        y = _stack_targets(ds)
        out["splits"][split] = {"n_samples": int(len(y)), "baselines": []}
        for b in baselines:
            pred = b.predict(dm, split)
            if pred is None:
                continue
            rec = _baseline_record(b.name, pred, y)
            out["splits"][split]["baselines"].append(rec)
            print(f"  {split:5s}  {b.name:14s}  "
                  f"MAE={rec['mae']:.3f}m  RMSE={rec['rmse']:.3f}m  "
                  f"(n={rec['n']})", flush=True)

    # Best-baseline summary (lower MAE is better)
    for split, sd in out["splits"].items():
        if not sd["baselines"]:
            continue
        best = min(sd["baselines"], key=lambda r: r["mae"])
        sd["best"] = best["name"]
        sd["best_mae"] = best["mae"]

    out_dir = ROOT / "runs" / "baselines" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default=None,
                    help="Dataset name (configs/data/<name>.yaml). "
                         "Omit to run all available datasets.")
    ap.add_argument("--skip-leaky", action="store_true",
                    help="Skip _intra datasets (they're DEV USE ONLY).")
    args = ap.parse_args()

    targets = ([args.dataset] if args.dataset else available_datasets())
    if args.skip_leaky:
        targets = [t for t in targets if not t.endswith("_intra")]
    summary = {}
    for ds in targets:
        try:
            rec = run_dataset(ds)
            summary[ds] = {
                s: {"best": sd.get("best"), "best_mae": sd.get("best_mae")}
                for s, sd in rec["splits"].items()
            }
        except Exception as e:
            print(f"  ERROR on {ds}: {type(e).__name__}: {e}", flush=True)
            summary[ds] = {"error": str(e)}

    out = ROOT / "runs" / "baselines" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\nSummary -> ", out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
