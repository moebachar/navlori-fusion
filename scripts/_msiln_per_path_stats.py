"""Throwaway helper for PLAN_02 step 5 + 6:

- Per-path MAE distribution (median, p25, p75, p90, max) for each
  baseline x split, on msiln_site1_b1.
- Per-sample vs per-waypoint MAE comparison for the centroid baseline
  (step 6 — gap should be < 20% if linear interpolation isn't biasing
  scores meaningfully).

Per-waypoint metric uses the ORIGINAL surveyor-clicked waypoint
timestamps from each trace's raw .txt (decoded via vendored io_f.py),
mapped to the nearest converted-CSV GT row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data.datamodule import FusionDataModule  # noqa: E402
from src.pipeline.fusion.builder import load_config  # noqa: E402
from scripts.baselines import (  # noqa: E402
    IMUKalmanBaseline,
    MeanTrainBaseline,
    WiFiKNNBaseline,
)


def _build_dm():
    cfg = load_config("msiln_site1_b1")
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
    return dm


def _per_path_distribution(pred: np.ndarray, y: np.ndarray,
                           pids: np.ndarray) -> dict:
    err = np.linalg.norm(pred - y, axis=1)
    per_path_mae = []
    for pid in np.unique(pids):
        mask = pids == pid
        per_path_mae.append(float(err[mask].mean()))
    arr = np.array(per_path_mae)
    return {
        "n_paths": int(len(arr)),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "max":    float(arr.max()),
    }


def _load_vendored_io():
    msiln_root = Path(r"C:\Users\FabLab\AppData\Local\Temp\msiln20")
    spec = importlib.util.spec_from_file_location(
        "msiln20_io_f", msiln_root / "io_f.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.read_data_file, msiln_root


def _waypoint_indices_for_val(dm) -> np.ndarray:
    """For each val sample, return True iff its sim_time matches an original
    waypoint timestamp from the raw .txt (within 50 ms tolerance after
    re-zeroing to t_lo_ms).

    The converted GT is 10 Hz interpolated; the original waypoints are at
    irregular times within each trace. We need them via the raw .txt
    because the converter doesn't preserve which rows are anchors.
    """
    read_data_file, msiln_root = _load_vendored_io()
    ds = dm.val_ds

    pids = np.array([r["path_id"] for r in ds._gt_rows])
    times = ds._timestamps.cpu().numpy()
    is_waypoint = np.zeros(len(pids), dtype=bool)

    for pid in np.unique(pids):
        # Find this path's source .txt via metadata.json
        path_dir = Path(ds._gt_rows[int(np.where(pids == pid)[0][0])]["path_dir"])
        meta = json.loads((path_dir / "metadata.json").read_text())
        src_name = meta["source_file"]
        site = meta["site"]
        floor = meta["floor"]
        raw_txt = msiln_root / "data" / site / floor / "path_data_files" / src_name
        if not raw_txt.exists():
            continue
        d = read_data_file(str(raw_txt))
        wp = d.waypoint
        if len(wp) == 0:
            continue
        t_lo_ms = float(wp[0, 0])
        # Original waypoint sim_times (seconds, re-zeroed)
        wp_sim_s = (wp[:, 0].astype(np.float64) - t_lo_ms) / 1000.0

        mask_pid = pids == pid
        idx_pid = np.where(mask_pid)[0]
        path_times = times[idx_pid]
        # For each waypoint, find the nearest converted-row in this path
        for wts in wp_sim_s:
            j = int(np.argmin(np.abs(path_times - wts)))
            if abs(path_times[j] - wts) < 0.05:  # 50 ms tolerance
                is_waypoint[idx_pid[j]] = True

    return is_waypoint


def main():
    print("Building datamodule…", flush=True)
    dm = _build_dm()

    baselines = [MeanTrainBaseline(), WiFiKNNBaseline(), IMUKalmanBaseline()]
    for b in baselines:
        b.fit(dm)

    out_rows = []
    for split in ("val", "test"):
        ds = getattr(dm, f"{split}_ds")
        y = ds._targets.cpu().numpy()
        pids = np.array([r["path_id"] for r in ds._gt_rows])
        for b in baselines:
            pred = b.predict(dm, split)
            if pred is None:
                continue
            agg = _per_path_distribution(pred, y, pids)
            agg["split"] = split
            agg["baseline"] = b.name
            out_rows.append(agg)

    print()
    print(f"{'split':<6}{'baseline':<16}{'n_paths':>9}{'mean':>9}{'median':>9}"
          f"{'p25':>9}{'p75':>9}{'p90':>9}{'max':>9}")
    for r in out_rows:
        print(f"{r['split']:<6}{r['baseline']:<16}"
              f"{r['n_paths']:>9}{r['mean']:>9.2f}{r['median']:>9.2f}"
              f"{r['p25']:>9.2f}{r['p75']:>9.2f}{r['p90']:>9.2f}{r['max']:>9.2f}")

    # ── per-sample vs per-waypoint (centroid baseline, val split) ───────────
    print("\nLocating original waypoint indices in val…", flush=True)
    is_wp = _waypoint_indices_for_val(dm)
    n_wp = int(is_wp.sum())
    n_total = len(is_wp)
    print(f"  {n_wp}/{n_total} val samples are at original waypoint timestamps "
          f"({100*n_wp/n_total:.1f}%)", flush=True)

    centroid = baselines[0]
    pred = centroid.predict(dm, "val")
    y = dm.val_ds._targets.cpu().numpy()
    err = np.linalg.norm(pred - y, axis=1)
    mae_all = float(err.mean())
    mae_wp = float(err[is_wp].mean()) if n_wp > 0 else float("nan")
    gap_pct = 100.0 * abs(mae_all - mae_wp) / mae_all if mae_all > 0 else float("nan")
    print(f"  centroid per-sample   MAE = {mae_all:.3f} m")
    print(f"  centroid per-waypoint MAE = {mae_wp:.3f} m  "
          f"(gap = {gap_pct:.1f}%; pass if < 20%)")

    out_dir = ROOT / "runs" / "overnight" / "iter_02"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_path_stats.json").write_text(json.dumps({
        "per_path": out_rows,
        "metric_gap_centroid_val": {
            "mae_per_sample": mae_all,
            "mae_per_waypoint": mae_wp,
            "n_waypoints": n_wp,
            "n_samples": n_total,
            "gap_pct": gap_pct,
        },
    }, indent=2))
    print(f"\nWrote {out_dir / 'per_path_stats.json'}")


if __name__ == "__main__":
    main()
