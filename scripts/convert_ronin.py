"""Convert a single RoNIN subject to navlori-fusion dataset format.

RoNIN sequences span many environments (one per subject), so a WiFi encoder
trained across subjects has nothing to learn — BSSIDs don't overlap.  This
converter restricts itself to sequences from ONE subject_identifier, where
BSSIDs do overlap, and emits a dataset with the same layout as
``data/async_collection/`` so the existing pipeline + notebook work unchanged.

Source layout
-------------
``Data/{train_dataset_1, train_dataset_2, seen_subjects_test_set,
unseen_subjects_test_set}/<subject>_<seq>/{data.hdf5, info.json}``

Each ``data.hdf5`` contains:
  - ``synced/{time, acce, gyro, rv, ...}`` at 200 Hz
  - ``pose/tango_pos`` (Tango VSLAM ground truth) at 200 Hz
  - ``raw/{tango|imu}/wifi_values`` (scan_number, ns_timestamp, rssi)
    and ``wifi_address`` (BSSIDs)
``info.json`` has ``start_frame`` (first valid synced index — must skip
earlier rows for calibration).

Outputs
-------
``<out_root>/ronin_<subject>[_intra]/``
    ├── path_NN/{imu.csv, wifi.csv, odometry.csv, ground_truth.csv,
    │            metadata.json, trajectory.png}
    ├── ap_vocab.json
    ├── split.json
    ├── metadata.json
    ├── convert_ronin.py
    └── configs_snippet.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ── tunables ────────────────────────────────────────────────────────────────
IMU_TARGET_HZ = 32
GT_TARGET_HZ = 10
MIN_GT_PER_PATH = 5
MIN_PATH_SECONDS = 5.0

# Native splits (folder name → split bucket)
NATIVE_SPLIT_DIRS = {
    "train_dataset_1": "train",
    "train_dataset_2": "train",
    "seen_subjects_test_set": "val",
    "unseen_subjects_test_set": "test",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def quat_xyzw_to_rpy_deg(q: np.ndarray) -> np.ndarray:
    """RoNIN rotation_vector is (x, y, z, w).  Returns (N, 3) of (roll, pitch, yaw) in degrees."""
    rot = R.from_quat(q)  # scipy uses (x, y, z, w) order — matches RoNIN
    return rot.as_euler("xyz", degrees=True)


# A sequence's split is decided by the most held-out folder it appears in.
# RoNIN's seen_subjects_test_set reuses sequence IDs that also live in
# train_dataset_* (e.g. a000_7 is in BOTH train_dataset_1 and
# seen_subjects_test_set). Without dedup the same walk lands in train AND
# val — the leak that made ronin_a000's 1.87m val_mae meaningless. Higher
# priority = more held-out; the held-out designation wins.
_SPLIT_PRIORITY = {"test": 3, "val": 2, "train": 1}


def find_subject_seqs(raw_root: Path, subject: str) -> list[tuple[Path, str]]:
    """Return [(seq_dir, native_split), ...], one entry per UNIQUE sequence.

    When a sequence name appears in multiple native folders, keep the
    most held-out split (test > val > train) and drop the duplicate, so a
    sequence is never assigned to two splits.
    """
    found: dict[str, tuple[Path, str]] = {}
    for sub, split in NATIVE_SPLIT_DIRS.items():
        d = raw_root / sub
        if not d.exists():
            continue
        for seq in sorted(d.glob(f"{subject}_*")):
            if not (seq / "data.hdf5").exists():
                continue
            name = seq.name
            prev = found.get(name)
            if prev is None:
                found[name] = (seq, split)
            elif _SPLIT_PRIORITY[split] > _SPLIT_PRIORITY[prev[1]]:
                print(f"  [dedup] {name}: in '{prev[1]}' and '{split}' folders; "
                      f"keeping held-out '{split}' (drops the train copy -> no leak)")
                found[name] = (seq, split)
            else:
                print(f"  [dedup] {name}: also in '{split}' folder; "
                      f"keeping existing '{prev[1]}'")
    out = list(found.values())
    out.sort(key=lambda x: int(x[0].name.split("_")[1]))
    return out


def get_wifi_arrays(f: h5py.File, synced_t0: float, synced_t1: float
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (wifi_values, wifi_address) using whichever group's timestamps
    overlap with the synced/time window.

    raw/imu/wifi_values shares the IMU device clock with synced/time, so
    that's the right source most of the time.  But on some sequences the
    IMU-side WiFi was logged with a frozen / pre-sync timestamp and only
    the tango-side WiFi is usable.  We pick whichever has timestamps in
    the synced range; if neither does, return None.
    """
    candidates = []
    for src in ("raw/imu", "raw/tango"):
        if f"{src}/wifi_values" in f and f"{src}/wifi_address" in f:
            wv = f[f"{src}/wifi_values"][:]
            wa = f[f"{src}/wifi_address"][:]
            if len(wv) == 0:
                continue
            wts = wv[:, 1] / 1e9
            # Need at least some scans whose timestamp falls inside the
            # synced/time window (with a 5 s grace period at each end).
            in_window = ((wts >= synced_t0 - 5) & (wts <= synced_t1 + 5)).sum()
            candidates.append((src, wv, wa, in_window))
    if not candidates:
        return None
    # Prefer the source with the most in-window scans; tiebreak: larger array
    candidates.sort(key=lambda c: (c[3], len(c[1])), reverse=True)
    src, wv, wa, in_window = candidates[0]
    if in_window == 0:
        return None
    return wv, wa


def load_sequence(seq_dir: Path):
    """Read one RoNIN sequence into the numeric arrays we need.

    Returns dict with:
      sim_time   (T,)            — synced time, re-zeroed to start_frame
      gt_xy      (T, 2)          — Tango (x, y) in meters
      acce       (T, 3)          — m/s^2
      gyro       (T, 3)          — rad/s
      rpy_deg    (T, 3)          — roll, pitch, yaw in degrees
      wifi_scans list of (scan_t_s, {bssid: rssi})
    or None if the sequence is unusable (no WiFi, too short, etc.).
    """
    info = json.loads((seq_dir / "info.json").read_text(encoding="utf-8"))
    start_frame = int(info.get("start_frame", 0))

    with h5py.File(seq_dir / "data.hdf5", "r") as f:
        synced_time = f["synced/time"][:]
        if start_frame >= len(synced_time) - 10:
            return None
        t = synced_time[start_frame:]
        t0 = float(t[0])
        sim_time = t - t0

        pos = f["pose/tango_pos"][start_frame:, :]
        acce = f["synced/acce"][start_frame:, :]
        gyro = f["synced/gyro"][start_frame:, :]
        rv = f["synced/rv"][start_frame:, :]  # quaternion (x, y, z, w)

        wifi = get_wifi_arrays(f, float(t[0]), float(t[-1]))

    if wifi is None:
        return None
    wifi_values, wifi_address = wifi
    bssids = [b.decode() if isinstance(b, bytes) else str(b) for b in wifi_address]

    rpy = quat_xyzw_to_rpy_deg(rv)

    # Group WIFI rows by scan_number.  wifi_values columns:
    #   (scan_number, last_timestep_ns, rssi)
    # Row i in wifi_values corresponds to BSSID wifi_address[i].  We use the
    # IMU device clock (last_timestep_ns / 1e9) which matches synced/time.
    scan_dict: dict[int, dict[str, float]] = {}
    scan_t_ns: dict[int, list[int]] = {}
    for i, row in enumerate(wifi_values):
        sn = int(row[0])
        ts_ns = int(row[1])
        rssi = float(row[2])
        bssid = bssids[i]
        scan_dict.setdefault(sn, {})
        cur = scan_dict[sn].get(bssid, -200.0)
        scan_dict[sn][bssid] = max(cur, rssi)
        scan_t_ns.setdefault(sn, []).append(ts_ns)

    wifi_scans: list[tuple[float, dict[str, float]]] = []
    for sn, rssi_map in scan_dict.items():
        ts_ns = float(np.median(scan_t_ns[sn]))
        scan_t_s = ts_ns / 1e9 - t0
        if scan_t_s < 0 or scan_t_s > sim_time[-1]:
            continue
        wifi_scans.append((scan_t_s, rssi_map))
    wifi_scans.sort(key=lambda x: x[0])

    return {
        "sim_time": sim_time,
        "gt_xy": pos[:, :2].astype(np.float32),
        "acce": acce.astype(np.float32),
        "gyro": gyro.astype(np.float32),
        "rpy_deg": rpy.astype(np.float32),
        "wifi_scans": wifi_scans,
    }


def downsample_imu(seq, target_hz: int) -> pd.DataFrame:
    t = seq["sim_time"]
    native_hz = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    step = max(int(round(native_hz / target_hz)), 1)
    idx = np.arange(0, len(t), step)
    return pd.DataFrame({
        "sim_time":  np.round(t[idx], 4),
        "accel_x":   seq["acce"][idx, 0],
        "accel_y":   seq["acce"][idx, 1],
        "accel_z":   seq["acce"][idx, 2],
        "gyro_x":    seq["gyro"][idx, 0],
        "gyro_y":    seq["gyro"][idx, 1],
        "gyro_z":    seq["gyro"][idx, 2],
        "roll_deg":  seq["rpy_deg"][idx, 0],
        "pitch_deg": seq["rpy_deg"][idx, 1],
        "yaw_deg":   seq["rpy_deg"][idx, 2],
    })


def downsample_gt(seq, target_hz: int) -> pd.DataFrame:
    t = seq["sim_time"]
    native_hz = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    step = max(int(round(native_hz / target_hz)), 1)
    idx = np.arange(0, len(t), step)
    return pd.DataFrame({
        "sim_time": np.round(t[idx], 3),
        "gt_x":     np.round(seq["gt_xy"][idx, 0], 4),
        "gt_y":     np.round(seq["gt_xy"][idx, 1], 4),
    })


def build_wifi_df(seq, bssid_vocab: list[str]) -> pd.DataFrame:
    rssi_cols = [f"wifi_rssi_{b.lower().replace(':', '')}" for b in bssid_vocab]
    bssid_to_col = {b: c for b, c in zip(bssid_vocab, rssi_cols)}
    rows = []
    for t_s, rssi_map in seq["wifi_scans"]:
        row = {c: np.nan for c in rssi_cols}
        visible = [(b, r) for b, r in rssi_map.items() if b in bssid_to_col]
        for b, r in visible:
            row[bssid_to_col[b]] = r
        if visible:
            best_b, best_r = max(visible, key=lambda x: x[1])
            best_mac = best_b.lower().replace(":", "")
        else:
            best_r, best_mac = np.nan, ""
        out = {
            "sim_time": round(t_s, 3),
            "wifi_visible_count": len(visible),
            "wifi_strongest_rssi": best_r,
            "wifi_strongest_mac": best_mac,
        }
        out.update(row)
        rows.append(out)
    cols = ["sim_time", "wifi_visible_count", "wifi_strongest_rssi",
            "wifi_strongest_mac"] + rssi_cols
    return pd.DataFrame(rows, columns=cols)


# ── chunking (same idea as convert_ipin2024) ─────────────────────────────────

def chunk_segment(gt_df, imu_df, wifi_df, chunk_seconds: float):
    if len(gt_df) < 2:
        return []
    t_min = float(gt_df["sim_time"].min())
    t_max = float(gt_df["sim_time"].max())
    duration = t_max - t_min
    if duration <= 0:
        return []
    n = max(1, int(round(duration / chunk_seconds)))
    bounds = np.linspace(t_min, t_max, n + 1)
    out = []
    for i in range(n):
        lo = float(bounds[i]); hi = float(bounds[i + 1])
        gt_mask = (gt_df["sim_time"] >= lo) & (
            (gt_df["sim_time"] <= hi) if i == n - 1 else (gt_df["sim_time"] < hi)
        )
        gt_c = gt_df.loc[gt_mask].reset_index(drop=True).copy()
        if len(gt_c) < 2:
            continue
        imu_c = imu_df.loc[(imu_df["sim_time"] >= lo) & (imu_df["sim_time"] <= hi)].reset_index(drop=True).copy()
        wifi_c = wifi_df.loc[(wifi_df["sim_time"] >= lo) & (wifi_df["sim_time"] <= hi)].reset_index(drop=True).copy()
        t0 = float(gt_c["sim_time"].iloc[0])
        gt_c["sim_time"] = (gt_c["sim_time"] - t0).round(3)
        if len(imu_c):
            imu_c["sim_time"] = (imu_c["sim_time"] - t0).round(4)
            imu_c = imu_c.loc[imu_c["sim_time"] >= 0].reset_index(drop=True)
        if len(wifi_c):
            wifi_c["sim_time"] = (wifi_c["sim_time"] - t0).round(3)
            wifi_c = wifi_c.loc[wifi_c["sim_time"] >= 0].reset_index(drop=True)
        out.append((gt_c, imu_c, wifi_c))
    return out


# ── writers ─────────────────────────────────────────────────────────────────

IMU_STUB = ("sim_time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,"
            "roll_deg,pitch_deg,yaw_deg\n")
WIFI_STUB = "sim_time,wifi_visible_count,wifi_strongest_rssi,wifi_strongest_mac\n"
ODOM_STUB = ("sim_time,odom_x,odom_y,odom_theta_deg,"
             "odom_linear_vel,odom_angular_vel,wheel_left_vel,wheel_right_vel\n")


def write_path(out_dir, gt_df, imu_df, wifi_df, source_name, path_id, native_split, seg_index) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(gt_df["sim_time"].max() - gt_df["sim_time"].min())
    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)
    if len(imu_df):
        imu_df.to_csv(out_dir / "imu.csv", index=False)
    else:
        (out_dir / "imu.csv").write_text(IMU_STUB, encoding="utf-8")
    if len(wifi_df):
        wifi_df.to_csv(out_dir / "wifi.csv", index=False)
    else:
        (out_dir / "wifi.csv").write_text(WIFI_STUB, encoding="utf-8")
    (out_dir / "odometry.csv").write_text(ODOM_STUB, encoding="utf-8")

    meta = {
        "dataset": "RoNIN",
        "source_seq": source_name,
        "segment_index": seg_index,
        "path_id": path_id,
        "native_split": native_split,
        "duration_s": round(duration, 2),
        "n_gt_samples": len(gt_df),
        "gt_rate_hz": round(len(gt_df) / duration, 2) if duration > 0 else 0,
        "n_wifi_scans": len(wifi_df),
        "wifi_rate_hz": round(len(wifi_df) / duration, 3) if duration > 0 else 0,
        "n_imu_samples": len(imu_df),
        "imu_rate_hz": round(len(imu_df) / duration, 1) if duration > 0 else 0,
        "x_range_m": [round(float(gt_df["gt_x"].min()), 3),
                      round(float(gt_df["gt_x"].max()), 3)],
        "y_range_m": [round(float(gt_df["gt_y"].min()), 3),
                      round(float(gt_df["gt_y"].max()), 3)],
        "modalities_available": ["ground_truth"]
            + (["wifi"] if len(wifi_df) > 0 else [])
            + (["imu"] if len(imu_df) > 0 else []),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(gt_df["gt_x"], gt_df["gt_y"], c=gt_df["sim_time"],
                    cmap="viridis", s=4, linewidths=0)
    ax.plot(float(gt_df["gt_x"].iloc[0]), float(gt_df["gt_y"].iloc[0]), "go", ms=8, label="start")
    ax.plot(float(gt_df["gt_x"].iloc[-1]), float(gt_df["gt_y"].iloc[-1]), "rs", ms=8, label="end")
    plt.colorbar(sc, ax=ax, label="sim_time (s)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"path_{path_id:02d}  |  {native_split}  |  {source_name} seg{seg_index}\n"
                 f"GT={len(gt_df)}  WiFi={len(wifi_df)}  IMU={len(imu_df)}  dur={duration:.1f}s")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=100)
    plt.close(fig)
    return meta


def write_summary(out_dir, dataset_name, splits, n_aps, per_path,
                  subject, split_mode, chunk_seconds, split_seed):
    total = sum(len(v) for v in splits.values())
    duration_sum = sum(m["duration_s"] for m in per_path)
    summary = {
        "dataset": "RoNIN",
        "subject": subject,
        "n_paths": total,
        "n_aps": n_aps,
        "splits": {k: len(v) for k, v in splits.items()},
        "total_duration_s": round(duration_sum, 2),
        "modalities": ["wifi", "imu", "ground_truth"],
        "imu_target_hz": IMU_TARGET_HZ,
        "gt_target_hz": GT_TARGET_HZ,
        "split_mode": split_mode,
    }
    if split_mode == "within-trial":
        summary["chunk_seconds"] = chunk_seconds
        summary["split_seed"] = split_seed
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    train = splits.get("train", []); val = splits.get("val", []); test = splits.get("test", [])
    if split_mode == "within-trial":
        header = (
            f"# RoNIN subject {subject} (within-trial split).\n"
            f"# {total} chunks total (~{chunk_seconds:.0f}s each, seed={split_seed}, "
            f"split: train={len(train)} / val={len(val)} / test={len(test)}).\n"
            f"# Train and val are drawn from the SAME sequences so WiFi BSSIDs and\n"
            f"# IMU device characteristics match across splits.\n"
        )
    else:
        header = (
            f"# RoNIN subject {subject}.\n"
            f"# {total} paths total (native split: train={len(train)} / "
            f"val={len(val)} / test={len(test)}).\n"
        )
    snippet = f"""{header}# Modalities: wifi + imu (no odom, no camera - smartphone dataset).
# Ground truth is Tango VSLAM (x, y) in meters, downsampled to {GT_TARGET_HZ} Hz.
data:
  name: {dataset_name}
  root: ${{oc.env:NAVLORI_DATA_ROOT,data}}
  collection_dir: {dataset_name}
  source: ronin
  modalities: [wifi, imu]

  split:
    train_paths: {train}
    val_paths:   {val}
    test_paths:  {test}

  preprocessing:
    normalize: true
    wifi_pca: 128   # reduce {n_aps} APs -> 128 dims

  windows:
    imu: 32    # ~1 s at {IMU_TARGET_HZ} Hz
    wifi: 1    # single scan
"""
    (out_dir / "configs_snippet.yaml").write_text(snippet, encoding="utf-8")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="a000",
                    help="Subject identifier prefix, e.g. 'a000'.")
    ap.add_argument("--raw-root", type=Path,
                    default=Path("data/FRDR_dataset_538_download_259_202604270443/Data"),
                    help="Path to the unzipped Data/ directory.")
    ap.add_argument("--out-root", type=Path, default=Path("data"),
                    help="Where to write ronin_<subject>[_intra]/.")
    ap.add_argument("--exclude", nargs="*", default=["a000_11"],
                    help="Sequence names to skip (e.g. a000_11 walks a different building).")
    ap.add_argument("--bssid-min-overlap", type=float, default=0.20,
                    help="Drop a sequence whose BSSID overlap with the largest "
                         "sequence is below this fraction.  Set to 0 to keep all.")
    ap.add_argument("--split-mode", choices=["trial-out", "within-trial"],
                    default="within-trial",
                    help="See convert_ipin2024.py.  Default: within-trial (proven to work).")
    ap.add_argument("--chunk-seconds", type=float, default=15.0)
    ap.add_argument("--split-ratios", type=str, default="0.7,0.15,0.15")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    ratios = [float(x) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error(f"--split-ratios must sum to 1, got {args.split_ratios!r}.")

    raw_root: Path = args.raw_root
    out_suffix = "_intra" if args.split_mode == "within-trial" else ""
    dataset_name = f"ronin_{args.subject}{out_suffix}"
    out_dir: Path = args.out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"RoNIN -> {out_dir}  (subject={args.subject}, split-mode={args.split_mode})")

    # 1. Discover sequences
    seq_meta = find_subject_seqs(raw_root, args.subject)
    seq_meta = [(p, s) for (p, s) in seq_meta if p.name not in set(args.exclude)]
    if not seq_meta:
        print("  [abort] no sequences found.")
        return
    print(f"  found {len(seq_meta)} sequences after exclude")

    # 2. Load sequences
    sequences = []  # list of (seq_name, native_split, seq_dict)
    for path, split in seq_meta:
        seq = load_sequence(path)
        if seq is None or len(seq["wifi_scans"]) < 2:
            print(f"    [skip] {path.name} (no usable WiFi)")
            continue
        sequences.append((path.name, split, seq))
    if not sequences:
        print("  [abort] no usable sequences.")
        return

    # 3. BSSID-overlap filter (find the sequence with the most BSSIDs as anchor)
    def bssids_of(seq):
        return set().union(*(rssi.keys() for _, rssi in seq["wifi_scans"]))
    bssid_sets = [(name, split, seq, bssids_of(seq)) for (name, split, seq) in sequences]
    anchor = max(bssid_sets, key=lambda x: len(x[3]))
    anchor_set = anchor[3]
    print(f"  anchor sequence: {anchor[0]}  ({len(anchor_set)} BSSIDs)")
    kept = []
    for name, split, seq, bset in bssid_sets:
        if not anchor_set:
            kept.append((name, split, seq, bset))
            continue
        overlap = len(bset & anchor_set) / max(len(bset), 1)
        if overlap < args.bssid_min_overlap:
            print(f"    [drop] {name} overlap={overlap:.2f} < {args.bssid_min_overlap}")
        else:
            kept.append((name, split, seq, bset))
    sequences = [(n, s, sq) for (n, s, sq, _) in kept]
    print(f"  kept {len(sequences)} sequences after BSSID overlap filter")

    # 4. Build BSSID vocab from kept sequences
    vocab_set: set[str] = set()
    for _, _, seq in sequences:
        vocab_set.update(bssids_of(seq))
    bssid_vocab = sorted(vocab_set)
    print(f"  BSSID vocab size: {len(bssid_vocab)}")

    # 5. Build candidate per-sequence dataframes
    @dataclass
    class Candidate:
        name: str
        split: str
        gt_df: pd.DataFrame
        imu_df: pd.DataFrame
        wifi_df: pd.DataFrame

    candidates: list[Candidate] = []
    for name, split, seq in sequences:
        gt_df = downsample_gt(seq, GT_TARGET_HZ)
        imu_df = downsample_imu(seq, IMU_TARGET_HZ)
        wifi_df = build_wifi_df(seq, bssid_vocab)
        if len(gt_df) < MIN_GT_PER_PATH:
            continue
        if (gt_df["sim_time"].max() - gt_df["sim_time"].min()) < MIN_PATH_SECONDS:
            continue
        candidates.append(Candidate(name, split, gt_df, imu_df, wifi_df))
    if not candidates:
        print("  [abort] no usable candidates.")
        return
    print(f"  candidates: {len(candidates)}")

    # 6. Emit (trial-out or within-trial)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_path: list[dict] = []

    if args.split_mode == "trial-out":
        for path_id, c in enumerate(candidates):
            meta = write_path(out_dir / f"path_{path_id:02d}",
                              c.gt_df, c.imu_df, c.wifi_df,
                              c.name, path_id, c.split, seg_index=0)
            splits[c.split].append(path_id)
            per_path.append(meta)
            print(f"    path_{path_id:02d}  [{c.split}]  {c.name}  "
                  f"{meta['duration_s']:.1f}s  GT={meta['n_gt_samples']}  "
                  f"WiFi={meta['n_wifi_scans']}  IMU={meta['n_imu_samples']}")
    else:
        @dataclass
        class Chunk:
            name: str; chunk_index: int
            gt_df: pd.DataFrame; imu_df: pd.DataFrame; wifi_df: pd.DataFrame

        chunks: list[Chunk] = []
        for c in candidates:
            for ci, (g, i, w) in enumerate(chunk_segment(
                    c.gt_df, c.imu_df, c.wifi_df, args.chunk_seconds)):
                chunks.append(Chunk(c.name, ci, g, i, w))
        if not chunks:
            print("  [abort] no chunks produced.")
            return
        rng = random.Random(args.split_seed)
        order = list(range(len(chunks)))
        rng.shuffle(order)
        n = len(order)
        n_train = max(1, min(n - 2, int(round(ratios[0] * n)))) if n >= 3 else max(1, int(round(ratios[0] * n)))
        n_val = max(1, min(n - n_train - 1, int(round(ratios[1] * n)))) if n >= 3 else max(0, int(round(ratios[1] * n)))
        train_idx = set(order[:n_train])
        val_idx = set(order[n_train:n_train + n_val])
        for path_id, ch in enumerate(chunks):
            assigned = "train" if path_id in train_idx else ("val" if path_id in val_idx else "test")
            meta = write_path(out_dir / f"path_{path_id:02d}",
                              ch.gt_df, ch.imu_df, ch.wifi_df,
                              ch.name, path_id, assigned, seg_index=ch.chunk_index)
            meta["chunk_index"] = ch.chunk_index
            splits[assigned].append(path_id)
            per_path.append(meta)
            print(f"    path_{path_id:02d}  [{assigned}]  {ch.name} chunk{ch.chunk_index}  "
                  f"{meta['duration_s']:.1f}s  GT={meta['n_gt_samples']}  "
                  f"WiFi={meta['n_wifi_scans']}  IMU={meta['n_imu_samples']}")

    # 7. Dataset-level files
    (out_dir / "ap_vocab.json").write_text(
        json.dumps({m: i for i, m in enumerate(bssid_vocab)}, indent=2), encoding="utf-8")
    write_summary(
        out_dir, dataset_name, splits, len(bssid_vocab), per_path,
        subject=args.subject, split_mode=args.split_mode,
        chunk_seconds=args.chunk_seconds if args.split_mode == "within-trial" else None,
        split_seed=args.split_seed if args.split_mode == "within-trial" else None,
    )
    shutil.copy2(Path(__file__), out_dir / "convert_ronin.py")

    n_emitted = sum(len(v) for v in splits.values())
    print()
    print(f"  wrote {n_emitted} paths to {out_dir}")
    print(f"  splits: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"  config snippet: {out_dir / 'configs_snippet.yaml'}")


if __name__ == "__main__":
    main()
