"""Convert IMUWiFine raw .txt files to navlori-fusion dataset format.

Single-floor mode.  Walks the train/ + val/ folders under
``raw_IUMIWiFi/<N>th_floor/`` and the pure-floor test files at
``IMU_DATA/test/test_<N>_*.txt``, writing each to its own
``path_XX/`` directory that matches the layout used by
``data/async_collection/``.

Outputs
-------
``<out_root>/imuwifine_floor<N>/``
    ├── path_00/
    │     ├── imu.csv          — ACCE+GYRO+AHRS downsampled to ~32 Hz
    │     ├── wifi.csv         — per-scan RSSI fingerprint grouped by SCAN_GAP_MS
    │     ├── odometry.csv     — header-only stub (not available in IMUWiFine)
    │     ├── ground_truth.csv — POSI measurements (~2.85 Hz)
    │     ├── metadata.json    — per-path stats + native_split
    │     └── trajectory.png
    ├── ...
    ├── ap_vocab.json          — global BSSID → column index map (floor-local)
    ├── split.json             — train/val/test path-id lists (native split)
    ├── metadata.json          — dataset-level summary
    ├── convert_imuwifine.py   — archived copy of this script
    └── configs_snippet.yaml   — ready-to-paste Hydra data config

Timestamp convention
--------------------
t0 = earliest timestamp across all modalities in the file.
sim_time = (raw_ms - t0) / 1000.0 → always >= 0.

WiFi scan grouping
------------------
Android WIFI lines arrive as bursts of BSSID readings separated by ~1-3 s.
Consecutive readings whose inter-reading gap < SCAN_GAP_MS are grouped into
one scan: mean timestamp, max RSSI per BSSID.

AHRS format (per IMUWiFine header)
----------------------------------
``AHRS;AppTimestamp;Timestamp;PitchX;RollY;AzimuthZ;...``
parts[3]=pitch  parts[4]=roll  parts[5]=yaw (azimuth 0-360°)

Two raw formats coexist in IMUWiFine
------------------------------------
**Train/val** (``raw_IUMIWiFi/<N>th_floor/{train,val}/DATA_*.txt``) carry the
Android logger header and use ms-since-epoch timestamps in ``parts[2]``.
WIFI layout: ``WIFI;appTs;sensorTs;SSID;MAC;freq;RSS``.

**Test** (``IMU_DATA/test/test_<N>_*.txt``) have no header comments and
encode timestamps as ``parts[1]=epoch_seconds``, ``parts[2]=ns_within_second``.
They contain only POSI + WIFI (no IMU tags) and WIFI is shifted:
``WIFI;epoch_s;ns;SSID;channel;MAC;freq_GHz;RSS``.  Format is detected by
scanning the file for a header line.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── configuration ───────────────────────────────────────────────────────────
# WiFi scan grouping threshold (ms).  Readings closer than this belong to the
# same scan.  Train/val files spread a single scan across ~100-500 ms, so the
# default 1000 ms window is appropriate.  Test files emit each scan as a burst
# of readings sharing one identical timestamp, with ~100 ms gaps to the next
# scan — so a tight 50 ms threshold gives correct per-scan grouping there.
SCAN_GAP_MS_DEFAULT = 1000
SCAN_GAP_MS_TEST = 50

IMU_KEEP_EVERY = 6          # native ~192 Hz → keep every 6th row ≈ 32 Hz
MIN_GT_SAMPLES = 10         # minimum POSI samples required to keep a path


# ── step 1: enumerate files for one floor ─────────────────────────────────

def find_floor_files(raw_root: Path, floor: int) -> list[tuple[Path, str]]:
    """Return (file_path, native_split) for all raw .txt files of a floor.

    Scans train/val under ``raw_IUMIWiFi/<N>th_floor/`` and flat
    ``IMU_DATA/test/test_<N>_*.txt``.  Cross-floor test files
    (e.g. ``test_454_*``) are ignored.
    """
    out: list[tuple[Path, str]] = []
    floor_dir = raw_root / "IMU_DATA" / "raw_IUMIWiFi" / f"{floor}th_floor"
    for split in ("train", "val"):
        d = floor_dir / split
        if d.is_dir():
            out.extend((p, split) for p in sorted(d.glob("DATA_*.txt")))
    test_dir = raw_root / "IMU_DATA" / "test"
    if test_dir.is_dir():
        out.extend((p, "test") for p in sorted(test_dir.glob(f"test_{floor}_*.txt")))
    return out


# ── step 2: detect format + build BSSID vocabulary ─────────────────────────

def detect_test_format(txt_path: Path) -> bool:
    """True if this file is the header-less test-set format.

    Test files start directly with a data tag (no ``%`` comment lines).
    """
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return not s.startswith("%")
    return False


def _wifi_fields(parts: list[str], is_test: bool) -> tuple[str, float] | None:
    """Extract (bssid, rssi) from a WIFI line in either format."""
    try:
        if is_test:
            # WIFI;epoch_s;ns;SSID;channel;MAC;freq_GHz;RSS
            if len(parts) >= 8:
                return parts[5].lower().strip(), float(parts[7])
        else:
            # WIFI;appTs;sensorTs;SSID;MAC;freq;RSS
            if len(parts) >= 6:
                return parts[4].lower().strip(), float(parts[5])
    except ValueError:
        return None
    return None


def build_bssid_vocab(files: list[tuple[Path, str]]) -> list[str]:
    """Sorted list of unique BSSIDs seen across all given files."""
    bssids: set[str] = set()
    for f, _ in files:
        is_test = detect_test_format(f)
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("WIFI"):
                    parts = line.strip().split(";")
                    fields = _wifi_fields(parts, is_test)
                    if fields is not None:
                        bssids.add(fields[0])
    return sorted(bssids)


# ── step 3: parse one raw file ────────────────────────────────────────────

def parse_raw(
    txt_path: Path,
    bssid_to_idx: dict[str, int],
    n_aps: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Return (gt_df, imu_df, wifi_df) or (None, None, None) if unusable."""
    is_test = detect_test_format(txt_path)
    acce_rows: list[list] = []
    gyro_rows: list[list] = []
    ahrs_rows: list[list] = []
    posi_rows: list[list] = []
    wifi_raw: list[tuple[int, str, float]] = []

    def ts_ms(parts: list[str]) -> int | None:
        """Unified millisecond timestamp from either format."""
        try:
            if is_test:
                # parts[1]=epoch_seconds, parts[2]=nanoseconds_within_second
                return int(parts[1]) * 1000 + int(parts[2]) // 1_000_000
            return int(parts[2])
        except (IndexError, ValueError):
            return None

    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.strip().split(";")
            tag = parts[0]
            ts = ts_ms(parts)
            if ts is None:
                continue

            if tag == "ACCE" and len(parts) >= 6:
                try:
                    acce_rows.append(
                        [ts, float(parts[3]), float(parts[4]), float(parts[5])]
                    )
                except ValueError:
                    pass
            elif tag == "GYRO" and len(parts) >= 6:
                try:
                    gyro_rows.append(
                        [ts, float(parts[3]), float(parts[4]), float(parts[5])]
                    )
                except ValueError:
                    pass
            elif tag == "AHRS" and len(parts) >= 6:
                # AHRS;...;PitchX;RollY;AzimuthZ(yaw)
                try:
                    ahrs_rows.append(
                        [
                            ts,
                            float(parts[3]),   # pitch
                            float(parts[4]),   # roll
                            float(parts[5]),   # yaw (azimuth 0-360°)
                        ]
                    )
                except ValueError:
                    pass
            elif tag == "POSI" and len(parts) >= 5:
                try:
                    posi_rows.append([ts, float(parts[3]), float(parts[4])])
                except ValueError:
                    pass
            elif tag == "WIFI":
                fields = _wifi_fields(parts, is_test)
                if fields is not None:
                    wifi_raw.append((ts, fields[0], fields[1]))

    if len(posi_rows) < MIN_GT_SAMPLES:
        return None, None, None

    # t0 — earliest timestamp across all modalities (WIFI lines aren't sorted)
    all_first: list[int] = []
    for rows in (acce_rows, gyro_rows, ahrs_rows, posi_rows):
        if rows:
            all_first.append(min(r[0] for r in rows))
    if wifi_raw:
        all_first.append(min(r[0] for r in wifi_raw))
    t0 = min(all_first)

    def to_rel(ts_ms: int) -> float:
        return (ts_ms - t0) / 1000.0

    # ── ground truth ─────────────────────────────────────────────────────
    gt_df = pd.DataFrame(posi_rows, columns=["ts_ms", "gt_x", "gt_y"])
    gt_df["sim_time"] = gt_df["ts_ms"].apply(to_rel)
    gt_df = gt_df[["sim_time", "gt_x", "gt_y"]].reset_index(drop=True)

    # ── IMU (ACCE + GYRO + AHRS orientation) ─────────────────────────────
    imu_df = None
    if acce_rows and gyro_rows:
        acce = pd.DataFrame(
            acce_rows, columns=["ts_ms", "accel_x", "accel_y", "accel_z"]
        )
        gyro = pd.DataFrame(
            gyro_rows, columns=["ts_ms", "gyro_x", "gyro_y", "gyro_z"]
        )
        acce = acce.iloc[::IMU_KEEP_EVERY].reset_index(drop=True)
        acce = acce.sort_values("ts_ms").reset_index(drop=True)
        gyro = gyro.sort_values("ts_ms").reset_index(drop=True)
        merged = pd.merge_asof(acce, gyro, on="ts_ms", direction="nearest")

        if ahrs_rows:
            ahrs = pd.DataFrame(
                ahrs_rows,
                columns=["ts_ms", "pitch_deg", "roll_deg", "yaw_deg"],
            )
            ahrs = ahrs.sort_values("ts_ms").reset_index(drop=True)
            merged = pd.merge_asof(merged, ahrs, on="ts_ms", direction="nearest")
        else:
            merged["roll_deg"] = 0.0
            merged["pitch_deg"] = 0.0
            merged["yaw_deg"] = 0.0

        merged["sim_time"] = merged["ts_ms"].apply(to_rel)
        imu_df = merged[
            [
                "sim_time",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z",
                "roll_deg", "pitch_deg", "yaw_deg",
            ]
        ].reset_index(drop=True)

    # ── WiFi: group consecutive readings within the format's scan window ─
    wifi_df = None
    if wifi_raw:
        scan_gap_ms = SCAN_GAP_MS_TEST if is_test else SCAN_GAP_MS_DEFAULT
        wifi_sorted = sorted(wifi_raw, key=lambda x: x[0])
        scans: list[list[tuple[int, str, float]]] = []
        current: list[tuple[int, str, float]] = [wifi_sorted[0]]
        for ts, bssid, rssi in wifi_sorted[1:]:
            if ts - current[-1][0] > scan_gap_ms:
                scans.append(current)
                current = []
            current.append((ts, bssid, rssi))
        if current:
            scans.append(current)

        sorted_bssids = sorted(bssid_to_idx, key=bssid_to_idx.get)
        col_names = [f"wifi_rssi_{b.replace(':', '')}" for b in sorted_bssids]

        wifi_rows: list[dict] = []
        for scan in scans:
            scan_ts_ms = int(np.mean([r[0] for r in scan]))
            bssid_max: dict[str, float] = {}
            for _, bssid, rssi in scan:
                if bssid not in bssid_max or rssi > bssid_max[bssid]:
                    bssid_max[bssid] = rssi

            rssi_vec = np.full(n_aps, np.nan, dtype=np.float32)
            for bssid, rssi in bssid_max.items():
                idx = bssid_to_idx.get(bssid)
                if idx is not None:
                    rssi_vec[idx] = rssi

            visible = int(np.sum(~np.isnan(rssi_vec)))
            filled = np.where(np.isnan(rssi_vec), -200.0, rssi_vec)
            best_idx = int(np.argmax(filled))
            best_mac = sorted_bssids[best_idx].replace(":", "")
            best_rssi = (
                float(filled[best_idx])
                if not np.isnan(rssi_vec[best_idx])
                else np.nan
            )

            row: dict = {
                "sim_time": to_rel(scan_ts_ms),
                "wifi_visible_count": visible,
                "wifi_strongest_rssi": best_rssi,
                "wifi_strongest_mac": best_mac,
            }
            for i, col in enumerate(col_names):
                row[col] = float(rssi_vec[i]) if not np.isnan(rssi_vec[i]) else np.nan
            wifi_rows.append(row)
        wifi_df = pd.DataFrame(wifi_rows)

    return gt_df, imu_df, wifi_df


# ── step 4: write one path directory ─────────────────────────────────────

IMU_STUB = (
    "sim_time,accel_x,accel_y,accel_z,"
    "gyro_x,gyro_y,gyro_z,roll_deg,pitch_deg,yaw_deg\n"
)
WIFI_STUB = "sim_time,wifi_visible_count,wifi_strongest_rssi,wifi_strongest_mac\n"
ODOM_STUB = (
    "sim_time,odom_x,odom_y,odom_theta_deg,"
    "odom_linear_vel,odom_angular_vel,wheel_left_vel,wheel_right_vel\n"
)


def write_path(
    out_dir: Path,
    gt_df: pd.DataFrame,
    imu_df: pd.DataFrame | None,
    wifi_df: pd.DataFrame | None,
    src_file: Path,
    floor: int,
    path_id: int,
    native_split: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(gt_df["sim_time"].max() - gt_df["sim_time"].min())

    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)
    if imu_df is not None and len(imu_df) > 0:
        imu_df.to_csv(out_dir / "imu.csv", index=False)
    else:
        (out_dir / "imu.csv").write_text(IMU_STUB)
    if wifi_df is not None and len(wifi_df) > 0:
        wifi_df.to_csv(out_dir / "wifi.csv", index=False)
    else:
        (out_dir / "wifi.csv").write_text(WIFI_STUB)
    (out_dir / "odometry.csv").write_text(ODOM_STUB)

    meta = {
        "dataset": "IMUWiFine",
        "source_file": src_file.name,
        "floor": floor,
        "path_id": path_id,
        "native_split": native_split,
        "duration_s": round(duration, 2),
        "n_gt_samples": len(gt_df),
        "gt_rate_hz": round(len(gt_df) / duration, 2) if duration > 0 else 0,
        "gt_sim_time_range": [
            round(float(gt_df["sim_time"].min()), 2),
            round(float(gt_df["sim_time"].max()), 2),
        ],
        "n_wifi_scans": len(wifi_df) if wifi_df is not None else 0,
        "wifi_rate_hz": (
            round(len(wifi_df) / duration, 3)
            if wifi_df is not None and duration > 0
            else 0
        ),
        "n_imu_samples": len(imu_df) if imu_df is not None else 0,
        "imu_rate_hz": (
            round(len(imu_df) / duration, 1)
            if imu_df is not None and duration > 0
            else 0
        ),
        "x_range_m": [
            round(float(gt_df["gt_x"].min()), 3),
            round(float(gt_df["gt_x"].max()), 3),
        ],
        "y_range_m": [
            round(float(gt_df["gt_y"].min()), 3),
            round(float(gt_df["gt_y"].max()), 3),
        ],
        "modalities_available": (
            ["ground_truth"]
            + (["wifi"] if wifi_df is not None and len(wifi_df) > 0 else [])
            + (["imu"] if imu_df is not None and len(imu_df) > 0 else [])
        ),
        "notes": {
            "odometry": "not available in IMUWiFine",
            "camera": "not available in IMUWiFine",
            "t0_anchor": "earliest timestamp across all modalities in source file",
            "wifi_scan_gap_ms": SCAN_GAP_MS_TEST if native_split == "test" else SCAN_GAP_MS_DEFAULT,
            "imu_downsample_factor": IMU_KEEP_EVERY,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    fig, ax = plt.subplots(figsize=(6, 6))
    t = gt_df["sim_time"].values
    sc = ax.scatter(
        gt_df["gt_x"], gt_df["gt_y"], c=t, cmap="viridis", s=4, linewidths=0
    )
    ax.plot(
        float(gt_df["gt_x"].iloc[0]),
        float(gt_df["gt_y"].iloc[0]),
        "go", ms=8, label="start",
    )
    ax.plot(
        float(gt_df["gt_x"].iloc[-1]),
        float(gt_df["gt_y"].iloc[-1]),
        "rs", ms=8, label="end",
    )
    plt.colorbar(sc, ax=ax, label="sim_time (s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"path_{path_id:02d}  |  floor {floor}  |  {native_split}  |  "
        f"{src_file.name}\nGT: {len(gt_df)} pts @ {meta['gt_rate_hz']} Hz  |  "
        f"WiFi: {meta['n_wifi_scans']} scans @ {meta['wifi_rate_hz']} Hz  |  "
        f"IMU: {meta['n_imu_samples']} rows @ {meta['imu_rate_hz']} Hz"
    )
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=100)
    plt.close(fig)
    return meta


# ── step 5: dataset-level summary + Hydra config snippet ─────────────────

def write_dataset_summary(
    out_dir: Path,
    floor: int,
    splits: dict[str, list[int]],
    n_aps: int,
    per_path: list[dict],
) -> None:
    total = sum(len(v) for v in splits.values())
    duration_sum = sum(m["duration_s"] for m in per_path)
    summary = {
        "dataset": "IMUWiFine",
        "floor": floor,
        "n_paths": total,
        "n_aps": n_aps,
        "splits": {k: len(v) for k, v in splits.items()},
        "total_duration_s": round(duration_sum, 2),
        "modalities": ["wifi", "imu", "ground_truth"],
        "wifi_scan_gap_ms_default": SCAN_GAP_MS_DEFAULT,
        "wifi_scan_gap_ms_test": SCAN_GAP_MS_TEST,
        "imu_downsample_factor": IMU_KEEP_EVERY,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "split.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )

    # Paste-ready Hydra config — keeps the same key layout as simulation.yaml
    yaml_lines = [
        f"# IMUWiFine dataset - floor {floor} only.",
        f"# {total} paths total (native split: train={len(splits['train'])} / "
        f"val={len(splits['val'])} / test={len(splits['test'])}).",
        "# No odometry or camera in IMUWiFine - modalities restricted to wifi+imu.",
        "# Test paths carry only WiFi + ground truth (no IMU); IMU windows will be zero-padded.",
        "data:",
        "  name: imuwifine",
        "  root: ${oc.env:NAVLORI_DATA_ROOT,data}",
        f"  collection_dir: imuwifine_floor{floor}",
        "  source: imuwifine",
        "  modalities: [wifi, imu]",
        "",
        "  split:",
        f"    train_paths: {splits['train']}",
        f"    val_paths:   {splits['val']}",
        f"    test_paths:  {splits['test']}",
        "",
        "  preprocessing:",
        "    normalize: true",
        f"    wifi_pca: 128   # reduce {n_aps} APs -> 128 dims (see FusionDataset)",
        "",
        "  windows:",
        "    imu: 32    # ~1 s at 32 Hz (downsampled from native 192 Hz)",
        "    wifi: 1    # single scan",
        "",
    ]
    (out_dir / "configs_snippet.yaml").write_text(
        "\n".join(yaml_lines), encoding="utf-8"
    )


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--floor", type=int, required=True, choices=[4, 5, 6])
    p.add_argument("--raw-root", type=Path, default=Path("X:/IMUWiFine"))
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("X:/navlori-fusion/data"),
        help="Output parent dir (dataset goes under imuwifine_floor{N}/)",
    )
    args = p.parse_args()

    out_dir = args.out_root / f"imuwifine_floor{args.floor}"

    print(f"Step 1/4 — Enumerating floor {args.floor} raw files under {args.raw_root}")
    files = find_floor_files(args.raw_root, args.floor)
    if not files:
        raise SystemExit(f"No files found for floor {args.floor} under {args.raw_root}")
    from collections import Counter
    per_split = Counter(s for _, s in files)
    print(f"  {len(files)} files  ({dict(per_split)})")

    # Sort: train → val → test, stable within split
    order = {"train": 0, "val": 1, "test": 2}
    files.sort(key=lambda x: (order[x[1]], x[0].name))

    print("Step 2/4 — Building BSSID vocabulary ...")
    bssids = build_bssid_vocab(files)
    n_aps = len(bssids)
    bssid_to_idx = {b: i for i, b in enumerate(bssids)}
    print(f"  {n_aps} unique BSSIDs on floor {args.floor}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy(__file__, out_dir / "convert_imuwifine.py")
    (out_dir / "ap_vocab.json").write_text(json.dumps(bssid_to_idx, indent=2))

    print(f"Step 3/4 — Parsing and writing paths to {out_dir}")
    splits_map: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_path_meta: list[dict] = []
    skipped = 0
    path_id = 0

    for i, (src, split) in enumerate(files):
        gt_df, imu_df, wifi_df = parse_raw(src, bssid_to_idx, n_aps)
        if gt_df is None:
            print(f"  [SKIP] ({split}) {src.name} — too few POSI samples")
            skipped += 1
            continue

        pdir = out_dir / f"path_{path_id:02d}"
        meta = write_path(pdir, gt_df, imu_df, wifi_df, src, args.floor, path_id, split)
        per_path_meta.append(meta)
        splits_map[split].append(path_id)

        if (i + 1) % 10 == 0 or i == 0:
            n_wifi = len(wifi_df) if wifi_df is not None else 0
            n_imu = len(imu_df) if imu_df is not None else 0
            print(
                f"  [{i+1:3d}/{len(files)}] path_{path_id:02d}  {split:5s}  "
                f"GT={len(gt_df)}pts  WiFi={n_wifi}  IMU={n_imu}  "
                f"dur={meta['duration_s']:.1f}s"
            )
        path_id += 1

    print("Step 4/4 — Dataset summary + Hydra config snippet ...")
    write_dataset_summary(out_dir, args.floor, splits_map, n_aps, per_path_meta)

    print()
    print(f"Done. {path_id} paths written, {skipped} skipped.")
    print(f"  train: {len(splits_map['train'])}   "
          f"val: {len(splits_map['val'])}   "
          f"test: {len(splits_map['test'])}")
    print(f"  → {out_dir}")
    print(f"  → Hydra snippet: {out_dir / 'configs_snippet.yaml'}")


if __name__ == "__main__":
    main()
