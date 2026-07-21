"""Load an original-dataset path (GT + IMU + WiFi) for replay in Webots.

Format reference (per project CLAUDE.md):
- ground_truth.csv : sim_time, gt_x, gt_y[, gt_z, gt_heading_rad, ...]
- imu.csv          : sim_time, accel_x/y/z, gyro_x/y/z, roll/pitch/yaw_deg
- wifi.csv         : sim_time, wifi_visible_count, wifi_strongest_rssi,
                     wifi_strongest_mac, wifi_rssi_<MAC>...

All datasets converted to async_collection format have the same column
families; missing modalities are tolerated (e.g. RoNIN has no camera).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Waypoint:
    t: float
    x: float
    y: float


@dataclass
class IMURow:
    t: float
    raw: dict  # keep all columns verbatim for write-through


@dataclass
class WiFiRow:
    t: float
    raw: dict


@dataclass
class ReplayPath:
    path_id: int
    path_dir: Path
    waypoints: list[Waypoint] = field(default_factory=list)
    imu_rows: list[IMURow] = field(default_factory=list)
    wifi_rows: list[WiFiRow] = field(default_factory=list)
    imu_columns: list[str] = field(default_factory=list)
    wifi_columns: list[str] = field(default_factory=list)

    @property
    def t_start(self) -> float:
        return self.waypoints[0].t

    @property
    def t_end(self) -> float:
        return self.waypoints[-1].t

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            rows.append(row)
    return cols, rows


def load_path(path_dir: str | Path, path_id: int) -> ReplayPath:
    """Load a single path directory. Raises FileNotFoundError if GT missing."""
    p = Path(path_dir)
    gt_csv = p / "ground_truth.csv"
    if not gt_csv.is_file():
        raise FileNotFoundError(f"ground_truth.csv not found in {p}")

    rp = ReplayPath(path_id=path_id, path_dir=p)

    # ─ Ground truth ─
    _, gt_rows = _read_csv(gt_csv)
    for r in gt_rows:
        try:
            rp.waypoints.append(Waypoint(
                t=float(r["sim_time"]),
                x=float(r["gt_x"]),
                y=float(r["gt_y"]),
            ))
        except (KeyError, ValueError):
            # skip malformed row silently — match async_collector tolerance
            continue
    if len(rp.waypoints) < 2:
        raise ValueError(f"path {p}: need >=2 waypoints, got {len(rp.waypoints)}")
    # enforce monotone time (some datasets have repeated stamps; we deduplicate)
    seen_t = set()
    dedup = []
    for w in rp.waypoints:
        if w.t in seen_t:
            continue
        seen_t.add(w.t)
        dedup.append(w)
    rp.waypoints = sorted(dedup, key=lambda w: w.t)

    # ─ IMU (optional but expected) ─
    imu_csv = p / "imu.csv"
    if imu_csv.is_file():
        cols, rows = _read_csv(imu_csv)
        rp.imu_columns = cols
        for r in rows:
            try:
                t = float(r["sim_time"])
            except (KeyError, ValueError):
                continue
            rp.imu_rows.append(IMURow(t=t, raw=dict(r)))

    # ─ WiFi (optional) ─
    wifi_csv = p / "wifi.csv"
    if wifi_csv.is_file():
        cols, rows = _read_csv(wifi_csv)
        rp.wifi_columns = cols
        for r in rows:
            try:
                t = float(r["sim_time"])
            except (KeyError, ValueError):
                continue
            rp.wifi_rows.append(WiFiRow(t=t, raw=dict(r)))

    return rp


def summarise(rp: ReplayPath) -> str:
    """One-line summary for CLI/log output."""
    return (
        f"path {rp.path_id}: {rp.n_waypoints} GT wps, "
        f"duration={rp.duration:.2f}s "
        f"({rp.t_start:.2f}->{rp.t_end:.2f}), "
        f"IMU={len(rp.imu_rows)}, WiFi={len(rp.wifi_rows)}"
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: path_loader.py <path_dir> [path_id]")
        sys.exit(1)
    pid = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    rp = load_path(sys.argv[1], pid)
    print(summarise(rp))
    print(f"  first WP: t={rp.waypoints[0].t:.3f} "
          f"({rp.waypoints[0].x:.3f}, {rp.waypoints[0].y:.3f})")
    print(f"  last WP:  t={rp.waypoints[-1].t:.3f} "
          f"({rp.waypoints[-1].x:.3f}, {rp.waypoints[-1].y:.3f})")
    if rp.imu_rows:
        print(f"  IMU cols: {rp.imu_columns[:6]}...")
    if rp.wifi_rows:
        print(f"  WiFi cols: {len(rp.wifi_columns)} total")
