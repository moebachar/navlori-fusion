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
import math
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


def imu_intensity(rp: ReplayPath, hp_tau: float = 0.8,
                  sm_tau: float = 0.6) -> tuple[list[float], list[float]]:
    """Walking-intensity signal s(t) from the verbatim IMU rows.

    s = smoothed |accel magnitude - slow EMA(accel magnitude)| -- the
    rectified high-pass of the accelerometer envelope. It is ~0 when the
    carrier stands still and grows with gait vigour. NO integration (that
    would drift); only the shape is used downstream, so bias cancels.
    """
    ts, mag = [], []
    for r in rp.imu_rows:
        d = r.raw
        try:
            if "accel_magnitude" in d and d["accel_magnitude"] != "":
                m = float(d["accel_magnitude"])
            else:
                m = math.hypot(float(d["accel_x"]),
                               math.hypot(float(d["accel_y"]),
                                          float(d["accel_z"])))
        except (KeyError, ValueError):
            continue
        ts.append(r.t)
        mag.append(m)
    if len(ts) < 4:
        return [], []
    # slow EMA -> gravity/bias estimate; rectify; smooth
    out = []
    ema = mag[0]
    for i in range(len(ts)):
        dt = max(1e-4, ts[i] - ts[i - 1]) if i else 0.032
        a = dt / (hp_tau + dt)
        ema += a * (mag[i] - ema)
        out.append(abs(mag[i] - ema))
    sm = out[0]
    for i in range(len(out)):
        dt = max(1e-4, ts[i] - ts[i - 1]) if i else 0.032
        a = dt / (sm_tau + dt)
        sm += a * (out[i] - sm)
        out[i] = sm
    return ts, out


def anchor_times(rp: ReplayPath) -> tuple[list[float], str]:
    """Times of the REAL waypoint presses -- the only true position/timing
    anchors. The time warp must map each exactly to itself.

    Priority:
      1. waypoints_raw.csv          (iln20 converts; exact press times)
      2. velocity kinks in the dense GT (msiln: GT is piecewise-linear
         interpolation, so bends/speed changes only happen at presses;
         both grid rows bounding a kink interval are pinned -- extra
         anchors are conservative, missing one would break constraint #5)
      3. [t_start, t_end]           (single segment)
    """
    raw_csv = rp.path_dir / "waypoints_raw.csv"
    if raw_csv.is_file():
        _, rows = _read_csv(raw_csv)
        ts = []
        for r in rows:
            try:
                ts.append(float(r["sim_time"]))
            except (KeyError, ValueError):
                continue
        ts = sorted(set(ts))
        ts = [t for t in ts if rp.t_start <= t <= rp.t_end]
        if ts and ts[0] > rp.t_start:
            ts.insert(0, rp.t_start)
        if ts and ts[-1] < rp.t_end:
            ts.append(rp.t_end)
        if len(ts) >= 2:
            return ts, "waypoints_raw"

    # kink detection on the dense polyline
    w = rp.waypoints
    if len(w) >= 4:
        anchors = {rp.t_start, rp.t_end}
        vlast = None
        for i in range(len(w) - 1):
            dt = w[i + 1].t - w[i].t
            if dt <= 0:
                continue
            v = ((w[i + 1].x - w[i].x) / dt, (w[i + 1].y - w[i].y) / dt)
            if vlast is not None:
                dv = math.hypot(v[0] - vlast[0], v[1] - vlast[1])
                ref = max(math.hypot(*v), math.hypot(*vlast), 0.05)
                if dv > 0.04 * ref:
                    anchors.add(w[i].t)          # both rows bounding the
                    anchors.add(w[i + 1].t)      # kink interval get pinned
            vlast = v
        ts = sorted(anchors)
        if len(ts) >= 2:
            return ts, "gt_kinks"

    return [rp.t_start, rp.t_end], "endpoints"


def build_trajectory(rp: ReplayPath, imu_speed_profile: bool = True,
                     v_max: float = 2.5, floor_frac: float = 0.05):
    """The ONE trajectory factory used by the controller AND verify_replay
    -- both sides must move/judge against the identical time law.

    Returns (traj, info). info["enabled"] tells whether the IMU warp is
    active; the controller persists info in per-path metadata.json so
    verify can rebuild the exact same trajectory.
    """
    from trajectory import HermiteTrajectory, IMUTimeWarp, WarpedTrajectory

    base = HermiteTrajectory([w.t for w in rp.waypoints],
                             [w.x for w in rp.waypoints],
                             [w.y for w in rp.waypoints])
    info = {"enabled": False, "v_max": v_max, "floor_frac": floor_frac}
    if not imu_speed_profile:
        info["reason"] = "disabled by config"
        return base, info

    its, s = imu_intensity(rp)
    if not its:
        info["reason"] = "no usable IMU rows"
        return base, info
    anchors, src = anchor_times(rp)
    info["anchor_source"] = src
    info["n_anchors"] = len(anchors)

    # per-segment base speed -> r_max = v_max / v_base (dense polyline length
    # between anchors over the segment duration)
    r_max = []
    wi = 0
    for j in range(len(anchors) - 1):
        t0, t1 = anchors[j], anchors[j + 1]
        dist = 0.0
        while wi < len(rp.waypoints) - 1 and rp.waypoints[wi + 1].t <= t1:
            if rp.waypoints[wi].t >= t0:
                dist += math.hypot(
                    rp.waypoints[wi + 1].x - rp.waypoints[wi].x,
                    rp.waypoints[wi + 1].y - rp.waypoints[wi].y)
            wi += 1
        v_base = dist / max(1e-6, t1 - t0)
        r_max.append(v_max / v_base if v_base > 1e-6 else float("inf"))

    warp = IMUTimeWarp(anchors, its, s, floor_frac=floor_frac,
                       r_max_per_seg=r_max)
    info["enabled"] = True
    info.update(warp.stats())
    return WarpedTrajectory(base, warp), info


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
