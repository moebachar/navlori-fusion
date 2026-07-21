"""NavLoRI - Replay Collector
================================
Webots controller that drives TIAGO++ along the EXACT positions of an
original-dataset path (RoNIN / IPIN / IMUWiFine / async_collection) and
emits aligned camera + odometry while preserving original WiFi + IMU.

Constraints (from project requirements 2026-05-29):
  1. Keep original WiFi & IMU from the source dataset verbatim.
  2. Speed/accel between waypoints are derived from waypoint geometry+timing
     (cubic Hermite spline through (t_i, x_i, y_i)).
  3. Camera + odometry are produced asynchronously from WiFi + IMU.
  4. Densified GT samples are added between original waypoints.
  5. Robot is at exactly (x_i, y_i) at sim_time = t_i for every original WP.
  6. No teleport -- per-step pose delta is small (sub-cm), looks like driving.
  7. No collisions -- worlds are visual-only except Floor (verified upstream).

Architecture:
  path_loader.py     : load original GT / IMU / WiFi
  trajectory.py      : time-parametrised cubic Hermite spline
  drive.py           : differential-drive tracker + OdometrySynthesizer
  replay_collector.py: this file -- main Webots loop, sensor scheduling, IO

Author: Mohamed -- NavLoRI Project, CESI LINEACT
"""

# ─── stdlib ───
import csv
import json
import math
import os
import sys
import time as pytime
from pathlib import Path

# ─── Webots venv shim (mirrors async_collector convention) ───
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_CANDIDATES = [
    r"x:\navlori-fusion\.venv\Lib\site-packages",
    r"C:\Users\Administrateur\navlori-fusion\.venv\Lib\site-packages",
]
for _vsp in _VENV_CANDIDATES:
    if os.path.isdir(_vsp) and _vsp not in sys.path:
        sys.path.insert(0, _vsp)
sys.path.insert(0, CONTROLLER_DIR)

from controller import Supervisor  # noqa: E402
from path_loader import load_path, summarise  # noqa: E402
from trajectory import HermiteTrajectory, feasibility  # noqa: E402
from drive import (  # noqa: E402
    DifferentialDriveTracker, DriveConfig, OdometrySynthesizer,
    wheel_velocities_for,
)


# ============================================================================
# Hardware constants (TIAGO++ defaults; we don't physically drive at the cap
# when pose-anchored, but the wheel motors still spin for visual realism)
# ============================================================================
WHEEL_RADIUS = 0.0985
AXLE_LENGTH = 0.4044
MAX_WHEEL_SPEED = 6.28  # rad/s (the PROTO cap; saturating is fine since we
                        # supervisor-anchor pose anyway)

CAMERA_NAME = "head_front_camera"
DEPTH_CAMERA_NAME = "head_front_camera_depth"

# Arm tuck pose (same as async_collector so visuals match prior dataset)
ARM_TUCK_POSITIONS = {
    "torso_lift_joint":    0.0,
    "arm_left_1_joint":    0.20, "arm_left_2_joint":   -1.10,
    "arm_left_3_joint":   -0.20, "arm_left_4_joint":    1.94,
    "arm_left_5_joint":   -1.57, "arm_left_6_joint":    1.37,
    "arm_left_7_joint":    0.0,
    "arm_right_1_joint":   0.20, "arm_right_2_joint":  -1.10,
    "arm_right_3_joint":   0.20, "arm_right_4_joint":   1.94,
    "arm_right_5_joint":  -1.57, "arm_right_6_joint":   1.37,
    "arm_right_7_joint":   0.0,
}


# ============================================================================
# Helpers
# ============================================================================
def init_motor(robot, name):
    m = robot.getDevice(name)
    if m is None:
        return None
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
    return m


def init_sensor(robot, name, timestep, label):
    d = robot.getDevice(name)
    if d is None:
        print(f"  [WARN] '{name}' not found ({label})")
        return None
    d.enable(timestep)
    print(f"  [OK] {label}: '{name}'")
    return d


def init_camera(robot, name, timestep):
    cam = robot.getDevice(name)
    if cam is None:
        return None
    cam.enable(timestep)
    print(f"  [OK] Camera '{name}' ({cam.getWidth()}x{cam.getHeight()})")
    return cam


def tuck_arms(robot, timestep):
    print("\n[Arms] Tucking arms...")
    for name, pos in ARM_TUCK_POSITIONS.items():
        m = robot.getDevice(name)
        if m is None:
            continue
        m.setVelocity(0.07 if name == "torso_lift_joint" else 1.0)
        m.setPosition(pos)
    for _ in range(80):
        robot.step(timestep)
    print("[Arms] Done.")


def set_wheels(lm, rm, v, omega):
    vl, vr = wheel_velocities_for(v, omega, AXLE_LENGTH, WHEEL_RADIUS,
                                   MAX_WHEEL_SPEED)
    lm.setVelocity(vl)
    rm.setVelocity(vr)


def supervisor_set_pose(node, x, y, yaw, keep_z=None):
    """Anchor the robot to (x, y, yaw) -- per-step delta is small so this
    LOOKS like driving, not teleporting. keep_z preserves the current
    floor-clearance (don't override Z or the robot may sink/jump)."""
    tf = node.getField("translation")
    rf = node.getField("rotation")
    cur = tf.getSFVec3f()
    z = keep_z if keep_z is not None else cur[2]
    tf.setSFVec3f([x, y, z])
    rf.setSFRotation([0.0, 0.0, 1.0, yaw])


def get_pose(node):
    pos = node.getPosition()
    ori = node.getOrientation()
    yaw = math.atan2(ori[3], ori[0])
    return pos[0], pos[1], pos[2], yaw


# ============================================================================
# CSV writer (mirrors async_collector.ModalityCSV)
# ============================================================================
class ModalityCSV:
    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = columns if columns[0] == "sim_time" else ["sim_time"] + columns
        self.file = None
        self.writer = None
        self.count = 0

    def open(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns,
                                      extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row):
        if self.writer is not None:
            self.writer.writerow(row)
            self.count += 1

    def close(self):
        if self.file is not None:
            self.file.close()
            print(f"  {os.path.basename(self.filepath):>30s}: {self.count} rows")


# ============================================================================
# Config loading
# ============================================================================
def load_config():
    p = Path(CONTROLLER_DIR) / "replay_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"replay_config.json not found in {CONTROLLER_DIR}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Normalise path_ids: accept [0, 1, 2] or "0-5" or "0,1,2"
    pids = cfg.get("path_ids", [0])
    if isinstance(pids, str):
        if "-" in pids:
            lo, hi = pids.split("-", 1)
            pids = list(range(int(lo), int(hi) + 1))
        else:
            pids = [int(x) for x in pids.split(",") if x.strip()]
    cfg["path_ids"] = list(pids)
    return cfg


# ============================================================================
# Per-path runner
# ============================================================================
def run_path(robot, node, timestep, cfg, path_id, sensors, motors, output_dir):
    dt_sim = timestep / 1000.0
    lm, rm = motors
    dataset_dir = Path(cfg["dataset_dir"])
    path_dir_src = dataset_dir / f"path_{path_id:02d}"

    # ── Load original modalities ──
    rp = load_path(path_dir_src, path_id)
    print(f"\n{'='*64}")
    print(f"  REPLAY {summarise(rp)}")
    print(f"{'='*64}")

    # ── Build trajectory ──
    ts = [w.t for w in rp.waypoints]
    xs = [w.x for w in rp.waypoints]
    ys = [w.y for w in rp.waypoints]
    traj = HermiteTrajectory(ts, xs, ys)

    # ── Feasibility report (informational; we pose-anchor anyway) ──
    feas = feasibility(traj,
                        max_speed=cfg["feasibility"]["max_speed_m_s"],
                        max_omega=cfg["feasibility"]["max_yaw_rate_rad_s"])
    print(f"  [feasibility] peak_speed={feas['peak_speed_m_s']:.3f} m/s, "
          f"peak_yaw_rate={feas['peak_yaw_rate_rad_s']:.3f} rad/s "
          f"-> within-limits={feas['feasible']} "
          f"(pose-anchored: {cfg.get('pose_anchor', True)})")

    # ── Output dirs ──
    path_dir_out = Path(output_dir) / f"path_{path_id:02d}"
    cam_dir = path_dir_out / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)

    # ── CSVs ──
    odom_cols = ["odom_x", "odom_y", "odom_theta_deg",
                  "odom_linear_vel", "odom_angular_vel",
                  "wheel_left_vel", "wheel_right_vel"]
    gt_cols = ["gt_x", "gt_y", "gt_z", "gt_heading_rad", "gt_heading_deg",
                "path_id", "waypoint_idx", "is_original"]
    cam_cols = ["frame_id", "rgb_path", "depth_path",
                 "cam_x", "cam_y", "cam_z"]

    csvs = {
        "ground_truth": ModalityCSV(str(path_dir_out / "ground_truth.csv"), gt_cols),
        "odometry":     ModalityCSV(str(path_dir_out / "odometry.csv"), odom_cols),
        "camera":       ModalityCSV(str(path_dir_out / "camera.csv"), cam_cols),
    }
    for c in csvs.values():
        c.open()

    # ── Verbatim copies of original WiFi/IMU (constraint #1) ──
    if rp.imu_columns:
        imu_out = ModalityCSV(str(path_dir_out / "imu.csv"),
                               rp.imu_columns)
        imu_out.open()
        for r in rp.imu_rows:
            imu_out.write(r.raw)
        imu_out.close()
    if rp.wifi_columns:
        wifi_out = ModalityCSV(str(path_dir_out / "wifi.csv"),
                                rp.wifi_columns)
        wifi_out.open()
        for r in rp.wifi_rows:
            wifi_out.write(r.raw)
        wifi_out.close()

    # ── Initial pose: spline at t_start (exactly the first GT WP) ──
    s0 = traj.evaluate(rp.t_start)
    _, _, z_keep, _ = get_pose(node)
    supervisor_set_pose(node, s0.x, s0.y, s0.yaw, keep_z=z_keep)
    node.resetPhysics()
    # let physics settle (touches floor, arm tuck stays)
    for _ in range(5):
        robot.step(timestep)

    # ── Drive + odom state ──
    drive_cfg = DriveConfig(
        dt_lookahead=float(cfg["drive"].get("dt_lookahead_s", 0.20)),
        k_v=float(cfg["drive"].get("k_v", 1.0)),
        k_yaw=float(cfg["drive"].get("k_yaw", 2.5)),
        k_lat=float(cfg["drive"].get("k_lat", 0.8)),
    )
    tracker = DifferentialDriveTracker(traj, drive_cfg)

    odom = OdometrySynthesizer(axle=AXLE_LENGTH, wheel_r=WHEEL_RADIUS)
    odom.reset(s0.x, s0.y, s0.yaw)

    # ── Sample interval bookkeeping (sim_time-based, not step-counter) ──
    rates = cfg["rates_hz"]
    gt_period = 1.0 / max(0.1, float(rates["ground_truth_dense"]))
    odom_period = 1.0 / max(0.1, float(rates["odometry"]))
    cam_period = 1.0 / max(0.1, float(rates["camera"]))

    next_gt = rp.t_start
    next_odom = rp.t_start
    next_cam = rp.t_start
    frame_count = 0

    # Map (original) waypoint timestamps -> waypoint_idx so we can mark
    # the synthesised dense GT rows that coincide with original WPs.
    original_ts = {round(w.t, 6): i for i, w in enumerate(rp.waypoints)}

    # ── Main loop ──
    t0_wall = pytime.time()
    pose_anchor = bool(cfg.get("pose_anchor", True))
    sim_t_start_offset = robot.getTime()
    # The path's sim_time starts at rp.t_start; the controller's sim clock
    # starts wherever Webots is. We map controller sim time -> path sim time
    # by: path_t = (robot.getTime() - sim_t_start_offset) + rp.t_start.

    last_print = -1.0
    step_count = 0
    finished = False
    extra_steps_after_finish = 30  # to flush late camera frames

    while robot.step(timestep) != -1:
        elapsed = robot.getTime() - sim_t_start_offset
        path_t = rp.t_start + elapsed

        # ── End condition ──
        if path_t > rp.t_end:
            if not finished:
                # snap to final WP exactly
                sf = traj.evaluate(rp.t_end)
                if pose_anchor:
                    supervisor_set_pose(node, sf.x, sf.y, sf.yaw, keep_z=z_keep)
                lm.setVelocity(0.0)
                rm.setVelocity(0.0)
                print(f"  [done] path_t={path_t:.3f} >= t_end={rp.t_end:.3f}, "
                       f"flushing {extra_steps_after_finish} extra steps")
                finished = True
            extra_steps_after_finish -= 1
            if extra_steps_after_finish <= 0:
                break

        # ── Spline state at current path_t ──
        st = traj.evaluate(path_t)

        # ── Apply pose anchor + drive command ──
        if pose_anchor:
            supervisor_set_pose(node, st.x, st.y, st.yaw, keep_z=z_keep)

        rx, ry, _, ryaw = get_pose(node)
        cmd = tracker.step(path_t, rx, ry, ryaw)
        set_wheels(lm, rm, cmd.v, cmd.omega)

        # ── Odometry synthesis (one row at the configured rate) ──
        if not finished and path_t >= next_odom:
            row = odom.step(cmd.v, cmd.omega, dt_sim)
            row["sim_time"] = round(path_t, 4)
            csvs["odometry"].write(row)
            next_odom += odom_period

        # ── GT-dense (one row at the configured rate, plus exact rows at
        #     original waypoint times -- handled below) ──
        if not finished and path_t >= next_gt:
            gt_row = {
                "sim_time":       round(path_t, 4),
                "gt_x":           round(st.x, 5),
                "gt_y":           round(st.y, 5),
                "gt_z":           round(z_keep, 5),
                "gt_heading_rad": round(st.yaw, 5),
                "gt_heading_deg": round(math.degrees(st.yaw), 3),
                "path_id":        path_id,
                "waypoint_idx":   -1,
                "is_original":    False,
            }
            csvs["ground_truth"].write(gt_row)
            next_gt += gt_period

        # ── Camera capture ──
        cam = sensors.get("camera")
        depth = sensors.get("depth")
        if not finished and path_t >= next_cam and cam is not None:
            frame_id = f"{frame_count:06d}"
            rgb_rel = f"camera/rgb_{frame_id}.png"
            depth_rel = ""
            cam.saveImage(str(cam_dir / f"rgb_{frame_id}.png"), 80)
            if depth is not None:
                depth_rel = f"camera/depth_{frame_id}.png"
                depth.saveImage(str(cam_dir / f"depth_{frame_id}.png"), 80)
            cam_row = {
                "sim_time": round(path_t, 4),
                "frame_id": frame_id,
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "cam_x": round(st.x, 5),
                "cam_y": round(st.y, 5),
                "cam_z": round(z_keep + 1.0, 5),  # camera ~1m above base
            }
            csvs["camera"].write(cam_row)
            frame_count += 1
            next_cam += cam_period

        # ── Progress print ──
        if path_t - last_print >= 1.0:
            print(f"  [{path_t:6.2f}/{rp.t_end:6.2f}s] "
                   f"pos=({st.x:+6.2f},{st.y:+6.2f}) "
                   f"v={cmd.v:.2f} w={cmd.omega:+.2f} "
                   f"frames={frame_count}")
            last_print = path_t

        step_count += 1

    # ── Add an "is_original=True" row for every original waypoint, exact
    #     timestamp/coord. Sort all GT rows by sim_time on close so the
    #     output CSV remains monotone. ──
    for i, w in enumerate(rp.waypoints):
        gt_row = {
            "sim_time":       round(w.t, 4),
            "gt_x":           round(w.x, 5),
            "gt_y":           round(w.y, 5),
            "gt_z":           round(0.0, 5),
            "gt_heading_rad": round(traj.evaluate(w.t).yaw, 5),
            "gt_heading_deg": round(math.degrees(traj.evaluate(w.t).yaw), 3),
            "path_id":        path_id,
            "waypoint_idx":   i,
            "is_original":    True,
        }
        csvs["ground_truth"].write(gt_row)

    # ── Close + tally ──
    lm.setVelocity(0.0)
    rm.setVelocity(0.0)
    print(f"\n  path {path_id} totals (wall {pytime.time() - t0_wall:.1f}s):")
    for c in csvs.values():
        c.close()

    # Sort & rewrite ground_truth.csv by sim_time so original + dense rows
    # are time-interleaved.
    _sort_csv_by_sim_time(str(path_dir_out / "ground_truth.csv"))

    # Per-path metadata
    meta = {
        "path_id": path_id,
        "dataset_dir": str(dataset_dir),
        "world": cfg.get("world_dataset_id", "unknown"),
        "n_original_waypoints": rp.n_waypoints,
        "duration_s": rp.duration,
        "frames_emitted": frame_count,
        "feasibility": feas,
        "pose_anchored": pose_anchor,
    }
    with open(path_dir_out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "path_id": path_id,
        "frames": frame_count,
        "duration_s": rp.duration,
        "n_original_waypoints": rp.n_waypoints,
    }


def _sort_csv_by_sim_time(path: str):
    """Sort an existing CSV in-place by sim_time, preserving the header."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    rows.sort(key=lambda r: float(r["sim_time"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================================
# Main
# ============================================================================
def main():
    cfg = load_config()
    print("=" * 64)
    print("  NavLoRI Replay Collector")
    print("  dataset = ", cfg["dataset_dir"])
    print("  output  = ", cfg["output_dir"])
    print("  paths   = ", cfg["path_ids"])
    print("=" * 64)

    robot = Supervisor()
    node = robot.getSelf()
    if node is None:
        print("ERROR: set 'supervisor TRUE' on the TIAGO++ node in the world.")
        return
    timestep = int(robot.getBasicTimeStep())

    # ── Sensors ──
    print("\n[Sensors]")
    sensors = {
        "accelerometer": init_sensor(robot, "accelerometer", timestep, "Accelerometer"),
        "gyro":          init_sensor(robot, "gyro", timestep, "Gyroscope"),
        "imu":           init_sensor(robot, "inertial unit", timestep, "InertialUnit"),
    }
    # cameras (may be absent)
    print("\n[Cameras]")
    cam = init_camera(robot, CAMERA_NAME, timestep)
    depth = init_camera(robot, DEPTH_CAMERA_NAME, timestep)
    if cam is None:
        # alt names
        for alt in ("Astra rgb", "camera", "head_2_camera"):
            cam = init_camera(robot, alt, timestep)
            if cam is not None:
                break
    sensors["camera"] = cam
    sensors["depth"] = depth

    # ── Motors ──
    print("\n[Motors]")
    lm = init_motor(robot, "wheel_left_joint")
    rm = init_motor(robot, "wheel_right_joint")
    if lm is None or rm is None:
        print("ERROR: wheel motors not found")
        return
    print("  [OK] wheel motors initialised")

    # ── Arms tucked (matches async_collector dataset visuals) ──
    tuck_arms(robot, timestep)

    # ── Run paths ──
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pid in cfg["path_ids"]:
        try:
            r = run_path(robot, node, timestep, cfg, pid, sensors, (lm, rm),
                          str(output_dir))
            summary[pid] = r
        except FileNotFoundError as e:
            print(f"[skip] path {pid}: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] path {pid}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Top-level summary ──
    print("\n" + "=" * 64)
    print("  REPLAY COMPLETE")
    for pid, s in summary.items():
        print(f"    path {pid:>2}: dur={s['duration_s']:6.2f}s  "
              f"frames={s['frames']:5d}  "
              f"original_wps={s['n_original_waypoints']}")
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "controller": "replay_collector",
            "config": cfg,
            "paths": summary,
            "created": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)


if __name__ == "__main__":
    main()
