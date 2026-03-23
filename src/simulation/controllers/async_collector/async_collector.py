"""
NavLoRI — Asynchronous Multi-Modal Data Collector
===================================================
Event-driven controller for TIAGO++ in Webots.
Each sensor fires at its own jittered rate, producing separate CSV files
per modality + PNG images for camera/depth.

Modalities & nominal rates:
  - IMU (accel + gyro + orientation): ~31 Hz (every 1 step)
  - Odometry (wheel encoders):        ~15 Hz (every 2 steps)
  - WiFi RSSI (GPR predictor):        ~1 Hz  (every 31 steps)
  - Camera RGB + Depth:               ~0.5 Hz  (every 62 steps)
  - Ground Truth:                      ~10 Hz (every 3 steps)

Jitter: ±20% on each interval to simulate real-world async arrivals.

Output structure (per path):
  async_data/path_{id}/
    imu.csv            — timestamp, accel_xyz, gyro_xyz, roll/pitch/yaw
    odometry.csv       — timestamp, odom_x/y/theta, velocities
    wifi.csv           — timestamp, per-AP RSSI columns
    ground_truth.csv   — timestamp, gt_x, gt_y, heading
    camera/
      rgb_{step:06d}.png
      depth_{step:06d}.png

Author: Mohamed — NavLoRI Project, CESI LINEACT
"""

import sys
import os
import math
import csv
import json
import time as pytime
import random
import numpy as np
from pathlib import Path
from controller import Supervisor

# Add parent controllers dir for wifi_predictor import
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CONTROLLER_DIR, "..", "wifi_supervisor"))

from dwa_planner import DWAPlanner, DWAConfig, depth_to_obstacles

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# --- TIAGo++ Physical Parameters ---
WHEEL_RADIUS = 0.0985
AXLE_LENGTH = 0.4044
MAX_WHEEL_SPEED = 6.28

# --- Navigation (DWA) ---
ARRIVAL_DIST = 0.3  # meters — waypoint reached threshold

# --- Batch Path Selection ---
# Change these before each run:
#   Batch 1: START_PATH=0,  END_PATH=9
#   Batch 2: START_PATH=10, END_PATH=19
#   Batch 3: START_PATH=20, END_PATH=29
#   All:     START_PATH=None, END_PATH=None
BATCH_START = 0
BATCH_END = 9

# --- Sensor Nominal Intervals (in simulation steps, 1 step = 32ms) ---
# Jitter of ±20% is applied each time a sensor fires
SENSOR_INTERVALS = {
    "imu":          1,    # ~31 Hz
    "odometry":     2,    # ~15 Hz
    "wifi":         31,   # ~1 Hz
    "camera":       62,   # ~0.5 Hz
    "ground_truth": 3,    # ~10 Hz
}
JITTER_FRACTION = 0.20  # ±20%

# --- Camera Settings ---
CAMERA_NAME = "head_front_camera"       # TIAGO++ head camera device name
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
DEPTH_CAMERA_NAME = "head_front_camera_depth"  # Will try RangeFinder if available

# --- Arm Tuck ---
ARM_TUCK_POSITIONS = {
    "torso_lift_joint":    0.15,
    "arm_left_1_joint":    0.20, "arm_left_2_joint":   -1.10,
    "arm_left_3_joint":   -0.20, "arm_left_4_joint":    1.94,
    "arm_left_5_joint":   -1.57, "arm_left_6_joint":    1.37,
    "arm_left_7_joint":    0.0,
    "arm_right_1_joint":   0.20, "arm_right_2_joint":  -1.10,
    "arm_right_3_joint":   0.20, "arm_right_4_joint":   1.94,
    "arm_right_5_joint":  -1.57, "arm_right_6_joint":   1.37,
    "arm_right_7_joint":   0.0,
}

# --- WiFi ---
RSSI_VISIBILITY_THRESHOLD = -85.0

# --- InfluxDB (optional) ---
ENABLE_VIZ = True
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "navlori-influx-token-2026"
INFLUXDB_ORG = "navlori"
INFLUXDB_BUCKET = "async_data"

# --- Output ---
# Data saves to <project_root>/data/async_collection/
# Resolved at runtime relative to this script
OUTPUT_DIR = os.path.normpath(os.path.join(CONTROLLER_DIR, "..", "..", "..", "..", "data", "async_collection"))


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_pose(node):
    pos = node.getPosition()
    ori = node.getOrientation()
    heading = math.atan2(ori[3], ori[0])
    return pos[0], pos[1], pos[2], heading


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def set_wheels(lm, rm, v_linear, v_angular):
    vl = (v_linear - v_angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vr = (v_linear + v_angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vl = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vl))
    vr = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vr))
    lm.setVelocity(vl)
    rm.setVelocity(vr)


def stop_wheels(lm, rm):
    lm.setVelocity(0.0)
    rm.setVelocity(0.0)


def tuck_arms(robot, timestep):
    print("\n[Arms] Tucking arms...")
    for name, pos in ARM_TUCK_POSITIONS.items():
        motor = robot.getDevice(name)
        if motor is None:
            continue
        motor.setVelocity(0.07 if name == "torso_lift_joint" else 1.0)
        motor.setPosition(pos)
    for _ in range(80):
        robot.step(timestep)
    print("[Arms] Tucked.")


def init_sensor(robot, name, timestep, label="Sensor"):
    try:
        dev = robot.getDevice(name)
        if dev is None:
            print(f"  [WARN] '{name}' not found")
            return None
        dev.enable(timestep)
        print(f"  [OK] {label}: '{name}'")
        return dev
    except Exception as e:
        print(f"  [WARN] '{name}': {e}")
        return None


def init_wheel_encoder(robot, motor_name, timestep):
    sensor_name = motor_name + "_sensor"
    try:
        sensor = robot.getDevice(sensor_name)
        if sensor is None:
            sensor = robot.getDevice(motor_name.replace("_joint", "") + "_sensor")
        if sensor is None:
            print(f"  [WARN] Encoder '{sensor_name}' not found")
            return None
        sensor.enable(timestep)
        print(f"  [OK] Encoder: '{sensor_name}'")
        return sensor
    except Exception as e:
        print(f"  [WARN] Encoder '{sensor_name}': {e}")
        return None


def load_wifi_predictor():
    search_paths = [
        os.path.join(CONTROLLER_DIR, "webots_export"),
        os.path.join(CONTROLLER_DIR, "..", "wifi_supervisor", "webots_export"),
        os.path.join(CONTROLLER_DIR, "..", "tiago_unified_collector", "webots_export"),
    ]
    for p in search_paths:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "rssi_grid.npz")):
            try:
                from wifi_predictor import WiFiPredictor
                pred = WiFiPredictor(p)
                print(f"  [OK] WiFi predictor loaded from {p}")
                return pred, pred.ap_names
            except Exception as e:
                print(f"  [WARN] WiFi predictor error: {e}")
                return None, []
    print("  [WARN] WiFi predictor not found")
    return None, []


# ═══════════════════════════════════════════════════════════════════════
# ASYNC SENSOR SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class SensorScheduler:
    """Manages jittered firing times for each sensor modality."""

    def __init__(self, intervals, jitter_frac, seed=42):
        self.intervals = dict(intervals)
        self.jitter_frac = jitter_frac
        self.rng = random.Random(seed)
        self.next_fire = {}
        self.reset()

    def reset(self):
        """Reset all sensors to fire at step 0."""
        for name in self.intervals:
            self.next_fire[name] = 0

    def should_fire(self, sensor_name, step):
        """Check if sensor should fire at this step."""
        return step >= self.next_fire[sensor_name]

    def advance(self, sensor_name, step):
        """Schedule next firing for this sensor with jitter."""
        base = self.intervals[sensor_name]
        jitter = self.rng.uniform(-self.jitter_frac, self.jitter_frac)
        interval = max(1, round(base * (1.0 + jitter)))
        self.next_fire[sensor_name] = step + interval


# ═══════════════════════════════════════════════════════════════════════
# PER-MODALITY CSV WRITERS
# ═══════════════════════════════════════════════════════════════════════

class ModalityCSV:
    """Writes a CSV for a single modality."""

    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = ["sim_time"] + columns
        self.file = None
        self.writer = None
        self.count = 0

    def open(self):
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns,
                                     extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row):
        if self.writer:
            self.writer.writerow(row)
            self.count += 1

    def close(self):
        if self.file:
            self.file.close()
            print(f"  {os.path.basename(self.filepath)}: {self.count} events")


# ═══════════════════════════════════════════════════════════════════════
# INFLUXDB WRITER (optional)
# ═══════════════════════════════════════════════════════════════════════

class InfluxWriter:
    def __init__(self, url, token, org, bucket):
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            self.Point = Point
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.bucket = bucket
            self.org = org
            self.ready = True
            print(f"[InfluxDB] Connected: {url}, bucket={bucket}")
        except Exception as e:
            print(f"[InfluxDB] Not available: {e}")
            self.ready = False
            self.Point = None

    def write_sensor_event(self, measurement, tags, fields):
        if not self.ready:
            return
        try:
            pt = self.Point(measurement)
            for k, v in tags.items():
                pt = pt.tag(k, str(v))
            for k, v in fields.items():
                if isinstance(v, (int, float)):
                    pt = pt.field(k, v)
            self.write_api.write(bucket=self.bucket, org=self.org, record=pt)
        except Exception as e:
            print(f"[InfluxDB] Write error: {e}")

    def write_points(self, points):
        """Write multiple points at once."""
        if not self.ready or not points:
            return
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            print(f"[InfluxDB] Write error: {e}")

    def close(self):
        if self.ready:
            self.client.close()


# ═══════════════════════════════════════════════════════════════════════
# SINGLE PATH COLLECTION
# ═══════════════════════════════════════════════════════════════════════

GRAVITY = 9.81


def run_path(robot, node, timestep, path_id, path_info,
             sensors, motors, predictor, ap_names,
             influx, output_dir, scheduler):
    """Navigate one path, collecting async sensor events."""

    dt = timestep / 1000.0
    lm, rm = motors
    wps = path_info["waypoints"]
    path_name = path_info["name"]
    path_dir = os.path.join(output_dir, f"path_{path_id:02d}")
    cam_dir = os.path.join(path_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)

    # ── Teleport to start ──
    sx, sy = wps[0]
    h0 = math.atan2(wps[1][1] - sy, wps[1][0] - sx) if len(wps) > 1 else 0.0
    tf = node.getField("translation")
    rf = node.getField("rotation")
    cur = tf.getSFVec3f()
    tf.setSFVec3f([sx, sy, cur[2]])
    rf.setSFRotation([0, 0, 1, h0])
    node.resetPhysics()

    for _ in range(20):
        robot.step(timestep)

    print(f"\n{'='*60}")
    print(f"  PATH {path_id}: {path_name}  ({len(wps)} waypoints)")
    print(f"{'='*60}")

    # ── Open per-modality CSV files ──
    imu_cols = ["accel_x", "accel_y", "accel_z", "accel_magnitude",
                "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude",
                "roll_deg", "pitch_deg", "yaw_deg"]
    odom_cols = ["odom_x", "odom_y", "odom_theta_deg",
                 "odom_linear_vel", "odom_angular_vel",
                 "wheel_left_vel", "wheel_right_vel"]
    wifi_cols = ["wifi_visible_count", "wifi_strongest_rssi", "wifi_strongest_mac"]
    wifi_cols += [f"wifi_rssi_{m.replace(':', '')}" for m in ap_names]
    gt_cols = ["gt_x", "gt_y", "gt_z", "gt_heading_rad", "gt_heading_deg",
               "path_id", "waypoint_idx"]
    cam_cols = ["frame_id", "rgb_path", "depth_path"]

    csvs = {
        "imu":          ModalityCSV(os.path.join(path_dir, "imu.csv"), imu_cols),
        "odometry":     ModalityCSV(os.path.join(path_dir, "odometry.csv"), odom_cols),
        "wifi":         ModalityCSV(os.path.join(path_dir, "wifi.csv"), wifi_cols),
        "ground_truth": ModalityCSV(os.path.join(path_dir, "ground_truth.csv"), gt_cols),
        "camera":       ModalityCSV(os.path.join(path_dir, "camera.csv"), cam_cols),
    }
    for c in csvs.values():
        c.open()

    # ── State ──
    scheduler.reset()
    prev_left_pos = None
    prev_right_pos = None
    odom_x = odom_y = odom_theta = 0.0
    step_count = 0
    wp_idx = 1
    pause_steps = 0
    path_finished = False
    extra_steps = 0
    frame_count = 0

    # ── DWA planner ──
    dwa = DWAPlanner()
    cur_v = 0.0
    cur_omega = 0.0

    camera = sensors.get("camera")
    depth = sensors.get("depth")

    while robot.step(timestep) != -1:
        sim_time = round(robot.getTime(), 4)

        # ══════════════════════════════════════════════════════════════
        # SENSOR EVENTS (async — each modality checks independently)
        # ══════════════════════════════════════════════════════════════

        # ── IMU ──
        if scheduler.should_fire("imu", step_count):
            row = {"sim_time": sim_time}
            accel = sensors.get("accelerometer")
            gyro = sensors.get("gyro")
            imu_dev = sensors.get("imu")

            if accel:
                a = accel.getValues()
                ax, ay, az = round(a[0], 5), round(a[1], 5), round(a[2] - GRAVITY, 5)
                row.update(accel_x=ax, accel_y=ay, accel_z=az,
                           accel_magnitude=round(math.sqrt(ax**2 + ay**2 + az**2), 5))
            if gyro:
                g = gyro.getValues()
                gx, gy, gz = round(g[0], 6), round(g[1], 6), round(g[2], 6)
                row.update(gyro_x=gx, gyro_y=gy, gyro_z=gz,
                           gyro_magnitude=round(math.sqrt(gx**2 + gy**2 + gz**2), 6))
            if imu_dev:
                rpy = imu_dev.getRollPitchYaw()
                row.update(roll_deg=round(math.degrees(rpy[0]), 3),
                           pitch_deg=round(math.degrees(rpy[1]), 3),
                           yaw_deg=round(math.degrees(rpy[2]), 3))

            csvs["imu"].write(row)
            if influx:
                influx.write_sensor_event("imu", {"robot": "tiago", "path_id": path_id},
                                          {k: v for k, v in row.items() if isinstance(v, (int, float))})
            scheduler.advance("imu", step_count)

        # ── Odometry ──
        if scheduler.should_fire("odometry", step_count):
            l_enc = sensors.get("left_encoder")
            r_enc = sensors.get("right_encoder")
            if l_enc and r_enc:
                l_pos = l_enc.getValue()
                r_pos = r_enc.getValue()

                if prev_left_pos is not None:
                    d_left = l_pos - prev_left_pos
                    d_right = r_pos - prev_right_pos
                    l_dist = d_left * WHEEL_RADIUS
                    r_dist = d_right * WHEEL_RADIUS
                    lin_dist = (l_dist + r_dist) / 2.0
                    ang_dist = (r_dist - l_dist) / AXLE_LENGTH

                    odom_theta += ang_dist
                    odom_x += lin_dist * math.cos(odom_theta)
                    odom_y += lin_dist * math.sin(odom_theta)

                    row = {
                        "sim_time": sim_time,
                        "odom_x": round(odom_x, 4),
                        "odom_y": round(odom_y, 4),
                        "odom_theta_deg": round(math.degrees(odom_theta), 2),
                        "odom_linear_vel": round(lin_dist / dt, 5),
                        "odom_angular_vel": round(ang_dist / dt, 5),
                        "wheel_left_vel": round(d_left / dt * WHEEL_RADIUS, 5),
                        "wheel_right_vel": round(d_right / dt * WHEEL_RADIUS, 5),
                    }
                    csvs["odometry"].write(row)
                    if influx:
                        influx.write_sensor_event("odometry",
                                                  {"robot": "tiago", "path_id": path_id},
                                                  {k: v for k, v in row.items() if isinstance(v, (int, float))})

                prev_left_pos = l_pos
                prev_right_pos = r_pos

            scheduler.advance("odometry", step_count)

        # ── WiFi ──
        if scheduler.should_fire("wifi", step_count):
            gt_x, gt_y, _, _ = get_pose(node)
            if predictor and predictor.is_in_bounds(gt_x, gt_y):
                scan = predictor.predict(gt_x, gt_y, add_noise=True)
                row = {"sim_time": sim_time}
                visible = sum(1 for r in scan.values() if r > RSSI_VISIBILITY_THRESHOLD)
                row["wifi_visible_count"] = visible
                if scan:
                    best_mac = max(scan, key=scan.get)
                    row["wifi_strongest_rssi"] = round(scan[best_mac], 1)
                    row["wifi_strongest_mac"] = best_mac
                for ap in ap_names:
                    row[f"wifi_rssi_{ap.replace(':', '')}"] = round(scan.get(ap, -200.0), 1)
                csvs["wifi"].write(row)
                if influx:
                    # Write summary measurement
                    influx.write_sensor_event("wifi",
                                              {"robot": "tiago", "path_id": path_id},
                                              {k: v for k, v in row.items() if isinstance(v, (int, float))})
                    # Write per-AP measurements (for Grafana bar chart)
                    P = influx.Point
                    ap_points = []
                    for ap_mac, rssi in scan.items():
                        ap_points.append(
                            P("wifi_rssi")
                            .tag("robot", "tiago")
                            .tag("path_id", str(path_id))
                            .tag("ap_mac", ap_mac)
                            .tag("ap_short", ap_mac[-8:])
                            .field("rssi", round(rssi, 1))
                            .field("robot_x", round(gt_x, 3))
                            .field("robot_y", round(gt_y, 3))
                        )
                    influx.write_points(ap_points)
            scheduler.advance("wifi", step_count)

        # ── Camera (RGB + Depth) ──
        if scheduler.should_fire("camera", step_count):
            saved_rgb = False
            saved_depth = False
            frame_id = f"{step_count:06d}"

            if camera is not None:
                rgb_path = os.path.join(cam_dir, f"rgb_{frame_id}.png")
                camera.saveImage(rgb_path, 80)  # quality 80
                saved_rgb = True

            if depth is not None:
                depth_path = os.path.join(cam_dir, f"depth_{frame_id}.png")
                depth.saveImage(depth_path, 80)
                saved_depth = True

            if saved_rgb or saved_depth:
                row = {
                    "sim_time": sim_time,
                    "frame_id": frame_id,
                    "rgb_path": f"camera/rgb_{frame_id}.png" if saved_rgb else "",
                    "depth_path": f"camera/depth_{frame_id}.png" if saved_depth else "",
                }
                csvs["camera"].write(row)
                frame_count += 1

                if influx:
                    influx.write_sensor_event("camera",
                                              {"robot": "tiago", "path_id": path_id},
                                              {"sim_time": sim_time, "frame_id": int(frame_id)})

            scheduler.advance("camera", step_count)

        # ── Ground Truth ──
        if scheduler.should_fire("ground_truth", step_count):
            gt_x, gt_y, gt_z, gt_h = get_pose(node)
            row = {
                "sim_time": sim_time,
                "gt_x": round(gt_x, 5), "gt_y": round(gt_y, 5), "gt_z": round(gt_z, 5),
                "gt_heading_rad": round(gt_h, 5),
                "gt_heading_deg": round(math.degrees(gt_h), 3),
                "path_id": path_id,
                "waypoint_idx": wp_idx,
            }
            csvs["ground_truth"].write(row)
            if influx:
                influx.write_sensor_event("ground_truth",
                                          {"robot": "tiago", "path_id": path_id},
                                          {k: v for k, v in row.items() if isinstance(v, (int, float))})
            scheduler.advance("ground_truth", step_count)

        step_count += 1

        # ══════════════════════════════════════════════════════════════
        # NAVIGATION — DWA (Dynamic Window Approach)
        # ══════════════════════════════════════════════════════════════
        if path_finished:
            stop_wheels(lm, rm)
            extra_steps += 1
            if extra_steps >= 50:
                break
            continue

        if wp_idx >= len(wps):
            stop_wheels(lm, rm)
            path_finished = True
            print(f"  Path {path_id} complete, collecting 50 extra samples...")
            continue

        if pause_steps > 0:
            pause_steps -= 1
            continue

        x, y, _, heading = get_pose(node)
        tx, ty = wps[wp_idx]
        dist = math.hypot(tx - x, ty - y)

        if dist < ARRIVAL_DIST:
            stop_wheels(lm, rm)
            print(f"   WP [{wp_idx}/{len(wps)}] Reached ({tx:.1f}, {ty:.1f})")
            wp_idx += 1
            pause_steps = 5
            cur_v = 0.0
            cur_omega = 0.0
            continue

        # Get obstacle points from depth camera
        obstacles = depth_to_obstacles(depth, x, y, heading)

        # DWA: compute optimal (v, omega)
        state = [x, y, heading, cur_v, cur_omega]
        goal = [tx, ty]
        cur_v, cur_omega, _ = dwa.plan(state, goal, obstacles)

        set_wheels(lm, rm, cur_v, cur_omega)

        # Console log every ~3 seconds
        if step_count % max(1, int(3.0 / dt)) == 0:
            n_obs = len(obstacles) if obstacles is not None else 0
            print(f"  [DWA] t={sim_time:7.2f}s  wp=[{wp_idx}/{len(wps)}]  "
                  f"v={cur_v:.2f}  w={cur_omega:.2f}  obs={n_obs}  "
                  f"frames={frame_count}  imu={csvs['imu'].count}")

    stop_wheels(lm, rm)

    # ── Close CSVs ──
    print(f"\n  Path {path_id} results:")
    for c in csvs.values():
        c.close()

    # ── Write path metadata ──
    meta = {
        "path_id": path_id,
        "path_name": path_name,
        "waypoints": wps,
        "timestep_ms": timestep,
        "sensor_intervals": dict(SENSOR_INTERVALS),
        "jitter_fraction": JITTER_FRACTION,
        "samples": {name: c.count for name, c in csvs.items()},
        "frames": frame_count,
    }
    with open(os.path.join(path_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {name: c.count for name, c in csvs.items()}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    robot = Supervisor()
    node = robot.getSelf()
    if node is None:
        print("ERROR: Set 'supervisor' to TRUE on the robot node.")
        return

    timestep = int(robot.getBasicTimeStep())

    print("=" * 60)
    print("  NavLoRI Async Multi-Modal Data Collector")
    print("  IMU + Odometry + WiFi + Camera + Ground Truth")
    print("  Event-driven, jittered rates, separate CSV per modality")
    print("  DWA navigation — obstacle avoidance")
    print("=" * 60)

    # ── Batch config (edit BATCH_START / BATCH_END at top of file) ──
    START_PATH = BATCH_START
    END_PATH = BATCH_END

    # ── Init Sensors ──
    print("\n[Sensors]")
    sensors = {
        "accelerometer": init_sensor(robot, "accelerometer", timestep, "Accelerometer"),
        "gyro":          init_sensor(robot, "gyro", timestep, "Gyroscope"),
        "imu":           init_sensor(robot, "inertial unit", timestep, "InertialUnit"),
        "left_encoder":  init_wheel_encoder(robot, "wheel_left_joint", timestep),
        "right_encoder": init_wheel_encoder(robot, "wheel_right_joint", timestep),
    }

    # ── Camera ──
    print("\n[Camera]")
    camera = robot.getDevice(CAMERA_NAME)
    if camera is None:
        # Try alternative names from TIAGO++ proto
        for alt in ["Astra rgb", "camera", "head_2_camera"]:
            camera = robot.getDevice(alt)
            if camera is not None:
                print(f"  [OK] Camera found as '{alt}'")
                break
    if camera is not None:
        camera.enable(timestep)
        print(f"  [OK] RGB camera enabled ({camera.getWidth()}x{camera.getHeight()})")
        sensors["camera"] = camera
    else:
        print("  [WARN] No RGB camera found — will skip camera capture")
        sensors["camera"] = None

    # Depth (RangeFinder)
    depth = robot.getDevice(DEPTH_CAMERA_NAME)
    if depth is None:
        for alt in ["Astra depth", "range-finder", "head_2_depth"]:
            depth = robot.getDevice(alt)
            if depth is not None:
                print(f"  [OK] Depth found as '{alt}'")
                break
    if depth is not None:
        depth.enable(timestep)
        print(f"  [OK] Depth sensor enabled ({depth.getWidth()}x{depth.getHeight()})")
        sensors["depth"] = depth
    else:
        print("  [WARN] No depth sensor found — will skip depth capture")
        sensors["depth"] = None

    # ── Motors ──
    lm = robot.getDevice("wheel_left_joint")
    rm = robot.getDevice("wheel_right_joint")
    if not lm or not rm:
        print("ERROR: Motors not found.")
        return
    lm.setPosition(float("inf"))
    rm.setPosition(float("inf"))
    stop_wheels(lm, rm)

    # ── Tuck Arms ──
    tuck_arms(robot, timestep)

    # ── WiFi ──
    print("\n[WiFi]")
    predictor, ap_names = load_wifi_predictor()

    # ── InfluxDB ──
    influx = None
    if ENABLE_VIZ:
        influx = InfluxWriter(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)

    # ── Paths ──
    from paths import ALL_PATHS
    all_ids = sorted(ALL_PATHS.keys())

    if START_PATH is not None and END_PATH is not None:
        path_ids = [pid for pid in all_ids if START_PATH <= pid <= END_PATH]
        print(f"\n[Batch] Running paths {START_PATH} to {END_PATH} "
              f"({len(path_ids)} paths)")
    elif START_PATH is not None:
        path_ids = [pid for pid in all_ids if pid >= START_PATH]
        print(f"\n[Batch] Running paths {START_PATH}+ ({len(path_ids)} paths)")
    else:
        path_ids = all_ids
        print(f"\n[Batch] Running ALL {len(path_ids)} paths")

    # ── Output Dir ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Output] {OUTPUT_DIR}")

    # ── Scheduler ──
    scheduler = SensorScheduler(SENSOR_INTERVALS, JITTER_FRACTION)

    # ── Run ──
    total_stats = {}
    for pid in path_ids:
        if pid not in ALL_PATHS:
            print(f"WARNING: Path {pid} not found, skipping.")
            continue
        stats = run_path(
            robot=robot, node=node, timestep=timestep,
            path_id=pid, path_info=ALL_PATHS[pid],
            sensors=sensors, motors=(lm, rm),
            predictor=predictor, ap_names=ap_names,
            influx=influx, output_dir=OUTPUT_DIR,
            scheduler=scheduler,
        )
        total_stats[pid] = stats

    if influx:
        influx.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print(f"  Paths: {len(path_ids)}")
    for pid, stats in total_stats.items():
        print(f"    Path {pid:2d}: imu={stats.get('imu',0):5d}  odom={stats.get('odometry',0):5d}  "
              f"wifi={stats.get('wifi',0):4d}  cam={stats.get('camera',0):4d}  gt={stats.get('ground_truth',0):4d}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Write global metadata ──
    global_meta = {
        "collector": "async_collector",
        "timestep_ms": timestep,
        "sensor_intervals": dict(SENSOR_INTERVALS),
        "jitter_fraction": JITTER_FRACTION,
        "paths_collected": len(path_ids),
        "path_stats": {str(k): v for k, v in total_stats.items()},
        "created": pytime.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(global_meta, f, indent=2)


if __name__ == "__main__":
    main()
