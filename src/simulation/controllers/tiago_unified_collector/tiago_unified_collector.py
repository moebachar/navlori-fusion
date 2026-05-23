"""
NavLoRI TIAGO++ Unified Data Collector
========================================
Single controller that runs ON the TIAGO++ robot (supervisor=TRUE).
Merges navigation, IMU collection, WiFi prediction, and ground-truth
logging into one synchronized pipeline.

Data Sources (all timestamped to sim_time):
  1. Ground Truth   — Supervisor API (x, y, z, heading)
  2. IMU             — Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz),
                       Inertial Unit (roll, pitch, yaw)
  3. Odometry        — Wheel encoders → dead-reckoned pose & velocities
  4. WiFi RSSI       — GPR predictor → per-AP signal strengths

Outputs:
  • InfluxDB         — measurement "unified_sample" with ALL fields per timestep
  • CSV file         — one row per timestep with all sensor columns,
                       ready for offline LSTM training

Author: Mohamed — NavLoRI Project, CESI LINEACT
"""

import sys
import os
import math
import csv
import time as pytime
import numpy as np
from pathlib import Path
from controller import Supervisor

# ═════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

# --- Path Selection ---
SELECTED_PATH = 1       # 1 to 5  (set to 0 to run ALL paths sequentially)
RUN_ALL_PATHS = True    # Override: set True to collect data on ALL 5 paths

# --- TIAGo++ Physical Parameters ---
WHEEL_RADIUS = 0.0985          # meters
AXLE_LENGTH = 0.4044           # meters (track width)
MAX_WHEEL_SPEED = 6.28

# --- Navigation Control ---
DRIVE_SPEED = 0.35             # Forward speed (m/s)
TURN_SPEED = 1.0               # Turning speed (rad/s)
ARRIVAL_DIST = 0.3             # Waypoint proximity threshold (m)
HEADING_TOLERANCE = 0.15       # Radians (~8°)

# --- Sensor Device Names ---
ACCEL_NAME = "accelerometer"
GYRO_NAME = "gyro"
IMU_NAME = "inertial unit"
LEFT_WHEEL_MOTOR = "wheel_left_joint"
RIGHT_WHEEL_MOTOR = "wheel_right_joint"
GRAVITY = 9.81

# --- Arm Tuck (avoid wall collisions) ---
ARM_TUCK_POSITIONS = {
    "torso_lift_joint":    0.15,
    "arm_left_1_joint":    0.20,
    "arm_left_2_joint":   -1.34,
    "arm_left_3_joint":   -0.20,
    "arm_left_4_joint":    1.94,
    "arm_left_5_joint":   -1.57,
    "arm_left_6_joint":    1.37,
    "arm_left_7_joint":    0.0,
    "arm_right_1_joint":   0.20,
    "arm_right_2_joint":  -1.34,
    "arm_right_3_joint":   0.20,
    "arm_right_4_joint":   1.94,
    "arm_right_5_joint":  -1.57,
    "arm_right_6_joint":   1.37,
    "arm_right_7_joint":   0.0,
}

# --- WiFi Predictor ---
WIFI_MODELS_PATH = None  # Auto-detected below
TOP_N_APS = 0                          # 0 = collect ALL APs (weak signals carry info too)
RSSI_VISIBILITY_THRESHOLD = -85.0   # dBm

# --- Visualization (InfluxDB + Grafana) ---
ENABLE_VIZ = False              # Set True to write to InfluxDB for Grafana dashboards

# --- InfluxDB Settings (only used if ENABLE_VIZ=True). Token from env. ---
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "navlori")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "wifi_data")

# --- Output ---
WRITE_INTERVAL = 1             # Write every N simulation steps (1 = every step)
CSV_OUTPUT_DIR = None          # Auto-set below
CSV_FILENAME_TEMPLATE = "unified_path{path_id}_{timestamp}.csv"

# ═════════════════════════════════════════════════════════════════════════
# 2. PATH DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════

PATHS = {
    1: {"name": "Top-to-Bottom",
        "waypoints": [(-1.5, 0.1), (-0.7, -3.9), (-5.5, -8.7), (-3.1, -14.3),
                       (-2.3, -14.3), (-3.1, -15.9), (-6.3, -17.5)]},
    2: {"name": "Left Side",
        "waypoints": [(-2.3, -0.7), (-0.7, -3.1), (-3.1, -5.5), (-5.5, -7.1),
                       (-10.3, -7.1), (-7.1, -8.7), (-4.7, -7.1)]},
    3: {"name": "Zigzag",
        "waypoints": [(-1.5, 0.1), (-0.7, -3.1), (-4.7, -7.9), (-5.5, -8.7),
                       (-3.9, -12.7), (-5.5, -9.5), (-7.1, -8.7), (-10.3, -8.7)]},
    4: {"name": "Perimeter",
        "waypoints": [(-1.5, 0.1), (-0.7, -3.1), (-5.5, -8.7), (-3.1, -14.3),
                       (-2.3, -15.1), (-3.9, -15.9), (-2.3, -15.1), (-6.3, -8.7),
                       (-8.7, -5.5), (-6.3, -7.1), (-4.7, -7.1), (-0.7, -3.1),
                       (-2.3, -1.5)]},
    5: {"name": "Center Exploration",
        "waypoints": [(-4.7, -7.9), (-7.1, -6.3), (-8.7, -10.3), (-7.1, -8.7),
                       (-5.5, -8.7), (-2.3, -14.3), (-3.1, -15.9), (-2.3, -15.1),
                       (-5.5, -8.7), (-0.7, -3.1), (-1.5, 0.1)]},
}

# ═════════════════════════════════════════════════════════════════════════
# 3. HELPER CLASSES
# ═════════════════════════════════════════════════════════════════════════

class InfluxWriter:
    """Writes data points to InfluxDB."""

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
            print(f"[InfluxDB] Connected to {url}, bucket={bucket}")
        except ImportError:
            print("[InfluxDB] WARNING: influxdb-client not installed.")
            print("  Run: pip install influxdb-client")
            print("  Continuing with CSV-only output.")
            self.ready = False
            self.Point = None
        except Exception as e:
            print(f"[InfluxDB] Connection error: {e}")
            print("  Continuing with CSV-only output.")
            self.ready = False
            self.Point = None

    def write_points(self, points):
        if not self.ready or not points:
            return
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            print(f"[InfluxDB] Write error: {e}")

    def close(self):
        if self.ready:
            self.client.close()


class CSVLogger:
    """
    Writes unified sensor rows to a CSV file.
    Each row = one timestep with ALL sensor channels aligned.
    """

    def __init__(self, filepath, ap_names=None):
        self.filepath = filepath
        self.ap_names = ap_names or []
        self.file = None
        self.writer = None
        self.row_count = 0

    def open(self):
        """Build header and open file."""
        self.columns = self._build_columns()
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns,
                                      extrasaction="ignore")
        self.writer.writeheader()
        print(f"[CSV] Opened {self.filepath}")
        print(f"[CSV] {len(self.columns)} columns")

    def _build_columns(self):
        cols = [
            # ── Identifiers ──
            "sim_time", "path_id", "path_name", "waypoint_idx",

            # ── Ground Truth (Supervisor) ──
            "gt_x", "gt_y", "gt_z", "gt_heading_rad", "gt_heading_deg",

            # ── IMU: Accelerometer ──
            "accel_x", "accel_y", "accel_z", "accel_magnitude",

            # ── IMU: Gyroscope ──
            "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude",

            # ── IMU: Orientation (Inertial Unit) ──
            "roll_deg", "pitch_deg", "yaw_deg",

            # ── Odometry (Dead-Reckoning) ──
            "odom_x", "odom_y", "odom_theta_deg",
            "odom_linear_vel", "odom_angular_vel",
            "wheel_left_vel", "wheel_right_vel",

            # ── WiFi Summary ──
            "wifi_visible_count", "wifi_strongest_rssi", "wifi_strongest_mac",
        ]

        # ── Per-AP WiFi RSSI (one column per known AP) ──
        for ap_mac in self.ap_names:
            safe_name = ap_mac.replace(":", "")
            cols.append(f"wifi_rssi_{safe_name}")

        return cols

    def write_row(self, data: dict):
        """Write a single unified row."""
        if self.writer is None:
            return
        self.writer.writerow(data)
        self.row_count += 1

    def close(self):
        if self.file:
            self.file.close()
            print(f"[CSV] Closed — {self.row_count} rows written to {self.filepath}")


# ═════════════════════════════════════════════════════════════════════════
# 4. SENSOR & NAVIGATION UTILITIES
# ═════════════════════════════════════════════════════════════════════════

def init_sensor(robot, name, timestep, sensor_type="generic"):
    """Enable a sensor device and return it (or None)."""
    try:
        device = robot.getDevice(name)
        if device is None:
            print(f"  [WARN] Device '{name}' not found")
            return None
        device.enable(timestep)
        print(f"  [OK] {sensor_type}: '{name}' enabled")
        return device
    except Exception as e:
        print(f"  [WARN] Could not init '{name}': {e}")
        return None


def init_wheel_encoder(robot, motor_name, timestep):
    """Enable wheel encoder position sensor."""
    sensor_name = motor_name + "_sensor"
    try:
        sensor = robot.getDevice(sensor_name)
        if sensor is None:
            sensor = robot.getDevice(motor_name.replace("_joint", "") + "_sensor")
        if sensor is None:
            print(f"  [WARN] Wheel encoder '{sensor_name}' not found")
            return None
        sensor.enable(timestep)
        print(f"  [OK] Wheel encoder: '{sensor_name}' enabled")
        return sensor
    except Exception as e:
        print(f"  [WARN] Could not init wheel encoder '{sensor_name}': {e}")
        return None


def get_pose(node):
    """Get (x, y, heading) from Supervisor node."""
    pos = node.getPosition()
    ori = node.getOrientation()
    # Orientation matrix → heading (yaw around Z)
    heading = math.atan2(ori[3], ori[0])
    return pos[0], pos[1], pos[2], heading


def normalize_angle(a):
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def set_wheels(left_motor, right_motor, v_linear, v_angular):
    """Set differential-drive wheel speeds."""
    vl = (v_linear - v_angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vr = (v_linear + v_angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vl = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vl))
    vr = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vr))
    left_motor.setVelocity(vl)
    right_motor.setVelocity(vr)


def stop_wheels(left_motor, right_motor):
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)


def tuck_arms(robot, timestep):
    """Move arms to a tucked position close to the body to avoid wall collisions."""
    print("\n[Arms] Tucking arms...")
    for joint_name, target_pos in ARM_TUCK_POSITIONS.items():
        motor = robot.getDevice(joint_name)
        if motor is None:
            print(f"  [WARN] Joint '{joint_name}' not found, skipping.")
            continue
        if joint_name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1.0)
        motor.setPosition(target_pos)

    # Wait for arms to reach tucked position
    for _ in range(80):
        robot.step(timestep)
    print("[Arms] Tucked.")


# ═════════════════════════════════════════════════════════════════════════
# 5. WiFi PREDICTOR LOADER
# ═════════════════════════════════════════════════════════════════════════

def load_wifi_predictor(controller_dir):
    """
    Try to load the WiFi GPR predictor.
    Returns (predictor, ap_names) or (None, []) if unavailable.
    """
    global WIFI_MODELS_PATH

    # Auto-detect export directory
    search_paths = [
        os.path.join(controller_dir, "webots_export"),
        os.path.join(controller_dir, "..", "wifi_supervisor", "webots_export"),
        os.path.join(controller_dir, "..", "..", "webots_export"),
    ]

    if WIFI_MODELS_PATH and os.path.isdir(WIFI_MODELS_PATH):
        search_paths.insert(0, WIFI_MODELS_PATH)

    export_dir = None
    for p in search_paths:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "rssi_grid.npz")):
            export_dir = p
            break

    if export_dir is None:
        print("[WiFi] WARNING: Could not find webots_export/rssi_grid.npz")
        print("  Searched:", search_paths)
        print("  WiFi data will be EMPTY in the output.")
        return None, []

    # Import predictor — also add the directory containing wifi_predictor.py
    sys.path.insert(0, controller_dir)
    wifi_supervisor_dir = os.path.join(controller_dir, "..", "wifi_supervisor")
    if os.path.isdir(wifi_supervisor_dir):
        sys.path.insert(0, os.path.abspath(wifi_supervisor_dir))
    try:
        from wifi_predictor import WiFiPredictor
        predictor = WiFiPredictor(export_dir)
        return predictor, predictor.ap_names
    except Exception as e:
        print(f"[WiFi] ERROR loading predictor: {e}")
        return None, []


# ═════════════════════════════════════════════════════════════════════════
# 6. SINGLE-PATH COLLECTION
# ═════════════════════════════════════════════════════════════════════════

def run_path(robot, node, timestep, path_id, path_info,
             sensors, motors, predictor, ap_names,
             writer, csv_logger):
    """
    Navigate one path and collect unified data at every timestep.

    Args:
        robot:      Supervisor instance
        node:       Robot's own node (getSelf())
        timestep:   Simulation timestep (ms)
        path_id:    Integer path identifier
        path_info:  Dict with 'name' and 'waypoints'
        sensors:    Dict of sensor devices
        motors:     (left_motor, right_motor) tuple
        predictor:  WiFiPredictor instance (or None)
        ap_names:   List of AP MAC strings
        writer:     InfluxWriter
        csv_logger: CSVLogger
    """
    dt = timestep / 1000.0
    lm, rm = motors
    wps = path_info["waypoints"]
    path_name = path_info["name"]

    # Teleport to starting position
    sx, sy = wps[0]
    h0 = math.atan2(wps[1][1] - sy, wps[1][0] - sx) if len(wps) > 1 else 0.0

    tf = node.getField("translation")
    rf = node.getField("rotation")
    cur = tf.getSFVec3f()
    tf.setSFVec3f([sx, sy, cur[2]])
    rf.setSFRotation([0, 0, 1, h0])
    node.resetPhysics()

    # Let physics settle (20 steps)
    for _ in range(20):
        robot.step(timestep)

    print(f"\n{'='*60}")
    print(f"  PATH {path_id}: {path_name}  ({len(wps)} waypoints)")
    print(f"  Start: ({sx:.1f}, {sy:.1f})  Heading: {math.degrees(h0):.1f}°")
    print(f"{'='*60}")

    # State variables
    prev_left_pos = None
    prev_right_pos = None
    odom_x = odom_y = odom_theta = 0.0
    step_count = 0
    write_count = 0
    wp_idx = 1
    pause_steps = 0
    path_finished = False
    extra_steps = 0

    P = writer.Point if writer is not None else None

    while robot.step(timestep) != -1:
        sim_time = round(robot.getTime(), 4)

        # ════════════════════════════════════════════════════════════
        # A. COLLECT DATA (every WRITE_INTERVAL steps)
        # ════════════════════════════════════════════════════════════
        if step_count % max(1, WRITE_INTERVAL) == 0:
            row = {
                "sim_time": sim_time,
                "path_id": path_id,
                "path_name": path_name,
                "waypoint_idx": wp_idx,
            }
            influx_points = []

            # ── A1. Ground Truth (Supervisor) ──
            gt_x, gt_y, gt_z, gt_heading = get_pose(node)
            gt_heading_deg = round(math.degrees(gt_heading), 3)

            row["gt_x"] = round(gt_x, 5)
            row["gt_y"] = round(gt_y, 5)
            row["gt_z"] = round(gt_z, 5)
            row["gt_heading_rad"] = round(gt_heading, 5)
            row["gt_heading_deg"] = gt_heading_deg

            # ── A2. Accelerometer ──
            accel = sensors["accelerometer"]
            if accel is not None:
                a_vals = accel.getValues()
                ax = round(a_vals[0], 5)
                ay = round(a_vals[1], 5)
                az = round(a_vals[2] - GRAVITY, 5)  # gravity-compensated
                a_mag = round(math.sqrt(ax**2 + ay**2 + az**2), 5)
                row["accel_x"] = ax
                row["accel_y"] = ay
                row["accel_z"] = az
                row["accel_magnitude"] = a_mag

            # ── A3. Gyroscope ──
            gyro = sensors["gyro"]
            if gyro is not None:
                g_vals = gyro.getValues()
                gx = round(g_vals[0], 6)
                gy = round(g_vals[1], 6)
                gz = round(g_vals[2], 6)
                g_mag = round(math.sqrt(gx**2 + gy**2 + gz**2), 6)
                row["gyro_x"] = gx
                row["gyro_y"] = gy
                row["gyro_z"] = gz
                row["gyro_magnitude"] = g_mag

            # ── A4. Inertial Unit (Roll, Pitch, Yaw) ──
            imu = sensors["imu"]
            if imu is not None:
                rpy = imu.getRollPitchYaw()
                row["roll_deg"] = round(math.degrees(rpy[0]), 3)
                row["pitch_deg"] = round(math.degrees(rpy[1]), 3)
                row["yaw_deg"] = round(math.degrees(rpy[2]), 3)

            # ── A5. Odometry (Wheel Encoders) ──
            l_enc = sensors["left_encoder"]
            r_enc = sensors["right_encoder"]
            if l_enc is not None and r_enc is not None:
                l_pos = l_enc.getValue()
                r_pos = r_enc.getValue()

                if prev_left_pos is not None and prev_right_pos is not None:
                    d_left = l_pos - prev_left_pos
                    d_right = r_pos - prev_right_pos
                    l_dist = d_left * WHEEL_RADIUS
                    r_dist = d_right * WHEEL_RADIUS
                    lin_dist = (l_dist + r_dist) / 2.0
                    ang_dist = (r_dist - l_dist) / AXLE_LENGTH

                    odom_theta += ang_dist
                    odom_x += lin_dist * math.cos(odom_theta)
                    odom_y += lin_dist * math.sin(odom_theta)

                    lin_vel = lin_dist / dt
                    ang_vel = ang_dist / dt
                    l_vel = d_left / dt * WHEEL_RADIUS
                    r_vel = d_right / dt * WHEEL_RADIUS

                    row["odom_x"] = round(odom_x, 4)
                    row["odom_y"] = round(odom_y, 4)
                    row["odom_theta_deg"] = round(math.degrees(odom_theta), 2)
                    row["odom_linear_vel"] = round(lin_vel, 5)
                    row["odom_angular_vel"] = round(ang_vel, 5)
                    row["wheel_left_vel"] = round(l_vel, 5)
                    row["wheel_right_vel"] = round(r_vel, 5)

                prev_left_pos = l_pos
                prev_right_pos = r_pos

            # ── A6. WiFi RSSI (ALL APs) ──
            if predictor is not None and predictor.is_in_bounds(gt_x, gt_y):
                scan = predictor.predict(gt_x, gt_y, add_noise=True)

                visible_count = sum(1 for _, rssi in scan.items()
                                     if rssi > RSSI_VISIBILITY_THRESHOLD)
                row["wifi_visible_count"] = visible_count

                # Find strongest AP for summary
                if scan:
                    strongest_mac = max(scan, key=scan.get)
                    row["wifi_strongest_rssi"] = round(scan[strongest_mac], 1)
                    row["wifi_strongest_mac"] = strongest_mac

                # Write ALL APs (one column per known AP)
                for ap_mac in ap_names:
                    safe_name = ap_mac.replace(":", "")
                    row[f"wifi_rssi_{safe_name}"] = round(scan.get(ap_mac, -200.0), 1)
            else:
                # Out of WiFi bounds — fill with sentinel
                row["wifi_visible_count"] = 0
                row["wifi_strongest_rssi"] = -200.0
                row["wifi_strongest_mac"] = ""
                for ap_mac in ap_names:
                    safe_name = ap_mac.replace(":", "")
                    row[f"wifi_rssi_{safe_name}"] = -200.0

            # ── Write to CSV ──
            csv_logger.write_row(row)

            # ── Write to InfluxDB (unified measurement) ──
            if P is not None:
                pt = (P("unified_sample")
                      .tag("robot", "tiago")
                      .tag("path_id", str(path_id))
                      .tag("path_name", path_name))

                # Add all numeric fields
                for key, val in row.items():
                    if key in ("path_name", "wifi_strongest_mac"):
                        continue  # skip string fields (already tags or not needed)
                    if isinstance(val, (int, float)):
                        pt = pt.field(key, val)

                # Also add individual wifi_rssi measurements for Grafana drill-down
                if predictor is not None and predictor.is_in_bounds(gt_x, gt_y):
                    scan = predictor.predict(gt_x, gt_y, add_noise=False)
                    for mac, rssi in scan.items():
                        influx_points.append(
                            P("wifi_rssi")
                            .tag("robot", "tiago")
                            .tag("path_id", str(path_id))
                            .tag("ap_mac", mac)
                            .tag("ap_short", mac[-8:])
                            .field("rssi", round(rssi, 1))
                            .field("robot_x", round(gt_x, 3))
                            .field("robot_y", round(gt_y, 3))
                        )

                influx_points.insert(0, pt)
                writer.write_points(influx_points)

            write_count += 1

            # Console log (every ~2 seconds of sim time)
            if step_count % max(1, int(2.0 / dt)) == 0:
                wifi_str = ""
                if predictor is not None:
                    wifi_str = f"  wifi={row.get('wifi_visible_count', 0)} APs"
                status = "NAV" if not path_finished else "DONE"
                print(
                    f"  [{status}] t={sim_time:7.2f}s  "
                    f"GT=({gt_x:6.2f}, {gt_y:6.2f})  "
                    f"wp=[{wp_idx}/{len(wps)}]  "
                    f"odom=({row.get('odom_x', 0):6.2f}, {row.get('odom_y', 0):6.2f})"
                    f"{wifi_str}  writes={write_count}"
                )

        step_count += 1

        # ════════════════════════════════════════════════════════════
        # B. NAVIGATION CONTROL
        # ════════════════════════════════════════════════════════════
        if path_finished:
            # Collect a few extra stationary samples then exit
            stop_wheels(lm, rm)
            extra_steps += 1
            if extra_steps >= 50:
                break
            continue

        if wp_idx >= len(wps):
            stop_wheels(lm, rm)
            path_finished = True
            print(f"\n  ✓ Path {path_id} complete! Collecting {50} extra samples...")
            continue

        if pause_steps > 0:
            pause_steps -= 1
            continue

        x, y, _, heading = get_pose(node)
        tx, ty = wps[wp_idx]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        if dist < ARRIVAL_DIST:
            stop_wheels(lm, rm)
            print(f"   ✓ WP [{wp_idx}/{len(wps)}] Reached ({tx:.1f}, {ty:.1f})")
            wp_idx += 1
            pause_steps = 5
            continue

        angle_err = normalize_angle(math.atan2(dy, dx) - heading)

        if abs(angle_err) > HEADING_TOLERANCE:
            turn_dir = TURN_SPEED if angle_err > 0 else -TURN_SPEED
            set_wheels(lm, rm, 0.0, turn_dir)
        else:
            set_wheels(lm, rm, DRIVE_SPEED, 1.5 * angle_err)

    stop_wheels(lm, rm)
    print(f"\n  Path {path_id} collection done: {write_count} unified samples.")
    return write_count


# ═════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def main():
    robot = Supervisor()
    node = robot.getSelf()
    if node is None:
        print("ERROR: Set 'supervisor' to TRUE on the robot node in Webots.")
        return

    timestep = int(robot.getBasicTimeStep())
    controller_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  NavLoRI TIAGO++ Unified Data Collector")
    print("  IMU + Odometry + WiFi + Ground Truth → InfluxDB + CSV")
    print("=" * 60)

    # ── Init Sensors ──
    print("\n[Sensors]")
    sensors = {
        "accelerometer": init_sensor(robot, ACCEL_NAME, timestep, "Accelerometer"),
        "gyro":          init_sensor(robot, GYRO_NAME, timestep, "Gyroscope"),
        "imu":           init_sensor(robot, IMU_NAME, timestep, "InertialUnit"),
        "left_encoder":  init_wheel_encoder(robot, LEFT_WHEEL_MOTOR, timestep),
        "right_encoder": init_wheel_encoder(robot, RIGHT_WHEEL_MOTOR, timestep),
    }

    # ── Init Motors ──
    lm = robot.getDevice(LEFT_WHEEL_MOTOR)
    rm = robot.getDevice(RIGHT_WHEEL_MOTOR)
    if not lm or not rm:
        print("ERROR: Motors not found.")
        return
    lm.setPosition(float("inf"))
    rm.setPosition(float("inf"))
    stop_wheels(lm, rm)

    # ── Init WiFi Predictor ──
    print("\n[WiFi Predictor]")
    predictor, ap_names = load_wifi_predictor(controller_dir)

    # ── Init InfluxDB (only if viz enabled) ──
    if ENABLE_VIZ:
        print("\n[InfluxDB]")
        writer = InfluxWriter(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    else:
        print("\n[InfluxDB] Skipped (ENABLE_VIZ=False, CSV-only mode)")
        writer = None

    # ── Determine which paths to run ──
    if RUN_ALL_PATHS:
        path_ids = sorted(PATHS.keys())
    else:
        path_ids = [SELECTED_PATH]

    # ── Output directory ──
    global CSV_OUTPUT_DIR
    if CSV_OUTPUT_DIR is None:
        CSV_OUTPUT_DIR = os.path.join(controller_dir, "unified_data")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    total_samples = 0

    for pid in path_ids:
        if pid not in PATHS:
            print(f"WARNING: Path {pid} not defined, skipping.")
            continue

        # Create CSV logger for this path
        ts = pytime.strftime("%Y%m%d_%H%M%S")
        csv_filename = CSV_FILENAME_TEMPLATE.format(path_id=pid, timestamp=ts)
        csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
        csv_logger = CSVLogger(csv_path, ap_names)
        csv_logger.open()

        # Run the path
        n_samples = run_path(
            robot=robot,
            node=node,
            timestep=timestep,
            path_id=pid,
            path_info=PATHS[pid],
            sensors=sensors,
            motors=(lm, rm),
            predictor=predictor,
            ap_names=ap_names,
            writer=writer,
            csv_logger=csv_logger,
        )
        total_samples += n_samples
        csv_logger.close()

    if writer is not None:
        writer.close()

    print("\n" + "=" * 60)
    print(f"  COLLECTION COMPLETE")
    print(f"  Paths run: {path_ids}")
    print(f"  Total unified samples: {total_samples}")
    print(f"  CSV files saved to: {CSV_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
