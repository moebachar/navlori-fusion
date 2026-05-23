"""
NavLoRI — Asynchronous Multi-Modal Data Collector
===================================================
Event-driven controller for TIAGO++ in Webots.
Each sensor fires at its own jittered rate, producing separate CSV files
per modality + PNG images for camera/depth.

Navigation: proportional steering toward waypoints.
  Smooth speed ramp. No obstacle avoidance (paths are pre-cleared).

Modalities & nominal rates:
  - IMU (accel + gyro + orientation): ~31 Hz (every 1 step)
  - Odometry (wheel encoders):        ~15 Hz (every 2 steps)
  - WiFi RSSI (GPR predictor):        ~1 Hz  (every 31 steps)
  - Camera RGB + Depth:               ~5 Hz   (every 6 steps) — dense for ACE/SCR training
  - Ground Truth:                      ~10 Hz (every 3 steps)

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

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENV_SITE = str(_PROJECT_ROOT / ".venv" / "Lib" / "site-packages")
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[WARN] Pillow not found — depth will fall back to 8-bit PNG (loses metric scale)")

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CONTROLLER_DIR, "..", "wifi_supervisor"))

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

WHEEL_RADIUS = 0.0985
AXLE_LENGTH = 0.4044
MAX_WHEEL_SPEED = 6.28

# Navigation — proportional steering
MAX_SPEED = 0.35       # m/s cruise speed
MIN_SPEED = 0.05       # m/s near-goal creep
ACCEL = 0.10           # m/s² ramp up
DECEL = 0.12           # m/s² ramp down
STEER_GAIN = 1.5       # proportional steering gain (same as old controller)
ARRIVAL_DIST = 0.40    # m — waypoint reached (robot base diameter)
APPROACH_DIST = 0.8    # m — start decelerating near final wp

# Batch
BATCH_START = 0
BATCH_END = 17

# Sensor intervals (in sim steps, 1 step = 32ms)
SENSOR_INTERVALS = {
    "imu":          1,
    "odometry":     2,
    "wifi":         31,
    "camera":       6,
    "ground_truth": 3,
}
JITTER_FRACTION = 0.20

# Camera
CAMERA_NAME = "head_front_camera"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEPTH_CAMERA_NAME = "head_front_camera_depth"

# Arm tuck positions
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

# WiFi
RSSI_VISIBILITY_THRESHOLD = -85.0

# InfluxDB (optional). Token is read from the INFLUXDB_TOKEN env var so it
# never lands in git. See .env.example for the full set of variables.
ENABLE_VIZ = True
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://127.0.0.1:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "navlori")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "async_data")

# Output
OUTPUT_DIR = os.path.normpath(os.path.join(CONTROLLER_DIR, "..", "..", "..", "..", "data", "async_collection"))

# Set to True to write CSVs, camera images, and metadata to disk.
# Set to False for demo/visualization runs — InfluxDB streaming stays active.
COLLECT_MODE = False


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

GRAVITY = 9.81


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


def set_wheels(lm, rm, v, omega):
    vl = (v - omega * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vr = (v + omega * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
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
    print("[Arms] Done.")


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


def resolve_camera_node(robot, self_node, camera, camera_name):
    """Best-effort lookup of the Solid Node that carries `camera`.

    Webots `Supervisor.getFromDevice()` silently returns None when the camera
    lives inside a PROTO (TIAGO++ head_front_camera is one such case). We try
    that first, then PROTO-internal DEF names, then a walk of the robot's
    children field. Returns (Node, source_tag) or (None, "fail").
    """
    try:
        n = robot.getFromDevice(camera)
        if n is not None:
            return n, "getFromDevice"
    except Exception:
        pass

    # World-scope DEF we added to the .wbt cameraSlot override. Pulls the
    # Astra Solid directly — its world pose is the camera mount (optical
    # centre is +4 cm along the Astra's local x, applied at dataset load).
    # Fallback candidates exist in case the .wbt DEF is ever renamed.
    candidates = ["HEAD_ASTRA", "CAMERA", "HEAD", "HEAD_FRONT_CAMERA",
                  camera_name, camera_name.upper()]
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            n = self_node.getFromProtoDef(name)
            if n is not None:
                return n, f"getFromProtoDef({name!r})"
        except Exception:
            pass
        try:
            n = robot.getFromDef(name)
            if n is not None:
                return n, f"getFromDef({name!r})"
        except Exception:
            pass
    return None, "fail"


def compute_intrinsics(camera):
    """Pinhole intrinsics from a Webots Camera/RangeFinder.

    Webots reports horizontal FOV; assume square pixels (fx == fy) and the
    principal point at image centre, which matches Webots' pinhole model.
    """
    w = camera.getWidth()
    h = camera.getHeight()
    fov = camera.getFov()
    fx = 0.5 * w / math.tan(0.5 * fov)
    return {
        "width":     w,
        "height":    h,
        "fx":        fx,
        "fy":        fx,
        "cx":        w / 2.0,
        "cy":        h / 2.0,
        "fov_x_rad": fov,
    }


def save_depth_metric(depth_device, path, scale_mm=1000.0):
    """Save Webots RangeFinder output as a 16-bit PNG in millimetres.

    Webots' `RangeFinder.saveImage()` writes an 8-bit normalised PNG which
    destroys metric scale. `getRangeImage()` returns float32 metres; we store
    as uint16 mm (max 65.535 m) — matches NYU / TUM / ScanNet convention.
    Invalid (inf / NaN) pixels are clipped to 0.
    """
    if not _HAS_PIL:
        depth_device.saveImage(path, 80)
        return False
    w = depth_device.getWidth()
    h = depth_device.getHeight()
    arr = np.asarray(depth_device.getRangeImage(), dtype=np.float32).reshape(h, w)
    arr[~np.isfinite(arr)] = 0.0
    depth_u16 = np.clip(arr * scale_mm, 0.0, 65535.0).astype(np.uint16)
    # Pillow >=12 picks the right mode (I;16) automatically from a uint16
    # array; passing mode= explicitly is deprecated in Pillow 13.
    Image.fromarray(depth_u16).save(path)
    return True


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
# SENSOR SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

class SensorScheduler:
    def __init__(self, intervals, jitter_frac, seed=42):
        self.intervals = dict(intervals)
        self.jitter_frac = jitter_frac
        self.rng = random.Random(seed)
        self.next_fire = {}
        self.reset()

    def reset(self):
        for name in self.intervals:
            self.next_fire[name] = 0

    def should_fire(self, sensor_name, step):
        return step >= self.next_fire[sensor_name]

    def advance(self, sensor_name, step):
        base = self.intervals[sensor_name]
        jitter = self.rng.uniform(-self.jitter_frac, self.jitter_frac)
        interval = max(1, round(base * (1.0 + jitter)))
        self.next_fire[sensor_name] = step + interval


# ═══════════════════════════════════════════════════════════════════════
# CSV WRITERS
# ═══════════════════════════════════════════════════════════════════════

class ModalityCSV:
    def __init__(self, filepath, columns):
        self.filepath = filepath
        self.columns = ["sim_time"] + columns
        self.file = None
        self.writer = None
        self.count = 0

    def open(self):
        if not COLLECT_MODE:
            return
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns,
                                     extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row):
        if not COLLECT_MODE:
            return
        if self.writer:
            self.writer.writerow(row)
            self.count += 1

    def close(self):
        if not COLLECT_MODE:
            return
        if self.file:
            self.file.close()
            print(f"  {os.path.basename(self.filepath)}: {self.count} events")


# ═══════════════════════════════════════════════════════════════════════
# INFLUXDB (optional)
# ═══════════════════════════════════════════════════════════════════════

class InfluxWriter:
    def __init__(self, url, token, org, bucket):
        self._write_count = 0
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            self.Point = Point
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.bucket = bucket
            self.org = org
            self.ready = True
            test_pt = Point("_connection_test").field("ok", 1)
            self.write_api.write(bucket=bucket, org=org, record=test_pt)
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
            self._write_count += 1
            if self._write_count <= 3:
                print(f"[InfluxDB] Write #{self._write_count}: {measurement}")
        except Exception as e:
            print(f"[InfluxDB] Write error: {e}")

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


# ═══════════════════════════════════════════════════════════════════════
# TRAJECTORY PLOT
# ═══════════════════════════════════════════════════════════════════════

_RAW_WALLS = [
    (1.48, -1.47, 2.2327, 0.2, 2.7),
    (1.27, -3.35, -2.5454, 0.2, 2.707),
    (-0.61, -6.52, 2.2301, 0.2, 6.9709),
    (-10.33, -5.40, 2.15, 0.2, 7.5),
    (-2.16, -10.77, -2.6475, 0.2, 5.0),
    (1.10, -11.53, -1.0, 0.2, 5.1671),
    (3.99, -11.34, -2.6168, 0.2, 3.0),
    (-1.06, -16.10, 2.1194, 0.2, 13.8),
    (-8.01, -18.25, 0.6777, 0.2, 1.9),
    (-6.55, -16.13, -1.0151, 0.2, 5.1559),
    (-5.62, -12.74, 0.5114, 0.2, 4.6),
    (-8.43, -11.76, 2.1214, 0.2, 3.8606),
    (-12.77, -8.61, 0.5831, 0.2, 2.9),
    (-10.79, -11.60, 0.5831, 0.2, 2.7),
    (-6.30, -4.47, -2.482, 0.2, 3.1004),
    (-4.05, -4.58, -0.8897, 0.2, 3.6),
    (-3.72, -1.98, 0.5912, 0.2, 3.4447),
    (-2.75, 1.41, -0.8125, 0.2, 6.0),
    (0.84, 1.52, -2.5178, 0.2, 5.3),
]


def _wall_polygon(tx, ty, angle, sx, sy, inflate=0.0):
    hx = sx / 2 + inflate
    hy = sy / 2 + inflate
    ca, sa = math.cos(angle), math.sin(angle)
    return [(tx + ca * cx - sa * cy, ty + sa * cx + ca * cy)
            for cx, cy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def draw_path_on_floor(robot, waypoints, path_id):
    """Draw waypoint markers and connecting lines on the Webots floor."""
    root = robot.getRoot()
    children = root.getField("children")
    floor_z = 0.005  # just above the floor

    # Colours: start=green, end=red, middle=blue, lines=yellow
    mat_wp = 'Material { diffuseColor 0.2 0.4 1.0 }'
    mat_start = 'Material { diffuseColor 0.0 0.8 0.2 }'
    mat_end = 'Material { diffuseColor 0.9 0.1 0.1 }'
    mat_line = 'Material { diffuseColor 1.0 0.8 0.0 }'

    # Waypoint spheres
    for i, (wx, wy) in enumerate(waypoints):
        if i == 0:
            mat = mat_start
            r = 0.08
        elif i == len(waypoints) - 1:
            mat = mat_end
            r = 0.08
        else:
            mat = mat_wp
            r = 0.05
        sphere = (
            f'DEF PATH{path_id}_WP{i} Pose {{'
            f'  translation {wx} {wy} {floor_z}'
            f'  children ['
            f'    Shape {{'
            f'      appearance Appearance {{ material {mat} }}'
            f'      geometry Sphere {{ radius {r} }}'
            f'    }}'
            f'  ]'
            f'}}'
        )
        children.importMFNodeFromString(-1, sphere)

    # Lines between waypoints (thin boxes)
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(y2 - y1, x2 - x1)
        line = (
            f'DEF PATH{path_id}_L{i} Pose {{'
            f'  translation {mx} {my} {floor_z}'
            f'  rotation 0 0 1 {angle}'
            f'  children ['
            f'    Shape {{'
            f'      appearance Appearance {{ material {mat_line} }}'
            f'      geometry Box {{ size {length} 0.03 0.005 }}'
            f'    }}'
            f'  ]'
            f'}}'
        )
        children.importMFNodeFromString(-1, line)

    print(f"  [Floor] Drew {len(waypoints)} markers + {len(waypoints)-1} lines")


def remove_path_from_floor(robot, waypoints, path_id):
    """Remove previously drawn path markers from the scene."""
    for i in range(len(waypoints)):
        n = robot.getFromDef(f"PATH{path_id}_WP{i}")
        if n:
            n.remove()
    for i in range(len(waypoints) - 1):
        n = robot.getFromDef(f"PATH{path_id}_L{i}")
        if n:
            n.remove()


def plot_trajectory(path_dir, path_id, waypoints, gt_csv_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        import seaborn as sns
        import pandas as pd

        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

        gt = pd.read_csv(gt_csv_path)
        if "gt_x" not in gt.columns or "gt_y" not in gt.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 15))

        for tx, ty, angle, sx, sy in _RAW_WALLS:
            poly = _wall_polygon(tx, ty, angle, sx, sy)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor="#888888",
                                     edgecolor="#333333", linewidth=0.6, alpha=0.6))

        clearance = 0.855
        for tx, ty, angle, sx, sy in _RAW_WALLS:
            poly = _wall_polygon(tx, ty, angle, sx, sy, inflate=clearance)
            ax.add_patch(MplPolygon(poly, closed=True, facecolor="#ff0000",
                                     edgecolor="none", alpha=0.06))

        wp_x = [w[0] for w in waypoints]
        wp_y = [w[1] for w in waypoints]
        ax.plot(wp_x, wp_y, "--", color="#3388ff", linewidth=1.2, alpha=0.5, label="Planned")
        ax.scatter(wp_x, wp_y, s=30, color="#3388ff", zorder=4, alpha=0.6)
        ax.scatter(wp_x[0], wp_y[0], s=100, color="#00cc44", marker="s",
                   edgecolors="black", linewidths=0.8, zorder=5, label="Start")
        ax.scatter(wp_x[-1], wp_y[-1], s=100, color="#e6194b", marker="D",
                   edgecolors="black", linewidths=0.8, zorder=5, label="End")

        scatter = ax.scatter(gt["gt_x"], gt["gt_y"], c=gt["sim_time"],
                             cmap="viridis", s=4, zorder=6, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Sim time (s)", fontsize=10)

        ax.set_xlim(-14.5, 5.5)
        ax.set_ylim(-19.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_title(f"Path {path_id} — Ground Truth Trajectory", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)

        out_path = os.path.join(path_dir, "trajectory.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [Plot] Saved → {out_path}")
    except Exception as e:
        print(f"  [Plot] Error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PATH RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_path(robot, node, timestep, path_id, path_info,
             sensors, motors, predictor, ap_names,
             influx, output_dir, scheduler):

    dt = timestep / 1000.0
    lm, rm = motors
    wps = path_info["waypoints"]
    path_name = path_info["name"]
    path_dir = os.path.join(output_dir, f"path_{path_id:02d}")
    cam_dir = os.path.join(path_dir, "camera")
    if COLLECT_MODE:
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

    # ── Draw path on floor ──
    draw_path_on_floor(robot, wps, path_id)

    print(f"\n{'='*60}")
    print(f"  PATH {path_id}: {path_name}  ({len(wps)} waypoints)")
    print(f"{'='*60}")

    # ── CSV files ──
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
    # Camera pose: world-frame translation + row-major 3x3 rotation of the
    # camera Solid. Webots camera local frame: +x = optical axis (forward),
    # +y = left, +z = up. Downstream code converting to OpenCV convention
    # (+z forward, +x right, +y down) must apply the fixed axis swap.
    cam_cols = [
        "frame_id", "rgb_path", "depth_path",
        "cam_x", "cam_y", "cam_z",
        "cam_r00", "cam_r01", "cam_r02",
        "cam_r10", "cam_r11", "cam_r12",
        "cam_r20", "cam_r21", "cam_r22",
    ]

    csvs = {
        "imu":          ModalityCSV(os.path.join(path_dir, "imu.csv"), imu_cols),
        "odometry":     ModalityCSV(os.path.join(path_dir, "odometry.csv"), odom_cols),
        "wifi":         ModalityCSV(os.path.join(path_dir, "wifi.csv"), wifi_cols),
        "ground_truth": ModalityCSV(os.path.join(path_dir, "ground_truth.csv"), gt_cols),
        "camera":       ModalityCSV(os.path.join(path_dir, "camera.csv"), cam_cols),
    }
    for c in csvs.values():
        c.open()

    # ── Navigation state ──
    scheduler.reset()
    prev_left_pos = None
    prev_right_pos = None
    odom_x = odom_y = odom_theta = 0.0
    step_count = 0
    wp_idx = 1           # heading to waypoint 1 (teleported to 0)
    speed = 0.0          # current linear speed (smooth ramping)
    path_done = False
    extra_steps = 0
    frame_count = 0

    camera = sensors.get("camera")
    depth = sensors.get("depth")

    # Resolve the supervisor Node for the RGB camera so we can read its
    # world-frame pose. TIAGO++'s head_front_camera is inside a PROTO, so
    # `getFromDevice` often returns None — `resolve_camera_node` falls back
    # to `getFromProtoDef`. If every lookup fails, we log robot-base pose in
    # the cam_* columns and flag it in camera_info.json for post-processing.
    cam_node = None
    cam_source = "base_fallback"
    if camera is not None:
        cam_node, cam_source = resolve_camera_node(robot, node, camera, CAMERA_NAME)
        if cam_node is None:
            print("  [WARN] Could not resolve camera node via any strategy."
                  " cam_* columns will hold ROBOT BASE pose as a fallback.")
        else:
            print(f"  [OK] Camera node resolved via {cam_source}")

    # Persist intrinsics + depth format once per path.
    if COLLECT_MODE and camera is not None:
        cam_info = compute_intrinsics(camera)
        cam_info["depth_format"] = "uint16_png_mm" if _HAS_PIL else "uint8_png_normalised"
        cam_info["depth_scale_m_per_unit"] = 1.0 / 1000.0 if _HAS_PIL else None
        cam_info["axis_convention"] = "webots (+x forward, +y left, +z up)"
        cam_info["pose_source"] = cam_source
        cam_info["pose_is_camera_frame"] = cam_node is not None
        if depth is not None:
            cam_info["depth"] = compute_intrinsics(depth)
            cam_info["depth"]["max_range_m"] = depth.getMaxRange()
        with open(os.path.join(path_dir, "camera_info.json"), "w") as f:
            json.dump(cam_info, f, indent=2)

    while robot.step(timestep) != -1:
        sim_time = round(robot.getTime(), 4)

        # ══════════════════════════════════════════════════════════════
        # SENSOR COLLECTION (unchanged — fires at jittered intervals)
        # ══════════════════════════════════════════════════════════════

        # ── IMU ──
        if scheduler.should_fire("imu", step_count):
            row = {"sim_time": sim_time}
            acc = sensors.get("accelerometer")
            gyr = sensors.get("gyro")
            imu = sensors.get("imu")
            if acc:
                a = acc.getValues()
                ax, ay, az = round(a[0], 5), round(a[1], 5), round(a[2] - GRAVITY, 5)
                row.update(accel_x=ax, accel_y=ay, accel_z=az,
                           accel_magnitude=round(math.sqrt(ax**2 + ay**2 + az**2), 5))
            if gyr:
                g = gyr.getValues()
                gx, gy, gz = round(g[0], 6), round(g[1], 6), round(g[2], 6)
                row.update(gyro_x=gx, gyro_y=gy, gyro_z=gz,
                           gyro_magnitude=round(math.sqrt(gx**2 + gy**2 + gz**2), 6))
            if imu:
                rpy = imu.getRollPitchYaw()
                row.update(roll_deg=round(math.degrees(rpy[0]), 3),
                           pitch_deg=round(math.degrees(rpy[1]), 3),
                           yaw_deg=round(math.degrees(rpy[2]), 3))
            csvs["imu"].write(row)
            if influx and influx.ready:
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
                    dl = (l_pos - prev_left_pos) * WHEEL_RADIUS
                    dr = (r_pos - prev_right_pos) * WHEEL_RADIUS
                    lin = (dl + dr) / 2.0
                    ang = (dr - dl) / AXLE_LENGTH
                    odom_theta += ang
                    odom_x += lin * math.cos(odom_theta)
                    odom_y += lin * math.sin(odom_theta)
                    row = {
                        "sim_time": sim_time,
                        "odom_x": round(odom_x, 4), "odom_y": round(odom_y, 4),
                        "odom_theta_deg": round(math.degrees(odom_theta), 2),
                        "odom_linear_vel": round(lin / dt, 5),
                        "odom_angular_vel": round(ang / dt, 5),
                        "wheel_left_vel": round(dl / dt, 5),
                        "wheel_right_vel": round(dr / dt, 5),
                    }
                    csvs["odometry"].write(row)
                    if influx and influx.ready:
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
                if influx and influx.ready:
                    influx.write_sensor_event("wifi",
                                              {"robot": "tiago", "path_id": path_id},
                                              {k: v for k, v in row.items() if isinstance(v, (int, float))})
                    P = influx.Point
                    ap_points = []
                    for ap_mac, rssi in scan.items():
                        ap_points.append(
                            P("wifi_rssi")
                            .tag("robot", "tiago").tag("path_id", str(path_id))
                            .tag("ap_mac", ap_mac).tag("ap_short", ap_mac[-8:])
                            .field("rssi", round(rssi, 1))
                            .field("robot_x", round(gt_x, 3))
                            .field("robot_y", round(gt_y, 3))
                        )
                    influx.write_points(ap_points)
            scheduler.advance("wifi", step_count)

        # ── Camera ──
        if scheduler.should_fire("camera", step_count):
            saved_rgb = saved_depth = False
            frame_id = f"{step_count:06d}"
            if camera is not None:
                if COLLECT_MODE:
                    camera.saveImage(os.path.join(cam_dir, f"rgb_{frame_id}.png"), 100)
                saved_rgb = True
            if depth is not None:
                if COLLECT_MODE:
                    save_depth_metric(depth, os.path.join(cam_dir, f"depth_{frame_id}.png"))
                saved_depth = True
            if saved_rgb or saved_depth:
                row = {
                    "sim_time": sim_time, "frame_id": frame_id,
                    "rgb_path": f"camera/rgb_{frame_id}.png" if saved_rgb else "",
                    "depth_path": f"camera/depth_{frame_id}.png" if saved_depth else "",
                }
                pose_node = cam_node if cam_node is not None else node
                p = pose_node.getPosition()
                R = pose_node.getOrientation()
                row.update(
                    cam_x=round(p[0], 5), cam_y=round(p[1], 5), cam_z=round(p[2], 5),
                    cam_r00=round(R[0], 6), cam_r01=round(R[1], 6), cam_r02=round(R[2], 6),
                    cam_r10=round(R[3], 6), cam_r11=round(R[4], 6), cam_r12=round(R[5], 6),
                    cam_r20=round(R[6], 6), cam_r21=round(R[7], 6), cam_r22=round(R[8], 6),
                )
                csvs["camera"].write(row)
                frame_count += 1
                if influx and influx.ready:
                    influx.write_sensor_event("camera",
                                              {"robot": "tiago", "path_id": path_id},
                                              {"sim_time": sim_time, "frame_id": int(frame_id)})
            scheduler.advance("camera", step_count)

        # ── Ground Truth ──
        if scheduler.should_fire("ground_truth", step_count):
            gx, gy, gz, gh = get_pose(node)
            row = {
                "sim_time": sim_time,
                "gt_x": round(gx, 5), "gt_y": round(gy, 5), "gt_z": round(gz, 5),
                "gt_heading_rad": round(gh, 5),
                "gt_heading_deg": round(math.degrees(gh), 3),
                "path_id": path_id, "waypoint_idx": wp_idx,
            }
            csvs["ground_truth"].write(row)
            if influx and influx.ready:
                influx.write_sensor_event("ground_truth",
                                          {"robot": "tiago", "path_id": path_id},
                                          {k: v for k, v in row.items() if isinstance(v, (int, float))})
            scheduler.advance("ground_truth", step_count)

        step_count += 1

        # ══════════════════════════════════════════════════════════════
        # NAVIGATION — proportional steering (like old tiago_unified)
        # ══════════════════════════════════════════════════════════════

        # Path finished — collect 50 extra sensor samples then exit
        if path_done:
            stop_wheels(lm, rm)
            extra_steps += 1
            if extra_steps >= 50:
                break
            continue

        # All waypoints done — decelerate to stop
        if wp_idx >= len(wps):
            speed = max(0.0, speed - DECEL * dt)
            set_wheels(lm, rm, speed, 0.0)
            if speed <= 0.0:
                stop_wheels(lm, rm)
                path_done = True
                print(f"  Path {path_id} complete, collecting extra samples...")
            continue

        x, y, _, heading = get_pose(node)
        tx, ty = wps[wp_idx]
        dist = math.hypot(tx - x, ty - y)
        angle_to_wp = math.atan2(ty - y, tx - x)
        angle_err = normalize_angle(angle_to_wp - heading)

        # ── Waypoint reached → advance ──
        if dist < ARRIVAL_DIST:
            print(f"   WP [{wp_idx}/{len(wps)-1}] ({tx:.1f}, {ty:.1f})")
            wp_idx += 1
            continue

        # ── Target speed ──
        # Slow down near final waypoint
        dist_to_goal = math.hypot(wps[-1][0] - x, wps[-1][1] - y)
        if wp_idx == len(wps) - 1 and dist_to_goal < APPROACH_DIST:
            target_speed = max(MIN_SPEED, MAX_SPEED * (dist_to_goal / APPROACH_DIST))
        else:
            target_speed = MAX_SPEED

        # Reduce speed when heading is off (smooth, not a hard cutoff)
        abs_err = abs(angle_err)
        if abs_err > 0.3:
            target_speed *= max(0.15, 1.0 - abs_err / math.pi)

        # Smooth ramp
        if speed < target_speed:
            speed = min(target_speed, speed + ACCEL * dt)
        else:
            speed = max(target_speed, speed - DECEL * dt)

        # ── Steering: proportional ──
        omega = STEER_GAIN * angle_err

        set_wheels(lm, rm, speed, omega)

        # Log every ~3s
        if step_count % max(1, int(3.0 / dt)) == 0:
            print(f"  [NAV] t={sim_time:.1f}s wp={wp_idx}/{len(wps)-1} "
                  f"d={dist:.2f}m v={speed:.2f} err={math.degrees(angle_err):.1f}°")

    stop_wheels(lm, rm)

    # ── Close CSVs ──
    print(f"\n  Path {path_id} results:")
    for c in csvs.values():
        c.close()

    # ── Metadata ──
    if COLLECT_MODE:
        meta = {
            "path_id": path_id, "path_name": path_name,
            "waypoints": wps, "timestep_ms": timestep,
            "sensor_intervals": dict(SENSOR_INTERVALS),
            "jitter_fraction": JITTER_FRACTION,
            "samples": {name: c.count for name, c in csvs.items()},
            "frames": frame_count,
        }
        with open(os.path.join(path_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Clean up floor markers ──
    remove_path_from_floor(robot, wps, path_id)

    # ── Trajectory plot ──
    if COLLECT_MODE:
        gt_csv = os.path.join(path_dir, "ground_truth.csv")
        if os.path.exists(gt_csv):
            plot_trajectory(path_dir, path_id, wps, gt_csv)

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
    print("  Proportional steering waypoint follower")
    print("=" * 60)

    START_PATH = BATCH_START
    END_PATH = BATCH_END

    # ── Sensors ──
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
        for alt in ["Astra rgb", "camera", "head_2_camera"]:
            camera = robot.getDevice(alt)
            if camera is not None:
                break
    if camera is not None:
        camera.enable(timestep)
        print(f"  [OK] RGB camera ({camera.getWidth()}x{camera.getHeight()})")
        sensors["camera"] = camera
    else:
        print("  [WARN] No RGB camera")
        sensors["camera"] = None

    depth = robot.getDevice(DEPTH_CAMERA_NAME)
    if depth is None:
        for alt in ["Astra depth", "range-finder", "head_2_depth"]:
            depth = robot.getDevice(alt)
            if depth is not None:
                break
    if depth is not None:
        depth.enable(timestep)
        print(f"  [OK] Depth sensor ({depth.getWidth()}x{depth.getHeight()})")
        sensors["depth"] = depth
    else:
        print("  [WARN] No depth sensor")
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

    # ── Load Paths ──
    from paths import ALL_PATHS
    all_ids = sorted(ALL_PATHS.keys())
    if START_PATH is not None and END_PATH is not None:
        path_ids = [pid for pid in all_ids if START_PATH <= pid <= END_PATH]
    elif START_PATH is not None:
        path_ids = [pid for pid in all_ids if pid >= START_PATH]
    else:
        path_ids = all_ids
    print(f"\n[Batch] {len(path_ids)} paths: {path_ids[0]}..{path_ids[-1]}")

    if COLLECT_MODE:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    if influx and influx.ready:
        influx.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    for pid, stats in total_stats.items():
        print(f"    Path {pid:2d}: imu={stats.get('imu',0):5d}  odom={stats.get('odometry',0):5d}  "
              f"wifi={stats.get('wifi',0):4d}  cam={stats.get('camera',0):4d}  gt={stats.get('ground_truth',0):4d}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    if COLLECT_MODE:
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
