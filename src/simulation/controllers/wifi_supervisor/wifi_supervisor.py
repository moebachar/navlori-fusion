"""
NavLoRI WiFi Simulation — InfluxDB Supervisor
================================================
Reads TIAGO++ position via Supervisor API, predicts WiFi signals
using GPR models, and writes all WiFi data to InfluxDB.

Measurements written:
  - robot_state:     x, y, z, heading, sim_time
  - wifi_rssi:       per-AP RSSI (tagged by MAC)
  - wifi_summary:    visible_aps count, strongest RSSI

Pipeline: Webots Supervisor → InfluxDB → Grafana

Install:
    pip install influxdb-client

Author: Mohamed — NavLoRI Project, CESI LINEACT
"""

import sys
import os
import math
import numpy as np

from controller import Supervisor

controller_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, controller_dir)

from wifi_predictor import WiFiPredictor

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

WIFI_MODELS_PATH = os.path.join(controller_dir, "webots_export")
TIAGO_DEF_NAME = "TIAGO"
TOP_N_APS = 15
RSSI_VISIBILITY_THRESHOLD = -85.0   # dBm — APs weaker than this are "not visible"
WRITE_INTERVAL = 3                  # write every N simulation steps

# InfluxDB settings — update INFLUXDB_TOKEN after setup
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "hTxBvNi8eVj1f4gkclf01cr7uSnVGOfXKPkKmnyRSMC5H2QsiFqQZe4leOyh2GcLoUOmtrLXfkiFMsBKR2EpSg=="
INFLUXDB_ORG = "navlori"
INFLUXDB_BUCKET = "wifi_data"

# ═════════════════════════════════════════════════════════════════════════
# InfluxDB Writer
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
            print("[InfluxDB] ERROR: influxdb-client not installed!")
            print("  Run: pip install influxdb-client")
            self.ready = False
        except Exception as e:
            print(f"[InfluxDB] Connection error: {e}")
            self.ready = False

    def write_points(self, points):
        """Write a list of Point objects."""
        if not self.ready or not points:
            return
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
        except Exception as e:
            print(f"[InfluxDB] Write error: {e}")

    def close(self):
        if self.ready:
            self.client.close()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    print("=" * 60)
    print("NavLoRI WiFi Supervisor — InfluxDB + Grafana")
    print("=" * 60)

    # Get TIAGO++ via Supervisor API
    tiago_node = robot.getFromDef(TIAGO_DEF_NAME)
    if tiago_node is None:
        print(f"ERROR: Cannot find DEF '{TIAGO_DEF_NAME}' in scene.")
        return

    tiago_translation = tiago_node.getField("translation")
    tiago_rotation = tiago_node.getField("rotation")
    print(f"[Main] Found TIAGO++ (DEF={TIAGO_DEF_NAME})")

    # WiFi predictor
    if not os.path.isdir(WIFI_MODELS_PATH):
        print(f"ERROR: WiFi models not found at {WIFI_MODELS_PATH}")
        return
    predictor = WiFiPredictor(WIFI_MODELS_PATH)

    # InfluxDB
    writer = InfluxWriter(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    P = writer.Point  # shortcut

    step_count = 0
    write_count = 0
    print(f"\n  InfluxDB: {INFLUXDB_URL}  |  Bucket: {INFLUXDB_BUCKET}")
    print(f"  Grafana:  http://localhost:3000")
    print("-" * 60)

    while robot.step(timestep) != -1:
        if step_count % WRITE_INTERVAL != 0:
            step_count += 1
            continue

        # ── Read robot state ──
        pos = tiago_translation.getSFVec3f()
        rot = tiago_rotation.getSFRotation()
        sim_time = round(robot.getTime(), 3)

        robot_x = pos[0]
        robot_y = pos[1]
        robot_z = pos[2]

        # Extract heading from rotation axis-angle
        if abs(rot[1]) > 0.5:
            heading_rad = rot[3]
        elif abs(rot[2]) > 0.5:
            heading_rad = -rot[3]
        else:
            heading_rad = 0.0
        heading_deg = round(math.degrees(heading_rad), 1)

        # ── Robot state measurement ──
        points = []
        points.append(
            P("robot_state")
            .tag("robot", "tiago")
            .tag("source", "supervisor")
            .field("x", round(robot_x, 4))
            .field("y", round(robot_y, 4))
            .field("z", round(robot_z, 4))
            .field("heading_deg", heading_deg)
            .field("heading_rad", round(heading_rad, 4))
            .field("sim_time", sim_time)
        )

        # ── WiFi scan ──
        if predictor.is_in_bounds(robot_x, robot_y):
            scan = predictor.predict(robot_x, robot_y, add_noise=True)
            sorted_aps = sorted(scan.items(), key=lambda kv: kv[1], reverse=True)
            top_aps = sorted_aps[:TOP_N_APS]

            # Count visible APs (above threshold)
            visible_count = sum(1 for _, rssi in scan.items() if rssi > RSSI_VISIBILITY_THRESHOLD)

            # Per-AP RSSI
            for mac, rssi in top_aps:
                points.append(
                    P("wifi_rssi")
                    .tag("robot", "tiago")
                    .tag("ap_mac", mac)
                    .tag("ap_short", mac[-8:])
                    .field("rssi", round(rssi, 1))
                    .field("robot_x", round(robot_x, 3))
                    .field("robot_y", round(robot_y, 3))
                )

            # Summary: strongest + visible count
            if top_aps:
                strongest_mac, strongest_rssi = top_aps[0]
                points.append(
                    P("wifi_summary")
                    .tag("robot", "tiago")
                    .tag("strongest_mac", strongest_mac)
                    .tag("strongest_short", strongest_mac[-8:])
                    .field("strongest_rssi", round(strongest_rssi, 1))
                    .field("visible_aps", visible_count)
                    .field("total_aps", len(scan))
                    .field("robot_x", round(robot_x, 3))
                    .field("robot_y", round(robot_y, 3))
                )

            writer.write_points(points)
            write_count += 1

            # Console log (every ~30 steps)
            if step_count % (WRITE_INTERVAL * 10) == 0:
                s_mac, s_rssi = top_aps[0] if top_aps else ("N/A", -100)
                print(
                    f"  t={sim_time:7.1f}s  pos=({robot_x:6.2f}, {robot_y:6.2f})  "
                    f"heading={heading_deg:6.1f}°  "
                    f"strongest={s_mac[-8:]} {s_rssi:.0f}dBm  "
                    f"visible={visible_count}  writes={write_count}"
                )
        else:
            # Still write robot state even if out of WiFi bounds
            writer.write_points(points)
            write_count += 1

        step_count += 1

    writer.close()
    print("[WiFi Supervisor] Controller terminated.")


if __name__ == "__main__":
    main()