"""Log 6-DoF poses of Vive controllers/trackers to CSV for ground-truth acquisition.

Runs against a live SteamVR instance (see README for setup). Stop with Ctrl+C
(or close the viz window); a session summary is printed on exit.

Usage:
    python vive_logger.py --out sessions --rate 100
    python vive_logger.py --out sessions --rate 100 --viz          # live 3D view
    python vive_logger.py --viz --trail 15                          # longer fading trail
"""

import argparse
import collections
import csv
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import openvr

DEVICE_CLASS_NAMES = {
    openvr.TrackedDeviceClass_HMD: "hmd",
    openvr.TrackedDeviceClass_Controller: "controller",
    openvr.TrackedDeviceClass_GenericTracker: "tracker",
}

CSV_FIELDS = [
    "t_unix", "t_mono", "device", "serial", "valid", "tracking_result",
    "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "trigger",
]


def matrix34_to_pose(m):
    """3x4 row-major pose matrix -> (x, y, z, qw, qx, qy, qz)."""
    x, y, z = m[0][3], m[1][3], m[2][3]
    qw = math.sqrt(max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) / 2.0
    qx = math.sqrt(max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) / 2.0
    qy = math.sqrt(max(0.0, 1.0 - m[0][0] + m[1][1] - m[2][2])) / 2.0
    qz = math.sqrt(max(0.0, 1.0 - m[0][0] - m[1][1] + m[2][2])) / 2.0
    qx = math.copysign(qx, m[2][1] - m[1][2])
    qy = math.copysign(qy, m[0][2] - m[2][0])
    qz = math.copysign(qz, m[1][0] - m[0][1])
    return x, y, z, qw, qx, qy, qz


def discover_devices(vr, include_hmd=False):
    """Return {index: (class_name, serial)} for devices worth logging."""
    devices = {}
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        cls = vr.getTrackedDeviceClass(i)
        if cls not in DEVICE_CLASS_NAMES:
            continue
        if cls == openvr.TrackedDeviceClass_HMD and not include_hmd:
            continue
        serial = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
        devices[i] = (DEVICE_CLASS_NAMES[cls], serial)
    return devices


def trigger_pressed(vr, index):
    got_state, state = vr.getControllerState(index)
    if not got_state:
        return 0
    mask = 1 << openvr.k_EButton_SteamVR_Trigger
    return 1 if (state.ulButtonPressed & mask) else 0


class SharedState:
    """Trail + status handed from the logging thread to the viz."""

    def __init__(self, trail_seconds):
        self.trail_seconds = trail_seconds
        self.trail = collections.deque()          # (t_mono, x, y, z) of primary device
        self.markers = collections.deque(maxlen=200)  # (x, y, z) at trigger presses
        self.status = "waiting for poses..."
        self.lock = threading.Lock()

    def push(self, t_mono, x, y, z, trigger_edge):
        with self.lock:
            self.trail.append((t_mono, x, y, z))
            while self.trail and t_mono - self.trail[0][0] > self.trail_seconds:
                self.trail.popleft()
            if trigger_edge:
                self.markers.append((x, y, z))

    def snapshot(self):
        with self.lock:
            return list(self.trail), list(self.markers), self.status


def logging_loop(vr, devices, primary_idx, writer, file_handle, rate, shared, stop_event, counters):
    period = 1.0 / rate
    last_trigger = {}
    last_status = time.monotonic()
    t_start = time.monotonic()
    next_tick = time.monotonic()

    while not stop_event.is_set():
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        t_unix = time.time()
        t_mono = time.monotonic()
        status_pos = None
        for idx, (cls, serial) in devices.items():
            pose = poses[idx]
            valid = int(pose.bPoseIsValid)
            x, y, z, qw, qx, qy, qz = matrix34_to_pose(pose.mDeviceToAbsoluteTracking)
            vel = pose.vVelocity
            trig = trigger_pressed(vr, idx) if cls == "controller" else 0
            trigger_edge = trig and not last_trigger.get(idx, 0)
            if trigger_edge:
                print(f"\n[marker] trigger on {serial} at t_unix={t_unix:.3f}", flush=True)
            last_trigger[idx] = trig
            writer.writerow({
                "t_unix": f"{t_unix:.6f}", "t_mono": f"{t_mono:.6f}",
                "device": cls, "serial": serial, "valid": valid,
                "tracking_result": int(pose.eTrackingResult),
                "x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}",
                "qw": f"{qw:.6f}", "qx": f"{qx:.6f}", "qy": f"{qy:.6f}", "qz": f"{qz:.6f}",
                "vx": f"{vel[0]:.4f}", "vy": f"{vel[1]:.4f}", "vz": f"{vel[2]:.4f}",
                "trigger": trig,
            })
            counters["samples"] += 1
            counters["valid"] += valid
            if valid and status_pos is None:
                status_pos = (x, y, z)
            if valid and idx == primary_idx:
                shared.push(t_mono, x, y, z, trigger_edge)

        if t_mono - last_status >= 1.0:
            last_status = t_mono
            file_handle.flush()
            pct = 100.0 * counters["valid"] / max(1, counters["samples"])
            pos_txt = ("x=%.3f y=%.3f z=%.3f" % status_pos) if status_pos else "NO VALID POSE"
            shared.status = f"{t_mono - t_start:.0f}s   {pos_txt}   valid={pct:.1f}%"
            print(f"\r{t_mono - t_start:7.1f}s  {pos_txt}  valid={pct:.1f}%   ",
                  end="", flush=True)

        next_tick += period
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_tick = time.monotonic()


def run_viz(shared, stop_event):
    """Live 3D view: current position + trail fading out over --trail seconds."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plt.ion()
    fig = plt.figure("Vive live position", figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.mpl_connect("close_event", lambda event: stop_event.set())
    trail_rgb = np.array([0.0, 0.85, 0.45])
    mins = None
    maxs = None

    while not stop_event.is_set():
        trail, markers, status = shared.snapshot()
        ax.cla()
        now = time.monotonic()
        if len(trail) >= 2:
            step = max(1, len(trail) // 300)  # cap segments so redraw stays fluid
            pts = np.asarray(trail[::step])
            # OpenVR is y-up: plot floor plane (x, z) horizontally, height y vertically
            xs, hs, zs = pts[:, 1], pts[:, 2], pts[:, 3]
            p3 = np.column_stack([xs, zs, hs])
            segs = np.stack([p3[:-1], p3[1:]], axis=1)
            age = now - pts[1:, 0]
            alpha = np.clip(1.0 - age / shared.trail_seconds, 0.03, 1.0)
            colors = np.zeros((len(alpha), 4))
            colors[:, :3] = trail_rgb
            colors[:, 3] = alpha
            ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=2.5))
            ax.scatter([xs[-1]], [zs[-1]], [hs[-1]], color="#ff3030", s=60, depthshade=False)

            lo = p3.min(axis=0) - 0.3
            hi = p3.max(axis=0) + 0.3
            mins = lo if mins is None else np.minimum(mins, lo)
            maxs = hi if maxs is None else np.maximum(maxs, hi)

        if markers:
            m = np.asarray(markers)
            ax.scatter(m[:, 0], m[:, 2], m[:, 1], color="orange", marker="x", s=50,
                       depthshade=False)

        if mins is not None:
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(min(0.0, mins[2]), maxs[2])
            ax.set_box_aspect(maxs - np.array([mins[0], mins[1], min(0.0, mins[2])]))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_zlabel("height [m]")
        ax.set_title(status)
        plt.pause(0.05)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="sessions", help="output directory")
    parser.add_argument("--rate", type=float, default=100.0, help="sampling rate in Hz")
    parser.add_argument("--include-hmd", action="store_true", help="also log the headset pose")
    parser.add_argument("--viz", action="store_true", help="open a live 3D view with fading trail")
    parser.add_argument("--trail", type=float, default=10.0,
                        help="seconds of path kept visible in the viz trail")
    args = parser.parse_args()

    vr = openvr.init(openvr.VRApplication_Other)
    devices = discover_devices(vr, include_hmd=args.include_hmd)
    if not devices:
        print("No controllers/trackers found. Is SteamVR running and a controller on (green)?")
        openvr.shutdown()
        return 1
    for idx, (cls, serial) in sorted(devices.items()):
        print(f"logging device {idx}: {cls} {serial}", flush=True)
    controller_idxs = [i for i, (cls, _) in devices.items() if cls == "controller"]
    primary_idx = min(controller_idxs) if controller_idxs else min(devices)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vive_gt_{datetime.now():%Y%m%d_%H%M%S}.csv"

    shared = SharedState(args.trail)
    stop_event = threading.Event()
    counters = {"samples": 0, "valid": 0}
    t_start = time.monotonic()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        print(f"writing {out_path} at {args.rate:g} Hz — Ctrl+C to stop", flush=True)
        try:
            if args.viz:
                logger = threading.Thread(
                    target=logging_loop,
                    args=(vr, devices, primary_idx, writer, f, args.rate,
                          shared, stop_event, counters),
                    daemon=True,
                )
                logger.start()
                run_viz(shared, stop_event)   # returns when window closed or stop set
                logger.join(timeout=2.0)
            else:
                logging_loop(vr, devices, primary_idx, writer, f, args.rate,
                             shared, stop_event, counters)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            openvr.shutdown()

    duration = time.monotonic() - t_start
    pct = 100.0 * counters["valid"] / max(1, counters["samples"])
    print(f"\nsession done: {out_path}")
    print(f"  duration {duration:.1f}s, {counters['samples']} samples, {pct:.1f}% valid poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
