"""Render per-path replay videos: 4 synchronized real-time panels.

  1. IMU (linear acceleration, gravity removed) + odometry (v, omega) --
     scrolling time-series with a "now" cursor.
  2. Camera stream (the frames the replay controller saved).
  3. WiFi: top-N strongest APs of the latest scan as a live bar chart,
     with scan age + visible-AP count.
  4. 2D map: walls from the .wbt, full path faint, traversed part + current
     pose live.

"Real-time" = the video clock IS the sim clock: duration equals the path's
duration, at --fps frames per sim second.

Gravity removal: exponential low-pass (tau = 1 s) of the raw accelerometer
estimates the gravity vector in the device frame; linear = accel - lowpass.
Same approach as Android's linear-acceleration virtual sensor -- robust to
axis conventions, no assumptions about the rotation-vector frame.

Output: <replay>/videos/path_XX.mp4 (mp4v via OpenCV, no external ffmpeg).

Usage (PowerShell, from repo root):
    .venv\\Scripts\\python.exe scripts\\render_replay_video.py \\
        --replay data/iln20_5d27099f_F2_replay \\
        --dataset data/iln20_5d27099f_F2 \\
        --world src/simulation/worlds/iln20_5d27099f_F2.wbt \\
        [--paths 0-4] [--fps 10]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time as pytime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

import cv2  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_replay import parse_path_ids  # noqa: E402

ABSENT_RSSI = -150.0     # source uses -200 for "AP not seen"
SCROLL_S = 8.0           # time-series window width
TOP_N_AP = 15


def read_table(p: Path) -> dict[str, np.ndarray]:
    """CSV -> dict of float arrays (non-numeric columns kept as object)."""
    with open(p, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for c in rows[0]:
        col = [r[c] for r in rows]
        try:
            out[c] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[c] = np.array(col, dtype=object)
    return out


def parse_walls(wbt: Path) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    text = wbt.read_text(encoding="utf-8")
    segs = []
    for m in re.finditer(
            r"DEF P?WALL_\d+ Solid \{\s*\n\s*"
            r"translation\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\n\s*"
            r"rotation 0 0 1 (-?[\d.]+)", text):
        tail = text[m.end():m.end() + 700]
        mb = re.search(r"geometry Box \{ size (-?[\d.]+)", tail)
        if not mb:
            continue
        cx, cy, yaw = float(m.group(1)), float(m.group(2)), float(m.group(4))
        half = float(mb.group(1)) / 2
        dx, dy = math.cos(yaw) * half, math.sin(yaw) * half
        segs.append(((cx - dx, cy - dy), (cx + dx, cy + dy)))
    return segs


def remove_gravity(t: np.ndarray, acc: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """acc[N,3] -> linear acceleration via per-sample EMA gravity estimate."""
    g = np.empty_like(acc)
    g[0] = acc[0]
    for i in range(1, len(acc)):
        dt = max(1e-4, t[i] - t[i - 1])
        a = dt / (tau + dt)
        g[i] = g[i - 1] + a * (acc[i] - g[i - 1])
    return acc - g


def render_path(rep_dir: Path, walls, out_mp4: Path, fps: float,
                title_prefix: str) -> str:
    # ── Load everything ──
    gt = read_table(rep_dir / "ground_truth.csv")
    imu = read_table(rep_dir / "imu.csv")
    odom = read_table(rep_dir / "odometry.csv")
    wifi = read_table(rep_dir / "wifi.csv")
    cam = read_table(rep_dir / "camera.csv")
    if not gt or not cam:
        return "missing ground_truth/camera data"

    t0, t1 = float(np.nanmin(gt["sim_time"])), float(np.nanmax(gt["sim_time"]))
    gt_t, gt_x, gt_y = gt["sim_time"], gt["gt_x"], gt["gt_y"]
    order = np.argsort(gt_t)
    gt_t, gt_x, gt_y = gt_t[order], gt_x[order], gt_y[order]
    yaw = np.deg2rad(gt["gt_heading_deg"][order])

    imu_t = imu["sim_time"]
    lin = remove_gravity(imu_t, np.stack(
        [imu["accel_x"], imu["accel_y"], imu["accel_z"]], axis=1))
    od_t = odom["sim_time"]

    wifi_t = wifi.get("sim_time", np.array([]))
    rssi_cols = [c for c in wifi if c.startswith("wifi_rssi_")]
    rssi = (np.stack([wifi[c] for c in rssi_cols], axis=1)
            if len(wifi_t) else np.zeros((0, 0)))
    ap_labels = [c[len("wifi_rssi_"):][-6:] for c in rssi_cols]  # MAC suffix

    cam_t = cam["sim_time"]
    cam_paths = [rep_dir / p for p in cam["rgb_path"]]

    # ── Figure: cam+wifi | imu+odom | tall map ──
    fig = plt.figure(figsize=(12.8, 9.6), dpi=100)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.15, 0.85],
                  hspace=0.28, wspace=0.25,
                  left=0.05, right=0.98, top=0.93, bottom=0.06)
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_wifi = fig.add_subplot(gs[1, 0])
    ax_imu = fig.add_subplot(gs[0, 1])
    ax_od = fig.add_subplot(gs[1, 1])
    ax_map = fig.add_subplot(gs[:, 2])

    # camera panel
    ax_cam.set_title("camera (head, 640x480)", fontsize=10)
    ax_cam.axis("off")
    im = ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

    # IMU panel (static series + scrolling xlim + cursor)
    for k, (lbl, c) in enumerate((("ax", "tab:red"), ("ay", "tab:green"),
                                  ("az", "tab:blue"))):
        ax_imu.plot(imu_t, lin[:, k], lw=0.6, color=c, label=lbl)
    ax_imu.set_title("IMU linear accel (gravity removed) [m/s²]", fontsize=10)
    ax_imu.legend(loc="upper right", fontsize=7, ncol=3)
    lim = float(np.nanpercentile(np.abs(lin), 99.5)) * 1.2 + 0.2
    ax_imu.set_ylim(-lim, lim)
    cur_imu = ax_imu.axvline(t0, color="k", lw=1)

    # odometry panel
    ax_od.plot(od_t, odom["odom_linear_vel"], lw=0.8, color="tab:blue",
               label="v [m/s]")
    ax_od.plot(od_t, odom["odom_angular_vel"], lw=0.8, color="tab:orange",
               label="ω [rad/s]")
    ax_od.set_title("odometry (synthesized encoders)", fontsize=10)
    ax_od.legend(loc="upper right", fontsize=7)
    ax_od.set_xlabel("sim time [s]", fontsize=8)
    cur_od = ax_od.axvline(t0, color="k", lw=1)

    # WiFi panel (redrawn on scan change; ~1 Hz -> cheap)
    ax_wifi.set_title("WiFi RSSI — top APs (latest scan)", fontsize=10)
    last_scan_idx = -2

    # map panel
    if walls:
        ax_map.add_collection(LineCollection(walls, colors="k", lw=0.8))
    ax_map.plot(gt_x, gt_y, "-", color="0.75", lw=1.0, zorder=1)
    trav, = ax_map.plot([], [], "-", color="tab:red", lw=1.8, zorder=2)
    pos, = ax_map.plot([], [], "o", color="tab:red", ms=7, zorder=3)
    head, = ax_map.plot([], [], "-", color="tab:red", lw=1.6, zorder=3)
    ax_map.set_aspect("equal")
    ax_map.set_title("live pose vs walls", fontsize=10)
    pad = 3.0
    ax_map.set_xlim(min(s[0][0] for s in walls) - pad if walls else gt_x.min() - pad,
                    max(s[1][0] for s in walls) + pad if walls else gt_x.max() + pad)
    ax_map.set_ylim(min(min(s[0][1], s[1][1]) for s in walls) - pad if walls else gt_y.min() - pad,
                    max(max(s[0][1], s[1][1]) for s in walls) + pad if walls else gt_y.max() + pad)
    ax_map.tick_params(labelsize=7)

    sup = fig.suptitle("", fontsize=11)

    # ── Writer (size from the actually-rendered canvas) ──
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    if not vw.isOpened():
        plt.close(fig)
        return "cv2.VideoWriter failed to open (mp4v)"

    n_frames = int((t1 - t0) * fps) + 1
    last_cam_idx = -1
    for k in range(n_frames):
        t = t0 + k / fps

        # camera
        ci = int(np.searchsorted(cam_t, t, side="right") - 1)
        if ci >= 0 and ci != last_cam_idx:
            img = cv2.imread(str(cam_paths[ci]))
            if img is not None:
                im.set_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            last_cam_idx = ci

        # scrolling series + cursors
        for ax in (ax_imu, ax_od):
            ax.set_xlim(t - SCROLL_S, t + 0.5)
        cur_imu.set_xdata([t, t])
        cur_od.set_xdata([t, t])

        # wifi (only when a new scan appears)
        wi = int(np.searchsorted(wifi_t, t, side="right") - 1) if len(wifi_t) else -1
        if wi >= 0 and wi != last_scan_idx:
            ax_wifi.clear()
            row = rssi[wi]
            top = np.argsort(row)[::-1][:TOP_N_AP]
            top = [i for i in top if row[i] > ABSENT_RSSI]
            vals = row[top][::-1]
            labs = [ap_labels[i] for i in top][::-1]
            colors = plt.cm.viridis((np.clip(vals, -95, -35) + 95) / 60)
            ax_wifi.barh(range(len(vals)), vals + 100, left=-100, color=colors)
            ax_wifi.set_yticks(range(len(vals)), labs, fontsize=6.5)
            ax_wifi.set_xlim(-100, -30)
            ax_wifi.set_xlabel("RSSI [dBm]", fontsize=8)
            ax_wifi.tick_params(axis="x", labelsize=7)
            last_scan_idx = wi
        age = (t - wifi_t[wi]) if wi >= 0 else float("nan")
        n_vis = int(wifi["wifi_visible_count"][wi]) if wi >= 0 else 0
        ax_wifi.set_title(
            f"WiFi — top {TOP_N_AP} APs | visible {n_vis} | scan age {age:.1f}s",
            fontsize=10)

        # map
        gi = int(np.searchsorted(gt_t, t, side="right"))
        trav.set_data(gt_x[:gi], gt_y[:gi])
        if gi > 0:
            px, py, pyaw = gt_x[gi - 1], gt_y[gi - 1], yaw[gi - 1]
            pos.set_data([px], [py])
            head.set_data([px, px + 1.2 * math.cos(pyaw)],
                          [py, py + 1.2 * math.sin(pyaw)])

        sup.set_text(f"{title_prefix}   t = {t - t0:6.2f} / {t1 - t0:.2f} s")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    vw.release()
    plt.close(fig)
    return f"ok ({n_frames} frames, {t1 - t0:.1f}s)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="replay output root")
    ap.add_argument("--dataset", default=None,
                    help="source dataset root (unused for now; kept for parity)")
    ap.add_argument("--world", required=True, help=".wbt to draw walls from")
    ap.add_argument("--paths", default=None, help="ids spec, e.g. '0-4' (default: all)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="video frames per sim second (default 10)")
    args = ap.parse_args()

    rep = Path(args.replay).resolve()
    world = Path(args.world).resolve()
    if not rep.is_dir():
        sys.exit(f"replay dir not found: {rep}")
    if not world.is_file():
        sys.exit(f"world not found: {world}")

    walls = parse_walls(world)
    print(f"[video] {len(walls)} wall segments from {world.name}")

    if args.paths:
        ids = parse_path_ids(args.paths)
    else:
        ids = sorted(int(d.name[5:]) for d in rep.iterdir()
                     if d.is_dir() and d.name.startswith("path_")
                     and d.name[5:].isdigit())

    out_dir = rep / "videos"
    n_ok = 0
    for pid in ids:
        pdir = rep / f"path_{pid:02d}"
        if not (pdir / "camera.csv").is_file():
            print(f"[video] path_{pid:02d}: no camera.csv -- skipped")
            continue
        t_w = pytime.time()
        msg = render_path(pdir, walls, out_dir / f"path_{pid:02d}.mp4",
                          args.fps, f"{rep.name} / path_{pid:02d}")
        ok = msg.startswith("ok")
        n_ok += ok
        print(f"[video] path_{pid:02d}: {msg}  [{pytime.time() - t_w:.1f}s wall]",
              flush=True)

    print(f"[video] {n_ok}/{len(ids)} videos -> {out_dir}")
    return 0 if n_ok == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
