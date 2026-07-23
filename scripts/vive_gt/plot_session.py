"""Inspect a vive_logger.py session: top-down trajectory plot + quality report.

Usage:
    python plot_session.py sessions\\vive_gt_YYYYMMDD_HHMMSS.csv [--serial SN] [--still-seconds 5]

The PNG is written next to the CSV. OpenVR is y-up, so the floor plane is (x, z).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(csv_path, serial=None):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    serials = np.unique(data["serial"])
    if serial is None:
        if len(serials) > 1:
            print(f"multiple devices {list(serials)}; using {serials[0]} (pick with --serial)")
        serial = serials[0]
    d = data[data["serial"] == serial]
    return d, str(serial)


def report(d, still_seconds):
    t = d["t_mono"] - d["t_mono"][0]
    valid = d["valid"].astype(bool)
    duration = t[-1] if len(t) else 0.0
    pct_valid = 100.0 * valid.mean() if len(d) else 0.0

    # longest run of consecutive invalid samples
    longest_gap = 0.0
    gap_start = None
    for ti, vi in zip(t, valid):
        if not vi and gap_start is None:
            gap_start = ti
        elif vi and gap_start is not None:
            longest_gap = max(longest_gap, ti - gap_start)
            gap_start = None
    if gap_start is not None:
        longest_gap = max(longest_gap, t[-1] - gap_start)

    x, z = d["x"][valid], d["z"][valid]
    bbox = (x.max() - x.min(), z.max() - z.min()) if valid.any() else (0, 0)

    still = valid & (t < still_seconds)
    jitter_mm = float("nan")
    if still.sum() > 10:
        pts = np.stack([d["x"][still], d["z"][still]], axis=1)
        jitter_mm = 1000.0 * np.linalg.norm(pts.std(axis=0))

    print(f"duration        {duration:.1f} s ({len(d)} samples, {len(d) / max(duration, 1e-9):.0f} Hz)")
    print(f"valid poses     {pct_valid:.1f} %")
    print(f"longest dropout {longest_gap * 1000:.0f} ms")
    print(f"bounding box    {bbox[0]:.2f} x {bbox[1]:.2f} m")
    print(f"still jitter    {jitter_mm:.2f} mm (first {still_seconds:g} s)"
          if np.isfinite(jitter_mm) else "still jitter    n/a (device moving at start?)")
    return pct_valid, longest_gap, jitter_mm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--serial", default=None, help="device serial to plot")
    parser.add_argument("--still-seconds", type=float, default=5.0,
                        help="assume the device is stationary for this long at start (jitter estimate)")
    args = parser.parse_args()

    d, serial = load(args.csv, args.serial)
    print(f"device {serial}, file {args.csv.name}")
    report(d, args.still_seconds)

    valid = d["valid"].astype(bool)
    t = d["t_mono"] - d["t_mono"][0]
    fig, (ax, ax_t) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    sc = ax.scatter(d["x"][valid], d["z"][valid], c=t[valid], s=2, cmap="viridis")
    trig = valid & (d["trigger"] == 1)
    if trig.any():
        ax.scatter(d["x"][trig], d["z"][trig], marker="x", s=60, color="red", label="trigger")
        ax.legend()
    fig.colorbar(sc, ax=ax, label="time [s]")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_title(f"top-down trajectory — {serial}")

    ax_t.plot(t, d["y"], lw=0.5, label="height y [m]")
    ax_t.plot(t, valid.astype(float), lw=0.8, alpha=0.6, label="pose valid")
    ax_t.set_xlabel("time [s]")
    ax_t.legend()
    ax_t.set_title("height + validity over time")

    out = args.csv.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"plot saved: {out}")


if __name__ == "__main__":
    main()
