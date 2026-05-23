"""Side-by-side comparison plots for ACE SCR vs DPVO on the (x, y) task.

Reads:
    runs/<ace_run>/pnp_eval_{val,test}_frames.csv   <- ACE+PnP per-frame
    runs/<dpvo_run>/dpvo_eval_frames.csv            <- DPVO Sim(3)-aligned per-frame

Writes (into the DPVO run's plots/ dir for consistency with the comparison):
    01_methods_baseline_bar.png   - 5-bar comparison incl. DPVO
    02_per_path_comparison.png    - bar chart, ACE vs DPVO per path
    03_error_cdf_comparison.png   - CDF curves, val and test, ACE vs DPVO
    04_trajectories_comparison.png - GT (blue) + ACE (orange) + DPVO (green) per path

Usage:
    python scripts/plot_dpvo_vs_ace.py
    python scripts/plot_dpvo_vs_ace.py --ace runs/ace_scr_... --dpvo runs/dpvo_...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon as MplPolygon

DATA = ROOT / "data" / "async_collection"

VAL = {2, 13, 14}
TEST = {15, 16, 17}

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


def _wall_polygon(tx, ty, angle, sx, sy):
    hx, hy = sx / 2, sy / 2
    ca, sa = math.cos(angle), math.sin(angle)
    return [(tx + ca * cx - sa * cy, ty + sa * cx + ca * cy)
            for cx, cy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def _draw_walls(ax) -> None:
    for tx, ty, angle, sx, sy in _RAW_WALLS:
        ax.add_patch(MplPolygon(
            _wall_polygon(tx, ty, angle, sx, sy), closed=True,
            facecolor="#888", edgecolor="#333", linewidth=0.5, alpha=0.55,
        ))


def _cdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def _pick_run(prefix: str, override: str | None) -> Path:
    if override:
        return Path(override)
    cands = sorted(
        r for r in (ROOT / "runs").glob(f"{prefix}*")
        if not r.name.startswith(f"{prefix}overfit")
        and not r.name.startswith(f"{prefix}smoketest")
    )
    cands = [r for r in cands if r.is_dir()]
    if not cands:
        raise SystemExit(f"No {prefix}* run found")
    return cands[-1]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_baselines(out: Path,
                   ace_val: float, ace_test: float,
                   dpvo_val: float, dpvo_test: float) -> None:
    methods = [
        ("DINOv2+LoRA\nlinear probe",   3.64,      "#8c564b"),
        ("ACE\nlinear probe",           3.49,      "#ff7f0e"),
        ("ACE SCR + PnP\n(val)",        ace_val,   "#1f77b4"),
        ("ACE SCR + PnP\n(test)",       ace_test,  "#1f77b4"),
        ("DPVO\n(val, Sim(3))",         dpvo_val,  "#2ca02c"),
        ("DPVO\n(test, Sim(3))",        dpvo_test, "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(methods))
    bars = ax.bar(xs, [m[1] for m in methods],
                  color=[m[2] for m in methods],
                  edgecolor="#222", linewidth=0.6)
    for b, (lbl, y, c) in zip(bars, methods):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.05,
                f"{y:.2f} m", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([m[0] for m in methods], fontsize=9)
    ax.set_ylabel("(x, y) error (m)")
    ax.set_title("Robot (x, y) localisation error — vision-only methods")
    ax.axhline(1.0, color="#555", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.98, 0.98,
            "Linear probe: MAE\nACE SCR+PnP: median euclid (no alignment)\n"
            "DPVO: median euclid after Sim(3) alignment",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#f5f5f5", edgecolor="#ccc"))
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_per_path_compare(out: Path,
                          ace_run: Path, dpvo_run: Path) -> None:
    a_val = json.loads((ace_run / "pnp_eval_val.json").read_text())["per_path"]
    a_tst = json.loads((ace_run / "pnp_eval_test.json").read_text())["per_path"]
    d = json.loads((dpvo_run / "dpvo_eval.json").read_text())["per_path"]

    rows = []
    for pid, info in a_val.items():
        rows.append((int(pid), "val", info["euclid_xy_to_gt"]["median"],
                     d.get(pid, {}).get("median_xy", float("nan"))))
    for pid, info in a_tst.items():
        rows.append((int(pid), "test", info["euclid_xy_to_gt"]["median"],
                     d.get(pid, {}).get("median_xy", float("nan"))))
    rows.sort()

    fig, ax = plt.subplots(figsize=(10, 4.7))
    xs = np.arange(len(rows))
    w = 0.36
    ace_vals  = [r[2] for r in rows]
    dpvo_vals = [r[3] for r in rows]
    ax.bar(xs - w/2, ace_vals,  width=w, label="ACE SCR + PnP", color="#1f77b4")
    ax.bar(xs + w/2, dpvo_vals, width=w, label="DPVO (Sim(3))", color="#2ca02c")
    for i, (pid, split, a, d_) in enumerate(rows):
        ax.text(i - w/2, a + 0.02, f"{a:.2f}",
                ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, d_ + 0.02, f"{d_:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(
        [f"p{r[0]}\n({r[1]})" for r in rows])
    ax.set_ylabel("median (x, y) error (m)")
    ax.set_title("Per-path comparison: ACE SCR+PnP  vs  DPVO (Sim(3)-aligned)")
    ax.axhline(1.0, color="#555", linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_cdf_compare(out: Path, ace_run: Path, dpvo_run: Path) -> None:
    a_val = pd.read_csv(ace_run / "pnp_eval_val_frames.csv")
    a_tst = pd.read_csv(ace_run / "pnp_eval_test_frames.csv")
    d = pd.read_csv(dpvo_run / "dpvo_eval_frames.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, split, label_a, label_d in [
            (axes[0], "val",  "ACE val",  "DPVO val"),
            (axes[1], "test", "ACE test", "DPVO test"),
    ]:
        ace_df = a_val if split == "val" else a_tst
        ace_errs = ace_df.loc[ace_df["ok"] & ace_df["euclid_xy_to_gt"].notna(),
                              "euclid_xy_to_gt"].values
        dpvo_errs = d.loc[d["split"] == split, "euclid_xy"].values

        if ace_errs.size:
            xs, ys = _cdf(ace_errs)
            ax.plot(xs, ys, lw=2, color="#1f77b4",
                    label=f"{label_a} (n={len(ace_errs)})")
        if dpvo_errs.size:
            xs, ys = _cdf(dpvo_errs)
            ax.plot(xs, ys, lw=2, color="#2ca02c",
                    label=f"{label_d} (n={len(dpvo_errs)})")

        ax.axvline(1.0, color="#555", linestyle=":", alpha=0.5)
        ax.axvline(0.5, color="#aaa", linestyle=":", alpha=0.5)
        ax.set_xlabel("(x, y) euclid error (m)")
        ax.set_ylabel("fraction of frames ≤ x")
        ax.set_title(f"{split.upper()} split")
        ax.set_xlim(0, np.percentile(np.concatenate([ace_errs, dpvo_errs]), 99))
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

    fig.suptitle("Error CDF — ACE SCR+PnP vs DPVO (Sim(3)-aligned)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_trajectories_compare(out: Path,
                              ace_run: Path, dpvo_run: Path) -> None:
    """3-line per-path: GT (blue), ACE PnP scatter (orange), DPVO line (green)."""
    a_val = pd.read_csv(ace_run / "pnp_eval_val_frames.csv")
    a_tst = pd.read_csv(ace_run / "pnp_eval_test_frames.csv")
    d = pd.read_csv(dpvo_run / "dpvo_eval_frames.csv")
    paths = sorted(set(a_val["path_id"]) | set(a_tst["path_id"]))

    cols = 3
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 5.5),
                             squeeze=False)
    for k, pid in enumerate(paths):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        _draw_walls(ax)

        gt_csv = DATA / f"path_{pid:02d}" / "ground_truth.csv"
        if gt_csv.exists():
            gt = pd.read_csv(gt_csv)
            ax.plot(gt["gt_x"], gt["gt_y"],
                    color="#1f77b4", lw=2.2, alpha=0.9, label="GT")

        a_df = pd.concat([a_val[a_val["path_id"] == pid],
                          a_tst[a_tst["path_id"] == pid]],
                         ignore_index=True)
        a_df = a_df[a_df["ok"]]
        if not a_df.empty:
            ax.scatter(a_df["pred_cam_x"], a_df["pred_cam_y"],
                       s=6, color="#ff7f0e", alpha=0.6, label="ACE+PnP")

        dd = d[d["path_id"] == pid].sort_values("sim_time")
        if not dd.empty:
            ax.plot(dd["pred_x_aligned"], dd["pred_y_aligned"],
                    color="#2ca02c", lw=1.4, alpha=0.85,
                    label="DPVO (Sim(3))")

        med_a = (a_df["euclid_xy_to_gt"].median() if not a_df.empty else float("nan"))
        med_d = (dd["euclid_xy"].median() if not dd.empty else float("nan"))
        split = "val" if pid in VAL else ("test" if pid in TEST else "?")
        ax.set_aspect("equal")
        ax.set_xlim(-14.5, 5.5); ax.set_ylim(-19.5, 3.5)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"path {pid:02d} ({split})\n"
                     f"ACE med={med_a:.2f} m  |  DPVO med={med_d:.2f} m",
                     fontsize=10)
        ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    for k in range(len(paths), rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    fig.suptitle("ACE SCR+PnP  vs  DPVO  vs  GT  —  val + test", y=1.00)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ace",  default=None, help="ACE SCR run dir")
    p.add_argument("--dpvo", default=None, help="DPVO run dir")
    args = p.parse_args()

    ace_run = _pick_run("ace_scr_", args.ace)
    dpvo_run = _pick_run("dpvo_", args.dpvo)
    print(f"ACE  run: {ace_run}")
    print(f"DPVO run: {dpvo_run}")

    ace_val_med = json.loads((ace_run / "pnp_eval_val.json").read_text())[
        "overall"]["euclid_xy_to_gt"]["median"]
    ace_tst_med = json.loads((ace_run / "pnp_eval_test.json").read_text())[
        "overall"]["euclid_xy_to_gt"]["median"]
    d_overall = json.loads((dpvo_run / "dpvo_eval.json").read_text())["overall"]
    dpvo_val_med = d_overall.get("val_median_xy", float("nan"))
    dpvo_tst_med = d_overall.get("test_median_xy", float("nan"))

    out_dir = dpvo_run / "plots"
    out_dir.mkdir(exist_ok=True)
    print(f"plots out: {out_dir}\n")

    plot_baselines(out_dir / "01_methods_baseline_bar.png",
                   ace_val_med, ace_tst_med,
                   dpvo_val_med, dpvo_tst_med)
    plot_per_path_compare(out_dir / "02_per_path_comparison.png",
                          ace_run, dpvo_run)
    plot_cdf_compare(out_dir / "03_error_cdf_comparison.png",
                     ace_run, dpvo_run)
    plot_trajectories_compare(out_dir / "04_trajectories_comparison.png",
                              ace_run, dpvo_run)


if __name__ == "__main__":
    main()
