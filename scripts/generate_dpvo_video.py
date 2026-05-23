"""Generate side-by-side video of image stream and DPVO trajectory prediction."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "async_collection"

# ---------------------------------------------------------------------------
# Sim(3) Alignment 
# ---------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve Sim(3) such that dst ≈ s · R · src + t."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / sigma_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t

def read_tum_trajectory(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 8:
            continue
        rows.append([float(x) for x in toks[:8]])
    arr = np.array(rows)
    return pd.DataFrame(arr, columns=["t", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

def _pick_run(override: str | None) -> Path:
    if override:
        return Path(override)
    runs = sorted((ROOT / "runs").glob("dpvo_*"))
    runs = [r for r in runs if (r / "saved_trajectories").exists()]
    if not runs:
        sys.exit("No dpvo_* run with saved_trajectories/ found.")
    return runs[-1]


def generate_video(pid: int, run_dir: Path, output_file: str):
    pdir = DATA / f"path_{pid:02d}"
    cam_csv = pdir / "camera.csv"
    gt_csv = pdir / "ground_truth.csv"
    traj_file = run_dir / "saved_trajectories" / f"path_{pid:02d}.txt"

    if not traj_file.exists():
        print(f"Skipping Path {pid:02d}: DPVO trajectory {traj_file} does not exist. Run scripts/run_dpvo_paths.py first.")
        return
    if not cam_csv.exists() or not gt_csv.exists():
        print(f"Skipping Path {pid:02d}: Ground truth or camera csv missing in {pdir}")
        return

    print("Loading trajectories...")
    pred = read_tum_trajectory(traj_file)
    cam_df = pd.read_csv(cam_csv).sort_values("sim_time").reset_index(drop=True)
    gt_df = pd.read_csv(gt_csv).sort_values("sim_time").reset_index(drop=True)

    if len(pred) > len(cam_df):
        pred = pred.iloc[:len(cam_df)].reset_index(drop=True)

    # Time map DPVO frames to Sim_Time
    pred_sim_t = cam_df["sim_time"].iloc[:len(pred)].to_numpy()
    gt_t = gt_df["sim_time"].to_numpy()
    
    idx = np.searchsorted(gt_t, pred_sim_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    
    left = idx - 1
    right = idx
    pick = np.where(np.abs(gt_t[left] - pred_sim_t) < np.abs(gt_t[right] - pred_sim_t), left, right)
    
    gt_xyz = gt_df.iloc[pick][["gt_x", "gt_y", "gt_z"]].to_numpy()
    pred_xyz = pred[["tx", "ty", "tz"]].to_numpy()

    print("Aligning trajectories...")
    s, R, t_offset = umeyama(pred_xyz, gt_xyz)
    pred_aligned = (s * (R @ pred_xyz.T)).T + t_offset

    # Determine stable 3D axes limits for smooth rendering
    full_gt_xyz = gt_df[["gt_x", "gt_y", "gt_z"]].to_numpy()
    x_min = min(full_gt_xyz[:,0].min(), pred_xyz[:,0].min(), pred_aligned[:,0].min()) - 1
    x_max = max(full_gt_xyz[:,0].max(), pred_xyz[:,0].max(), pred_aligned[:,0].max()) + 1
    y_min = min(full_gt_xyz[:,1].min(), pred_xyz[:,1].min(), pred_aligned[:,1].min()) - 1
    y_max = max(full_gt_xyz[:,1].max(), pred_xyz[:,1].max(), pred_aligned[:,1].max()) + 1
    z_min = min(full_gt_xyz[:,2].min(), pred_xyz[:,2].min(), pred_aligned[:,2].min()) - 1
    z_max = max(full_gt_xyz[:,2].max(), pred_xyz[:,2].max(), pred_aligned[:,2].max()) + 1

    # First image to grab dimensions
    first_img_path = pdir / cam_df["rgb_path"].iloc[0]
    img = cv2.imread(str(first_img_path))
    if img is None:
        sys.exit(f"Could not read first image: {first_img_path}")
    ih, iw, ic = img.shape
    
    # Initialize Matplotlib Figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    fig.canvas.draw()
    plot_img_rgba = np.asarray(fig.canvas.buffer_rgba())
    ph, pw = plot_img_rgba.shape[0], plot_img_rgba.shape[1]

    # Target width to match image height
    target_pw = int(pw * (ih / ph))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_file, fourcc, 15.0, (iw + target_pw, ih))

    print(f"Rendering {len(pred)} frames to {output_file} ...")
    
    for i in tqdm(range(len(pred))):
        img_p = pdir / cam_df["rgb_path"].iloc[i]
        frame = cv2.imread(str(img_p))
        if frame is None:
            print(f"Warning: skipped missing frame to {img_p}")
            continue
            
        ax.clear()
        
        # Keep limits static so the camera doesn't jump
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"3D Path Prediction (sim_time: {pred_sim_t[i]:.2f}s)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Plot full Ground Truth footprint as light gray map
        ax.plot(full_gt_xyz[:,0], full_gt_xyz[:,1], full_gt_xyz[:,2], color='lightgray', alpha=0.5, linewidth=1, label='Full Path')
        
        # Plot exact current history
        cur_gt = gt_xyz[:i+1]
        cur_pred = pred_aligned[:i+1]
        cur_raw = pred_xyz[:i+1]
        
        ax.plot(cur_gt[:,0], cur_gt[:,1], cur_gt[:,2], color='black', linewidth=2, label='Ground Truth')
        ax.plot(cur_pred[:,0], cur_pred[:,1], cur_pred[:,2], color='red', linewidth=2, label='Aligned DPVO')
        ax.plot(cur_raw[:,0], cur_raw[:,1], cur_raw[:,2], color='blue', linestyle='--', linewidth=1.5, label='Raw DPVO')
        
        # Current Head Indicator
        ax.scatter([cur_gt[-1,0]], [cur_gt[-1,1]], [cur_gt[-1,2]], color='black', s=50)
        ax.scatter([cur_pred[-1,0]], [cur_pred[-1,1]], [cur_pred[-1,2]], color='red', s=50)
        ax.scatter([cur_raw[-1,0]], [cur_raw[-1,1]], [cur_raw[-1,2]], color='blue', s=40)
        
        if i == 0 or i == len(pred)-1:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

        fig.canvas.draw()
        plot_rgba = np.asarray(fig.canvas.buffer_rgba())
        plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
        
        plot_bgr_resized = cv2.resize(plot_bgr, (target_pw, ih))
        
        combined = np.hstack((frame, plot_bgr_resized))
        out_video.write(combined)
        
    out_video.release()
    plt.close(fig)
    print("Done! Video saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", type=int, required=True, help="Path IDs to render (e.g. 2 13 14)")
    parser.add_argument("--run", default=None, help="Specific dpvo run directory (default: latest dpvo_*)")
    args = parser.parse_args()
    
    run_dir = _pick_run(args.run)
    for p in args.paths:
        out_file = str(run_dir / f"trajectory_video_path_{p:02d}.mp4")
        generate_video(p, run_dir, out_file)
