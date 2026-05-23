"""PnP-based (x, y) localisation evaluation for a trained ACE SCR head.

For every val frame we:
    1. Predict a 60x80 scene-coord map (world-frame XYZ per patch) with the
       trained ACEScrRegressor.
    2. Pair each patch centre pixel (u, v) with its predicted world XYZ and
       feed the correspondences to ``cv2.solvePnPRansac`` to recover the
       camera pose.
    3. Extract the camera centre in world coords, compare to
         - the camera pose saved in camera.csv (tight geometry check)
         - the robot ground-truth (x, y) (the real task metric)

Usage (from repo root):
    python scripts/eval_ace_scr_pnp.py                              # latest ace_scr_* run, val split
    python scripts/eval_ace_scr_pnp.py --run runs/ace_scr_... --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipeline.data.scr_dataset import SCRDataset
from src.pipeline.encoders import ACEScrRegressor


TRAIN_PATHS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
VAL_PATHS = [2, 13, 14]
TEST_PATHS = [15, 16, 17]


def pnp_camera_centre(
    pred_sc: np.ndarray,     # (3, h, w) predicted world XYZ
    K: np.ndarray,            # (3, 3)
    stride: int = 8,
    valid_mask: np.ndarray | None = None,   # (h, w) bool, optional
    reprojection_err: float = 8.0,
    iterations: int = 200,
) -> tuple[np.ndarray | None, int, bool]:
    """Run solvePnPRansac. Returns (camera_centre_world_xyz, n_inliers, ok)."""
    h, w = pred_sc.shape[1:]
    ys = np.arange(h, dtype=np.float32) * stride + stride / 2.0
    xs = np.arange(w, dtype=np.float32) * stride + stride / 2.0
    u_grid, v_grid = np.meshgrid(xs, ys)  # (h, w)
    img_pts = np.stack([u_grid.ravel(), v_grid.ravel()], axis=-1).astype(np.float32)
    obj_pts = pred_sc.reshape(3, -1).T.astype(np.float32)

    if valid_mask is not None:
        m = valid_mask.ravel()
        img_pts = img_pts[m]
        obj_pts = obj_pts[m]

    if len(img_pts) < 4:
        return None, 0, False

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K.astype(np.float32), None,
            reprojectionError=reprojection_err,
            iterationsCount=iterations,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None, 0, False

    if not success or inliers is None or len(inliers) < 6:
        return None, 0, False

    R, _ = cv2.Rodrigues(rvec)
    centre = (-R.T @ tvec).flatten()
    return centre, int(len(inliers)), True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="Run dir (default: latest ace_scr_* with head.pt)")
    p.add_argument("--split", choices=["val", "test", "train"], default="val")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "async_collection"))
    p.add_argument("--weights",
                   default=str(ROOT / "runs" / "_weights" / "ace_encoder_pretrained.pt"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--use-valid-mask", action="store_true",
                   help="Restrict PnP to pixels where GT depth was valid "
                        "(optimistic — inference has no GT mask).")
    p.add_argument("--reproj-err", type=float, default=8.0)
    return p.parse_args()


def _pick_run(override: str | None) -> Path:
    if override:
        return Path(override)
    runs = sorted(
        r for r in (ROOT / "runs").glob("ace_scr_*")
        if (r / "head.pt").exists()
        and not r.name.startswith("ace_scr_overfit")
        and not r.name.startswith("ace_scr_smoketest")
    )
    if not runs:
        raise SystemExit("No ACE SCR production runs with head.pt found.")
    return runs[-1]


def main() -> None:
    args = _parse_args()

    run_dir = _pick_run(args.run)
    ckpt = run_dir / "head.pt"
    print(f"run dir: {run_dir}")
    print(f"using checkpoint: {ckpt}")

    split_paths = {"train": TRAIN_PATHS, "val": VAL_PATHS, "test": TEST_PATHS}[args.split]

    # Scene centre = mean of training camera positions (how the trainer did it).
    scene_ds = SCRDataset(
        data_dir=args.data_dir, path_ids=TRAIN_PATHS,
        stride=8, preload_depth=False,
    )
    mean_xyz = torch.from_numpy(scene_ds.mean_camera_translation)
    del scene_ds

    eval_ds = SCRDataset(
        data_dir=args.data_dir, path_ids=split_paths,
        stride=8, preload_depth=False,
    )
    print(f"{args.split}: {len(eval_ds)} frames across paths {split_paths}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACEScrRegressor(
        mean_xyz=mean_xyz, weights_path=args.weights,
    ).to(device).eval()
    model.head.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))

    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Collect per-frame results ---
    per_path: dict[int, dict[str, list]] = {}
    per_frame_rows: list[dict] = []  # saved to CSV for downstream plots/video
    n_pnp_fail = 0

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            pred = model(rgb).cpu().numpy()           # (B, 3, 60, 80)
            Ks = batch["K"].numpy()                   # (B, 3, 3)
            valids = batch["valid_mask"].numpy()      # (B, 60, 80)
            targets = batch["target"].numpy()         # (B, 2)
            T_cam_world = batch["T_cam_world"].numpy()  # (B, 4, 4) — saved pose w/ Astra offset
            path_ids = batch["path_id"].tolist()

            for i in range(rgb.shape[0]):
                centre, n_in, ok = pnp_camera_centre(
                    pred[i], Ks[i], stride=8,
                    valid_mask=valids[i] if args.use_valid_mask else None,
                    reprojection_err=args.reproj_err,
                )
                pid = path_ids[i]
                per_path.setdefault(pid, {
                    "euclid_xy_to_gt": [],
                    "euclid_xy_to_cam": [],
                    "mae_x_to_gt": [], "mae_y_to_gt": [],
                    "inliers": [], "n_ok": 0, "n_total": 0,
                })
                per_path[pid]["n_total"] += 1

                sim_time = float(batch["sim_time"][i])
                cam_saved = T_cam_world[i, :3, 3]

                if not ok:
                    n_pnp_fail += 1
                    per_frame_rows.append({
                        "path_id": pid, "sim_time": sim_time, "ok": False,
                        "pred_cam_x": float("nan"), "pred_cam_y": float("nan"),
                        "pred_cam_z": float("nan"),
                        "cam_saved_x": float(cam_saved[0]),
                        "cam_saved_y": float(cam_saved[1]),
                        "cam_saved_z": float(cam_saved[2]),
                        "gt_x": float(targets[i, 0]),
                        "gt_y": float(targets[i, 1]),
                        "inliers": 0,
                        "euclid_xy_to_gt": float("nan"),
                        "euclid_xy_to_cam": float("nan"),
                    })
                    continue

                per_path[pid]["n_ok"] += 1
                per_path[pid]["inliers"].append(n_in)

                d_cam = float(np.hypot(centre[0] - cam_saved[0], centre[1] - cam_saved[1]))
                d_gt = float(np.hypot(centre[0] - targets[i, 0], centre[1] - targets[i, 1]))
                per_path[pid]["euclid_xy_to_cam"].append(d_cam)
                per_path[pid]["euclid_xy_to_gt"].append(d_gt)
                per_path[pid]["mae_x_to_gt"].append(abs(centre[0] - targets[i, 0]))
                per_path[pid]["mae_y_to_gt"].append(abs(centre[1] - targets[i, 1]))

                per_frame_rows.append({
                    "path_id": pid, "sim_time": sim_time, "ok": True,
                    "pred_cam_x": float(centre[0]),
                    "pred_cam_y": float(centre[1]),
                    "pred_cam_z": float(centre[2]),
                    "cam_saved_x": float(cam_saved[0]),
                    "cam_saved_y": float(cam_saved[1]),
                    "cam_saved_z": float(cam_saved[2]),
                    "gt_x": float(targets[i, 0]),
                    "gt_y": float(targets[i, 1]),
                    "inliers": n_in,
                    "euclid_xy_to_gt": d_gt,
                    "euclid_xy_to_cam": d_cam,
                })

    # --- Summarise ---
    def _sum(xs: list[float]) -> dict:
        if not xs:
            return {"median": float("nan"), "mean": float("nan"),
                    "p95": float("nan"), "n": 0}
        x = np.asarray(xs)
        return {
            "median": float(np.median(x)),
            "mean":   float(np.mean(x)),
            "p95":    float(np.percentile(x, 95)),
            "n":      int(x.size),
        }

    report = {
        "run_dir": str(run_dir),
        "split": args.split,
        "used_valid_mask": args.use_valid_mask,
        "reproj_err": args.reproj_err,
        "n_frames_total": sum(p["n_total"] for p in per_path.values()),
        "n_frames_ok":    sum(p["n_ok"]    for p in per_path.values()),
        "n_frames_pnp_fail": n_pnp_fail,
        "per_path": {},
    }

    all_euclid_gt = []
    all_euclid_cam = []
    all_mae_x = []
    all_mae_y = []
    for pid in sorted(per_path):
        d = per_path[pid]
        report["per_path"][str(pid)] = {
            "n_total": d["n_total"],
            "n_ok":    d["n_ok"],
            "mean_inliers": float(np.mean(d["inliers"])) if d["inliers"] else 0.0,
            "euclid_xy_to_gt":  _sum(d["euclid_xy_to_gt"]),
            "euclid_xy_to_cam": _sum(d["euclid_xy_to_cam"]),
            "mae_x_to_gt":      _sum(d["mae_x_to_gt"]),
            "mae_y_to_gt":      _sum(d["mae_y_to_gt"]),
        }
        all_euclid_gt.extend(d["euclid_xy_to_gt"])
        all_euclid_cam.extend(d["euclid_xy_to_cam"])
        all_mae_x.extend(d["mae_x_to_gt"])
        all_mae_y.extend(d["mae_y_to_gt"])

    report["overall"] = {
        "euclid_xy_to_gt":  _sum(all_euclid_gt),
        "euclid_xy_to_cam": _sum(all_euclid_cam),
        "mae_x_to_gt":      _sum(all_mae_x),
        "mae_y_to_gt":      _sum(all_mae_y),
    }

    # --- Print + save ---
    print()
    print("=" * 66)
    print(f"PnP (x, y) evaluation — {args.split} split")
    print("=" * 66)
    o = report["overall"]
    print(
        f"frames: {report['n_frames_ok']}/{report['n_frames_total']} solved "
        f"({n_pnp_fail} PnP failures)"
    )
    e = o["euclid_xy_to_gt"]
    print(f"  vs robot GT (x, y)   median={e['median']:.3f}m  "
          f"mean={e['mean']:.3f}m  p95={e['p95']:.3f}m")
    e = o["euclid_xy_to_cam"]
    print(f"  vs camera-saved (x, y) median={e['median']:.3f}m  "
          f"mean={e['mean']:.3f}m  p95={e['p95']:.3f}m")
    print(f"  MAE x={o['mae_x_to_gt']['mean']:.3f}m  "
          f"y={o['mae_y_to_gt']['mean']:.3f}m")
    print()
    print("per path:")
    for pid, d in report["per_path"].items():
        e = d["euclid_xy_to_gt"]
        print(
            f"  path {pid:>2}: {d['n_ok']}/{d['n_total']} solved  "
            f"inliers={d['mean_inliers']:.0f}  "
            f"med_err={e['median']:.3f}m  p95={e['p95']:.3f}m"
        )

    out_path = run_dir / f"pnp_eval_{args.split}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved: {out_path}")

    # Per-frame CSV for plotting and video generation.
    import csv
    csv_path = run_dir / f"pnp_eval_{args.split}_frames.csv"
    if per_frame_rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_frame_rows)
        print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
