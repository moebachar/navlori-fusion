"""PLAN_08 Step 2 — TartanVO on TartanAir hospital P000.

Wraps the vendored TartanVO python3-branch runner with two
**runtime compat shims** (kept here per Demand #3, never edited into
the vendored source):

1. ``scipy.spatial.transform.Rotation.as_dcm`` was renamed to
   ``as_matrix`` in scipy 1.4. We monkey-patch ``as_dcm = as_matrix``
   to make the vendored ``Datasets/transformation.py:pos_quats2SEs``
   work without source edits.
2. ``cupy.util.memoize`` was removed; the python3 branch
   already has the offending decorator commented out, so no shim
   needed there. ``cupy.cuda.compile_with_cache`` still appears but
   isn't reached on the inference path we use.

Then calls TartanVO's ``vo_trajectory_from_folder``-style code
in-process so we don't have to chdir into the vendored repo from a
subprocess.

Outputs:
- ``runs/overnight/run2_iter_08/tartanvo_hospital.json`` with ATE
  (TartanAir scale-aligned), per-frame errors, and timing.
- ``runs/overnight/run2_iter_08/tartanvo_hospital.png`` traj plot
  (matplotlib agg backend).

Run: ``.venv/Scripts/python.exe scripts/_eval_tartanvo_hospital.py``
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.baselines import apply_tartanvo_shims, TARTANVO_ROOT  # noqa: E402

# All 3 TartanVO runtime shims (scipy as_dcm, numpy linalg.linalg, cupy
# compile_with_cache) live in src.pipeline.baselines._shims; Demand #3
# preserved (no vendored-source edits).
apply_tartanvo_shims()

TARTANVO_DIR = TARTANVO_ROOT
WEIGHTS = "tartanvo_1914.pkl"

SEQ_ROOT = ROOT / "data" / "tartanair_hospital" / "P000"
OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_08"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TARTANVO_DIR))
    os.chdir(TARTANVO_DIR)

    # Now import TartanVO + datasets.
    from Datasets.tartanTrajFlowDataset import TrajFolderDataset  # noqa: E402
    from Datasets.transformation import ses2poses_quat  # noqa: E402
    from Datasets.utils import (  # noqa: E402
        Compose, CropCenter, DownscaleFlow, ToTensor, dataset_intrinsics,
        plot_traj,
    )
    from evaluator.tartanair_evaluator import TartanAirEvaluator  # noqa: E402
    from TartanVO import TartanVO  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    test_dir = str(SEQ_ROOT / "image_left")
    pose_file = str(SEQ_ROOT / "pose_left.txt")

    print(f"loading TartanVO model {WEIGHTS}...", flush=True)
    vo = TartanVO(WEIGHTS)

    focalx, focaly, centerx, centery = dataset_intrinsics("tartanair")
    transform = Compose([CropCenter((448, 640)), DownscaleFlow(), ToTensor()])
    ds = TrajFolderDataset(test_dir, posefile=pose_file, transform=transform,
                            focalx=focalx, focaly=focaly,
                            centerx=centerx, centery=centery)
    print(f"sequence: {len(ds)} frame pairs from {test_dir}", flush=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    motionlist = []
    t0 = time.time()
    for i, sample in enumerate(dl):
        motions, _flow = vo.test_batch(sample)
        motionlist.extend(motions)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(ds)} pairs ({time.time()-t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"done {len(motionlist)} pairs in {elapsed:.1f}s "
          f"({elapsed*1000/max(1, len(motionlist)):.1f} ms/pair)", flush=True)

    poselist = ses2poses_quat(np.array(motionlist))
    # poselist shape: (N+1, 7) — first row is identity start pose.

    # Load GT poses (NED + scalar-last quat).
    gt_raw = np.loadtxt(pose_file)
    # TartanVO uses gt[0:N+1] aligned with poselist (cumulative from id).
    gt_xyz = gt_raw[: len(poselist), :3]
    est_xyz = poselist[:, :3]

    # Sim(3) ATE via Umeyama (the standard alignment).
    n = min(len(gt_xyz), len(est_xyz))
    g, e = gt_xyz[:n], est_xyz[:n]
    mu_g, mu_e = g.mean(0), e.mean(0)
    gc, ec = g - mu_g, e - mu_e
    var_e = (ec ** 2).sum() / n
    H = ec.T @ gc / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R_align = Vt.T @ D @ U.T
    s_align = (S * np.diag(D)).sum() / max(var_e, 1e-12)
    t_align = mu_g - s_align * R_align @ mu_e
    est_aligned = (s_align * (R_align @ ec.T)).T + mu_g

    errs = np.linalg.norm(est_aligned - g, axis=1)
    ate_rmse = float(np.sqrt((errs ** 2).mean()))
    ate_mean = float(errs.mean())
    ate_median = float(np.median(errs))
    ate_p90 = float(np.percentile(errs, 90))
    ate_max = float(errs.max())
    print(f"\n==> TartanVO on hospital_P000:", flush=True)
    print(f"    Umeyama-aligned ATE RMSE = {ate_rmse:.4f} m", flush=True)
    print(f"    mean    = {ate_mean:.4f} m", flush=True)
    print(f"    median  = {ate_median:.4f} m", flush=True)
    print(f"    p90     = {ate_p90:.4f} m", flush=True)
    print(f"    max     = {ate_max:.4f} m", flush=True)
    print(f"    scale   = {s_align:.4f}", flush=True)
    print(f"    n_pairs = {len(motionlist)}, n_aligned = {n}", flush=True)

    # Save artifacts.
    np.savetxt(str(OUT_DIR / "tartanvo_hospital_pred_poses.txt"), poselist)
    np.savetxt(str(OUT_DIR / "tartanvo_hospital_aligned_gt.txt"), g)
    np.savetxt(str(OUT_DIR / "tartanvo_hospital_aligned_est.txt"), est_aligned)
    np.savetxt(str(OUT_DIR / "tartanvo_hospital_errs.txt"), errs)

    try:
        plot_traj(g, est_aligned, vis=False,
                   savefigname=str(OUT_DIR / "tartanvo_hospital_traj.png"),
                   title=f"TartanVO hospital_P000  ATE={ate_rmse:.4f}m")
    except Exception as ee:
        print(f"plot failed: {ee}", flush=True)

    out = {
        "method": "TartanVO (python3 branch, tartanvo_1914.pkl)",
        "dataset": "TartanAir hospital P000",
        "n_pairs": int(len(motionlist)),
        "n_aligned_frames": int(n),
        "elapsed_s": elapsed,
        "latency_ms_per_pair": elapsed * 1000 / max(1, len(motionlist)),
        "ate_umeyama": {
            "rmse_m": ate_rmse, "mean_m": ate_mean, "median_m": ate_median,
            "p90_m": ate_p90, "max_m": ate_max, "scale": float(s_align),
        },
    }
    with open(OUT_DIR / "tartanvo_hospital.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'tartanvo_hospital.json'}", flush=True)


if __name__ == "__main__":
    main()
