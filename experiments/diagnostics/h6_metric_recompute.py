"""H6: Recompute MSILN site1/B1 test errors under alternate metrics.

We load the saved FusionTransformer (best Phase B/C model) and the best
'mamba_imu_place_conditioned_encoder' run if its checkpoint exists, and
report per-sample Euclidean error under several aggregations:
    - mean      (= our reported MAE)
    - median    (50th pct)
    - 75th pct
    - 90th pct
    - ATE       (RMSE of (pred - gt) after Umeyama rigid alignment)

We also compute an "ATE rigid-aligned" estimate by per-path Umeyama
alignment (since published papers occasionally evaluate after such
alignment, especially robotics/VIO crowd).
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path('x:/navlori-fusion')
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from src.pipeline.fusion.builder import (
    load_config, build_datamodule, build_encoders, build_model, build_trainer,
    extract_vision_tokens,
)

RUN_DIR = REPO / 'runs' / 'main_table' / 'msiln_site1_b1' / 'transformer' / 'fusion_20260604_145150'
CKPT = RUN_DIR / 'model.pt'


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Estimate s,R,t such that  s*R@src + t ~ dst (least squares).

    src, dst: (N, 2) arrays.  Returns aligned src (N,2).
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    if len(src) < 2:
        return src.copy()
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s; dc = dst - mu_d
    H = sc.T @ dc / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    var_s = (sc ** 2).sum() / len(src)
    s = (S * np.array([1.0, d])).sum() / max(var_s, 1e-9)
    t = mu_d - s * R @ mu_s
    return (s * (R @ src.T)).T + t


def report(name, preds: np.ndarray, gt: np.ndarray, path_ids=None):
    err = np.linalg.norm(preds - gt, axis=1)
    out = {
        'n': int(len(err)),
        'mean_MAE_m': float(err.mean()),
        'median_m': float(np.median(err)),
        'p75_m': float(np.percentile(err, 75)),
        'p90_m': float(np.percentile(err, 90)),
        'rmse_m': float(np.sqrt((err ** 2).mean())),
    }
    # Global Umeyama (treat full test set as one trajectory)
    aligned = umeyama(preds, gt)
    err_g = np.linalg.norm(aligned - gt, axis=1)
    out['ate_global_mean_m'] = float(err_g.mean())
    out['ate_global_rmse_m'] = float(np.sqrt((err_g ** 2).mean()))
    out['ate_global_median_m'] = float(np.median(err_g))
    # Per-path Umeyama
    if path_ids is not None:
        aligned_pp = np.zeros_like(preds)
        for pid in np.unique(path_ids):
            mask = path_ids == pid
            if mask.sum() >= 2:
                aligned_pp[mask] = umeyama(preds[mask], gt[mask])
            else:
                aligned_pp[mask] = preds[mask]
        err_p = np.linalg.norm(aligned_pp - gt, axis=1)
        out['ate_perpath_mean_m'] = float(err_p.mean())
        out['ate_perpath_rmse_m'] = float(np.sqrt((err_p ** 2).mean()))
        out['ate_perpath_median_m'] = float(np.median(err_p))
    print(f'== {name} ==')
    for k, v in out.items():
        print(f'  {k:>26s}: {v}')
    return out


def main():
    cfg = load_config('msiln_site1_b1')
    # Use the meta-recorded train config for shape parity
    cfg.temporal.n_instants = 4
    cfg.temporal.instant_stride = 9
    cfg.train.modality_dropout = 0.0
    cfg.train.instant_dropout = 0.0
    dm = build_datamodule(cfg)

    encoders = build_encoders(cfg, dm)
    extras = extract_vision_tokens(cfg, dm, encoders)
    model = build_model(cfg, encoders)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=False)

    trainer = build_trainer(cfg, model, dm, extra_inputs=extras,
                             run_dir=str(REPO / 'runs' / 'tmp_h6_eval'))
    # device
    device = next(trainer.model.parameters()).device
    print(f'device={device}')

    results = {}
    for split in ('val', 'test'):
        pv, tv = trainer.predict(split)
        preds = pv.numpy(); gt = tv.numpy()
        # We don't have direct path_ids here; per-path align is a nice-to-have.
        results[split] = report(f'transformer / {split}', preds, gt)
    out_path = REPO / 'experiments' / 'diagnostics' / 'h6_metric_recompute_transformer.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
