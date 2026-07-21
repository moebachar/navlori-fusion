"""H6: Recompute MSILN site1/B1 test errors under alternate metrics.

Loads the saved FusionTransformer (best Phase B/C model) and reports
per-sample Euclidean error under multiple aggregations:
    - mean      (= our reported MAE)
    - median    (50th pct)
    - 75th / 90th pct
    - RMSE
    - ATE global (after global Umeyama similarity alignment)
    - ATE per-path (after per-path Umeyama similarity alignment)

WiFi+IMU model only, so no vision extraction. Uses the actual builder API
(extract_vision_tokens has signature (dm, vision_encoder, device)).
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
)

RUN_DIR = REPO / 'runs' / 'main_table' / 'msiln_site1_b1' / 'transformer' / 'fusion_20260604_145150'
CKPT = RUN_DIR / 'model.pt'


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate s,R,t such that s*R@src + t ~ dst (least squares)."""
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
    aligned = umeyama(preds, gt)
    err_g = np.linalg.norm(aligned - gt, axis=1)
    out['ate_global_mean_m'] = float(err_g.mean())
    out['ate_global_rmse_m'] = float(np.sqrt((err_g ** 2).mean()))
    out['ate_global_median_m'] = float(np.median(err_g))
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
        out['n_paths'] = int(len(np.unique(path_ids)))
    print(f'== {name} ==')
    for k, v in out.items():
        print(f'  {k:>26s}: {v}')
    return out


def main():
    cfg = load_config('msiln_site1_b1')
    cfg.temporal.n_instants = 4
    cfg.temporal.instant_stride = 9
    cfg.train.modality_dropout = 0.0
    cfg.train.instant_dropout = 0.0
    dm = build_datamodule(cfg)

    encoders, vision = build_encoders(cfg, dm)
    extras = None
    if vision is not None:
        from src.pipeline.fusion.builder import extract_vision_tokens
        extras = extract_vision_tokens(dm, vision, device='cuda')

    model = build_model(cfg, encoders)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)

    trainer = build_trainer(cfg, model, dm, extra_inputs=extras,
                            run_dir=str(REPO / 'runs' / 'tmp_h6_eval'))

    # Try to grab per-sample path_ids if datamodule exposes them.
    # FusionDataModule keeps test_ds with .path_ids if available.
    def maybe_path_ids(split: str):
        ds_map = {'train': dm.train_ds, 'val': dm.val_ds, 'test': dm.test_ds}
        ds = ds_map.get(split)
        if ds is None:
            return None
        for attr in ('path_ids', 'path_id', 'paths'):
            v = getattr(ds, attr, None)
            if v is not None:
                arr = np.asarray(v)
                if arr.ndim == 1:
                    return arr
        return None

    results = {}
    for split in ('val', 'test'):
        pv, tv = trainer.predict(split)
        preds = pv.numpy(); gt = tv.numpy()
        pids = maybe_path_ids(split)
        results[split] = report(f'transformer / {split}', preds, gt, path_ids=pids)

    out_path = REPO / 'experiments' / 'diagnostics' / 'h6_metric_recompute_transformer.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
