"""PLAN_19 — load existing CNN1D/LSTM-attn IMUWiFine checkpoints and regenerate JSON.

Used to recover from the int64-JSON failure in `_train_imuwifine_2mod_arch.py`
without re-running training. Loads the latest fusion_* checkpoint under
``runs/overnight/run2_iter_19/<arch>/`` and emits the full result JSON +
per-path plots.

Run: ``.venv/Scripts/python.exe scripts/_iter19_postproc.py --arch cnn1d``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.bakeoff import CANDIDATES  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, extract_vision_tokens, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

OUT_DIR = ROOT / "runs" / "overnight" / "run2_iter_19"


def per_path_distribution(preds, gts, pid):
    errs = np.linalg.norm(preds - gts, axis=1)
    per_path = {}
    for p in np.unique(pid):
        m = pid == p; e = errs[m]
        per_path[int(p)] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)), "max": float(e.max()), "n": int(len(e)),
        }
    return {
        "aggregate": {
            "mean": float(errs.mean()), "median": float(np.median(errs)),
            "p25": float(np.percentile(errs, 25)), "p75": float(np.percentile(errs, 75)),
            "p90": float(np.percentile(errs, 90)), "max": float(errs.max()),
        },
        "per_path": per_path,
    }


def per_traj_smoothness(preds, gts, pid):
    per_path = {}
    for p in np.unique(pid):
        m = pid == p
        if m.sum() < 5: continue
        pp, gg = preds[m], gts[m]
        dp = np.linalg.norm(np.diff(pp, axis=0), axis=1)
        dg = np.linalg.norm(np.diff(gg, axis=0), axis=1)
        if dp.std() < 1e-9 or dg.std() < 1e-9:
            per_path[int(p)] = 0.0
        else:
            per_path[int(p)] = float(np.corrcoef(dp, dg)[0, 1])
    rs = list(per_path.values())
    return {"per_path": per_path,
             "median_r": float(np.median(rs)) if rs else 0.0,
             "min_r": float(min(rs)) if rs else 0.0,
             "max_r": float(max(rs)) if rs else 0.0}


def plot_path(preds, gts, pid, out_path, suffix):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gts[:, 0], gts[:, 1], "k-", label="GT", lw=1.5)
    ax.plot(preds[:, 0], preds[:, 1], "r-", label="pred", lw=1.0, alpha=0.75)
    ax.scatter(gts[0, 0], gts[0, 1], c="green", s=40, marker="o", label="start")
    ax.set_aspect("equal"); ax.set_title(f"path_{pid:02d} {suffix}")
    ax.legend(); ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(CANDIDATES.keys()))
    args = ap.parse_args()

    arch_dir = OUT_DIR / args.arch
    fusion_dirs = sorted(arch_dir.glob("fusion_*"))
    if not fusion_dirs:
        raise SystemExit(f"No fusion_* dir under {arch_dir}")
    run_path = fusion_dirs[-1]
    print(f"loading {args.arch} from {run_path}", flush=True)

    cfg = load_config("imuwifine")
    cfg.dataset.modalities = ["wifi", "imu"]
    cfg.temporal.n_instants = 4
    cfg.data.batch_size = 128

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    extra = extract_vision_tokens(dm, vision, device="cuda") if vision is not None else {}

    incumbent_kwargs = dict(
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.depth),
        n_heads=int(cfg.model.n_heads),
        ff_mult=int(cfg.model.ff_mult),
        dropout=float(cfg.model.dropout),
        use_time=bool(cfg.model.use_time),
        readout=str(cfg.model.readout),
        absolute_modalities=list(cfg.model.get("absolute_modalities", None) or ["wifi"]),
    )
    torch.manual_seed(42)
    model = CANDIDATES[args.arch](incumbent_kwargs, encs)
    state = torch.load(run_path / "model.pt", weights_only=True, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded ({n_params/1e6:.2f} M params)", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities),
        extra_inputs=extra,
        lr=float(cfg.train.lr),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        batch_size=128,
        run_dir=str(OUT_DIR / f"{args.arch}_postproc_skip"),
    )

    pred_v, gt_v = trainer.predict("val")
    pred_t, gt_t = trainer.predict("test")
    val_mae = float(torch.linalg.norm(pred_v - gt_v, dim=1).mean())
    test_mae = float(torch.linalg.norm(pred_t - gt_t, dim=1).mean())
    print(f"sanity: val {val_mae:.3f}  test {test_mae:.3f}", flush=True)

    subsets_val = trainer.evaluate_all_subsets("val")
    subsets_test = trainer.evaluate_all_subsets("test")
    for k, v in subsets_test.items():
        print(f"  test {k:30s} -> mae={v['mae']:.3f}", flush=True)

    pred = pred_t.numpy(); gt = gt_t.numpy()
    pids = np.array([r["path_id"] for r in dm.test_ds._gt_rows])[:len(pred)]
    test_dist = per_path_distribution(pred, gt, pids)
    test_smooth = per_traj_smoothness(pred, gt, pids)
    print(f"\n  smoothness median r = {test_smooth['median_r']:.3f}", flush=True)

    path_lens = {int(p): int((pids == p).sum()) for p in np.unique(pids)}
    top5 = [int(p) for p, _ in sorted(path_lens.items(), key=lambda kv: -kv[1])[:5]]
    print(f"  top5 longest test paths: {top5}", flush=True)
    for p in top5:
        m = pids == p
        if m.sum() > 5:
            plot_path(pred[m], gt[m], p,
                      OUT_DIR / "test_paths" / f"{args.arch}_path_{p:02d}.png",
                      f"({args.arch} IMUWiFine)")

    out = {
        "arch": args.arch,
        "checkpoint": str(run_path.relative_to(ROOT)),
        "n_params": int(n_params),
        "config": {"dataset": "imuwifine",
                    "modalities": list(model.modalities),
                    "n_instants": int(cfg.temporal.n_instants),
                    "batch_size": int(cfg.data.batch_size)},
        "sanity": {"val_mae": val_mae, "test_mae": test_mae},
        "subsets_val": {k: {"mae": float(v["mae"])} for k, v in subsets_val.items()},
        "subsets_test": {k: {"mae": float(v["mae"])} for k, v in subsets_test.items()},
        "test_dist": test_dist,
        "test_smoothness": test_smooth,
        "test_top5_longest_paths": top5,
    }
    with open(OUT_DIR / f"{args.arch}_imuwifine.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_DIR / f'{args.arch}_imuwifine.json'}", flush=True)


if __name__ == "__main__":
    main()
