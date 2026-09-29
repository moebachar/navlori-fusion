"""Log GT-vs-predicted trajectories (+ ablation) of a FINISHED run to W&B.
Rebuilds the pipeline, loads the run's model.pt, predicts the test split, and
logs one floor-plan overlay per test path as wandb.Image, plus a per-path MAE
bar and the modality ablation. Does NOT retrain; safe to run post-hoc.

Run: .venv/Scripts/python.exe scripts/_wandb_eval.py --run replay_f2_relstrong60 \
        --readout query --wifi-norm rel_strong --name query_relstrong_60ep
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.fusion.bakeoff import CANDIDATES
from src.pipeline.fusion.builder import (build_datamodule, build_encoders,
                                         extract_vision_tokens_fast, load_config)
from src.pipeline.training.fusion_trainer import FusionTrainer

DATASET = "iln20_5d27099f_F2_replay"
WORLD = ROOT / "src" / "simulation" / "worlds" / "iln20_5d27099f_F2.wbt"


def walls():
    out = []
    wt = WORLD.read_text(encoding="utf-8") if WORLD.is_file() else ""
    for m in re.finditer(r"DEF P?WALL_\d+ Solid \{\s*\n\s*translation\s+(-?[\d.]+)\s+"
                         r"(-?[\d.]+)\s+(-?[\d.]+)\s*\n\s*rotation 0 0 1 (-?[\d.]+)", wt):
        tail = wt[m.end():m.end() + 400]
        mb = re.search(r"geometry Box \{ size (-?[\d.]+)", tail)
        if not mb:
            continue
        cx, cy, yaw, half = float(m.group(1)), float(m.group(2)), float(m.group(4)), float(mb.group(1)) / 2
        dx, dy = math.cos(yaw) * half, math.sin(yaw) * half
        out.append(((cx - dx, cx + dx), (cy - dy, cy + dy)))
    return out


def fig_path(gp, pp, pid, mae, W):
    fig, ax = plt.subplots(figsize=(6, 6))
    for xs, ys in W:
        ax.plot(xs, ys, "-", color="0.8", lw=0.7, zorder=1)
    ax.plot(gp[:, 0], gp[:, 1], "-", color="0.35", lw=2.2, label="Ground truth", zorder=3)
    ax.plot(pp[:, 0], pp[:, 1], "-", color="tab:red", lw=1.5, label="Predicted", zorder=4)
    ax.scatter(gp[0, 0], gp[0, 1], c="green", s=60, zorder=5, label="Start")
    ax.set_aspect("equal"); ax.legend(loc="best"); ax.axis("off")
    ax.set_title(f"path {pid} — mean err {mae:.2f} m")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir under runs/")
    ap.add_argument("--readout", default="query", choices=["query", "decomposed", "cls"])
    ap.add_argument("--wifi-norm", default="rel_strong")
    ap.add_argument("--wifi", default="set_transformer")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    rundir = ROOT / "runs" / args.run
    ckpt = next((rundir / "trainer").glob("fusion_*/model.pt"))
    res = json.loads((rundir / "results.json").read_text()) if (rundir / "results.json").is_file() else {}

    cfg = load_config(DATASET)
    cfg.dataset.wifi_encoder_type = args.wifi
    cfg.dataset.preprocessing.wifi_norm = args.wifi_norm
    cfg.data.batch_size = 96
    cfg.temporal.n_instants = args.k
    cfg.model.readout = args.readout
    cfg.model.absolute_modalities = ["wifi"]

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    extra = (extract_vision_tokens_fast(dm, vision, device="cuda", subsample_stride=5)
             if vision is not None else None)
    kw = dict(embed_dim=int(cfg.model.embed_dim), depth=int(cfg.model.depth),
              n_heads=int(cfg.model.n_heads), ff_mult=int(cfg.model.ff_mult),
              dropout=float(cfg.model.dropout), use_time=bool(cfg.model.use_time),
              readout=args.readout, absolute_modalities=["wifi"])
    model = CANDIDATES["incumbent"](kw, encs)
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities), extra_inputs=extra,
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        huber_delta=float(cfg.train.huber_delta), grad_clip=float(cfg.train.grad_clip),
        patience=int(cfg.train.patience), batch_size=96,
        modality_dropout=0.0, instant_dropout=0.0,
        n_instants=args.k, instant_stride=int(cfg.temporal.instant_stride),
        modality_balanced_loss=bool(cfg.train.modality_balanced_loss),
        modality_balanced_weight=float(cfg.train.modality_balanced_weight),
        aux_abs_weight=float(cfg.train.aux_abs_weight), run_dir=str(rundir / "trainer"))

    pred_t, gt_t = trainer.predict("test")
    pred, gt = pred_t.numpy(), gt_t.numpy()
    rows = dm.test_ds._gt_rows[:len(pred)]
    pid = np.array([r["path_id"] for r in rows])
    ts = np.array([float(r.get("time", r.get("sim_time", i))) for i, r in enumerate(rows)])
    errs = np.linalg.norm(pred - gt, axis=1)
    W = walls()

    import wandb
    run = wandb.init(project="navlori-replay-f2",
                     name=(args.name or args.run) + "_eval",
                     config={"readout": args.readout, "wifi_norm": args.wifi_norm,
                             "source_run": args.run})
    per_path = {}
    imgs = []
    for p in sorted(np.unique(pid)):
        m = pid == p; order = np.argsort(ts[m])
        gp, pp = gt[m][order], pred[m][order]
        mae = float(errs[m].mean()); per_path[int(p)] = mae
        f = fig_path(gp, pp, int(p), mae, W)
        imgs.append(wandb.Image(f, caption=f"path {int(p)}  {mae:.2f} m"))
        plt.close(f)
    run.log({"gt_vs_pred": imgs})
    # per-path MAE bar
    tbl = wandb.Table(data=[[str(k), v] for k, v in sorted(per_path.items())],
                      columns=["path", "mae_m"])
    run.log({"per_path_mae": wandb.plot.bar(tbl, "path", "mae_m", title="Per-path test MAE (m)")})
    # ablation bar (from results.json if present)
    if res.get("ablation_test"):
        at = wandb.Table(data=[[k, v] for k, v in sorted(res["ablation_test"].items(), key=lambda x: x[1])],
                         columns=["subset", "mae_m"])
        run.log({"ablation": wandb.plot.bar(at, "subset", "mae_m", title="Modality ablation test MAE (m)")})
    run.summary["test_mae"] = float(errs.mean())
    run.summary["test_median"] = float(np.median(errs))
    run.finish()
    print(f"logged {len(imgs)} GT-vs-pred plots + ablation to W&B (test MAE {errs.mean():.3f} m)", flush=True)


if __name__ == "__main__":
    main()
