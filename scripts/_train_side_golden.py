"""Train the classical FusionTransformer on side_golden (12 real TurtleBot3 runs) [copy of _train_replay_f2.py]
and produce: training curves, final numbers, a modality ABLATION (subset
eval), per-path GT-vs-pred plots, and a GT-vs-pred trajectory VIDEO.

Run: .venv/Scripts/python.exe scripts/_train_side_golden.py [--epochs 80] [--no-camera]
Outputs under runs/side_golden/fusion_{3mod,4mod}[_tag]/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.bakeoff import CANDIDATES  # noqa: E402
from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, extract_vision_tokens,
    extract_vision_tokens_fast, load_config,
)
from src.pipeline.training.fusion_trainer import FusionTrainer  # noqa: E402

DATASET = "side_golden"
OUT = ROOT / "runs" / "side_golden" / "fusion"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_curve(hist, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist.val_mae, label="val MAE (m)", color="tab:blue", lw=1.8)
    ax.axvline(hist.best_epoch, color="k", ls="--", alpha=0.5,
               label=f"best ep {hist.best_epoch} = {hist.best_val_mae:.2f} m")
    ax.set_xlabel("epoch"); ax.set_ylabel("val MAE (m)"); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(hist.train_loss, color="tab:orange", alpha=0.5, label="train loss")
    ax2.plot(hist.val_loss, color="tab:red", alpha=0.4, label="val loss")
    ax2.set_ylabel("Huber loss")
    ax.legend(loc="upper right"); ax2.legend(loc="lower left")
    ax.set_title("side_golden — training"); fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(subsets, out):
    plt = _plt()
    items = sorted(subsets.items(), key=lambda kv: kv[1]["mae"])
    names = [k for k, _ in items]; maes = [v["mae"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
    ax.barh(range(len(names)), maes, color="tab:blue")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("test MAE (m)")
    ax.set_title("Modality ablation (test) — side_golden")
    for i, m in enumerate(maes):
        ax.text(m, i, f" {m:.2f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_path(pred, gt, pid, out, mae):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt[:, 0], gt[:, 1], "k-", label="GT", lw=1.8)
    ax.plot(pred[:, 0], pred[:, 1], "r-", label="pred", lw=1.1, alpha=0.8)
    ax.scatter(gt[0, 0], gt[0, 1], c="green", s=45, marker="o", zorder=5)
    ax.set_aspect("equal"); ax.legend()
    ax.set_title(f"path_{pid:02d}  mean err {mae:.2f} m"); ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def render_gt_pred_video(pred, gt, pid, out_mp4, fps=15):
    """Animate GT vs predicted position growing over time."""
    import cv2
    plt = _plt()
    xmin, xmax = min(gt[:, 0].min(), pred[:, 0].min()), max(gt[:, 0].max(), pred[:, 0].max())
    ymin, ymax = min(gt[:, 1].min(), pred[:, 1].min()), max(gt[:, 1].max(), pred[:, 1].max())
    pad = 1.0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    step = max(1, len(gt) // 300)          # cap ~300 frames
    for k in range(1, len(gt) + 1, step):
        ax.clear()
        ax.plot(gt[:k, 0], gt[:k, 1], "k-", lw=1.8, label="GT")
        ax.plot(pred[:k, 0], pred[:k, 1], "r-", lw=1.2, label="pred")
        ax.plot(gt[k - 1, 0], gt[k - 1, 1], "ko", ms=7)
        ax.plot(pred[k - 1, 0], pred[k - 1, 1], "ro", ms=7)
        err = np.linalg.norm(pred[k - 1] - gt[k - 1])
        ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal"); ax.legend(loc="upper right")
        ax.set_title(f"path_{pid:02d}  GT vs predicted   err {err:.2f} m")
        ax.grid(True, alpha=0.3)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release(); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--arch", default="incumbent", choices=list(CANDIDATES.keys()))
    ap.add_argument("--no-camera", action="store_true",
                    help="drop the camera modality (skips the slow DPVO token "
                         "extraction) -- fast WiFi+IMU+odom result")
    ap.add_argument("--k", type=int, default=4, help="temporal instants (K)")
    ap.add_argument("--wifi", default=None, choices=["wifi_net", "set_transformer"],
                    help="override WiFi encoder (wifi_net is much faster on 581 APs)")
    ap.add_argument("--tag", default=None, help="run-dir suffix")
    ap.add_argument("--batch", type=int, default=None, help="override batch size")
    ap.add_argument("--wifi-norm", default=None,
                    choices=["raw", "rel_strong", "whiten"],
                    help="override WiFi preprocessing (rel_strong = session-invariant)")
    args = ap.parse_args()
    global OUT
    base = "fusion_3mod" if args.no_camera else "fusion_4mod"
    OUT = ROOT / "runs" / "side_golden" / (base + (f"_{args.tag}" if args.tag else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_paths").mkdir(exist_ok=True)

    cfg = load_config(DATASET)
    if args.no_camera:
        cfg.dataset.modalities = [m for m in cfg.dataset.modalities if m != "camera"]
    if args.wifi:
        cfg.dataset.wifi_encoder_type = args.wifi
    if args.wifi_norm:
        cfg.dataset.preprocessing.wifi_norm = args.wifi_norm
    if args.batch:
        cfg.data.batch_size = args.batch
    cfg.temporal.n_instants = args.k
    print(f"=== side_golden training arch={args.arch} mods={list(cfg.dataset.modalities)} "
          f"{args.epochs} ep ===", flush=True)
    dm = build_datamodule(cfg)
    print(f"  samples train/val/test: {len(dm.train_ds)}/{len(dm.val_ds)}/{len(dm.test_ds)}",
          flush=True)

    encs, vision = build_encoders(cfg, dm)
    # reuse the 1 Hz (stride-5) DPVO cache built by scripts/_cache_camera.py:
    # extract_vision_tokens_fast reads {split}_s5.pt -> cache hit, no DPVO reruns
    extra = (extract_vision_tokens_fast(dm, vision, device="cuda", subsample_stride=5)
             if vision is not None else None)

    kw = dict(embed_dim=int(cfg.model.embed_dim), depth=int(cfg.model.depth),
              n_heads=int(cfg.model.n_heads), ff_mult=int(cfg.model.ff_mult),
              dropout=float(cfg.model.dropout), use_time=bool(cfg.model.use_time),
              readout=str(cfg.model.readout),
              absolute_modalities=list(cfg.model.get("absolute_modalities", None) or ["wifi"]))
    torch.manual_seed(42)
    model = CANDIDATES[args.arch](kw, encs)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)

    trainer = FusionTrainer(
        model=model, dm=dm, modalities=list(model.modalities), extra_inputs=extra,
        lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        huber_delta=float(cfg.train.huber_delta), grad_clip=float(cfg.train.grad_clip),
        patience=int(cfg.train.patience), batch_size=int(cfg.data.batch_size),
        modality_dropout=float(cfg.train.modality_dropout),
        instant_dropout=float(cfg.train.instant_dropout),
        n_instants=int(cfg.temporal.n_instants),
        instant_stride=int(cfg.temporal.instant_stride),
        modality_balanced_loss=bool(cfg.train.modality_balanced_loss),
        modality_balanced_weight=float(cfg.train.modality_balanced_weight),
        aux_abs_weight=float(cfg.train.aux_abs_weight),
        run_dir=str(OUT / "trainer"))

    t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  best val MAE {hist.best_val_mae:.3f} m (ep {hist.best_epoch}), "
          f"{elapsed:.0f}s", flush=True)

    # ablation (modality subsets) on val + test
    sub_val = trainer.evaluate_all_subsets("val")
    sub_test = trainer.evaluate_all_subsets("test")
    print("\n  ABLATION (test MAE by modality subset):", flush=True)
    for n, d in sorted(sub_test.items(), key=lambda kv: kv[1]["mae"]):
        print(f"    {n:32s} {d['mae']:.3f} m", flush=True)

    # per-path test predictions
    pred_t, gt_t = trainer.predict("test")
    pred, gt = pred_t.numpy(), gt_t.numpy()
    rows = dm.test_ds._gt_rows[:len(pred)]
    pid = np.array([r["path_id"] for r in rows])
    ts = np.array([float(r.get("time", r.get("sim_time", i)))
                   for i, r in enumerate(rows)])
    errs = np.linalg.norm(pred - gt, axis=1)
    test_mae = float(errs.mean())
    print(f"\n  TEST MAE (4-mod) = {test_mae:.3f} m  median {np.median(errs):.3f}",
          flush=True)

    per_path = {}
    for p in np.unique(pid):
        m = pid == p
        per_path[int(p)] = float(errs[m].mean())
    ranked = sorted(per_path.items(), key=lambda kv: kv[1])
    vid_pid = ranked[len(ranked) // 2][0]

    # ── write numbers FIRST (never lose them to a plotting glitch) ──
    out = {
        "dataset": DATASET, "arch": args.arch, "epochs": args.epochs,
        "modalities": list(cfg.dataset.modalities), "k": int(cfg.temporal.n_instants),
        "wifi_encoder": str(cfg.dataset.get("wifi_encoder_type")),
        "samples": {"train": len(dm.train_ds), "val": len(dm.val_ds),
                    "test": len(dm.test_ds)},
        "best_val_mae": float(hist.best_val_mae), "best_epoch": int(hist.best_epoch),
        "test_mae": test_mae, "test_median": float(np.median(errs)),
        "elapsed_s": elapsed,
        "ablation_test": {k: float(v["mae"]) for k, v in sub_test.items()},
        "ablation_val": {k: float(v["mae"]) for k, v in sub_val.items()},
        "per_path_test_mae": per_path, "video_path_id": int(vid_pid),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "pred_test.npz", pred=pred, gt=gt, path_id=pid, sim_time=ts)
    print(f"  wrote results.json", flush=True)

    # ── artifacts: each wrapped so one failure can't drop the rest ──
    for name, fn in [
        ("training_curve", lambda: plot_curve(hist, OUT / "training_curve.png")),
        ("ablation", lambda: plot_ablation(sub_test, OUT / "ablation_test.png")),
    ]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {name} plot failed: {e}", flush=True)
    for p, mae in ranked:
        try:
            m = pid == p; order = np.argsort(ts[m])
            plot_path(pred[m][order], gt[m][order], p,
                      OUT / "test_paths" / f"path_{p:02d}.png", mae)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] path {p} plot failed: {e}", flush=True)
    try:
        m = pid == vid_pid; order = np.argsort(ts[m])
        render_gt_pred_video(pred[m][order], gt[m][order], vid_pid,
                             OUT / f"gt_vs_pred_path_{vid_pid:02d}.mp4")
        print(f"  video: gt_vs_pred_path_{vid_pid:02d}.mp4", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] video failed: {e}", flush=True)
    print(f"\ndone -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
