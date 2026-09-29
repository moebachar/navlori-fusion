"""MINIMUM test of the anchor+correction architecture (idea #1) on the F2
replay dataset. Same pipeline/encoders/hyperparams as the 4-mod baseline
(readout='query', test MAE 7.11 m), but readout='decomposed':

    pred = p_abs(WiFi tokens) + gate * delta(motion tokens)

Structurally forbids relative motion from producing absolute position.
Run: .venv/Scripts/python.exe scripts/_test_decomposed.py [--epochs 20]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.fusion.bakeoff import CANDIDATES
from src.pipeline.fusion.builder import (build_datamodule, build_encoders,
                                         extract_vision_tokens_fast, load_config)
from src.pipeline.training.fusion_trainer import FusionTrainer

DATASET = "iln20_5d27099f_F2_replay"
BASELINE = 7.11
OUT = ROOT / "runs" / "replay_f2_decomposed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--wifi-norm", default="rel_strong",
                    choices=["raw", "rel_strong", "whiten"])
    args = ap.parse_args()
    global OUT
    if args.wifi_norm != "raw":
        OUT = ROOT / "runs" / f"replay_f2_decomposed_{args.wifi_norm}"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_config(DATASET)
    cfg.dataset.wifi_encoder_type = "set_transformer"
    cfg.dataset.preprocessing.wifi_norm = args.wifi_norm
    cfg.data.batch_size = 96
    cfg.temporal.n_instants = 4
    cfg.model.readout = "decomposed"
    cfg.model.absolute_modalities = ["wifi"]
    print(f"=== decomposed (anchor+correction) mods={list(cfg.dataset.modalities)} "
          f"{args.epochs} ep ; baseline query={BASELINE} m ===", flush=True)

    dm = build_datamodule(cfg)
    encs, vision = build_encoders(cfg, dm)
    extra = (extract_vision_tokens_fast(dm, vision, device="cuda", subsample_stride=5)
             if vision is not None else None)
    kw = dict(embed_dim=int(cfg.model.embed_dim), depth=int(cfg.model.depth),
              n_heads=int(cfg.model.n_heads), ff_mult=int(cfg.model.ff_mult),
              dropout=float(cfg.model.dropout), use_time=bool(cfg.model.use_time),
              readout="decomposed", absolute_modalities=["wifi"])
    torch.manual_seed(42)
    model = CANDIDATES["incumbent"](kw, encs)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M "
          f"readout={model.readout}", flush=True)

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
        aux_abs_weight=float(cfg.train.aux_abs_weight), run_dir=str(OUT / "trainer"))

    t0 = time.time()
    hist = trainer.fit(epochs=args.epochs, verbose=True)
    elapsed = time.time() - t0
    print(f"  best val MAE {hist.best_val_mae:.3f} m (ep {hist.best_epoch}), "
          f"{elapsed:.0f}s", flush=True)

    caps = {"p_abs": [], "delta": [], "gate": []}
    h1 = model.anchor_head.register_forward_hook(
        lambda m, i, o: caps["p_abs"].append(o.detach().cpu()))
    h2 = model.motion_head.register_forward_hook(
        lambda m, i, o: caps["delta"].append(o.detach().cpu()))
    h3 = model.gate.register_forward_hook(
        lambda m, i, o: caps["gate"].append(torch.sigmoid(o.detach().cpu())))
    pred_t, gt_t = trainer.predict("test")
    h1.remove(); h2.remove(); h3.remove()
    pred, gt = pred_t.numpy(), gt_t.numpy()
    p_abs = torch.cat(caps["p_abs"]).numpy()
    delta = torch.cat(caps["delta"]).numpy()
    gate = torch.cat(caps["gate"]).numpy().reshape(-1)

    recon = p_abs + gate[:, None] * delta
    align_err = float(np.abs(recon - pred).mean())

    rows = dm.test_ds._gt_rows[:len(pred)]
    pid = np.array([r["path_id"] for r in rows])
    errs = np.linalg.norm(pred - gt, axis=1)
    errs_abs = np.linalg.norm(p_abs - gt, axis=1)
    test_mae = float(errs.mean()); test_med = float(np.median(errs))
    anchor_mae = float(errs_abs.mean())
    corr_mag = float(np.linalg.norm(gate[:, None] * delta, axis=1).mean())

    per_path = {int(p): float(errs[pid == p].mean()) for p in np.unique(pid)}
    macro = float(np.mean(list(per_path.values())))

    print("\n" + "=" * 56, flush=True)
    print(f"  align check |recon-pred|={align_err:.4f} (should be ~0)", flush=True)
    print(f"  DECOMPOSED test MAE = {test_mae:.3f} m  median {test_med:.3f}  "
          f"macro {macro:.3f}", flush=True)
    print(f"  baseline   query    = {BASELINE:.3f} m", flush=True)
    print(f"  anchor-only (p_abs) = {anchor_mae:.3f} m", flush=True)
    print(f"  gate: mean {gate.mean():.3f}  p50 {np.median(gate):.3f}  "
          f"p90 {np.percentile(gate,90):.3f}  |correction|={corr_mag:.3f} m", flush=True)
    print("  per-path MAE:", {k: round(v, 2) for k, v in sorted(per_path.items())},
          flush=True)

    out = {"readout": "decomposed", "epochs": args.epochs,
           "best_val_mae": float(hist.best_val_mae), "best_epoch": int(hist.best_epoch),
           "test_mae": test_mae, "test_median": test_med, "macro_avg": macro,
           "anchor_only_mae": anchor_mae, "baseline_query_mae": BASELINE,
           "gate_mean": float(gate.mean()), "gate_p50": float(np.median(gate)),
           "gate_p90": float(np.percentile(gate, 90)),
           "correction_mag_m": corr_mag, "align_err": align_err,
           "per_path_test_mae": per_path, "elapsed_s": elapsed}
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ndone -> {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
