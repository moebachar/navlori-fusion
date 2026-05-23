"""Bake-off: query vs decomposed readout on the honest IPIN trial-out split.

Trains both readouts with identical everything-else, reports val_mae +
subset table + (for decomposed) the mean gate / motion_frac, and compares
each against the dataset's best baseline. This is the "does the decomposition
actually use motion and beat the centroid bar" test.

Run: .venv/Scripts/python.exe scripts/_bakeoff_decomposed.py [--dataset NAME] [--epochs N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fusion.builder import (  # noqa: E402
    build_datamodule, build_encoders, build_model, build_trainer, load_config,
)


def run_one(dataset, readout, epochs, dm, modalities=None):
    cfg = load_config(dataset)
    cfg.model.readout = readout
    if modalities is not None:
        cfg.dataset.modalities = list(modalities)
    encoders, _ = build_encoders(cfg, dm)
    model = build_model(cfg, encoders)
    trainer = build_trainer(cfg, model, dm)
    t0 = time.time()
    hist = trainer.fit(epochs=epochs, verbose=False)
    elapsed = time.time() - t0
    subsets = trainer.evaluate_subsets("val")

    extra = {}
    if readout == "decomposed":
        # Mean gate / motion contribution over the longest val path.
        counts = {}
        for r in dm.val_ds._gt_rows:
            counts[r["path_id"]] = counts.get(r["path_id"], 0) + 1
        pid = max(counts, key=counts.get)
        recs = trainer.log_attribution("val", path_id=pid, save=False,
                                       verbose=False)
        extra["mean_gate"] = round(float(np.mean([r["gate"] for r in recs])), 3)
        extra["mean_motion_frac"] = round(
            float(np.mean([r["motion_frac"] for r in recs])), 3)
    return {
        "readout": readout,
        "best_val_mae": round(hist.best_val_mae, 3),
        "best_epoch": hist.best_epoch,
        "elapsed_s": round(elapsed, 1),
        "subsets": {k: round(v["mae"], 3) for k, v in subsets.items()},
        **extra,
        "run_dir": str(trainer.run_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ipin2024_floor-2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--modalities", default=None,
                    help="comma-separated override, e.g. wifi,imu,odom "
                         "(skips camera/DPVO extraction)")
    args = ap.parse_args()
    mods = args.modalities.split(",") if args.modalities else None

    # Shared datamodule (same data for both runs).
    cfg0 = load_config(args.dataset)
    if mods is not None:
        cfg0.dataset.modalities = mods
    dm = build_datamodule(cfg0)

    # Best baseline for the gate.
    bpath = ROOT / "runs" / "baselines" / args.dataset / "baselines.json"
    best_baseline = None
    if bpath.exists():
        v = json.loads(bpath.read_text()).get("splits", {}).get("val", {})
        best_baseline = (v.get("best"), v.get("best_mae"))

    print(f"=== Bake-off on {args.dataset} ({args.epochs} epochs each) ===",
          flush=True)
    results = []
    for readout in ("query", "decomposed"):
        print(f"\n--- training readout={readout} ---", flush=True)
        r = run_one(args.dataset, readout, args.epochs, dm, modalities=mods)
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    print("\n=== SUMMARY ===", flush=True)
    if best_baseline and best_baseline[1] is not None:
        print(f"best baseline: {best_baseline[0]} @ {best_baseline[1]:.3f}m", flush=True)
    for r in results:
        gap = (r["best_val_mae"] - best_baseline[1]) if (best_baseline and best_baseline[1]) else float("nan")
        verdict = "PASS" if gap < 0 else "FAIL"
        extra = (f"  gate={r.get('mean_gate')}  motion_frac={r.get('mean_motion_frac')}"
                 if r["readout"] == "decomposed" else "")
        print(f"  {r['readout']:11s} val_mae={r['best_val_mae']:.3f}m  "
              f"(vs baseline {gap:+.3f}m [{verdict}]){extra}", flush=True)
        print(f"              subsets: {r['subsets']}", flush=True)

    out = ROOT / "runs" / "bakeoff_decomposed.json"
    out.write_text(json.dumps({"dataset": args.dataset, "epochs": args.epochs,
                               "baseline": best_baseline, "results": results},
                              indent=2))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
