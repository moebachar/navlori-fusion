"""M1 + M2 batch driver: (time_enc_mode x seed x dataset).

Runs sequentially on one GPU. Saves a manifest at
revision/ablation_m1_timeenc/manifest.json after every completed run, so
the user can spot-check progress and the workflow can resume on failure.

Skip already-completed (run-name has summary.json) — restart-safe.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
RUNNER = REPO / "revision" / "runners" / "train_one.py"
MANIFEST = REPO / "revision" / "ablation_m1_timeenc" / "manifest.json"

MODES = ["learned_continuous", "none", "binned", "posindex"]
SEEDS = [42, 7, 123]
DATASETS = ["simulation_2mod", "msiln_site1_b1"]


def main() -> None:
    plan = [(d, m, s) for d in DATASETS for m in MODES for s in SEEDS]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"[batch] {len(plan)} runs planned", flush=True)

    results: list[dict] = []
    if MANIFEST.exists():
        try:
            results = json.loads(MANIFEST.read_text())
            print(f"[batch] {len(results)} prior results loaded from manifest",
                  flush=True)
        except json.JSONDecodeError:
            results = []

    done_keys = {(r["dataset"], r["time_enc_mode"], r["seed"])
                 for r in results if r.get("status") == "ok"}

    for i, (dataset, mode, seed) in enumerate(plan, 1):
        key = (dataset, mode, seed)
        if key in done_keys:
            print(f"[{i}/{len(plan)}] SKIP dataset={dataset} mode={mode} "
                  f"seed={seed} (already done)", flush=True)
            continue
        run_name = f"m1_{dataset}_{mode}_s{seed}"
        print(f"[{i}/{len(plan)}] START {datetime.now().isoformat(timespec='seconds')} "
              f"dataset={dataset} mode={mode} seed={seed}", flush=True)
        t0 = time.time()
        r = subprocess.run([
            str(PYTHON), "-X", "faulthandler",
            str(RUNNER),
            "--dataset", dataset,
            "--time-enc-mode", mode,
            "--seed", str(seed),
            "--run-name", run_name,
        ], cwd=str(REPO))
        elapsed = time.time() - t0

        summary_p = REPO / "runs" / "revision" / run_name / "summary.json"
        if r.returncode == 0 and summary_p.is_file():
            s = json.loads(summary_p.read_text())
            s["status"] = "ok"
            s["elapsed_min"] = round(elapsed / 60.0, 2)
            results.append(s)
            print(f"  -> val={s['val_mae_m']:.3f}m test={s['test_mae_m']:.3f}m "
                  f"({elapsed/60:.1f}min)", flush=True)
        else:
            results.append({
                "dataset": dataset, "time_enc_mode": mode, "seed": seed,
                "status": "failed", "rc": r.returncode,
                "elapsed_min": round(elapsed / 60.0, 2),
            })
            print(f"  FAILED rc={r.returncode} ({elapsed/60:.1f}min)", flush=True)

        MANIFEST.write_text(json.dumps(results, indent=2))

    print(f"[batch] DONE — manifest: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
