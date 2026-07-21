"""D3 period-range sub-ablation.

For the learned-continuous time encoding (the paper's default), sweep
(time_min_period, time_max_period) at a single seed and report val/test MAE.
The (0.05, 120) baseline is NOT re-trained — it is read off the M1+M2 manifest.

Variants (range chosen to span "coarse but legitimate" sensitivity):
  - narrow:   (0.5, 10)   focused on common observation timescales
  - wide:     (0.01, 600) broader spectrum (raw IMU period .. 10 minutes)
  - shifted:  (0.1, 30)   middle ground (WiFi scan .. ~30 s history)

Runs at seed=42 on simulation_2mod + msiln_site1_b1.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
RUNNER = REPO / "revision" / "runners" / "train_one.py"
MANIFEST = REPO / "revision" / "k_period_d3" / "period_manifest.json"

PERIOD_VARIANTS = [
    ("narrow",  0.5, 10.0),
    ("wide",    0.01, 600.0),
    ("shifted", 0.1, 30.0),
]
SEED = 42
DATASETS = ["simulation_2mod", "msiln_site1_b1"]


def main() -> None:
    plan = [(d, v) for d in DATASETS for v in PERIOD_VARIANTS]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"[d3] {len(plan)} runs planned", flush=True)

    results: list[dict] = []
    if MANIFEST.exists():
        try:
            results = json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            results = []
    done_keys = {(r["dataset"], r["variant_label"])
                 for r in results if r.get("status") == "ok"}

    for i, (dataset, (label, lo, hi)) in enumerate(plan, 1):
        key = (dataset, label)
        if key in done_keys:
            print(f"[{i}/{len(plan)}] SKIP dataset={dataset} variant={label} (done)",
                  flush=True)
            continue
        run_name = f"d3_period_{dataset}_{label}_s{SEED}"
        print(f"[{i}/{len(plan)}] START {datetime.now().isoformat(timespec='seconds')} "
              f"dataset={dataset} variant={label} range=({lo},{hi})", flush=True)
        t0 = time.time()
        r = subprocess.run([
            str(PYTHON), "-X", "faulthandler",
            str(RUNNER),
            "--dataset", dataset,
            "--time-enc-mode", "learned_continuous",
            "--time-min-period", str(lo),
            "--time-max-period", str(hi),
            "--seed", str(SEED),
            "--run-name", run_name,
        ], cwd=str(REPO))
        elapsed = time.time() - t0

        summary_p = REPO / "runs" / "revision" / run_name / "summary.json"
        if r.returncode == 0 and summary_p.is_file():
            s = json.loads(summary_p.read_text())
            s["status"] = "ok"
            s["variant_label"] = label
            s["min_period"] = lo
            s["max_period"] = hi
            s["elapsed_min"] = round(elapsed / 60.0, 2)
            results.append(s)
            print(f"  -> val={s['val_mae_m']:.3f}m test={s['test_mae_m']:.3f}m "
                  f"({elapsed/60:.1f}min)", flush=True)
        else:
            results.append({
                "dataset": dataset, "variant_label": label,
                "min_period": lo, "max_period": hi,
                "status": "failed", "rc": r.returncode,
                "elapsed_min": round(elapsed / 60.0, 2),
            })
            print(f"  FAILED rc={r.returncode} ({elapsed/60:.1f}min)", flush=True)
        MANIFEST.write_text(json.dumps(results, indent=2))

    print(f"[d3] DONE - manifest: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
