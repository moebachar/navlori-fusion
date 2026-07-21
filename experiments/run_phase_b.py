"""Phase B — small lead tests.

Reads experiments/state/leads_top.json, which lists candidate lead runners
written under experiments/leads/. For each, runs a short MSILN training
(15 epochs, ~22 min) and records val_mae / test_mae. Writes a ranked
result JSON to experiments/state/phase_b_results.json.

Resilient: skips any lead whose result is already on disk; tolerates per-lead
failures and continues.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

LEADS_JSON = REPO / "experiments" / "state" / "leads_top.json"
RESULTS_JSON = REPO / "experiments" / "state" / "phase_b_results.json"
LOG_DIR = REPO / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PY = REPO / ".venv" / "Scripts" / "python.exe"

RESULT_RX = re.compile(
    r"RESULT:.*?val=([0-9.]+).*?test=([0-9.]+)",
    re.IGNORECASE,
)


def parse_result(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text(errors="replace")
    matches = RESULT_RX.findall(txt)
    if not matches:
        return None
    val, test = matches[-1]
    return {"val_mae_m": float(val), "test_mae_m": float(test)}


def main() -> None:
    if not LEADS_JSON.exists():
        print(f"[phase-b] missing {LEADS_JSON}", flush=True)
        sys.exit(2)
    leads = json.loads(LEADS_JSON.read_text())
    print(f"[phase-b] {len(leads)} leads queued for small tests", flush=True)

    results: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            results = json.loads(RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            results = []
    done = {r["name"]: r for r in results if r.get("status") == "ok"}

    for i, lead in enumerate(leads, 1):
        name = lead["name"]
        runner = REPO / lead["runner"]
        epochs = int(lead.get("epochs", 15))
        dataset = lead.get("dataset", "msiln_site1_b1")
        if name in done:
            print(f"[{i}/{len(leads)}] SKIP {name} (done)", flush=True)
            continue
        if not runner.exists():
            print(f"[{i}/{len(leads)}] MISSING runner {runner}", flush=True)
            results.append({"name": name, "status": "missing", "runner": str(runner)})
            continue

        log_p = LOG_DIR / f"phase_b_{name}.log"
        seed = int(lead.get("seed_override", 42))
        cmd = [str(PY), "-X", "faulthandler", "-u", str(runner),
               "--dataset", dataset, "--seed", str(seed),
               "--epochs", str(epochs)]
        print(f"[{i}/{len(leads)}] {datetime.now().isoformat(timespec='seconds')} "
              f"START {name} ({epochs} ep) -> {log_p}", flush=True)
        t0 = time.time()
        with log_p.open("w") as f:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
        elapsed_min = (time.time() - t0) / 60.0
        parsed = parse_result(log_p)
        if parsed is None:
            results.append({
                "name": name, "status": "failed", "rc": r.returncode,
                "elapsed_min": round(elapsed_min, 2), "log": str(log_p),
            })
            print(f"  FAILED rc={r.returncode} ({elapsed_min:.1f} min)", flush=True)
        else:
            results.append({
                "name": name, "status": "ok",
                "val_mae_m": parsed["val_mae_m"],
                "test_mae_m": parsed["test_mae_m"],
                "epochs": epochs, "elapsed_min": round(elapsed_min, 2),
                "log": str(log_p), "runner": str(runner),
            })
            print(f"  -> val={parsed['val_mae_m']:.3f} test={parsed['test_mae_m']:.3f} "
                  f"({elapsed_min:.1f} min)", flush=True)
        RESULTS_JSON.write_text(json.dumps(results, indent=2))

    # Rank by test_mae (lower is better)
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["test_mae_m"])
    print("\n[phase-b] RANKED RESULTS (lowest test MAE first):", flush=True)
    for r in ok:
        print(f"  {r['test_mae_m']:.3f} m  {r['name']}  "
              f"(val={r['val_mae_m']:.3f}, {r['elapsed_min']:.1f} min)",
              flush=True)
    print("[phase-b] DONE", flush=True)


if __name__ == "__main__":
    main()
