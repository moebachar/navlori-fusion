"""Phase B' (night 2) - small lead screen.

Reads experiments/state/night2_leads.json (an array of leads with at least
{name, runner, epochs, dataset}) and runs each. Each runner is expected to
print 'RESULT: dataset=X seed=Y val=V.V test=T.T' on stdout. Per-lead logs
go to experiments/logs/phase_b2_<name>.log.

Tolerant: failures are logged and the next lead runs.
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

LEADS_JSON = REPO / "experiments" / "state" / "night2_leads.json"
RESULTS_JSON = REPO / "experiments" / "state" / "phase_b2_results.json"
LOG_DIR = REPO / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PY = REPO / ".venv" / "Scripts" / "python.exe"

RESULT_RX = re.compile(r"RESULT:.*?val=([0-9.]+).*?test=([0-9.]+)", re.I)
PER_LEAD_TIMEOUT_S = 30 * 60  # 30-min ceiling per lead — protects the queue


def parse_result(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    txt = log_path.read_text(errors="replace")
    m = RESULT_RX.findall(txt)
    if not m:
        return None
    val, test = m[-1]
    return {"val_mae_m": float(val), "test_mae_m": float(test)}


def main() -> None:
    if not LEADS_JSON.exists():
        print(f"[phase-b2] missing {LEADS_JSON}", flush=True)
        sys.exit(2)
    leads = json.loads(LEADS_JSON.read_text())
    print(f"[phase-b2] {len(leads)} leads queued", flush=True)

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
        epochs = int(lead.get("epochs", 12))
        dataset = lead.get("dataset", "msiln_site1_b1")
        seed = int(lead.get("seed", 42))
        if name in done:
            print(f"[{i}/{len(leads)}] SKIP {name} (done)", flush=True)
            continue
        if not runner.exists():
            print(f"[{i}/{len(leads)}] MISSING runner {runner}", flush=True)
            results.append({"name": name, "status": "missing", "runner": str(runner)})
            RESULTS_JSON.write_text(json.dumps(results, indent=2))
            continue

        log_p = LOG_DIR / f"phase_b2_{name}.log"
        # Synth's leads have inconsistent CLI. Call with NO args; each lead
        # uses its own defaults. Epochs from the manifest is ignored at the
        # CLI level but printed for tracking.
        cmd = [str(PY), "-X", "faulthandler", "-u", str(runner)]
        print(f"[{i}/{len(leads)}] {datetime.now().isoformat(timespec='seconds')} "
              f"START {name} (~{epochs} ep) -> {log_p}", flush=True)
        t0 = time.time()
        with log_p.open("w") as f:
            try:
                r = subprocess.run(cmd, cwd=str(REPO), stdout=f,
                                    stderr=subprocess.STDOUT,
                                    timeout=PER_LEAD_TIMEOUT_S)
                rc = r.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                rc = -999
                timed_out = True
        elapsed_min = (time.time() - t0) / 60.0
        parsed = parse_result(log_p)
        if parsed is None or timed_out:
            results.append({
                "name": name, "status": "failed",
                "rc": rc, "timed_out": timed_out,
                "elapsed_min": round(elapsed_min, 2),
                "log": str(log_p), "family": lead.get("family", ""),
            })
            print(f"  FAILED rc={rc} timed_out={timed_out} "
                  f"({elapsed_min:.1f} min)", flush=True)
        else:
            results.append({
                "name": name, "status": "ok",
                "val_mae_m": parsed["val_mae_m"],
                "test_mae_m": parsed["test_mae_m"],
                "epochs": epochs, "elapsed_min": round(elapsed_min, 2),
                "log": str(log_p), "runner": str(runner),
                "family": lead.get("family", ""),
            })
            print(f"  -> val={parsed['val_mae_m']:.3f} test={parsed['test_mae_m']:.3f} "
                  f"({elapsed_min:.1f} min)", flush=True)
        RESULTS_JSON.write_text(json.dumps(results, indent=2))

    # Ranked summary at the end
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["test_mae_m"])
    print("\n[phase-b2] RANKED RESULTS (lowest test MAE first):", flush=True)
    for r in ok:
        print(f"  {r['test_mae_m']:.3f} m  {r['name']}  [{r.get('family', '')}]  "
              f"(val={r['val_mae_m']:.3f}, {r['elapsed_min']:.1f} min)",
              flush=True)
    print("[phase-b2] DONE", flush=True)


if __name__ == "__main__":
    main()
