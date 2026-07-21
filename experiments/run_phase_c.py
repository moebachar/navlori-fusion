"""Phase C - Full training of top leads from Phase B.

Reads experiments/state/phase_b_results.json, picks top N by test_mae,
runs each at 40 epochs on MSILN. Writes morning report.
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

PHASE_B_JSON = REPO / "experiments" / "state" / "phase_b_results.json"
PHASE_C_JSON = REPO / "experiments" / "state" / "phase_c_results.json"
REPORT = REPO / "experiments" / "MORNING_REPORT.md"
LOG_DIR = REPO / "experiments" / "logs"
PY = REPO / ".venv" / "Scripts" / "python.exe"

TOP_N = 2
FULL_EPOCHS = 40
BASE_TEST_MAE = 11.527
BASE_TEST_STD = 3.154

RESULT_RX = re.compile(r"RESULT:.*?val=([0-9.]+).*?test=([0-9.]+)", re.I)


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
    if not PHASE_B_JSON.exists():
        print(f"[phase-c] missing {PHASE_B_JSON}", flush=True); sys.exit(2)
    pb = json.loads(PHASE_B_JSON.read_text())
    ok = [r for r in pb if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["test_mae_m"])
    top = ok[:TOP_N]
    print(f"[phase-c] {len(top)} leads selected for full training", flush=True)
    for r in top:
        print(f"  {r['test_mae_m']:.3f} m  {r['name']}", flush=True)

    results: list[dict] = []
    if PHASE_C_JSON.exists():
        try:
            results = json.loads(PHASE_C_JSON.read_text())
        except json.JSONDecodeError:
            results = []
    done = {r["name"] for r in results if r.get("status") == "ok"}

    for i, lead in enumerate(top, 1):
        name = lead["name"]
        runner = REPO / lead["runner"]
        if name in done:
            print(f"[{i}/{len(top)}] SKIP {name} (done)", flush=True)
            continue
        log_p = LOG_DIR / f"phase_c_{name}.log"
        cmd = [str(PY), "-X", "faulthandler", "-u", str(runner),
               "--dataset", "msiln_site1_b1", "--seed", "42",
               "--epochs", str(FULL_EPOCHS)]
        print(f"[{i}/{len(top)}] {datetime.now().isoformat(timespec='seconds')} "
              f"START full train {name}", flush=True)
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
            print(f"  FAILED rc={r.returncode}", flush=True)
        else:
            results.append({
                "name": name, "status": "ok",
                "val_mae_m": parsed["val_mae_m"],
                "test_mae_m": parsed["test_mae_m"],
                "phase_b_test": lead["test_mae_m"],
                "epochs": FULL_EPOCHS,
                "elapsed_min": round(elapsed_min, 2),
                "log": str(log_p),
            })
            print(f"  -> val={parsed['val_mae_m']:.3f} test={parsed['test_mae_m']:.3f} "
                  f"({elapsed_min:.1f} min)", flush=True)
        PHASE_C_JSON.write_text(json.dumps(results, indent=2))

    # ----- Morning report -----
    lines = ["# Overnight Morning Report", ""]
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')}.")
    lines.append("")
    lines.append(f"**Baseline (M1+M2 paper config, MSILN test, 3 seeds): "
                 f"{BASE_TEST_MAE} ± {BASE_TEST_STD} m**")
    lines.append("")

    # idea1 + idea2 MSILN
    lines.append("## Ideas tested on MSILN (Phase A)")
    lines.append("")
    lines.append("| Idea | val MAE (m) | test MAE (m) | Δ vs baseline |")
    lines.append("|---|---:|---:|---:|")
    for fname, label in [("idea1_msiln.log", "Idea 1 — place-PE IMU"),
                          ("idea2_msiln.log", "Idea 2 — Gaussian splat")]:
        p = REPO / "experiments" / fname
        parsed = parse_result(p)
        if parsed:
            d = parsed["test_mae_m"] - BASE_TEST_MAE
            sign = "+" if d >= 0 else ""
            lines.append(f"| {label} | {parsed['val_mae_m']:.3f} | "
                         f"{parsed['test_mae_m']:.3f} | {sign}{d:.3f} m |")
        else:
            lines.append(f"| {label} | (no result) | — | — |")
    lines.append("")

    # Phase B
    lines.append("## Phase B — small lead screen (15 epochs)")
    lines.append("")
    lines.append("| Rank | Lead | val MAE (m) | test MAE (m) | Δ vs baseline | min |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for rank, r in enumerate(ok, 1):
        d = r["test_mae_m"] - BASE_TEST_MAE
        sign = "+" if d >= 0 else ""
        lines.append(f"| {rank} | {r['name']} | {r['val_mae_m']:.3f} | "
                     f"{r['test_mae_m']:.3f} | {sign}{d:.3f} | {r['elapsed_min']:.1f} |")
    lines.append("")

    # Phase C
    lines.append("## Phase C — top leads, full 40-epoch training")
    lines.append("")
    lines.append("| Lead | Phase B test | Phase C test | Δ vs baseline | min |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        if r.get("status") != "ok":
            continue
        d = r["test_mae_m"] - BASE_TEST_MAE
        sign = "+" if d >= 0 else ""
        lines.append(f"| {r['name']} | {r.get('phase_b_test', '?'):.3f} | "
                     f"{r['test_mae_m']:.3f} | {sign}{d:.3f} | "
                     f"{r['elapsed_min']:.1f} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- `experiments/state/leads_top.json` — original ranked leads from the research workflow")
    lines.append("- `experiments/state/phase_b_results.json` — Phase B raw results")
    lines.append("- `experiments/state/phase_c_results.json` — Phase C raw results")
    lines.append("- `experiments/logs/` — per-run training logs")
    lines.append("- `experiments/leads/` — small one-file runners for each lead")
    lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"[phase-c] wrote {REPORT}", flush=True)
    print("[phase-c] DONE", flush=True)


if __name__ == "__main__":
    main()
