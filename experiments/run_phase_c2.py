"""Phase C' (night 2) - full training of the top Phase B' leads.

Picks the top-N leads by Phase B' test_mae, trains each at 40 epochs with
seed 42, then writes MORNING_REPORT_v2.md.

Strict policy: candidate must beat baseline 11.5 m test by >= 0.5 m at
Phase B' to qualify for Phase C'. Otherwise we report Phase B' only.
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

PB_JSON = REPO / "experiments" / "state" / "phase_b2_results.json"
PC_JSON = REPO / "experiments" / "state" / "phase_c2_results.json"
REPORT = REPO / "experiments" / "MORNING_REPORT_v2.md"
LOG_DIR = REPO / "experiments" / "logs"
PY = REPO / ".venv" / "Scripts" / "python.exe"

TOP_N = 3
FULL_EPOCHS = 40
BASE_TEST_MAE = 11.527
BASE_TEST_STD = 3.154
IDEA1_TEST_MAE = 9.93
IDEA1_TEST_STD = 0.25
LEAD3_TEST_MAE = 9.11

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


def write_report(pb_results: list, pc_results: list) -> None:
    lines = ["# Night 2 Morning Report", ""]
    lines.append(f"Generated {datetime.now().isoformat(timespec='seconds')}.")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(f"**Baseline (M1+M2 paper config): test {BASE_TEST_MAE:.2f} ± {BASE_TEST_STD:.2f} m.**")
    lines.append(f"Yesterday's two winners: idea1 (test {IDEA1_TEST_MAE:.2f} ± {IDEA1_TEST_STD:.2f}, 4 seeds), "
                 f"lead3 JEPA (test {LEAD3_TEST_MAE:.2f}, 1 seed).")
    lines.append("")

    pb_ok = [r for r in pb_results if r.get("status") == "ok"]
    pb_ok.sort(key=lambda r: r["test_mae_m"])
    pc_ok = [r for r in pc_results if r.get("status") == "ok"]
    pc_ok.sort(key=lambda r: r["test_mae_m"])

    if pc_ok:
        best = pc_ok[0]
        lines.append(f"**Best Phase C' lead: {best['name']} (test {best['test_mae_m']:.2f} m, "
                     f"family={best.get('family', '?')}).**")
        lines.append("")
    if pb_ok:
        lines.append("Phase B' top 3:")
        for r in pb_ok[:3]:
            lines.append(f"- {r['name']} [{r.get('family', '')}]: test {r['test_mae_m']:.3f} m")
        lines.append("")

    # Phase B' table grouped by family
    lines.append("## Phase B' — full screen (12 epochs default unless noted)")
    lines.append("")
    lines.append("Ranked by test MAE.")
    lines.append("")
    lines.append("| Rank | Lead | Family | val MAE (m) | test MAE (m) | Δ vs base | min |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(pb_ok, 1):
        d = r["test_mae_m"] - BASE_TEST_MAE
        sign = "+" if d >= 0 else ""
        lines.append(f"| {i} | {r['name']} | {r.get('family', '')} | "
                     f"{r['val_mae_m']:.3f} | **{r['test_mae_m']:.3f}** | "
                     f"{sign}{d:.2f} m | {r['elapsed_min']:.1f} |")
    lines.append("")

    # Failed and missing
    failed = [r for r in pb_results if r.get("status") != "ok"]
    if failed:
        lines.append("## Failed / missing")
        lines.append("")
        for r in failed:
            lines.append(f"- `{r['name']}`: {r.get('status', '?')}"
                         + (f" (rc={r.get('rc')})" if r.get('rc') else "")
                         + (f" (timed out)" if r.get('timed_out') else ""))
        lines.append("")

    # Phase C
    if pc_ok:
        lines.append("## Phase C' — top leads at full 40 epochs")
        lines.append("")
        lines.append("| Lead | Phase B' test | Phase C' test | Δ vs base | min |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in pc_ok:
            d = r["test_mae_m"] - BASE_TEST_MAE
            sign = "+" if d >= 0 else ""
            lines.append(f"| {r['name']} | {r.get('phase_b_test', '?'):.3f} | "
                         f"{r['test_mae_m']:.3f} | {sign}{d:.2f} m | "
                         f"{r['elapsed_min']:.1f} |")
        lines.append("")

    # Family landscape
    lines.append("## Family landscape — what worked, what didn't")
    lines.append("")
    by_family: dict[str, list[dict]] = {}
    for r in pb_ok:
        f = r.get("family", "unknown")
        by_family.setdefault(f, []).append(r)
    for fam, rs in sorted(by_family.items(),
                          key=lambda kv: min(r["test_mae_m"] for r in kv[1])):
        best = min(rs, key=lambda r: r["test_mae_m"])
        d = best["test_mae_m"] - BASE_TEST_MAE
        sign = "+" if d >= 0 else ""
        lines.append(f"- **{fam}**: best = `{best['name']}` "
                     f"test {best['test_mae_m']:.3f} m ({sign}{d:.2f} vs base, "
                     f"{sign}{best['test_mae_m'] - IDEA1_TEST_MAE:.2f} vs idea1).")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- `experiments/state/night2_leads.json` — input list")
    lines.append("- `experiments/state/phase_b2_results.json` — Phase B' raw")
    lines.append("- `experiments/state/phase_c2_results.json` — Phase C' raw")
    lines.append("- `experiments/leads_night2/` — per-lead runners")
    lines.append("- `experiments/logs/phase_b2_*.log` / `phase_c2_*.log`")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {REPORT}", flush=True)


def main() -> None:
    if not PB_JSON.exists():
        print(f"[phase-c2] missing {PB_JSON}", flush=True)
        sys.exit(2)
    pb = json.loads(PB_JSON.read_text())
    pb_ok = [r for r in pb if r.get("status") == "ok"]
    pb_ok.sort(key=lambda r: r["test_mae_m"])

    qualifiers = [r for r in pb_ok
                  if r["test_mae_m"] <= (BASE_TEST_MAE - 0.5)]
    top = qualifiers[:TOP_N]
    print(f"[phase-c2] {len(qualifiers)} qualifiers; picking top {len(top)} "
          f"for full training", flush=True)
    for r in top:
        print(f"  {r['test_mae_m']:.3f}  {r['name']}", flush=True)

    pc_results: list[dict] = []
    if PC_JSON.exists():
        try:
            pc_results = json.loads(PC_JSON.read_text())
        except json.JSONDecodeError:
            pc_results = []
    done = {r["name"] for r in pc_results if r.get("status") == "ok"}

    for i, lead in enumerate(top, 1):
        name = lead["name"]
        if name in done:
            print(f"[{i}/{len(top)}] SKIP {name}", flush=True)
            continue
        runner = REPO / lead["runner"]
        log_p = LOG_DIR / f"phase_c2_{name}.log"
        cmd = [str(PY), "-X", "faulthandler", "-u", str(runner),
               "--dataset", "msiln_site1_b1", "--seed", "42",
               "--epochs", str(FULL_EPOCHS)]
        print(f"[{i}/{len(top)}] START {name}", flush=True)
        t0 = time.time()
        with log_p.open("w") as f:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=f,
                                stderr=subprocess.STDOUT)
        elapsed_min = (time.time() - t0) / 60.0
        parsed = parse_result(log_p)
        if parsed is None:
            rec = {"name": name, "status": "failed", "rc": r.returncode,
                   "elapsed_min": round(elapsed_min, 2)}
        else:
            rec = {
                "name": name, "status": "ok",
                "val_mae_m": parsed["val_mae_m"],
                "test_mae_m": parsed["test_mae_m"],
                "phase_b_test": lead["test_mae_m"],
                "elapsed_min": round(elapsed_min, 2),
                "family": lead.get("family", ""),
                "log": str(log_p),
            }
            print(f"  -> val={parsed['val_mae_m']:.3f} test={parsed['test_mae_m']:.3f} "
                  f"({elapsed_min:.1f} min)", flush=True)
        pc_results.append(rec)
        PC_JSON.write_text(json.dumps(pc_results, indent=2))

    write_report(pb, pc_results)
    print("[phase-c2] DONE", flush=True)


if __name__ == "__main__":
    main()
