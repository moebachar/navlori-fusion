"""Quick health check for the M1+M2 batch.

Reports:
  - last manifest entry + timestamp
  - last log line + how long ago
  - whether the batch process is alive
  - estimated time remaining

Designed for periodic invocation by a wakeup loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "revision" / "ablation_m1_timeenc" / "manifest.json"
BATCH_LOG = REPO / "revision" / "ablation_m1_timeenc" / "batch.log"
TOTAL_RUNS = 24


def is_batch_alive() -> bool:
    """True if a python process running batch_m1_m2 or train_one is alive."""
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
             "-and ($_.CommandLine -like '*batch_m1_m2*' "
             "     -or $_.CommandLine -like '*train_one*') }).Count"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0
    except Exception:
        return False


def main() -> None:
    now = time.time()
    n_ok = 0
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        n_ok = sum(1 for r in m if r.get("status") == "ok")

    log_mtime = (BATCH_LOG.stat().st_mtime if BATCH_LOG.exists()
                 else 0)
    log_age_min = (now - log_mtime) / 60.0 if log_mtime else float("inf")

    alive = is_batch_alive()

    last_line = ""
    if BATCH_LOG.exists():
        with BATCH_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
            for ln in reversed(tail):
                ln = ln.strip()
                if ln:
                    last_line = ln
                    break

    print(f"[health] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[health] manifest: {n_ok}/{TOTAL_RUNS} runs complete")
    print(f"[health] batch process alive: {alive}")
    print(f"[health] last log activity: {log_age_min:.1f} min ago")
    print(f"[health] last log line: {last_line[:160]}")

    verdict = "OK"
    if not alive and n_ok < TOTAL_RUNS:
        verdict = "PROCESS DEAD before batch finished"
    elif log_age_min > 90:
        verdict = f"SILENT for {log_age_min:.0f} min - likely hung"
    print(f"[health] verdict: {verdict}")

    sys.exit(0 if verdict == "OK" else 1)


if __name__ == "__main__":
    main()
