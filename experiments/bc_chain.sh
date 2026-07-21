#!/usr/bin/env bash
# Phase B → Phase C chain
set -u
cd /x/navlori-fusion

PY=.venv/Scripts/python.exe
log() { echo "[bc $(date +%H:%M:%S)] $*"; }

log "Phase B starting"
$PY -X faulthandler -u experiments/run_phase_b.py > experiments/phase_b.log 2>&1
log "Phase B done"

log "Phase C starting"
$PY -X faulthandler -u experiments/run_phase_c.py > experiments/phase_c.log 2>&1
log "Phase C done. Morning report at experiments/MORNING_REPORT.md"
