#!/usr/bin/env bash
# Night 2 chain: wait for leads, run Phase B', then Phase C'.
set -u
cd /x/navlori-fusion

PY=.venv/Scripts/python.exe
log() { echo "[night2 $(date +%H:%M:%S)] $*"; }

log "Stage 1: waiting for night2_leads.json"
until [ -f experiments/state/night2_leads.json ]; do
  sleep 30
done
log "Leads file present. Counts: $(jq length experiments/state/night2_leads.json 2>/dev/null || echo '?') leads queued"

log "Stage 2: Phase B' starting"
$PY -X faulthandler -u experiments/run_phase_b2.py > experiments/phase_b2.log 2>&1
log "Phase B' done"

log "Stage 3: Phase C' starting"
$PY -X faulthandler -u experiments/run_phase_c2.py > experiments/phase_c2.log 2>&1
log "Phase C' done. Report: experiments/MORNING_REPORT_v2.md"
