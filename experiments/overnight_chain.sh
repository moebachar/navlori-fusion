#!/usr/bin/env bash
# Overnight chain: idea1 MSILN → idea2 MSILN → wait for leads workflow → Phase B runs.
# Resilient: idempotent each stage via a sentinel file in experiments/state/.

set -u
cd /x/navlori-fusion
mkdir -p experiments/state experiments/logs

PY=.venv/Scripts/python.exe
log() { echo "[chain $(date +%H:%M:%S)] $*"; }

wait_for_pattern() {
  local file=$1; local pat=$2
  until grep -q "$pat" "$file" 2>/dev/null; do sleep 30; done
}

# ---- Stage 1: idea1 MSILN (already running in another process) ----
log "Stage 1: waiting for idea1 MSILN to finish ..."
wait_for_pattern experiments/idea1_msiln.log "\[idea1\] RESULT:"
log "idea1 done. Summary: $(grep '\[idea1\] RESULT:' experiments/idea1_msiln.log | tail -1)"
touch experiments/state/idea1_done

# ---- Stage 2: idea2 MSILN ----
log "Stage 2: launching idea2 MSILN ..."
$PY -X faulthandler -u experiments/idea2_gaussian_splat.py \
    --dataset msiln_site1_b1 --seed 42 --epochs 40 \
    > experiments/idea2_msiln.log 2>&1
log "idea2 done. Summary: $(grep '\[idea2\] RESULT:' experiments/idea2_msiln.log | tail -1)"
touch experiments/state/idea2_done

# ---- Stage 3: wait for leads workflow ----
log "Stage 3: waiting for leads_top.json (the leads workflow result) ..."
until [ -f experiments/state/leads_top.json ]; do
  sleep 60
done
log "leads_top.json present. Reading top leads ..."

# ---- Stage 4: Phase B small tests (one per top-lead runner) ----
log "Stage 4: Phase B - small tests of the top leads"
$PY -X faulthandler -u experiments/run_phase_b.py \
    > experiments/phase_b.log 2>&1
log "Phase B done. Summary: $(tail -5 experiments/phase_b.log)"
touch experiments/state/phase_b_done

# ---- Stage 5: Phase C - full training of top 1-3 from Phase B ----
log "Stage 5: Phase C - full training of top leads"
$PY -X faulthandler -u experiments/run_phase_c.py \
    > experiments/phase_c.log 2>&1
log "Phase C done. Summary: $(tail -5 experiments/phase_c.log)"
touch experiments/state/phase_c_done

log "OVERNIGHT CHAIN COMPLETE - see experiments/MORNING_REPORT.md"
