#!/usr/bin/env bash
exec > /root/navlori/logs/gt_sweep.log 2>&1
export MPLBACKEND=Agg QUICK=1
PY=/root/navlori/venv/bin/python; S=/root/navlori/scripts_local/gt_fused.py
echo "### A current  (1cm+2%, 0.1deg+1%)";  SXY0=0.01  SXY1=0.02  STH0_DEG=0.1  STH1=0.01  $PY $S | grep -E "=====|held-out"
echo "### B stiffer  (3mm+0.5%, 0.05deg+0.5%)"; SXY0=0.003 SXY1=0.005 STH0_DEG=0.05 STH1=0.005 $PY $S | grep -E "=====|held-out"
echo "### C v.stiff  (1mm+0.2%, 0.02deg+0.2%)"; SXY0=0.001 SXY1=0.002 STH0_DEG=0.02 STH1=0.002 $PY $S | grep -E "=====|held-out"
echo SWEEP_DONE
