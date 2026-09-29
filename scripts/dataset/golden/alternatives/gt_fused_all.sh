#!/usr/bin/env bash
exec > /root/navlori/logs/gt_fused_all.log 2>&1
export MPLBACKEND=Agg NOFIG=1
/root/navlori/venv/bin/python /root/navlori/scripts_local/gt_fused.py golden_run_1 golden_run_2 golden_run_3 golden_run_4 golden_run_5 golden_run_6 golden_run_7 golden_run_8 golden_run_9 golden_run_10 golden_run_11 golden_run_12
echo FUSED_EXIT=$?
