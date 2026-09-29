#!/usr/bin/env bash
exec > /root/navlori/logs/gt_fused.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/gt_fused.py
echo FUSED_EXIT=$?
