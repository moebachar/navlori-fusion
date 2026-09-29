#!/usr/bin/env bash
exec > /root/navlori/logs/gt_posegraph.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/gt_posegraph.py
echo PG_EXIT=$?
