#!/usr/bin/env bash
exec > /root/navlori/logs/batch_golden.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/batch_golden.py
echo BATCH_GOLDEN_EXIT=$?
