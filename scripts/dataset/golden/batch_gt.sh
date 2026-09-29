#!/usr/bin/env bash
exec > /root/navlori/logs/batch_gt.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/batch_gt.py
echo BATCH_GT_EXIT=$?
