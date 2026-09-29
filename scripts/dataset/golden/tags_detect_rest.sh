#!/usr/bin/env bash
exec > /root/navlori/logs/tags_detect_rest.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/tags_detect.py golden_run_1 golden_run_2 golden_run_3 golden_run_4 golden_run_5 golden_run_6 golden_run_8 golden_run_9 golden_run_12
echo TAGS_EXIT=$?
