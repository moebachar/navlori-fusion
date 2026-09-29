#!/usr/bin/env bash
exec > /root/navlori/logs/tags_detect.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/tags_detect.py
echo TAGS_EXIT=$?
