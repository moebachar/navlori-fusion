#!/usr/bin/env bash
exec > /root/navlori/logs/tag_ba.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/tag_ba.py
echo BA_EXIT=$?
