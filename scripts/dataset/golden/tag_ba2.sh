#!/usr/bin/env bash
exec > /root/navlori/logs/tag_ba2.log 2>&1
export MPLBACKEND=Agg
cd /root/navlori/scripts_local
/root/navlori/venv/bin/python tag_ba2.py
echo BA_EXIT=$?
