#!/usr/bin/env bash
exec > /root/navlori/logs/golden_package.log 2>&1
export MPLBACKEND=Agg
cd /root/navlori/scripts_local
/root/navlori/venv/bin/python golden_package.py
echo PKG_EXIT=$?
