#!/bin/zsh
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
until grep -q "UPROBE DONE" logs_queue4.txt 2>/dev/null; do sleep 20; done
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; sleep 30; done; }
guard; $PY run_capacity.py unsloth/gemma-3-270m-it --dtype fp32 > logs/unsloth__gemma-3-270m-it.log 2>&1; echo "$(date +%T) gemma main exit=$?"
echo "$(date +%T) Q5 DONE"
