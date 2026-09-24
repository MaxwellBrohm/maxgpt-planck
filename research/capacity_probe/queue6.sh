#!/bin/zsh
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
until grep -q "Q5 DONE" logs_queue5.txt 2>/dev/null; do sleep 20; done
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 40 ] && break; sleep 30; done; }
guard; $PY ft_test.py HuggingFaceTB/SmolLM2-135M-Instruct --steps 400 > logs/ft__SmolLM2-135M-Instruct.log 2>&1; echo "$(date +%T) ft 135M exit=$?"
guard; $PY ft_test.py HuggingFaceTB/SmolLM2-360M-Instruct --steps 400 > logs/ft__SmolLM2-360M-Instruct.log 2>&1; echo "$(date +%T) ft 360M exit=$?"
echo "$(date +%T) Q6 DONE"
