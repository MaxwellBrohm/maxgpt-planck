#!/bin/zsh
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
until grep -q "COPYSPLIT DONE" logs_queue3.txt 2>/dev/null; do sleep 20; done
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; sleep 30; done; }
for m in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct LiquidAI/LFM2-350M unsloth/gemma-3-270m-it Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B; do
  guard; $PY uprobe.py $m > logs/uprobe__${m//\//__}.log 2>&1; echo "$(date +%T) uprobe $m exit=$?"
done
guard; $PY uprobe.py LiquidAI/LFM2-2.6B --dtype bf16 > logs/uprobe__LiquidAI__LFM2-2.6B.log 2>&1; echo "$(date +%T) uprobe LFM2-2.6B exit=$?"
echo "$(date +%T) UPROBE DONE"
