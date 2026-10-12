#!/bin/zsh
# Reprioritized tail: fine-tune test first, then the remaining controls.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge ${1:-35} ] && break; sleep 20; done; }
guard 35; $PY ft_test.py HuggingFaceTB/SmolLM2-135M-Instruct --steps 400 > logs/ft__SmolLM2-135M-Instruct.log 2>&1; echo "$(date +%T) ft 135M exit=$?"
guard 35; $PY ft_test.py HuggingFaceTB/SmolLM2-360M-Instruct --steps 400 > logs/ft__SmolLM2-360M-Instruct.log 2>&1; echo "$(date +%T) ft 360M exit=$?"
echo "$(date +%T) FT DONE"
guard 40; $PY uprobe.py LiquidAI/LFM2-2.6B --dtype bf16 > logs/uprobe__LiquidAI__LFM2-2.6B.log 2>&1; echo "$(date +%T) uprobe LFM2-2.6B exit=$?"
guard; $PY uprobe.py LiquidAI/LFM2-350M > logs/uprobe__LiquidAI__LFM2-350M.log 2>&1; echo "$(date +%T) uprobe LFM2-350M exit=$?"
guard; $PY uprobe.py Qwen/Qwen3-0.6B > logs/uprobe__Qwen__Qwen3-0.6B.log 2>&1; echo "$(date +%T) uprobe Qwen3 exit=$?"
echo "$(date +%T) UPROBE2 DONE"
for m in HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct LiquidAI/LFM2-350M; do
  guard; $PY copysplit.py $m --n 444 > logs/copysplit__${m//\//__}.log 2>&1; echo "$(date +%T) copysplit $m exit=$?"
done
guard 40; $PY copysplit.py LiquidAI/LFM2-2.6B --dtype bf16 --n 444 > logs/copysplit__LiquidAI__LFM2-2.6B.log 2>&1; echo "$(date +%T) copysplit LFM2-2.6B exit=$?"
echo "$(date +%T) COPYSPLIT DONE"
guard; $PY run_capacity.py unsloth/gemma-3-270m-it --dtype fp32 > logs/unsloth__gemma-3-270m-it.log 2>&1; echo "$(date +%T) gemma main exit=$?"
echo "$(date +%T) TAIL2 DONE"
