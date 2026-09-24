#!/bin/zsh
# One model process at a time; each start waits for >= 35% free memory (other agents share this Mac).
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
mkdir -p logs
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; echo "$(date +%T) waiting for memory (free ${f}%)"; sleep 30; done; }
for m in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M-Instruct unsloth/gemma-3-270m-it LiquidAI/LFM2-350M LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B; do
  guard; $PY run_capacity.py $m --dtype fp32 > logs/${m//\//__}.log 2>&1; echo "$(date +%T) finished $m exit=$?"
done
guard; $PY run_capacity.py LiquidAI/LFM2-2.6B --dtype bf16 > logs/LiquidAI__LFM2-2.6B.log 2>&1; echo "$(date +%T) finished LFM2-2.6B exit=$?"
guard; PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 $PY run_capacity.py tiiuae/Falcon-H1-Tiny-90M-Instruct --dtype fp32 --limit-scen 16 > logs/tiiuae__Falcon-H1-Tiny-90M-Instruct.log 2>&1; echo "$(date +%T) finished Falcon exit=$?"
echo "$(date +%T) ALL DONE"
