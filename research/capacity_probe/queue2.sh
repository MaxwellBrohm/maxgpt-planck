#!/bin/zsh
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
export HF_HUB_OFFLINE=1
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
until grep -q "ALL DONE" logs_queue.txt 2>/dev/null; do sleep 20; done
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; sleep 30; done; }
for m in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M-Instruct unsloth/gemma-3-270m-it LiquidAI/LFM2-350M LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B; do
  guard; $PY khard.py $m > logs/khard__${m//\//__}.log 2>&1; echo "$(date +%T) khard $m exit=$?"
done
guard; $PY khard.py LiquidAI/LFM2-2.6B --dtype bf16 > logs/khard__LiquidAI__LFM2-2.6B.log 2>&1; echo "$(date +%T) khard LFM2-2.6B exit=$?"
guard; PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 $PY khard.py tiiuae/Falcon-H1-Tiny-90M-Instruct > logs/khard__Falcon.log 2>&1; echo "$(date +%T) khard Falcon exit=$?"
echo "$(date +%T) KHARD DONE"
