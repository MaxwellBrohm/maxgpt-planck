#!/bin/zsh
# E3 after everything else: the fixed-history sweep (short replies) with a one-line memory system prompt.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/probe
until grep -q "ALL DONE falcon" logs_queue_falcon.txt 2>/dev/null; do sleep 15; done
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; sleep 30; done; }
for m in HuggingFaceTB/SmolLM2-135M-Instruct unsloth/gemma-3-270m-it LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
  guard; $PY exp_fixed_history.py $m --memsys > logs/E3__${m//\//__}.log 2>&1; echo "$(date +%T) E3 finished $m exit=$?"
done
echo "$(date +%T) ALL DONE e3"
