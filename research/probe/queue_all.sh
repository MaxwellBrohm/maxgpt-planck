#!/bin/zsh
# Master queue. Strictly one model process at a time, and each start waits until
# the Mac has at least 35% memory free (other agents share this machine).
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/probe
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 35 ] && break; echo "$(date +%T) waiting for memory (free ${f}%)"; sleep 30; done; }
MODELS=(HuggingFaceTB/SmolLM2-135M-Instruct unsloth/gemma-3-270m-it LiquidAI/LFM2-350M LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B)
dev() { echo mps; }
# Falcon-H1-Tiny (Mamba2 torch fallback) is ~90 s/conversation on CPU and peaks near 9 GB on MPS,
# so it runs last, on MPS with a capped allocator (see queue_falcon.sh).
STAGE=${1:-all}
if [[ $STAGE == all || $STAGE == greedy ]]; then
for m in $MODELS; do guard; $PY run_model.py $m --mode greedy --device $(dev $m) > logs/${m//\//__}__greedy.log 2>&1; echo "$(date +%T) greedy finished $m exit=$?"; done
fi
if [[ $STAGE == all || $STAGE == exp ]]; then
for m in $MODELS; do guard; c=""; [[ $(dev $m) == cpu ]] && c="--cpu"; $PY exp_fixed_history.py $m $c > logs/E2__${m//\//__}.log 2>&1; echo "$(date +%T) E2 finished $m exit=$?"; done
guard; $PY exp_fixed_history.py HuggingFaceTB/SmolLM2-135M --plain > logs/E1__base_plain.log 2>&1; echo "$(date +%T) E1 finished base exit=$?"
guard; $PY exp_fixed_history.py HuggingFaceTB/SmolLM2-135M-Instruct --plain > logs/E1__instruct_plain.log 2>&1; echo "$(date +%T) E1 finished instruct exit=$?"
fi
if [[ $STAGE == all || $STAGE == sampled ]]; then
for m in $MODELS; do guard; $PY run_model.py $m --mode sampled --seeds 0 1 --device $(dev $m) > logs/${m//\//__}__sampled.log 2>&1; echo "$(date +%T) sampled finished $m exit=$?"; done
fi
echo "$(date +%T) ALL DONE $STAGE"
