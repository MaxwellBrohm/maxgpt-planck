#!/bin/zsh
# Falcon-H1-Tiny-90M on MPS with the MPS allocator capped, run after everything else.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-nano/research/probe
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35
guard() { while true; do f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'); [ "${f:-0}" -ge 45 ] && break; echo "$(date +%T) waiting for memory (free ${f}%)"; sleep 30; done; }
M=tiiuae/Falcon-H1-Tiny-90M-Instruct
guard; $PY run_model.py $M --mode greedy --device mps > logs/${M//\//__}__greedy.log 2>&1; echo "$(date +%T) greedy finished $M exit=$?"
guard; $PY exp_fixed_history.py $M > logs/E2__${M//\//__}.log 2>&1; echo "$(date +%T) E2 finished $M exit=$?"
echo "$(date +%T) ALL DONE falcon"
