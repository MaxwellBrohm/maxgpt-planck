#!/bin/zsh
# E002 queue: every model job goes through guard.py (one model process at a time, LM Studio unloaded,
# waits for >= 35% free memory, MPS watermarks 0.7/0.6, kills on wall-clock ceiling, < 15% free or
# > 768 MB swap growth). Jobs run strictly one after another; a memory kill stops the queue.
# Dry runs (5 steps at worst-case length) were done before this queue: logs/dry_135m*, logs/dry_360m*.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-planck/experiments/E002_ft_test/code
Q=../logs/queue.txt
echo "$(date +%T) QUEUE START" >> $Q
run() {
  local name=$1 ceil=$2; shift 2
  $PY guard.py --name $name --ceiling $ceil -- $PY -B "$@"
  local rc=$?
  local killed=$(python3 -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  echo "$(date +%T) $name exit=$rc killed=$killed" >> $Q
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    echo "$(date +%T) QUEUE STOPPED: $killed in $name" >> $Q
    exit 2
  fi
}
M135=HuggingFaceTB/SmolLM2-135M-Instruct
M360=HuggingFaceTB/SmolLM2-360M-Instruct
A135=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
A360=(--bs 2 --accum 8 --max-len 768 --grad-ckpt)
run base_135m 2400 ft_test.py $M135 --seed 0 --steps 0 $A135
for s in 0 1 2 3 4; do
  run ft_135m_s$s 5400 ft_test.py $M135 --seed $s $A135
done
run base_360m 4800 ft_test.py $M360 --seed 0 --steps 0 $A360
for s in 0 1 2; do
  run ft_360m_s$s 10800 ft_test.py $M360 --seed $s $A360
done
echo "$(date +%T) QUEUE DONE" >> $Q
