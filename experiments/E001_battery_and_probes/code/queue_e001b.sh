#!/bin/zsh
# E001 queue, part 2 (after stopping part 1 during the slow Falcon CPU job): same guard rules.
# Falcon's likelihood battery on CPU with the shared-prompt scorer (check_shared.json: max |diff|
# 1.5e-5 nats, 0 decision flips on 60 items, 2.1x faster), then the extras.
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
cd /Users/brohm/Documents/Projects/maxgpt-planck/experiments/E001_battery_and_probes/code
Q=../logs/queue.txt
echo "$(date +%T) QUEUE B START" >> $Q
run() {
  local name=$1 ceil=$2; shift 2
  $PY guard.py --name $name --ceiling $ceil -- $PY "$@"
  local rc=$?
  local killed=$(python3 -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  echo "$(date +%T) $name exit=$rc killed=$killed" >> $Q
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* ]]; then
    echo "$(date +%T) QUEUE STOPPED: memory kill in $name" >> $Q
    exit 2
  fi
}
run bat_new_falcon_cpu       10800 run_battery.py tiiuae/Falcon-H1-Tiny-90M-Instruct --device cpu --shared --sets new
run bat_oldkh_falcon_cpu      7200 run_battery.py tiiuae/Falcon-H1-Tiny-90M-Instruct --device cpu --shared --sets old,khard
run bat_new_lfm2_2_6b         7200 run_battery.py LiquidAI/LFM2-2.6B --sets new --dtype bf16
run bat_new_smollm2_135m_base 3600 run_battery.py HuggingFaceTB/SmolLM2-135M --sets new
echo "$(date +%T) QUEUE B DONE" >> $Q
