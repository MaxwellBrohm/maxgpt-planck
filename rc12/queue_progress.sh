#!/bin/bash
# Progress of the RC-12 dev-baseline queue (queue_dev_baselines.sh), read only. From the Mac:
#   <scratchpad>/pcwsl.sh rc12/queue_progress.sh 90
CODE=$HOME/planck/dev/rc12_q9; ROOT=$HOME/planck/runs/rc12_dev; LOGS=$HOME/planck/logs
date +%Y-%m-%dT%H:%M:%S
if pgrep -f queue_dev_baselines.sh > /dev/null; then echo "queue: RUNNING"; else echo "queue: not running"; fi
[ -e "$LOGS/rc12_dev_queue.DONE" ] && echo "queue: DONE marker present"
[ -e "$LOGS/rc12_dev_queue.STOP" ] && echo "queue: STOP file present"
echo "--- last queue log lines"; tail -4 "$LOGS/rc12_dev_queue.log" 2>/dev/null | cut -c1-200
echo "--- last status lines"; tail -6 "$ROOT/queue_status.txt" 2>/dev/null | cut -c1-240
echo "--- progress"
cd "$CODE" && "$HOME/planck/venv-vllm/bin/python" -B queue_status.py --root "$ROOT" --progress --models "$(
  sed -n '/^MODELS=/,/"}$/p' queue_dev_baselines.sh | tr -d '"}' | sed 's/^MODELS=\${Q_MODELS:-//')"
echo "--- gpu"; /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
