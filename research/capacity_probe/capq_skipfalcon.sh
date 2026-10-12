#!/bin/zsh
# Skip Falcon-H1-Tiny (slow Mamba fallback on MPS) in this probe's own queues only.
D=/Users/brohm/Documents/Projects/maxgpt-nano/research/capacity_probe
for i in {1..720}; do
  for p in $(pgrep -f "run_capacity.py tiiuae|khard.py tiiuae"); do
    c=$(lsof -a -p $p -d cwd -Fn 2>/dev/null | grep ^n | cut -c2-)
    [ "$c" = "$D" ] && kill $p && echo "$(date +%T) skipped falcon pid $p"
  done
  grep -q "KHARD DONE" $D/logs_queue2.txt 2>/dev/null && break
  sleep 5
done
