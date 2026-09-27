#!/bin/bash
# boot.d/30_rc12.sh: after a reboot, resume the RC-12 dev-baseline queue (rc12/queue_dev_baselines.sh, rc12/notes.txt
# STEP 9b; the Doge pass STEP 9c).
#   bash 30_rc12.sh [--dry-run]
# Resumes only when all hold: no stop file (bootlib.sh: ~/planck/boot.d/STOP, boot.d/rc12.STOP, ~/planck/STOP,
# ~/planck/PAUSE, plus the queue's own ~/planck/logs/rc12_dev_queue.STOP); nothing of the queue runs (the queue, the
# dg_follow.sh follow-up, a dev_batch.py process); no ~/planck/logs/rc12_dev_queue.DONE (the queue removes it at start
# and writes it at its end); and rc12_dev_queue.log's last "queue: start," has no "queue: end;" or "queue: STOP file"
# after it.
# Restart path (STEP 9c "Manual restart"): Q_CODE=CODE bash CODE/queue_dev_baselines.sh, CODE = ~/planck/dev/rc12_dg
# (rc12_q9 plus the Doge engine; same queue name, lock, status file and root, so finished runs are skipped and Doge's
# 8 processes run after the rest). When the queue moves to another code copy, change CODE here (or set RC12_CODE).
NAME=rc12
. "$(dirname "$0")/bootlib.sh"
CODE=${RC12_CODE:-$HOME/planck/dev/rc12_dg}
QL=$P/logs/rc12_dev_queue.log
DONE=$P/logs/rc12_dev_queue.DONE
ANY='queue_dev_baselines\.sh|dg_follow\.sh|dev_batch\.py'

blocked "$P/logs/rc12_dev_queue.STOP" && exit 0
if running "$ANY"; then say "running (pids $PIDS): nothing to do"; exit 0; fi
if [ -e "$DONE" ]; then say "$DONE present [$(head -1 "$DONE" 2>/dev/null)]: nothing to resume"; exit 0; fi
[ -f "$QL" ] || { say "no $QL: the queue never ran here, nothing to resume"; exit 0; }
start=$(last_start "$QL" 'queue: start,')
[ -n "$start" ] || { say "no queue start line in $QL: nothing to resume"; exit 0; }
end=$(ended_after "$QL" "$start" 'queue: end;|queue: STOP file, exiting')
if [ -n "$end" ]; then say "the last run ended by itself [$end]: nothing to resume"; exit 0; fi
[ -f "$CODE/queue_dev_baselines.sh" ] || { say "no $CODE/queue_dev_baselines.sh: cannot resume"; exit 1; }
say "cut off: last start at line $start, last line [$(tail -1 "$QL")]"
launch 'queue_dev_baselines\.sh' env Q_CODE="$CODE" bash "$CODE/queue_dev_baselines.sh"
