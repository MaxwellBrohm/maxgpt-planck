#!/bin/bash
# boot.d/20_e006.sh: after a reboot, resume the E006 queue (experiments/E006_three_objects) through its own wrapper,
# start_queue_e006.sh in the ~/planck/dev/e006 working copy (notes.txt PC QUEUE).
#   bash 20_e006.sh [--dry-run]
# Resumes only when all hold: no stop file (bootlib.sh: ~/planck/boot.d/STOP, boot.d/e006.STOP, ~/planck/STOP,
# ~/planck/PAUSE); nothing of E006 runs (the wrapper, queue_e006.sh, a guard_e006_pc.py job); logs/queue.txt's last
# "QUEUE E006 START" has no "QUEUE E006 DONE" or "QUEUE STOPPED" after it; ~/planck/logs/e006_queue.out has no
# "queue_e006 exit" after its last "start_queue_e006" line (the wrapper saw no exit: the queue was cut off); and no
# job was cut mid-run. A job cut mid-run leaves NAME.log and NAME.guard.log without NAME.guard.json; the guard refuses
# a name whose logs exist (logs are never overwritten), so the queue would retry it 12 times and stop. That case is
# left for a person (E006's rule: rerun by hand as <tag>_r2, a logged deviation).
# On a resume it first appends one line to queue.txt saying why the queue restarts, as the hand restarts did.
NAME=e006
. "$(dirname "$0")/bootlib.sh"
ROOT=${E006_ROOT:-$HOME/planck/dev/e006}
EL=${E006_LOGS:-$HOME/planck/e006_run/logs}
Q=$EL/queue.txt
OUT=$P/logs/e006_queue.out
W=$ROOT/experiments/E006_three_objects/code/start_queue_e006.sh
ANY='queue_e006\.sh|guard_e006_pc\.py'

blocked && exit 0
if running "$ANY"; then say "running (pids $PIDS): nothing to do"; exit 0; fi
[ -f "$Q" ] || { say "no $Q: E006 never ran here, nothing to resume"; exit 0; }
start=$(last_start "$Q" 'QUEUE E006 START')
[ -n "$start" ] || { say "no QUEUE E006 START in $Q: nothing to resume"; exit 0; }
end=$(ended_after "$Q" "$start" 'QUEUE E006 DONE|QUEUE STOPPED')
if [ -n "$end" ]; then say "the last run ended by itself [$end]: nothing to resume"; exit 0; fi
if [ -f "$OUT" ]; then
    wend=$(awk '/^start_queue_e006 /{e=""} /^queue_e006 exit /{e=$0} END{print e}' "$OUT")
    if [ -n "$wend" ]; then say "the wrapper saw the queue exit after its last start [$wend]: nothing to resume"; exit 0; fi
fi
cut=""
for g in "$EL"/*.guard.log; do
    [ -e "$g" ] || continue
    n=${g%.guard.log}
    [ -e "$n.guard.json" ] || cut="$cut ${n##*/}"
done
if [ -n "$cut" ]; then
    say "job(s) cut mid-run:$cut (guard.log, no guard.json). The queue refuses a job whose logs exist and would stop" \
        "there: not resuming, a person decides (E006 notes: rerun by hand as <tag>_r2, a logged deviation)"
    exit 0
fi
[ -f "$W" ] || { say "no wrapper $W: cannot resume (sync the working copy first)"; exit 1; }
say "cut off: last start at line $start, last line [$(tail -1 "$Q")]"
if [ $DRY = 0 ]; then
    echo "$(date '+%F %T') boot resume: the PC restarted (WSL up $(uptime_s) s) and this run had no QUEUE E006 DONE or" \
        "STOPPED line: it was cut off. Restarted by ~/planck/boot.d/20_e006.sh (pc/RUNBOOK.txt RESUME AFTER A REBOOT)." >> "$Q"
fi
launch 'queue_e006\.sh' bash "$W"
