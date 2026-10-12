#!/bin/bash
# boot.d/30_rc12.sh: after a reboot, resume the RC-12 dev-baseline queue (rc12/queue_dev_baselines.sh, rc12/notes.txt
# STEP 9b; the Doge pass STEP 9c; gpu.lock sessions STEP 9g).
#   bash 30_rc12.sh [--dry-run]
# Resumes only when all hold: no stop file (bootlib.sh: ~/planck/boot.d/STOP, boot.d/rc12.STOP, ~/planck/STOP,
# ~/planck/PAUSE, plus the queue's own ~/planck/logs/rc12_dev_queue.STOP); nothing of the queue runs (the queue, the
# dg_follow.sh follow-up, a dev_batch.py process); no ~/planck/logs/rc12_dev_queue.DONE (the queue removes it at start
# and writes it at its end); and rc12_dev_queue.log's last "queue: start," has no "queue: end;" or "queue: STOP file"
# after it.
# Restart path (STEP 9c "Manual restart"): Q_CODE=CODE bash CODE/queue_dev_baselines.sh, CODE = the code dir of the
# cut-off run, read from its "queue: start, code CODE," line: the queue has run from rc12_q9, rc12_dg, the session
# queue rc12_dg3 (STEP 9g) and, for the Doge pass (run_doge_after.sh), rc12_doge2. Same queue name, lock, status file
# and root, so finished runs are skipped. RC12_CODE, when set, overrides it. No queue script there: not resumed.
# Settings (rc12/notes.txt STEP 14X): a queue since STEP 14X names its run on the start line (", run ID") and writes
# ~/planck/logs/rc12_dev_queue.run before it (run=ID, then Q_MODELS, Q_RENDERS, Q_SEEDS, Q_ROOT, Q_ENGINES,
# Q_TIMEOUT, Q_SESSION_S, Q_SESSION_TAIL and Q_LIMIT as that run used them); the restart passes those, so a cut-off
# extras phase (e.g. cRia's shards with their 1,020 s cap) resumes as it ran, not as the default 13-model list. Not
# resumed: a run id without a run file of that id, a missing key, an empty value (Q_LIMIT may be empty), or text
# outside [A-Za-z0-9._/: -]. A start line without a run id (an older queue) restarts with the queue's defaults.
NAME=rc12
. "$(dirname "$0")/bootlib.sh"
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
CODE=${RC12_CODE:-$(sed -n "${start}s/.* queue: start, code \([^,]*\), root .*/\1/p" "$QL")}
[ -f "$CODE/queue_dev_baselines.sh" ] || { say "no queue_dev_baselines.sh in code dir [$CODE]: cannot resume"; exit 1; }
RUN=$P/logs/rc12_dev_queue.run
rid=$(sed -n "${start}p" "$QL" | tr -d '\000\r' | sed -n 's/.*, run \([0-9T.]*\)$/\1/p')
SET=()
if [ -n "$rid" ]; then
    [ "$(grep -a '^run=' "$RUN" 2>/dev/null | head -1 | tr -d '\000\r')" = "run=$rid" ] \
        || { say "run $rid: no run file of that run ($RUN): cannot resume"; exit 1; }
    for k in Q_MODELS Q_RENDERS Q_SEEDS Q_ROOT Q_ENGINES Q_TIMEOUT Q_SESSION_S Q_SESSION_TAIL Q_LIMIT; do
        v=$(grep -a "^$k=" "$RUN" | head -1 | tr -d '\000\r')
        [ -n "$v" ] || { say "run file $RUN has no $k line: cannot resume"; exit 1; }
        v=${v#*=}
        [ -n "$v" ] || [ $k = Q_LIMIT ] || { say "run file $RUN: $k is empty: cannot resume"; exit 1; }
        case $v in *[!A-Za-z0-9._/:\ -]*) say "run file $RUN: $k [$v] has unexpected text: cannot resume"; exit 1;; esac
        SET+=("$k=$v")
    done
    say "run $rid: the run file's settings [${SET[*]}]"
else
    say "the start line names no run (a queue older than STEP 14X): the queue's defaults"
fi
say "cut off: last start at line $start, last line [$(tail -1 "$QL")]"
launch 'queue_dev_baselines\.sh' env Q_CODE="$CODE" "${SET[@]}" bash "$CODE/queue_dev_baselines.sh"
