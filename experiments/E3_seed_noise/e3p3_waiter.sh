#!/bin/bash
# e3p3_waiter.sh: the one file pcdetach.sh starts for E3 Part 3 (notes.txt PART 3 REOPENED, QUEUE). pcdetach.sh copies
# only this file to planck-kit and runs it with no arguments; the commit holding Part 3 must be planck-kit/
# planck_e3p3.bundle (on the Mac: git bundle create planck_e3p3.bundle HEAD main, then copy it to planck-kit).
# It does what launch_e3.sh does, from that bundle instead of planck.bundle: clone its main into ~/planck/qcode/<commit12>,
# check plans/CURRENT (part3) and plans/part3.txt (its sha256, pinned below) there, and run that clone's queue_e3.sh
# on part3 IN THE FOREGROUND. Then it reads only what that queue appended to ~/planck/runs/E3/queue_e3.txt:
#   DONE      part3's start line at the bundle's commit, then "MARK E3 PART 3 DONE", "plan finished: part3.txt" and
#             marks/E3_PART_3_DONE: exit 0.
#   RELAUNCH  the start line, then "gpu.lock not free after 2 h" (or queue_lib's 30 min wait on another process's GPU
#             memory under the lock) and "queue stops at: train <run>": the same queue again from the same clone (it
#             skips scored runs), at most MAXR (24) relaunches in all.
#   GIVEUP    any other end (a refusal, a scoring failure, STOP, heat 6 times, a kill, no start line): exit 1, for a
#             person. Nothing is relaunched after an end it cannot name.
# Started again (a reboot or a power cut kills it and the queue): DONE if marks/E3_PART_3_DONE exists; WAIT while an E3
# queue, launcher or E3 train.py runs; else it launches part3 again, which resumes it: the queue skips SCORED, DIVERGED
# and GAP runs, and train.py resumes a cut run from its last durable checkpoint (latest.json, E3 DEVIATION D4). Nothing
# on the PC starts it after a reboot (boot.d has no E3 entry): the same pcdetach.sh command does, by a person or a
# boot hook (~/planck/logs/e3p3_waiter.started exists from the first launch on).
# Never twice: one waiter (flock on ~/planck/logs/e3p3_waiter.lock) and the queue's own runs/E3/queue.lock.
# Log ~/planck/logs/e3p3_waiter.log. Cancel: touch ~/planck/logs/e3p3_waiter.CANCEL (read before every launch).
# E3W_* variables exist for the tests only; the PC run sets none.
P=${E3W_HOME:-$HOME/planck}
OUT=$P/runs/E3; QLOG=$OUT/queue_e3.txt; LOGD=$P/logs
POLL=${E3W_POLL:-300}; MAXR=${E3W_MAXR:-24}
PAT=${E3W_PAT:-'queue_e2\.sh --exp E3|queue_e3\.sh|launch_e3\.sh|train\.py [^ ]*/E3_seed_noise/configs/'}
KIT=${E3W_KIT:-}
XREL=experiments/E3_seed_noise; BUNDLE=planck_e3p3.bundle
PLAN_SHA=07018f0fec5729944f648b20d499d8b2d346426c81fec7cbc892fb894d1c2417
MARKF=E3_PART_3_DONE
TS='[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
CANCEL=$LOGD/e3p3_waiter.CANCEL; STARTED=$LOGD/e3p3_waiter.started
REF=; Q=

say() { echo "$(date '+%F %T') $*"; }

running() {    # 1 an E3 queue, launcher or E3 train.py runs, 0 none, E pgrep could not tell
    pgrep -f "$PAT" > /dev/null 2>&1
    case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac
}

classify() {   # classify RC TEXT: DONE, RELAUNCH or GIVEUP (first word) for the lines one queue run appended
    local rc=$1 txt last prev
    txt=$(printf '%s\n' "$2" | tr -d '\000' | grep -av '^[[:space:]]*$')
    if ! printf '%s\n' "$txt" | grep -aqE "^$TS queue start: exp E3, plan part3\.txt \(sha256 ${PLAN_SHA:0:16}\), code commit $REF$"; then
        echo "GIVEUP no part3 start line at ${REF:0:12} in what the queue wrote (exit $rc): [$(printf '%s\n' "$txt" | tail -1)]"
        return
    fi
    last=$(printf '%s\n' "$txt" | tail -1)
    prev=$(printf '%s\n' "$txt" | tail -2 | head -1)
    if [ "$rc" = 0 ] && printf '%s\n' "$prev" | grep -aqE "^$TS MARK E3 PART 3 DONE$" &&
        printf '%s\n' "$last" | grep -aqE "^$TS plan finished: part3\.txt$"; then
        if [ -e "$OUT/marks/$MARKF" ]; then echo "DONE part3 ended cleanly at its mark"
        else echo "GIVEUP a clean end but no marks/$MARKF"; fi
        return
    fi
    if [ "$rc" = 1 ] && printf '%s\n' "$last" | grep -aqE "^$TS queue stops at: train e3_5m_A_[A-Za-z0-9_]+$"; then
        if printf '%s\n' "$prev" | grep -aqE "^$TS gpu.lock not free after 2 h$"; then
            echo "RELAUNCH part3 stopped after a 2 h gpu.lock wait: [$last]"; return
        fi
        if printf '%s\n' "$prev" | grep -aqE "^$TS GPU memory in use \[[0-9]*\] MiB > 3000 with the lock held: waiting$"; then
            echo "RELAUNCH part3 stopped after 30 min of another process's GPU memory under the lock: [$last]"; return
        fi
    fi
    echo "GIVEUP the queue ended another way (exit $rc): [$prev] [$last]"
}

clone() {      # launch_e3.sh's clone step from planck-kit/BUNDLE, with part3's checks: sets REF and Q, or exits 1
    local W B PLAN
    if [ -z "$KIT" ]; then
        W=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r'); KIT=/mnt/c/Users/$W/planck-kit
    fi
    B=$KIT/$BUNDLE
    [ -f "$B" ] || { say "GIVEUP: no bundle $B"; exit 1; }
    REF=$(git bundle list-heads "$B" refs/heads/main | cut -d' ' -f1)
    [ -n "$REF" ] || { say "GIVEUP: $B has no main"; exit 1; }
    Q=$P/qcode/${REF:0:12}
    [ -d "$Q/.git" ] || git clone -q "$B" "$Q" || { say "GIVEUP: git clone of $B into $Q failed"; exit 1; }
    git -C "$Q" checkout -q "$REF" && [ "$(git -C "$Q" rev-parse HEAD)" = "$REF" ] || {
        say "GIVEUP: checkout of $REF in $Q failed"; exit 1; }
    PLAN=$(grep -v '^#' "$Q/$XREL/plans/CURRENT" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ "$PLAN" = part3 ] || { say "GIVEUP: plans/CURRENT at ${REF:0:12} names [$PLAN], not part3"; exit 1; }
    [ "$(sha256sum "$Q/$XREL/plans/part3.txt" 2>/dev/null | cut -c1-64)" = "$PLAN_SHA" ] || {
        say "GIVEUP: plans/part3.txt at ${REF:0:12} is missing or not sha256 ${PLAN_SHA:0:16}"; exit 1; }
    [ -f "$Q/$XREL/queue_e3.sh" ] || { say "GIVEUP: no queue_e3.sh at ${REF:0:12}"; exit 1; }
}

main() {
    set -u
    mkdir -p "$LOGD" || exit 1
    exec >> "$LOGD/e3p3_waiter.log" 2>&1
    exec 7>> "$LOGD/e3p3_waiter.lock"
    flock -n 7 || { say "another e3p3_waiter holds $LOGD/e3p3_waiter.lock: exiting"; exit 1; }
    say "=== e3p3_waiter (pid $$): E3 Part 3 from $BUNDLE; poll $POLL s, at most $MAXR relaunches"
    local n=0 r rc before new waited=
    while :; do
        [ -e "$CANCEL" ] && { say "GIVEUP: $CANCEL exists"; exit 1; }
        [ -e "$OUT/marks/$MARKF" ] && { say "DONE: marks/$MARKF exists"; exit 0; }
        [ -e "$P/STOP" ] && { say "GIVEUP: $P/STOP exists"; exit 1; }
        r=$(running)
        [ "$r" = E ] && { say "GIVEUP: pgrep failed: cannot tell whether an E3 queue runs"; exit 1; }
        if [ "$r" = 1 ]; then
            [ -n "$waited" ] || say "WAIT an E3 queue, launcher or train.py runs"
            waited=1; sleep "$POLL"; continue
        fi
        waited=
        [ -n "$Q" ] || clone
        before=$(wc -c < "$QLOG" 2>/dev/null || echo 0)
        [ -e "$STARTED" ] || echo "$(date '+%F %T') first launch of part3 from ${REF:0:12}" > "$STARTED"
        say "launch $((n + 1)): queue_e3.sh on part3 from qcode/${REF:0:12} (code $REF)"
        bash "$Q/$XREL/queue_e3.sh" --plan "$Q/$XREL/plans/part3.txt" 7>&-
        rc=$?
        new=$(tail -c +$((before + 1)) "$QLOG" 2>/dev/null)
        r=$(classify "$rc" "$new")
        say "queue ended, exit $rc: $r"
        case $r in
            DONE*) exit 0 ;;
            RELAUNCH*) n=$((n + 1))
                       [ "$n" -le "$MAXR" ] || { say "GIVEUP: $MAXR relaunches after 2 h gpu.lock waits"; exit 1; } ;;
            *) exit 1 ;;
        esac
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
