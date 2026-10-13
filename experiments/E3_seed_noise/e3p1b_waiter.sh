#!/bin/bash
# e3p1b_waiter.sh: the one file pcdetach.sh starts for E3 Part 1b and Part 3 at 20M (notes.txt ADDENDUM 2026-10-12,
# QUEUE). pcdetach.sh copies only this file to planck-kit and runs it with no arguments; the commit holding the addendum
# must be planck-kit/planck_e3p1b.bundle (on the Mac: git bundle create planck_e3p1b.bundle HEAD main, copied to
# planck-kit). Every POLL s (300) decide() reads files and processes only, and says one of:
#   WAIT    an E3 queue, launcher, waiter or E3 train.py runs; a SCREENS queue or launcher, robust_waiter.sh,
#           chain_waiter.sh or the C6 runner (diag_queue.sh, diag_runner.py) runs; marks/SCREENS_ROBUSTNESS_K5_DONE is
#           not there yet; robust_waiter.log has no DONE line for the C6 robust group yet; or under MINFREE (41) GB free.
#   LAUNCH  none of those, the SCREENS log's last robustness_k5 lines are its clean end ("MARK SCREENS ROBUSTNESS K5
#           DONE" then "plan finished: robustness_k5.txt"), and E3's last queue either ended cleanly or is this plan
#           (a restart resumes it). Decided once more, then, as launch_e3.sh does but from planck_e3p1b.bundle: clone
#           its main into ~/planck/qcode/<commit12>; check plans/CURRENT (part1b_p3at20m), the plan's sha256 (pinned
#           below) and the five code trees (git tree ids pinned below: the trees PART 1b PARITY compared, equal to Part
#           3's commit 4962426); run that clone's queue_e3.sh on the plan IN THE FOREGROUND with QUEUE_MIN_FREE_GB=41
#           (a run starts only with 41 GB free; one run writes under 0.6 GB).
#   DONE    marks/E3_PART_3_AT_20M_DONE exists: exit 0.
#   GIVEUP  CANCEL or ~/planck/STOP, pgrep failing, a robustness mark without the log's clean end, E3's last queue being
#           another plan that did not end cleanly, a wrong bundle: exit 1, for a person.
# After a launch it reads only what that queue appended to ~/planck/runs/E3/queue_e3.txt:
#   DONE      the plan's start line at the bundle's commit, then "MARK E3 PART 3 AT 20M DONE", "plan finished:
#             part1b_p3at20m.txt", exit 0 and the mark file: exit 0.
#   RELAUNCH  exit 1 after "gpu.lock not free after 2 h" (or queue_lib's 30 min wait on another process's GPU memory
#             under the lock) and "queue stops at: train <run>": the same queue again from the same clone (it skips
#             scored runs) once no E3 process runs, without waiting on SCREENS again; at most MAXR (24) relaunches.
#   GIVEUP    any other end (a refusal, a scoring failure, STOP, heat 6 times, disk, a kill): exit 1, for a person.
# Started again (a reboot or a power cut kills it and the queue): DONE from the mark, or the same checks and a launch,
# which resumes (the queue skips SCORED, DIVERGED and GAP runs; train.py resumes a cut run from latest.json, D4).
# Nothing on the PC starts it after a reboot (boot.d has no E3 entry): the same pcdetach.sh command does, by a person or a
# later session; until then the GPU sits idle for E3.
# Never twice: one waiter (flock on ~/planck/logs/e3p1b_waiter.lock) and the queue's own runs/E3/queue.lock.
# Log ~/planck/logs/e3p1b_waiter.log. Cancel: touch ~/planck/logs/e3p1b_waiter.CANCEL (read at every poll).
#   bash e3p1b_waiter.sh --decide   prints decide()'s line for the real state and exits (reads only; launches nothing)
# E3W_* variables exist for the tests and the scratch dry run only; the PC run sets none.
P=${E3W_HOME:-$HOME/planck}
OUT=$P/runs/E3; QLOG=$OUT/queue_e3.txt; LOGD=$P/logs
SOUT=$P/runs/SCREENS; SLOG=$SOUT/queue_screens.txt; RWLOG=$LOGD/robust_waiter.log
POLL=${E3W_POLL:-300}; MAXR=${E3W_MAXR:-24}; MINFREE=${E3W_MINFREE:-41}
PAT=${E3W_PAT:-'queue_e2\.sh --exp E3|queue_e3\.sh|launch_e3\.sh|e3p3_waiter\.sh|train\.py [^ ]*/E3_seed_noise/configs/'}
SPAT=${E3W_SPAT:-'queue_screens\.sh|launch_screens\.sh|robust_waiter\.sh|chain_waiter\.sh|diag_queue\.sh|diag_runner\.py'}
KIT=${E3W_KIT:-}
XREL=experiments/E3_seed_noise; BUNDLE=planck_e3p1b.bundle; PLAN=part1b_p3at20m
PLAN_SHA=17c939a5e86f3d63f08ef88bbde6c85e780bab9635d6252f224f1026774a38e0
TREES=${E3W_TREES:-"harness:67282e87457ef3da20707843fec2563bdf4aff7a data_prep:307117591dcb88bea0b359fc1e7a28463399ccb0 \
tokenizer:f90a86b9e2d17b5ea4a365f6c2d095e84e11d3d3 corpus:7785d282489d1f4abf1b7f2ff03a5329775fbcc0 \
experiments/E2_lr_transfer:ec65e287386d553686bf53baa7a468eb8470ecb5"}
MARKF=E3_PART_3_AT_20M_DONE; RMARK=SCREENS_ROBUSTNESS_K5_DONE
TS='[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
CANCEL=$LOGD/e3p1b_waiter.CANCEL; STARTED=$LOGD/e3p1b_waiter.started
REF=; Q=

say() { echo "$(date '+%F %T') $*"; }

running() {    # running PATTERN: 1 a process matches, 0 none, E pgrep could not tell
    pgrep -f "$1" > /dev/null 2>&1
    case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac
}

free_gb() { df -BG --output=avail "$P" 2>/dev/null | tail -1 | tr -dc '0-9'; }

e3_last() {    # OK or GIVEUP: E3's last queue ended cleanly, or it is this plan (resumed by a launch)
    local n sline plan tail last prev
    [ -f "$QLOG" ] || { echo "GIVEUP no E3 queue log $QLOG"; return; }
    n=$(tr -d '\000' < "$QLOG" | grep -anE "^$TS queue start: exp E3, plan " | tail -1 | cut -d: -f1)
    [ -n "$n" ] || { echo "GIVEUP no queue start line in $QLOG"; return; }
    sline=$(tr -d '\000' < "$QLOG" | sed -n "${n}p")
    plan=$(printf '%s\n' "$sline" | sed -E 's/^.* queue start: exp E3, plan ([^ ]+\.txt) .*$/\1/')
    [ "$plan" = "$PLAN.txt" ] && { echo "OK E3's last queue is $PLAN.txt: a launch resumes it"; return; }
    tail=$(tr -d '\000' < "$QLOG" | tail -n +$((n + 1)) | grep -av '^[[:space:]]*$')
    last=$(printf '%s\n' "$tail" | tail -1); prev=$(printf '%s\n' "$tail" | tail -2 | head -1)
    if printf '%s\n' "$last" | grep -aqE "^$TS plan finished: ${plan//./\\.}$" && [ "$prev" != "$last" ] && \
       printf '%s\n' "$prev" | grep -aqE "^$TS MARK [A-Z0-9 ]+$"; then
        echo "OK E3's last queue ($plan) ended cleanly"; return
    fi
    echo "GIVEUP E3's last queue ($plan) did not end cleanly: [$prev] [$last]"
}

decide() {     # decide E3_RUNNING SCREENS_RUNNING [relaunch]: one line, first word LAUNCH, WAIT, GIVEUP or DONE
    local f e m
    [ -e "$CANCEL" ] && { echo "GIVEUP $CANCEL exists"; return; }
    [ -e "$OUT/marks/$MARKF" ] && { echo "DONE marks/$MARKF exists"; return; }
    [ -e "$P/STOP" ] && { echo "GIVEUP $P/STOP exists"; return; }
    { [ "$1" = E ] || [ "$2" = E ]; } && { echo "GIVEUP pgrep failed: cannot tell what runs"; return; }
    [ "$1" = 1 ] && { echo "WAIT an E3 queue, launcher, waiter or train.py runs"; return; }
    [ "${3:-}" = relaunch ] && { echo "LAUNCH relaunch after a lock wait"; return; }
    [ "$2" = 1 ] && { echo "WAIT a SCREENS queue or launcher, robust_waiter, chain_waiter or the C6 runner runs"; return; }
    [ -e "$SOUT/marks/$RMARK" ] || { echo "WAIT no marks/$RMARK yet"; return; }
    m=$(tr -d '\000' < "$SLOG" 2>/dev/null | grep -aE -A1 "^$TS MARK SCREENS ROBUSTNESS K5 DONE$" | tail -1)
    printf '%s\n' "$m" | grep -aqE "^$TS plan finished: robustness_k5\.txt$" || {
        echo "GIVEUP marks/$RMARK exists but the SCREENS log has no clean robustness_k5 end: [$m]"; return; }
    grep -aqE "^$TS DONE: robustness_k5 ended cleanly at its mark and the C6 robust group is done$" "$RWLOG" 2>/dev/null || {
        echo "WAIT robust_waiter.log has no DONE line for the C6 robust group yet"; return; }
    e=$(e3_last)
    case $e in OK*) ;; *) echo "$e"; return ;; esac
    f=$(free_gb)
    [ "${f:-0}" -ge "$MINFREE" ] || { echo "WAIT disk: [$f] GB free (< $MINFREE)"; return; }
    echo "LAUNCH robustness_k5 and the C6 robust group ended; $e; $f GB free"
}

classify() {   # classify RC TEXT: DONE, RELAUNCH or GIVEUP (first word) for the lines one queue run appended
    local rc=$1 txt last prev
    txt=$(printf '%s\n' "$2" | tr -d '\000' | grep -av '^[[:space:]]*$')
    if ! printf '%s\n' "$txt" | grep -aqE "^$TS queue start: exp E3, plan $PLAN\.txt \(sha256 ${PLAN_SHA:0:16}\), code commit $REF$"; then
        echo "GIVEUP no $PLAN start line at ${REF:0:12} in what the queue wrote (exit $rc): [$(printf '%s\n' "$txt" | tail -1)]"
        return
    fi
    last=$(printf '%s\n' "$txt" | tail -1); prev=$(printf '%s\n' "$txt" | tail -2 | head -1)
    if [ "$rc" = 0 ] && printf '%s\n' "$prev" | grep -aqE "^$TS MARK E3 PART 3 AT 20M DONE$" &&
        printf '%s\n' "$last" | grep -aqE "^$TS plan finished: $PLAN\.txt$"; then
        if [ -e "$OUT/marks/$MARKF" ]; then echo "DONE $PLAN ended cleanly at its mark"
        else echo "GIVEUP a clean end but no marks/$MARKF"; fi
        return
    fi
    if [ "$rc" = 1 ] && printf '%s\n' "$last" | grep -aqE "^$TS queue stops at: train e3_(5m|20m)_A_[A-Za-z0-9_]+$"; then
        if printf '%s\n' "$prev" | grep -aqE "^$TS gpu.lock not free after 2 h$"; then
            echo "RELAUNCH $PLAN stopped after a 2 h gpu.lock wait: [$last]"; return
        fi
        if printf '%s\n' "$prev" | grep -aqE "^$TS GPU memory in use \[[0-9]*\] MiB > 3000 with the lock held: waiting$"; then
            echo "RELAUNCH $PLAN stopped after 30 min of another process's GPU memory under the lock: [$last]"; return
        fi
    fi
    echo "GIVEUP the queue ended another way (exit $rc): [$prev] [$last]"
}

clone() {      # launch_e3.sh's clone step from planck-kit/BUNDLE, with this plan's checks: sets REF and Q, or exits 1
    local W B CUR t got
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
    CUR=$(grep -v '^#' "$Q/$XREL/plans/CURRENT" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ "$CUR" = "$PLAN" ] || { say "GIVEUP: plans/CURRENT at ${REF:0:12} names [$CUR], not $PLAN"; exit 1; }
    [ "$(sha256sum "$Q/$XREL/plans/$PLAN.txt" 2>/dev/null | cut -c1-64)" = "$PLAN_SHA" ] || {
        say "GIVEUP: plans/$PLAN.txt at ${REF:0:12} is missing or not sha256 ${PLAN_SHA:0:16}"; exit 1; }
    [ -f "$Q/$XREL/queue_e3.sh" ] || { say "GIVEUP: no queue_e3.sh at ${REF:0:12}"; exit 1; }
    for t in $TREES; do
        got=$(git -C "$Q" rev-parse "$REF:${t%%:*}" 2>/dev/null)
        [ "$got" = "${t##*:}" ] || {
            say "GIVEUP: ${t%%:*} at ${REF:0:12} is tree [$got], not ${t##*:} (the tree PART 1b PARITY compared)"; exit 1; }
    done
}

main() {
    set -u
    if [ "${1:-}" = --decide ]; then
        decide "$(running "$PAT")" "$(running "$SPAT")"; exit 0
    fi
    mkdir -p "$LOGD" || exit 1
    exec >> "$LOGD/e3p1b_waiter.log" 2>&1
    exec 7>> "$LOGD/e3p1b_waiter.lock"
    flock -n 7 || { say "another e3p1b_waiter holds $LOGD/e3p1b_waiter.lock: exiting"; exit 1; }
    say "=== e3p1b_waiter (pid $$): E3 $PLAN from $BUNDLE; poll $POLL s, at most $MAXR relaunches, $MINFREE GB free"
    local n=0 r rc before new said= i=0 mode=
    while :; do
        r=$(decide "$(running "$PAT")" "$(running "$SPAT")" "$mode")
        { [ "$r" != "$said" ] || [ $((i % 12)) = 0 ]; } && say "$r"
        said=$r; i=$((i + 1))
        case $r in
            DONE*) exit 0 ;;
            WAIT*) sleep "$POLL"; continue ;;
            LAUNCH*) ;;
            *) exit 1 ;;
        esac
        r=$(decide "$(running "$PAT")" "$(running "$SPAT")" "$mode")      # once more, just before the launch
        case $r in LAUNCH*) ;; *) say "recheck: $r"; sleep "$POLL"; continue ;; esac
        [ -n "$Q" ] || clone
        before=$(wc -c < "$QLOG" 2>/dev/null || echo 0)
        [ -e "$STARTED" ] || echo "$(date '+%F %T') first launch of $PLAN from ${REF:0:12}" > "$STARTED"
        say "launch $((n + 1)): queue_e3.sh on $PLAN from qcode/${REF:0:12} (code $REF)"
        QUEUE_MIN_FREE_GB=$MINFREE bash "$Q/$XREL/queue_e3.sh" --plan "$Q/$XREL/plans/$PLAN.txt" 7>&-
        rc=$?
        new=$(tail -c +$((before + 1)) "$QLOG" 2>/dev/null)
        r=$(classify "$rc" "$new")
        say "queue ended, exit $rc: $r"; said=
        case $r in
            DONE*) exit 0 ;;
            RELAUNCH*) n=$((n + 1)); mode=relaunch
                       [ "$n" -le "$MAXR" ] || { say "GIVEUP: $MAXR relaunches after lock waits"; exit 1; } ;;
            *) exit 1 ;;
        esac
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
