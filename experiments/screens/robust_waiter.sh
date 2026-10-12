#!/bin/bash
# robust_waiter.sh: the one file pcdetach.sh starts for plans/robustness_k5 (experiments/SCREENS.txt AMENDMENT
# SEEDS-103-105, STARTING IT UNATTENDED). pcdetach.sh copies only this file to planck-kit and runs it with no arguments:
# chain_waiter.sh must already sit beside it, byte for byte the committed file (its sha256 is pinned here; anything else
# is refused), and the commit holding the amendment must be planck-kit/planck_robust.bundle. One detached process:
#  0. E3 first: while an E3 queue, launcher or waiter runs (pgrep), it waits. Once none runs, E3's queue log
#     (~/planck/runs/E3/queue_e3.txt) must end cleanly: after its last "queue start" line, the last two lines are
#     "MARK <text>" and "plan finished: <that start line's plan>". Any other end: GIVEUP. No log: no E3 queue ran, go on.
#  1. chain_waiter.sh as a child: once stage2_replay (at 2aa3787) has ended cleanly at its mark (the queue log's state
#     since 2026-10-10 23:51:19), it clones planck_robust.bundle's main into ~/planck/qcode/<commit12>, checks
#     plans/CURRENT (robustness_k5) and the plan's wait line there and runs queue_screens.sh on robustness_k5; it returns
#     when that queue ends, or when it gives up. It waits while any SCREENS queue or launcher runs.
#  2. replay_waiter.sh's relaunch: every POLL s chain_waiter.sh's decide() (sourced) reads the queue log for
#     robustness_k5 at the bundle's commit:
#       LAUNCH  robustness_k5 ended cleanly at its mark: step 3.
#       HOLD    after a 2 h gpu.lock wait: queue_screens.sh again, in the foreground, from the same clone and plan (the
#               queue skips scored runs; train.py resumes a cut run), at most MAXR (24) times.
#       HOLD    with no end line (a kill): held at most GRACE s (2700) for a new start, then GIVEUP.
#       WAIT    a SCREENS queue or launcher runs. If stage2_replay is still the last start line, step 1 launched
#               nothing: GIVEUP.
#       other   GIVEUP (another plan or commit, another end, STOP, a cancel file): exit 1, for a person.
#  3. The C6 runner's robust group (AMENDMENT SEEDS-103-105, C6 RUNNER): that clone's diag_queue.sh --code <clone>
#     --groups robust, in the foreground (it takes gpu.lock per hold and waits out lock waits itself). Exit 0: DONE.
#     Any other exit: GIVEUP. Started again, done jobs are skipped.
# Log: ~/planck/logs/robust_waiter.log (every step; chain_waiter.sh's own lines in robust_waiter.log too, since NAME is
# robust). One robust_waiter at a time (flock on robust_waiter_all.lock, not passed to the queues it starts). Cancel: touch ~/planck/logs/robust_waiter.CANCEL (read before each step's launch and every poll). A reboot
# kills this process (boot.d/40_screens.sh resumes a cut SCREENS run); the same pcdetach.sh command then watches again
# (step 1 says DONE from its stamp). CW_* variables exist for the tests only; the PC run sets none.
H=$(cd "$(dirname "$0")" && pwd)
SHA=920db4c14819bec872971f53f1cac815d93b1c4e9398ef5c5114c004c3f51642
REPLAY_REF=2aa37874b34fc49b6e564e795482b46662212c44
MAXR=${CW_MAXR:-24}
EPAT=${CW_EPAT:-'queue_e2\.sh --exp E3|queue_e3\.sh|launch_e3\.sh|e3p3_waiter\.sh'}
mkdir -p "${CW_HOME:-$HOME/planck}/logs" || exit 1
exec >> "${CW_HOME:-$HOME/planck}/logs/robust_waiter.log" 2>&1
if [ "$(sha256sum "$H/chain_waiter.sh" 2>/dev/null | cut -c1-64)" != "$SHA" ]; then
    echo "$(date '+%F %T') GIVEUP: $H/chain_waiter.sh is missing or not sha256 ${SHA:0:16}"
    exit 1
fi
. "$H/chain_waiter.sh"                 # its settings and functions (say, args, running, decide); main does not run
CANCEL0=$LOGD/robust_waiter.CANCEL
E3LOG=$P/runs/E3/queue_e3.txt
exec 5>> "$LOGD/robust_waiter_all.lock"        # one robust_waiter at a time (its children do not inherit it)
flock -n 5 || { say "another robust_waiter holds $LOGD/robust_waiter_all.lock: exiting"; exit 1; }

e3_state() {   # one line: WAIT, OK or GIVEUP, from pgrep and E3's queue log (read only)
    local n sline plan tail last prev
    pgrep -f "$EPAT" > /dev/null 2>&1
    case $? in 0) echo "WAIT an E3 queue, launcher or waiter runs"; return ;; 1) ;;
               *) echo "GIVEUP pgrep failed: cannot tell whether an E3 queue runs"; return ;; esac
    [ -f "$E3LOG" ] || { echo "OK no E3 queue log: no E3 queue ran"; return; }
    n=$(tr -d '\000' < "$E3LOG" | grep -anE "^$TS queue start: exp E3, plan " | tail -1 | cut -d: -f1)
    [ -n "$n" ] || { echo "GIVEUP no queue start line in $E3LOG"; return; }
    sline=$(tr -d '\000' < "$E3LOG" | sed -n "${n}p")
    plan=$(printf '%s\n' "$sline" | sed -E 's/^.* queue start: exp E3, plan ([^ ]+\.txt) .*$/\1/')
    tail=$(tr -d '\000' < "$E3LOG" | tail -n +$((n + 1)) | grep -av '^[[:space:]]*$')
    last=$(printf '%s\n' "$tail" | tail -1)
    prev=$(printf '%s\n' "$tail" | tail -2 | head -1)
    if printf '%s\n' "$last" | grep -aqE "^$TS plan finished: ${plan//./\\.}$" && \
       printf '%s\n' "$prev" | grep -aqE "^$TS MARK [A-Z0-9 ]+$" && [ "$prev" != "$last" ]; then
        echo "OK E3's last queue ($plan) ended cleanly: [$prev] [$last]"; return
    fi
    echo "GIVEUP E3's last queue ($plan) did not end cleanly: [$prev] [$last]"
}

# step 0
said=
while :; do
    [ -e "$CANCEL0" ] && { say "GIVEUP: $CANCEL0 exists"; exit 1; }
    r=$(e3_state)
    [ "$r" = "$said" ] || say "step 0: $r"
    said=$r
    case $r in
        OK*) break ;;
        WAIT*) sleep "$POLL" ;;
        *) say "GIVEUP: step 0"; exit 1 ;;
    esac
done

# step 1
bash "$H/chain_waiter.sh" robust stage2_replay $REPLAY_REF "SCREENS BASE REPLAY DONE" robustness_k5 planck_robust.bundle 5>&-
rc=$?
say "step 1 (chain_waiter.sh) ended, exit $rc; step 2 watches robustness_k5"

# step 2
REF=$(git bundle list-heads "${KIT:-$H}/planck_robust.bundle" refs/heads/main 2>/dev/null | cut -d' ' -f1)
args robust_requeue robustness_k5 "$REF" "SCREENS ROBUSTNESS K5 DONE" robust_requeue_none planck_robust.bundle \
    "stage2_replay@$REPLAY_REF" || { say "GIVEUP: no main in ${KIT:-$H}/planck_robust.bundle [$REF]"; exit 1; }
exec 6>> "$LOGD/robust_requeue.lock"
flock -n 6 || { say "another robust_waiter watches robustness_k5: exiting"; exit 1; }
Q=$P/qcode/${REF:0:12}
n=0 hold= said=
while :; do
    [ -e "$CANCEL0" ] && { say "GIVEUP: $CANCEL0 exists"; exit 1; }
    r=$(decide "$(running)")
    [ "$r" = "$said" ] || say "$r"
    said=$r
    case $r in
        LAUNCH*) say "robustness_k5 ended cleanly at its mark"; break ;;
        HOLD*"stopped after a 2 h gpu.lock wait"*)
            hold=
            [ "$n" -lt "$MAXR" ] || { say "GIVEUP: $MAXR relaunches after 2 h gpu.lock waits"; exit 1; }
            if [ "$(git -C "$Q" rev-parse HEAD 2>/dev/null)" != "$REF" ] || [ ! -f "$Q/$XREL/plans/robustness_k5.txt" ]; then
                say "GIVEUP: no clone of ${REF:0:12} with plans/robustness_k5.txt in qcode"; exit 1
            fi
            n=$((n + 1)); said=
            say "relaunch $n of robustness_k5 from qcode/${REF:0:12}"
            bash "$Q/$XREL/queue_screens.sh" --code "$Q" --plan "$Q/$XREL/plans/robustness_k5.txt" 5>&- 6>&-
            say "relaunched queue ended, exit $?" ;;
        HOLD*) [ -n "$hold" ] || hold=$(date +%s)
               if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then
                   say "GIVEUP: no new robustness_k5 start within $GRACE s"; exit 1
               fi ;;
        WAIT*"earlier in the chain"*) say "GIVEUP: step 1 launched nothing"; exit 1 ;;
        WAIT*) hold= ;;
        *) say "GIVEUP: $r"; exit 1 ;;
    esac
    sleep "$POLL"
done

# step 3
[ -e "$CANCEL0" ] && { say "GIVEUP: $CANCEL0 exists"; exit 1; }
if [ "$(git -C "$Q" rev-parse HEAD 2>/dev/null)" != "$REF" ] || [ ! -f "$Q/$XREL/diag_queue.sh" ]; then
    say "GIVEUP: no clone of ${REF:0:12} with diag_queue.sh in qcode"; exit 1
fi
say "step 3: the C6 runner's robust group from qcode/${REF:0:12}"
bash "$Q/$XREL/diag_queue.sh" --code "$Q" --groups robust 5>&- 6>&-
rc=$?
if [ "$rc" = 0 ]; then
    say "DONE: robustness_k5 ended cleanly at its mark and the C6 robust group is done"; exit 0
fi
say "GIVEUP: the C6 robust group's queue ended, exit $rc (its log: ~/planck/logs/c6diag.log)"
exit 1
