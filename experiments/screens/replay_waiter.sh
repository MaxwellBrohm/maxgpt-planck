#!/bin/bash
# replay_waiter.sh: the one file pcdetach.sh starts for plans/stage2_replay (experiments/SCREENS.txt AMENDMENT
# BASE-REPLAY, LAUNCH, and INTEGRATION 2026-10-10 after it). pcdetach.sh copies only this file to planck-kit and runs
# it with no arguments: chain_waiter.sh must already sit beside it, byte for byte the committed file (its sha256 is
# pinned here; anything else is refused), and the commit holding the entry must be planck-kit/planck_replay.bundle.
# One detached process, two steps:
#  1. chain_waiter.sh as a child, with the entry's arguments: once stage2_base_diag (at 27d8cb7) has ended cleanly at
#     its mark (on the PC since 2026-10-07 03:03:58), it clones planck_replay.bundle's main into ~/planck/qcode/
#     <commit12>, checks plans/CURRENT and the plan's wait line there and runs queue_screens.sh on stage2_replay; it
#     returns when that queue ends, or when it gives up.
#  2. The relaunch chain_waiter.sh leaves to a person. Every POLL s (300) chain_waiter.sh's own decide() (sourced; its
#     main runs only when the file is executed) reads the queue log for stage2_replay at the bundle's commit, with
#     stage2_base_diag as the earlier plan and a next plan that never runs. Its first word:
#       LAUNCH  stage2_replay ended cleanly at its mark: DONE, exit 0.
#       HOLD    after a 2 h gpu.lock wait: queue_screens.sh again, in the foreground, from the same clone and plan
#               (boot.d/40_screens.sh's restart command; the queue skips scored runs), at most MAXR (24) times.
#       HOLD    with no end line (a kill): held at most GRACE s (2700) for a new start, then GIVEUP.
#       WAIT    a SCREENS queue or launcher runs. If stage2_base_diag is still the last start line, step 1 launched
#               nothing: GIVEUP.
#       other   GIVEUP (another plan or commit, another end, STOP, a cancel file): exit 1, for a person.
# Log: ~/planck/logs/replay_waiter.log (both steps). Cancel: touch ~/planck/logs/replay_waiter.CANCEL. A reboot kills
# this process (boot.d/40_screens.sh resumes a cut run); the same pcdetach.sh command then watches again (step 1 says
# DONE from its stamp). CW_* variables exist for the tests only; the PC run sets none.
H=$(cd "$(dirname "$0")" && pwd)
SHA=920db4c14819bec872971f53f1cac815d93b1c4e9398ef5c5114c004c3f51642
BASE_REF=27d8cb7505248ff4391a1dee9a51bad617938195
MAXR=${CW_MAXR:-24}
mkdir -p "${CW_HOME:-$HOME/planck}/logs" || exit 1
exec >> "${CW_HOME:-$HOME/planck}/logs/replay_waiter.log" 2>&1
if [ "$(sha256sum "$H/chain_waiter.sh" 2>/dev/null | cut -c1-64)" != "$SHA" ]; then
    echo "$(date '+%F %T') GIVEUP: $H/chain_waiter.sh is missing or not sha256 ${SHA:0:16}"
    exit 1
fi
bash "$H/chain_waiter.sh" replay stage2_base_diag $BASE_REF "SCREENS BASE DIAG DONE" stage2_replay planck_replay.bundle
rc=$?
. "$H/chain_waiter.sh"                 # its settings and functions (say, args, running, decide)
say "step 1 (chain_waiter.sh) ended, exit $rc; step 2 watches stage2_replay"
REF=$(git bundle list-heads "${KIT:-$H}/planck_replay.bundle" refs/heads/main 2>/dev/null | cut -d' ' -f1)
args replay_requeue stage2_replay "$REF" "SCREENS BASE REPLAY DONE" replay_requeue_none planck_replay.bundle \
    "stage2_base_diag@$BASE_REF" || { say "GIVEUP: no main in ${KIT:-$H}/planck_replay.bundle [$REF]"; exit 1; }
exec 6>> "$LOGD/replay_requeue.lock"
flock -n 6 || { say "another replay_waiter watches stage2_replay: exiting"; exit 1; }
Q=$P/qcode/${REF:0:12}
n=0 hold= said=
while :; do
    [ -e "$LOGD/replay_waiter.CANCEL" ] && { say "GIVEUP: $LOGD/replay_waiter.CANCEL exists"; exit 1; }
    r=$(decide "$(running)")
    [ "$r" = "$said" ] || say "$r"
    said=$r
    case $r in
        LAUNCH*) say "DONE: stage2_replay ended cleanly at its mark"; exit 0 ;;
        HOLD*"stopped after a 2 h gpu.lock wait"*)
            hold=
            [ "$n" -lt "$MAXR" ] || { say "GIVEUP: $MAXR relaunches after 2 h gpu.lock waits"; exit 1; }
            if [ "$(git -C "$Q" rev-parse HEAD 2>/dev/null)" != "$REF" ] || [ ! -f "$Q/$XREL/plans/stage2_replay.txt" ]; then
                say "GIVEUP: no clone of ${REF:0:12} with plans/stage2_replay.txt in qcode"; exit 1
            fi
            n=$((n + 1)); said=
            say "relaunch $n of stage2_replay from qcode/${REF:0:12}"
            bash "$Q/$XREL/queue_screens.sh" --code "$Q" --plan "$Q/$XREL/plans/stage2_replay.txt" 6>&-
            say "relaunched queue ended, exit $?" ;;
        HOLD*) [ -n "$hold" ] || hold=$(date +%s)
               if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then
                   say "GIVEUP: no new stage2_replay start within $GRACE s"; exit 1
               fi ;;
        WAIT*"earlier in the chain"*) say "GIVEUP: step 1 launched nothing"; exit 1 ;;
        WAIT*) hold= ;;
        *) say "GIVEUP: $r"; exit 1 ;;
    esac
    sleep "$POLL"
done
