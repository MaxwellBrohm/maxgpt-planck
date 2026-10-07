#!/bin/bash
# s006_waiter.sh: start the SCREENS queue on plans/stage2_s006_seeds (experiments/SCREENS.txt AMENDMENT S006-REINSTATE)
# once the running stage2_seeds queue has ended cleanly at its mark. Self-contained: pcdetach.sh copies only this file
# to the PC and runs it in WSL with no arguments, detached from any SSH session (install: that entry). It needs the
# commit holding that entry as a git bundle at planck-kit/planck_s006.bundle; planck-kit/planck.bundle stays at
# d21df3d, the commit stage2_seeds was launched from, which the stage 2 autopilot's W6 relaunch check reads.
# Every POLL s (300) it reads ~/planck/runs/SCREENS, read only, and decide() says one of:
#   LAUNCH  marks/SCREENS_STAGE_2_SEEDS_DONE exists, no SCREENS queue or launcher runs (pgrep), and the queue log's
#           last "queue start" line is stage2_seeds at d21df3d, after which the last two lines are its clean end,
#           "MARK SCREENS STAGE 2 SEEDS DONE" then "plan finished: stage2_seeds.txt". It decides once more, then does
#           what launch_screens.sh does, from planck_s006.bundle: clone its main into ~/planck/qcode/<commit12>, read
#           plans/CURRENT there (it must name stage2_s006_seeds), exec queue_screens.sh on that plan.
#   WAIT    a SCREENS queue or launcher runs.
#   HOLD    no queue runs and the log ends with no end line (a kill) or after a 2 h gpu.lock wait (the autopilot's W6
#           relaunches it): held at most GRACE s (45 min) for a new stage2_seeds start, then a GIVEUP.
#   GIVEUP  any other end (STOP, a refusal, a scoring failure, heat, "plan says stop", ...), a last start line for
#           another plan or commit, a clean end with no mark file, ~/planck/STOP, the cancel file, pgrep failing:
#           logs why and exits 1. A person then launches S006 (launch_screens.sh with a bundle of the commit).
#   DONE    already launched (the stamp file, or a stage2_s006_seeds start line in the queue log): exits 0.
# Never twice: one waiter at a time (flock on s006_waiter.lock), the stamp written just before the launch, the start
# line check, and the queue's own queue.lock. A reboot kills it: boot.d/40_screens.sh then resumes a cut stage2_seeds
# run, and a person starts this again (it waits for that queue's clean end) or launches S006 by hand.
# Files in ~/planck/logs: s006_waiter.log (this log), .lock, .launched (the stamp), .CANCEL (touch it: GIVEUP).
# S006W_* variables exist for the tests and the scratch dry run only; the PC run sets none.
P=${S006W_HOME:-$HOME/planck}
OUT=$P/runs/SCREENS; QLOG=$OUT/queue_screens.txt; LOGD=$P/logs
POLL=${S006W_POLL:-300}; GRACE=${S006W_GRACE:-2700}
PAT=${S006W_PAT:-'queue_screens\.sh|launch_screens\.sh'}
KIT=${S006W_KIT:-}
XREL=experiments/screens
WANT_PLAN=stage2_seeds; WANT_REF=d21df3de007146c2d619ba7c3f46f264286d318c; NEXT_PLAN=stage2_s006_seeds
MARK=SCREENS_STAGE_2_SEEDS_DONE
STAMP=$LOGD/s006_waiter.launched; CANCEL=$LOGD/s006_waiter.CANCEL
TS='[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
ENDS="^$TS (plan finished: |plan says stop|queue stops at: |STOP while waiting for |refusing: |bad plan line: )"

say() { echo "$(date '+%F %T') $*"; }

log_text() { tr -d '\000' < "$QLOG"; }     # a power cut can leave NUL bytes in the log

running() {    # 1 a SCREENS queue or launcher runs, 0 none, E pgrep could not tell
    pgrep -f "$PAT" > /dev/null 2>&1
    case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac
}

decide() {     # decide RUNNING: one line whose first word is LAUNCH, WAIT, HOLD, GIVEUP or DONE. Reads files only.
    local n sline tail end last prev
    [ -e "$CANCEL" ] && { echo "GIVEUP the cancel file $CANCEL exists"; return; }
    [ -e "$STAMP" ] && { echo "DONE already launched: $(head -1 "$STAMP")"; return; }
    if [ -f "$QLOG" ] && log_text | grep -aqE "^$TS queue start: exp SCREENS, plan $NEXT_PLAN\.txt "; then
        echo "DONE already launched: the queue log has a $NEXT_PLAN start line"; return
    fi
    [ -e "$P/STOP" ] && { echo "GIVEUP $P/STOP exists"; return; }
    [ "$1" = E ] && { echo "GIVEUP pgrep failed: cannot tell whether a SCREENS queue runs"; return; }
    [ "$1" = 1 ] && { echo "WAIT a SCREENS queue or launcher runs"; return; }
    [ -f "$QLOG" ] || { echo "GIVEUP no queue log $QLOG"; return; }
    n=$(log_text | grep -anE "^$TS queue start: exp SCREENS, plan " | tail -1 | cut -d: -f1)
    [ -n "$n" ] || { echo "GIVEUP no queue start line in $QLOG"; return; }
    sline=$(log_text | sed -n "${n}p")
    case $sline in
        *" queue start: exp SCREENS, plan $WANT_PLAN.txt (sha256 "*"), code commit $WANT_REF") ;;
        *) echo "GIVEUP the last start line is not $WANT_PLAN at ${WANT_REF:0:12}: [$sline]"; return ;;
    esac
    tail=$(log_text | tail -n +$((n + 1)) | grep -av '^[[:space:]]*$')
    end=$(printf '%s\n' "$tail" | grep -aE "$ENDS" | tail -1)
    last=$(printf '%s\n' "$tail" | tail -1)
    [ -n "$end" ] || { echo "HOLD no queue runs and no end line after the last start: [${last:-the start line}]"; return; }
    [ "$end" = "$last" ] || { echo "GIVEUP lines after the end line [$end]: [$last]"; return; }
    prev=$(printf '%s\n' "$tail" | tail -2 | head -1)
    [ "$prev" = "$end" ] && prev=
    case $end in
        *" plan finished: $WANT_PLAN.txt")
            if ! printf '%s\n' "$prev" | grep -aqE "^$TS MARK SCREENS STAGE 2 SEEDS DONE$"; then
                echo "GIVEUP plan finished with no MARK line just before it: [$prev]"; return
            fi
            [ -e "$OUT/marks/$MARK" ] || { echo "GIVEUP a clean end but no marks/$MARK"; return; }
            echo "LAUNCH $WANT_PLAN ended cleanly: [$prev] [$end]"; return ;;
        *" queue stops at: "*)
            if printf '%s\n' "$prev" | grep -aqE "^$TS gpu.lock not free after 2 h$"; then
                echo "HOLD $WANT_PLAN stopped after a 2 h gpu.lock wait (the autopilot relaunches it): [$end]"; return
            fi ;;
    esac
    echo "GIVEUP $WANT_PLAN ended another way: [$prev] [$end]"
}

launch() {     # launch_screens.sh's steps from planck_s006.bundle; any failure is a GIVEUP (exit 1)
    local W B REF Q PLAN
    if [ -z "$KIT" ]; then
        W=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r'); KIT=/mnt/c/Users/$W/planck-kit
    fi
    B=$KIT/planck_s006.bundle
    [ -f "$B" ] || { say "GIVEUP: no bundle $B"; exit 1; }
    REF=$(git bundle list-heads "$B" refs/heads/main | cut -d' ' -f1)
    [ -n "$REF" ] || { say "GIVEUP: $B has no main"; exit 1; }
    Q=$P/qcode/${REF:0:12}
    [ -d "$Q/.git" ] || git clone -q "$B" "$Q" || { say "GIVEUP: git clone of $B into $Q failed"; exit 1; }
    git -C "$Q" checkout -q "$REF" || { say "GIVEUP: checkout of $REF in $Q failed"; exit 1; }
    PLAN=$(grep -v '^#' "$Q/$XREL/plans/CURRENT" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ "$PLAN" = "$NEXT_PLAN" ] || { say "GIVEUP: plans/CURRENT at ${REF:0:12} names [$PLAN], not $NEXT_PLAN"; exit 1; }
    [ -f "$Q/$XREL/plans/$PLAN.txt" ] && [ -f "$Q/$XREL/queue_screens.sh" ] || {
        say "GIVEUP: no plans/$PLAN.txt or queue_screens.sh at ${REF:0:12}"; exit 1; }
    echo "$(date '+%F %T') launched $NEXT_PLAN from ${REF:0:12} (pid $$)" > "$STAMP" || {
        say "GIVEUP: cannot write $STAMP"; exit 1; }
    say "code $REF in qcode/${REF:0:12}, plan $PLAN"
    exec 7>&-
    exec bash "$Q/$XREL/queue_screens.sh" --code "$Q" --plan "$Q/$XREL/plans/$PLAN.txt"
}

main() {
    set -u
    mkdir -p "$LOGD" || exit 1
    exec >> "$LOGD/s006_waiter.log" 2>&1
    exec 7>> "$LOGD/s006_waiter.lock"
    flock -n 7 || { say "another s006_waiter holds $LOGD/s006_waiter.lock: exiting"; exit 1; }
    say "=== s006_waiter (pid $$): $WANT_PLAN at ${WANT_REF:0:12}, then $NEXT_PLAN; poll $POLL s, grace $GRACE s"
    local r said= hold= i=0
    while :; do
        r=$(decide "$(running)")
        { [ "$r" != "$said" ] || [ $((i % 12)) = 0 ]; } && say "$r"
        said=$r; i=$((i + 1))
        case $r in
            LAUNCH*) r=$(decide "$(running)")                  # once more, just before the launch
                     case $r in LAUNCH*) launch ;; *) say "recheck: $r" ;; esac ;;
            WAIT*) hold= ;;
            HOLD*) [ -n "$hold" ] || hold=$(date +%s)
                   if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then
                       say "GIVEUP: no new $WANT_PLAN start within $GRACE s ($r)"; exit 1
                   fi ;;
            DONE*) exit 0 ;;
            *) exit 1 ;;
        esac
        sleep "$POLL"
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main
fi
