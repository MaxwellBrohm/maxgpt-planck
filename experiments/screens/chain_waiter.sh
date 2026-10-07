#!/bin/bash
# chain_waiter.sh: start the next SCREENS plan of a chain once the plan before it has ended cleanly at its mark
# (experiments/SCREENS.txt AMENDMENT BASE-DIAG). s006_waiter.sh with its constants made arguments, an EARLIER list,
# and the next plan's wait line checked at launch; the s006 waiter already running on the PC is not this file.
#   bash chain_waiter.sh NAME AFTER_PLAN AFTER_REF "AFTER_MARK" NEXT_PLAN BUNDLE [EARLIER_PLAN@REF ...]
# A one-step wrapper (base_diag_waiter.sh) passes them: pcdetach.sh copies only the wrapper and runs it with no
# arguments, so this file must already be in planck-kit beside it (INSTALL in the amendment). NAME names the files in
# ~/planck/logs: NAME_waiter.log (this log), .lock, .launched (the stamp), .CANCEL (touch it: GIVEUP).
# Every POLL s (300) it reads ~/planck/runs/SCREENS, read only, and decide() says one of:
#   LAUNCH  marks/<AFTER_MARK, spaces as _> exists, no SCREENS queue or launcher runs (pgrep), and the queue log's last
#           "queue start" line is AFTER_PLAN at AFTER_REF, after which the last two lines are its clean end, "MARK
#           AFTER_MARK" then "plan finished: AFTER_PLAN.txt". It decides once more, then does what launch_screens.sh
#           does, from planck-kit/BUNDLE: clone its main into ~/planck/qcode/<commit12>, read plans/CURRENT there (it
#           must name NEXT_PLAN, whose first plan line must be "wait_mark SCREENS AFTER_MARK"), exec queue_screens.sh.
#   WAIT    a SCREENS queue or launcher runs, or the last start line is an EARLIER plan at its commit (the chain has
#           not reached AFTER_PLAN yet; whatever starts AFTER_PLAN is not this waiter's).
#   HOLD    no queue runs and AFTER_PLAN's log ends with no end line (a kill) or after a 2 h gpu.lock wait: held at
#           most GRACE s (45 min) for a new AFTER_PLAN start (nothing here relaunches it), then a GIVEUP.
#   GIVEUP  any other end (STOP, a refusal, a scoring failure, heat, "plan says stop", ...), a last start line for
#           another plan or commit, a clean end with no mark file, ~/planck/STOP, the cancel file, pgrep failing:
#           logs why and exits 1. A person then launches NEXT_PLAN (queue_screens.sh from a clone of BUNDLE).
#   DONE    already launched (the stamp file, or a NEXT_PLAN start line in the queue log): exits 0.
# Never twice: one waiter per NAME (flock on NAME_waiter.lock), the stamp written just before the launch, the start
# line check, and the queue's own queue.lock. A reboot kills it; a person starts it again (it re-reads the log).
# Bad arguments: a usage line in ~/planck/logs/chain_waiter.log, exit 2. CW_* variables exist for the tests and the
# scratch dry run only; the PC run sets none.
P=${CW_HOME:-$HOME/planck}
OUT=$P/runs/SCREENS; QLOG=$OUT/queue_screens.txt; LOGD=$P/logs
POLL=${CW_POLL:-300}; GRACE=${CW_GRACE:-2700}
PAT=${CW_PAT:-'queue_screens\.sh|launch_screens\.sh'}
KIT=${CW_KIT:-}
XREL=experiments/screens
TS='[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
ENDS="^$TS (plan finished: |plan says stop|queue stops at: |STOP while waiting for |refusing: |bad plan line: )"
NAME=; AFTER_PLAN=; AFTER_REF=; AFTER_MARK=; NEXT_PLAN=; BUNDLE=; EARLIER=; MARKF=; STAMP=; CANCEL=

say() { echo "$(date '+%F %T') $*"; }

log_text() { tr -d '\000' < "$QLOG"; }     # a power cut can leave NUL bytes in the log

args() {       # args NAME AFTER_PLAN AFTER_REF AFTER_MARK NEXT_PLAN BUNDLE [EARLIER_PLAN@REF ...]: 0 = set, 1 = refused
    local id='^[a-z0-9_]+$' ref='^[0-9a-f]{40}$' mark='^[A-Z0-9]+( [A-Z0-9]+)*$' bun='^[a-z0-9_]+\.bundle$'
    local ear='^[a-z0-9_]+@[0-9a-f]{40}$' e
    [ $# -ge 6 ] || return 1
    [[ $1 =~ $id && $2 =~ $id && $3 =~ $ref && $4 =~ $mark && $5 =~ $id && $6 =~ $bun && $2 != "$5" ]] || return 1
    for e in "${@:7}"; do
        [[ $e =~ $ear && $e != "$2@$3" && ${e%@*} != "$5" ]] || return 1
    done
    NAME=$1; AFTER_PLAN=$2; AFTER_REF=$3; AFTER_MARK=$4; NEXT_PLAN=$5; BUNDLE=$6; EARLIER="${*:7}"
    MARKF=$(printf '%s' "$AFTER_MARK" | tr ' ' '_')      # queue_screens.sh's slug of an [A-Z0-9 ] text
    STAMP=$LOGD/${NAME}_waiter.launched; CANCEL=$LOGD/${NAME}_waiter.CANCEL
}

running() {    # 1 a SCREENS queue or launcher runs, 0 none, E pgrep could not tell
    pgrep -f "$PAT" > /dev/null 2>&1
    case $? in 0) echo 1 ;; 1) echo 0 ;; *) echo E ;; esac
}

decide() {     # decide RUNNING (after args): one line, first word LAUNCH, WAIT, HOLD, GIVEUP or DONE. Reads files only.
    local n sline tail end last prev e
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
    for e in $EARLIER; do
        case $sline in
            *" queue start: exp SCREENS, plan ${e%@*}.txt (sha256 "*"), code commit ${e#*@}")
                echo "WAIT the last start line is ${e%@*} at $(printf '%s' "${e#*@}" | cut -c1-12), earlier in the chain" \
                     "than $AFTER_PLAN"; return ;;
        esac
    done
    case $sline in
        *" queue start: exp SCREENS, plan $AFTER_PLAN.txt (sha256 "*"), code commit $AFTER_REF") ;;
        *) echo "GIVEUP the last start line is not $AFTER_PLAN at ${AFTER_REF:0:12}: [$sline]"; return ;;
    esac
    tail=$(log_text | tail -n +$((n + 1)) | grep -av '^[[:space:]]*$')
    end=$(printf '%s\n' "$tail" | grep -aE "$ENDS" | tail -1)
    last=$(printf '%s\n' "$tail" | tail -1)
    [ -n "$end" ] || { echo "HOLD no queue runs and no end line after the last start: [${last:-the start line}]"; return; }
    [ "$end" = "$last" ] || { echo "GIVEUP lines after the end line [$end]: [$last]"; return; }
    prev=$(printf '%s\n' "$tail" | tail -2 | head -1)
    [ "$prev" = "$end" ] && prev=
    case $end in
        *" plan finished: $AFTER_PLAN.txt")
            if ! printf '%s\n' "$prev" | grep -aqE "^$TS MARK $AFTER_MARK$"; then
                echo "GIVEUP plan finished with no MARK line just before it: [$prev]"; return
            fi
            [ -e "$OUT/marks/$MARKF" ] || { echo "GIVEUP a clean end but no marks/$MARKF"; return; }
            echo "LAUNCH $AFTER_PLAN ended cleanly: [$prev] [$end]"; return ;;
        *" queue stops at: "*)
            if printf '%s\n' "$prev" | grep -aqE "^$TS gpu.lock not free after 2 h$"; then
                echo "HOLD $AFTER_PLAN stopped after a 2 h gpu.lock wait (only a person relaunches it): [$end]"; return
            fi ;;
    esac
    echo "GIVEUP $AFTER_PLAN ended another way: [$prev] [$end]"
}

launch() {     # launch_screens.sh's steps from planck-kit/BUNDLE; any failure is a GIVEUP (exit 1)
    local W B REF Q PLAN WL
    if [ -z "$KIT" ]; then
        W=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r'); KIT=/mnt/c/Users/$W/planck-kit
    fi
    B=$KIT/$BUNDLE
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
    WL=$(grep -v '^#' "$Q/$XREL/plans/$PLAN.txt" | grep -v '^[[:space:]]*$' | head -1)
    [ "$WL" = "wait_mark SCREENS $AFTER_MARK" ] || {
        say "GIVEUP: plans/$PLAN.txt starts [$WL], not wait_mark SCREENS $AFTER_MARK"; exit 1; }
    echo "$(date '+%F %T') launched $NEXT_PLAN from ${REF:0:12} (pid $$)" > "$STAMP" || {
        say "GIVEUP: cannot write $STAMP"; exit 1; }
    say "code $REF in qcode/${REF:0:12}, plan $PLAN"
    exec 7>&-
    exec bash "$Q/$XREL/queue_screens.sh" --code "$Q" --plan "$Q/$XREL/plans/$PLAN.txt"
}

main() {
    set -u
    mkdir -p "$LOGD" || exit 1
    if ! args "$@"; then
        say "usage: chain_waiter.sh NAME AFTER_PLAN AFTER_REF \"AFTER_MARK\" NEXT_PLAN BUNDLE [EARLIER_PLAN@REF ...];" \
            "refused: [$*]" >> "$LOGD/chain_waiter.log"
        exit 2
    fi
    exec >> "$LOGD/${NAME}_waiter.log" 2>&1
    exec 7>> "$LOGD/${NAME}_waiter.lock"
    flock -n 7 || { say "another ${NAME}_waiter holds $LOGD/${NAME}_waiter.lock: exiting"; exit 1; }
    say "=== ${NAME}_waiter (pid $$): $AFTER_PLAN at ${AFTER_REF:0:12} (mark $AFTER_MARK), then $NEXT_PLAN from" \
        "$BUNDLE; earlier: ${EARLIER:-none}; poll $POLL s, grace $GRACE s"
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
                       say "GIVEUP: no new $AFTER_PLAN start within $GRACE s ($r)"; exit 1
                   fi ;;
            DONE*) exit 0 ;;
            *) exit 1 ;;
        esac
        sleep "$POLL"
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
