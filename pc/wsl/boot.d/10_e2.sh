#!/bin/bash
# boot.d/10_e2.sh: after a reboot, resume the E2 queue (experiments/E2_lr_transfer) through launch_e2.sh.
#   bash 10_e2.sh [--dry-run]        (boot_resume.sh runs it; by hand it is safe too: it never starts a second copy)
# Resumes only when all hold: no stop file (bootlib.sh: ~/planck/boot.d/STOP, boot.d/e2.STOP, ~/planck/STOP,
# ~/planck/PAUSE); nothing of E2 runs (queue, launcher, an E2 train.py or scorer); and ~/planck/runs/E2/queue_e2.txt's
# last "queue start: exp E2" has no line after it that ends a run (plan finished, plan says stop, queue stops at,
# STOP while waiting, refusing, bad plan line): a power cut leaves no such line, an ending the queue chose does.
# Restart path (launch_e2.sh header): the planck-kit copy of launch_e2.sh clones the newest bundle into
# ~/planck/qcode/<commit12> and runs plans/CURRENT; the queue skips every finished, DIVERGED or GAP run. If
# launch_e2.sh never reaches the queue (no bundle, or cmd.exe interop missing in the boot session), this falls back
# to queue_e2.sh from the checkout and plan named in that last start line.
NAME=e2
. "$(dirname "$0")/bootlib.sh"
Q=$P/runs/E2/queue_e2.txt
LL=$P/logs/launch_e2.log
KIT=${PLANCK_KIT:-$(ls -d /mnt/c/Users/*/planck-kit 2>/dev/null | head -1)}
ANY='queue_e2\.sh --exp E2|launch_e2\.sh|train\.py [^ ]*E2_lr_transfer/configs/|bpb\.py [^ ]*/runs/E2/'
QUEUE='queue_e2\.sh --exp E2'
ENDS=' (plan finished|plan says stop|queue stops at|STOP while waiting|refusing:|bad plan line)'

blocked && exit 0
if running "$ANY"; then say "running (pids $PIDS): nothing to do"; exit 0; fi
[ -f "$Q" ] || { say "no $Q: E2 never ran here, nothing to resume"; exit 0; }
start=$(last_start "$Q" ' queue start: exp E2,')
[ -n "$start" ] || { say "no queue start line in $Q: nothing to resume"; exit 0; }
end=$(ended_after "$Q" "$start" "$ENDS")
if [ -n "$end" ]; then say "the last run ended by itself [$end]: nothing to resume"; exit 0; fi
say "cut off: last start at line $start, last line [$(tail -1 "$Q")]"

# fallback command from the last start line: "... queue start: exp E2, plan P.txt (sha256 X), code commit REF"
sline=$(sed -n "${start}p" "$Q")
ref=$(echo "$sline" | sed -n 's/.*code commit \([0-9a-f]\{12,\}\).*/\1/p')
plan=$(echo "$sline" | sed -n 's/.*, plan \([^ ]*\) (sha256.*/\1/p')
C=$P/qcode/$(echo "$ref" | cut -c1-12)
X=$C/experiments/E2_lr_transfer
fallback() {
    if [ -z "$ref" ] || [ ! -f "$X/queue_e2.sh" ] || [ ! -f "$X/plans/$plan" ]; then
        say "no fallback: checkout [$C] or plan [$plan] missing"; return 1
    fi
    LAUNCH_LOG=$LL launch "$QUEUE" bash "$X/queue_e2.sh" --exp E2 --code "$C" --plan "$X/plans/$plan"
}

if [ -z "$KIT" ] || [ ! -f "$KIT/launch_e2.sh" ]; then
    say "no launch_e2.sh in planck-kit [$KIT]: using the fallback"
    fallback; exit $?
fi
[ $DRY = 1 ] && say "(fallback if launch_e2.sh cannot reach the queue: bash $X/queue_e2.sh --exp E2 --code $C --plan $X/plans/$plan)"
before=$(wc -l < "$LL" 2>/dev/null || echo 0)
launch "$QUEUE" bash "$KIT/launch_e2.sh" && exit 0
# launch_e2.sh writes "code <commit> in qcode/..., plan <name>" just before it hands over to the queue
if tail -n +"$((before + 1))" "$LL" 2>/dev/null | grep -aq '^code .* plan '; then
    say "launch_e2.sh reached the queue and the queue exited: last queue line [$(tail -1 "$Q")]"
    exit 1
fi
say "launch_e2.sh did not reach the queue: [$(tail -1 "$LL" 2>/dev/null)]; trying the fallback"
fallback
