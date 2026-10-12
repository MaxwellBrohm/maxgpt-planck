#!/bin/bash
# launch_screens.sh: start the SCREENS queue on the PC, detached from any SSH session (modeled on launch_e3.sh).
# Self-contained: pcdetach.sh copies only this file to the PC and runs it in WSL with no arguments. From the Mac,
# after the commit:
#   bash pc/mac/send_kit.sh                          # the committed repo as a git bundle in planck-kit/
#   pcdetach.sh experiments/screens/launch_screens.sh
# It clones the bundle's main into ~/planck/qcode/<commit12> (a directory only queues use, never ~/planck/dev),
# then runs experiments/screens/queue_screens.sh on the plan named in experiments/screens/plans/CURRENT at that
# commit, so what was queued is always a committed file. Log: ~/planck/logs/launch_screens.log; the queue's own
# log is ~/planck/runs/SCREENS/queue_screens.txt. Stop: touch ~/planck/STOP (the running train.py checkpoints
# and exits). Before the first launch: SCREENS ORDER 0 preconditions (plans/CURRENT says which), then the commit.
exec >> "$HOME/planck/logs/launch_screens.log" 2>&1
set -eu
EXP=SCREENS; XREL=experiments/screens
echo "=== launch $EXP $(date '+%F %T')"
W=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
B=/mnt/c/Users/$W/planck-kit/planck.bundle
[ -f "$B" ] || { echo "no bundle in planck-kit (run pc/mac/send_kit.sh on the Mac)"; exit 1; }
REF=$(git bundle list-heads "$B" refs/heads/main | cut -d' ' -f1)
[ -n "$REF" ] || { echo "the bundle has no main"; exit 1; }
Q=$HOME/planck/qcode/${REF:0:12}
[ -d "$Q/.git" ] || git clone -q "$B" "$Q"
git -C "$Q" checkout -q "$REF"
PLAN=$(grep -v '^#' "$Q/$XREL/plans/CURRENT" | head -1 | tr -d '[:space:]')
[ -f "$Q/$XREL/plans/$PLAN.txt" ] || { echo "plans/CURRENT names [$PLAN], no such plan"; exit 1; }
echo "code $REF in qcode/${REF:0:12}, plan $PLAN"
exec bash "$Q/$XREL/queue_screens.sh" --code "$Q" --plan "$Q/$XREL/plans/$PLAN.txt"
