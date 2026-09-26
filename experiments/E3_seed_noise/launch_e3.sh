#!/bin/bash
# launch_e3.sh: start the E3 queue on the PC, detached from any SSH session. Self-contained: pcdetach.sh copies
# only this file to the PC and runs it in WSL with no arguments. From the Mac, after the commit:
#   bash pc/mac/send_kit.sh                          # the committed repo as a git bundle in planck-kit/
#   pcdetach.sh experiments/E3_seed_noise/launch_e3.sh
# It clones the bundle's main into ~/planck/qcode/<commit12> (a directory only queues use, never ~/planck/dev,
# which another process cleaned once), then runs queue_e2.sh --exp E3 on the plan named in plans/CURRENT at that commit,
# so what was queued is always a committed file. Log: ~/planck/logs/launch_e3.log; the queue's own log is
# ~/planck/runs/E3/queue_e3.txt. Stop: touch ~/planck/STOP (the running train.py checkpoints and exits).
exec >> "$HOME/planck/logs/launch_e3.log" 2>&1
set -eu
EXP=E3; XREL=experiments/E3_seed_noise
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
exec bash "$Q/experiments/E2_lr_transfer/queue_e2.sh" --exp "$EXP" --code "$Q" --plan "$Q/$XREL/plans/$PLAN.txt"
