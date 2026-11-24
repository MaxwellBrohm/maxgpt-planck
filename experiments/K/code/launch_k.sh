#!/bin/bash
# launch_k.sh: start the Track K queue on the PC, detached from any SSH session (launch_screens.sh's shape; K3 round
# 16, 2026-11-21, never run yet). Self-contained: pcdetach.sh copies only this file to the PC and runs it in WSL with
# no arguments. From the Mac, after the commit, K ships its own bundle (never planck.bundle, which other queues use):
#   git bundle create k.bundle main, copied to planck-kit/k.bundle on the PC
#   pcdetach.sh experiments/K/code/launch_k.sh
# It clones the bundle's main into ~/planck/qcode/<commit12> (a directory only queues use), then runs
# experiments/K/code/queue_k.sh on the plan experiments/K/plans/CURRENT names at that commit, with the K data root
# ~/planck/data/k (gen.py's, checked: gen.py check exit 0). Log: ~/planck/logs/launch_k.log; the queue's own log is
# ~/planck/runs/K/queue_k.txt. Stop K: touch ~/planck/runs/K/STOP (the running ktrain.py checkpoints and exits; a
# waiting queue ends); ~/planck/STOP would stop every Planck queue (2026-11-22, port review note 6).
exec >> "$HOME/planck/logs/launch_k.log" 2>&1
set -eu
XREL=experiments/K
echo "=== launch K $(date '+%F %T')"
W=$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')
B=/mnt/c/Users/$W/planck-kit/k.bundle
[ -f "$B" ] || { echo "no k.bundle in planck-kit"; exit 1; }
REF=$(git bundle list-heads "$B" refs/heads/main | cut -d' ' -f1)
[ -n "$REF" ] || { echo "the bundle has no main"; exit 1; }
Q=$HOME/planck/qcode/${REF:0:12}
[ -d "$Q/.git" ] || git clone -q "$B" "$Q"
git -C "$Q" checkout -q "$REF"
PLAN=$(grep -v '^#' "$Q/$XREL/plans/CURRENT" | head -1 | tr -d '[:space:]')
[ -f "$Q/$XREL/plans/$PLAN.txt" ] || { echo "plans/CURRENT names [$PLAN], no such plan"; exit 1; }
D=$HOME/planck/data/k
[ -f "$D/check.json" ] || { echo "no $D/check.json: generate and check the K data first (code/notes.txt)"; exit 1; }
echo "code $REF in qcode/${REF:0:12}, plan $PLAN, data $D"
exec bash "$Q/$XREL/code/queue_k.sh" --code "$Q" --plan "$Q/$XREL/plans/$PLAN.txt" --data "$D"
