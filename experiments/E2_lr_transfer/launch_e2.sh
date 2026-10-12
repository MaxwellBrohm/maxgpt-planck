#!/bin/bash
# launch_e2.sh: start the E2 queue on the PC, detached from any SSH session. Self-contained: pcdetach.sh copies
# only this file to the PC and runs it in WSL with no arguments. From the Mac, after the commit:
#   bash pc/mac/send_kit.sh                          # the committed repo as a git bundle in planck-kit/
#   pcdetach.sh experiments/E2_lr_transfer/launch_e2.sh
# It clones the bundle's main into ~/planck/qcode/<commit12> (a directory only queues use, never ~/planck/dev,
# which another process cleaned once), then runs queue_e2.sh on the plan named in plans/CURRENT at that commit,
# so what was queued is always a committed file. Log: ~/planck/logs/launch_e2.log; the queue's own log is
# ~/planck/runs/E2/queue_e2.txt. Stop: touch ~/planck/STOP (the running train.py checkpoints and exits).
# Before the FIRST launch: the FIXING step (configs/engine.yaml values and engine_fixed set, FIXED log in
# notes.txt), then the commit; preflight.py --strict refuses every run until engine_fixed is set.
# Stage flow (each plan ends with a mark and the queue exits; nothing advances a stage by itself):
#   stageA done -> python3 e2pick.py stage A --runs <copy of ~/planck/runs/E2>   (pick, extensions)
#   -> python3 e2plan.py arm --size 5m --eta ETA --r R ... ; python3 e2plan.py plan stageB ARM ... [--mark ...]
#   -> plans/CURRENT = stageB -> commit -> send_kit.sh -> pcdetach.sh launch_e2.sh; the same for C, extensions
#   ("mark E2 5M DONE" ends the last 5M plan) and the 20M grid (its plan starts "wait_mark E3 E3 PART 1 DONE").
#   E3's arms come from e2pick.py too: stage C's lrs.pick / lrs.runner_up are e3plan.py --a5 / --b5, and q2's
#   grid_20m gives A20 / B20 (argmin_g, runner_up_g, times eta_5) and any 20M extension (extend_with_g).
exec >> "$HOME/planck/logs/launch_e2.log" 2>&1
set -eu
EXP=E2; XREL=experiments/E2_lr_transfer
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
