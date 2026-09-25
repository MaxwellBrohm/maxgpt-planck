#!/usr/bin/env bash
# runner_start.sh: what the Windows PlanckRunner task runs inside WSL (RUNBOOK step 11).
# Stays in the foreground forever: the task's wsl.exe process keeps the WSL VM alive only
# while this runs. Restarts the runner 60 s after a crash; exits if another runner holds
# the lock (rc 3) or the runner was shut down cleanly (rc 0).
#
# Optional ~/planck/runner.env (plain settings, NEVER secrets), for example:
#   PLANCK_NVIDIA_SMI=/mnt/c/Windows/System32/nvidia-smi.exe   # only if WSL's lacks temps
#   PLANCK_STATUS_REPO=$HOME/planck/status-repo                 # local clone for heartbeats
#   RUNNER_ARGS="--push"                                        # push heartbeats there
set -u
PLANCK_HOME="${PLANCK_HOME:-$HOME/planck}"
export PLANCK_HOME
LOG="$PLANCK_HOME/logs/runner.log"
mkdir -p "$PLANCK_HOME/logs"
export PATH="/usr/lib/wsl/lib:$PATH"          # nvidia-smi in WSL
if [ -f "$PLANCK_HOME/runner.env" ]; then
    set -a; . "$PLANCK_HOME/runner.env"; set +a
fi
RUNNER="$PLANCK_HOME/repo/pc/wsl/queue_runner.py"

echo "$(date "+%Y-%m-%dT%H:%M:%S%z") runner_start: boot (pid $$, uptime $(cut -d" " -f1 /proc/uptime 2>/dev/null || echo "?") s)" >> "$LOG"
while true; do
    # shellcheck disable=SC2086  # RUNNER_ARGS is a word list on purpose
    python3 "$RUNNER" --home "$PLANCK_HOME" ${RUNNER_ARGS:-} >> "$LOG" 2>&1
    rc=$?
    echo "$(date "+%Y-%m-%dT%H:%M:%S%z") runner_start: runner exited rc=$rc" >> "$LOG"
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]; then
        exit "$rc"
    fi
    sleep 60
done
