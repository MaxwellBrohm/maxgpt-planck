#!/bin/bash
# diag_queue.sh: the PC queue for diag_runner.py (SCREENS C6 diagnostics, E3 Part 4d). Started detached (pcdetach.sh).
#   bash diag_queue.sh --bundle NAME.bundle [--groups G,..] [--only RUN,..]   clone the bundle's main into
#                                        ~/planck/qcode/<commit12>, then run this file from that clone with --code
#   bash diag_queue.sh --code Q [--groups G,..] [--only RUN,..]   run diag_runner.py from the code root Q (--only:
#                                                            a dry run or a partial rerun; passed to the runner)
# 1. The CPU pass (modules jobs) without gpu.lock. 2. GPU holds: each takes ~/planck/locks/gpu.lock (flock -w, 2 h),
# checks that no other process holds over 3,000 MiB of GPU memory, runs diag_runner.py with a soft budget (1,500 s)
# under a hard timeout (1,980 s, so a hold stays under 35 min), then releases the lock; a run with jobs left
# (runner exit 3) takes the next hold. A 2 h lock wait is NOT an exit: the queue logs it and waits again (flock is not
# FIFO; nothing has to relaunch it). Done jobs are skipped, so after a power cut or a kill the same command resumes.
# Before each hold: ~/planck/STOP or ~/planck/c6diag/CANCEL ends the queue; ~/planck/PAUSE, under 40 GB free on the
# runs disk or the GPU above 75 C waits. Ends: DONE (exit 0, ~/planck/c6diag/DONE written); a refusal or a failed job
# (runner exit 2 or 1), two hard timeouts, STOP or CANCEL (exit 1, for a person). Log: ~/planck/logs/c6diag.log.
# Test hooks (env): PLANCK_HOME, PLANCK_PY, C6_SMI, C6_LOCK_WAIT_S, C6_POLL_S, C6_HOLD_S, C6_BUDGET_S, C6_DEVICE,
# C6_KIT. The runner opens no result file of any run (diag_runner.py).
P=${PLANCK_HOME:-$HOME/planck}
PY=${PLANCK_PY:-$P/venv/bin/python}
SMI=${C6_SMI:-nvidia-smi}
LOCKWAIT=${C6_LOCK_WAIT_S:-7200}
POLL=${C6_POLL_S:-60}
HOLD=${C6_HOLD_S:-1980}
BUDGET=${C6_BUDGET_S:-1500}
DEVICE=${C6_DEVICE:-cuda}
ST=$P/c6diag
LOG=$P/logs/c6diag.log
MINFREE=40
CODE= BUNDLE= GRP= ONLY=

say() { echo "$(date '+%F %T') c6diag: $*" >> "$LOG"; }
LASTW=
waiting() {    # a wait reason, logged once per change (not once per poll), then sleep
    [ "$1" = "$LASTW" ] || say "$1"
    LASTW=$1
    sleep "$POLL"
}
stopped() {    # 0 (and a log line) when STOP or CANCEL is present
    [ -e "$P/STOP" ] && { say "STOP present: exiting"; return 0; }
    [ -e "$ST/CANCEL" ] && { say "CANCEL present: exiting"; return 0; }
    return 1
}
gpu_q() { $SMI --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_gb() { python3 -c 'import shutil, sys; print(shutil.disk_usage(sys.argv[1]).free // 2**30)' "$P/runs"; }

while [ $# -gt 0 ]; do
    case $1 in
        --code) CODE=$2; shift 2 ;;
        --bundle) BUNDLE=$2; shift 2 ;;
        --groups) GRP=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        *) echo "usage: diag_queue.sh (--code Q | --bundle NAME.bundle) [--groups G,..] [--only RUN,..]"; exit 2 ;;
    esac
done
mkdir -p "$ST" "$P/logs" "$P/locks" || exit 1
GA=()
[ -n "$GRP" ] && GA=(--groups "$GRP")
[ -n "$ONLY" ] && GA=("${GA[@]}" --only "$ONLY")

if [ -n "$BUNDLE" ]; then     # launch: a named bundle in planck-kit, never send_kit.sh's planck.bundle
    case $BUNDLE in planck.bundle|*/*|"") say "refusing bundle [$BUNDLE]"; exit 2 ;; esac
    KIT=${C6_KIT:-/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/planck-kit}
    B=$KIT/$BUNDLE
    REF=$(git bundle list-heads "$B" refs/heads/main 2>/dev/null | cut -d' ' -f1)
    [ -n "$REF" ] || { say "no main in $B"; exit 1; }
    Q=$P/qcode/${REF:0:12}
    [ -d "$Q/.git" ] || git clone -q "$B" "$Q" || { say "git clone of $B failed"; exit 1; }
    git -C "$Q" checkout -q "$REF" || { say "checkout of $REF in $Q failed"; exit 1; }
    say "code $REF from $BUNDLE in qcode/${REF:0:12}"
    exec bash "$Q/experiments/screens/diag_queue.sh" --code "$Q" "${GA[@]}"
fi
[ -n "$CODE" ] && [ -f "$CODE/experiments/screens/diag_runner.py" ] || { say "no diag_runner.py under [$CODE]"; exit 2; }
exec 8> "$P/locks/c6diag.lock"
flock -n 8 || { say "another c6diag queue holds locks/c6diag.lock: exiting"; exit 1; }
R=(run --runs-root "$P/runs" --evalset "$P/data/evalsets/v0" "${GA[@]}")
say "start (pid $$), code $(git -C "$CODE" rev-parse HEAD 2>/dev/null), groups [${GRP:-all}], only [${ONLY:-all}]"
rm -f "$ST/DONE"
stopped && exit 1
"$PY" "$CODE/experiments/screens/diag_runner.py" "${R[@]}" --cpu-only >> "$LOG" 2>&1
rc=$?
[ $rc = 0 ] || { say "CPU pass: runner exit $rc: stopping for a person"; exit 1; }
say "CPU pass done"

kills=0
while true; do
    stopped && exit 1
    if [ -e "$P/PAUSE" ]; then waiting "PAUSE present: waiting"; continue; fi
    f=$(free_gb)
    if [ "${f:-0}" -lt $MINFREE ]; then waiting "disk: [$f] GB free (< $MINFREE): waiting"; continue; fi
    t=$(gpu_q temperature.gpu)
    if [ -z "$t" ] || [ "$t" -gt 75 ]; then waiting "GPU temperature [$t] not <= 75: waiting"; continue; fi
    exec 9> "$P/locks/gpu.lock"
    say "waiting for gpu.lock"
    if ! flock -w "$LOCKWAIT" 9; then
        exec 9>&-
        say "gpu.lock not free after $LOCKWAIT s: waiting again"
        continue
    fi
    used=$(gpu_q memory.used)
    if [ -z "$used" ] || [ "$used" -gt 3000 ]; then
        flock -u 9; exec 9>&-
        waiting "GPU memory in use [$used] MiB > 3000 with the lock held: released, waiting"; continue
    fi
    LASTW=
    say "gpu.lock held; GPU memory in use $used MiB"
    timeout -k 30 "$HOLD" "$PY" "$CODE/experiments/screens/diag_runner.py" "${R[@]}" --device "$DEVICE" \
        --budget-s "$BUDGET" >> "$LOG" 2>&1
    rc=$?
    flock -u 9; exec 9>&-
    say "gpu.lock released (runner exit $rc)"
    case $rc in
        0) date '+%F %T' > "$ST/DONE"; say "DONE"; exit 0 ;;
        3) ;;
        124|137) kills=$((kills + 1))
                 [ $kills -ge 2 ] && { say "the runner hit the $HOLD s hold limit twice: stopping for a person"; exit 1; } ;;
        *) say "runner exit $rc: stopping for a person"; exit 1 ;;
    esac
done
