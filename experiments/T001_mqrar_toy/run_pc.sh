#!/bin/bash
# run_pc.sh: the T001 toy queue on the PC 5070 (PREREG.txt GPU PLAN), run detached by pcdetach.sh, which copies only
# this file into planck-kit. ship_pc.sh puts the code there first as t001_code.tgz with its sha256 (never
# planck.bundle). Code is unpacked to ~/planck/dev/t001/code_<sha12>; runs write ~/planck/dev/t001/out/<run>.json.
# runplan.py names the next run from the files already in out/, so the queue resumes anywhere after a stop.
# Runs go one at a time under gpu.lock, each under timeout 1800 s (-k 60). Amendment A2 (scheduling only): after a
# run, the same hold starts the next run if the hold is younger than HOLD_MORE s, STOP and PAUSE are absent, the GPU is
# at or below 80 C, the plan is not DONE and at least 600 s remain before HOLD_CAP; a later run's timeout is cut to end
# by HOLD_CAP s of hold age, so no hold passes 35 min. Before each hold: ~/planck/STOP ends the queue, PAUSE
# waits, 45 GB free on / (40 GB kept free plus margin), the GPU at or below 75 C; with the lock held, at most 3,000 MiB
# in use (else release, wait a minute, retry). A 2 h lock wait is not an exit: the queue logs it and waits again, so
# no relaunch watcher is needed. A run at 87 C on 3 samples (5 s apart) is stopped and the queue ends: TERM goes to
# timeout, which passes it to python and KILLs python 60 s later (-k 60); the lock is released only once both are
# gone (critique D6: the old kill -9 of timeout could orphan python). A non-zero exit of a run ends the queue (a
# person reads the log). Only this queue's own child and its python are ever signalled.
# Log: ~/planck/logs/t001.log. Single instance: ~/planck/locks/t001.lock.
exec >> "$HOME/planck/logs/t001.log" 2>&1
set -u
P=$HOME/planck; T=$P/dev/t001; PY=$P/venv/bin/python; LOCK=$P/locks/gpu.lock
HOLD_MORE=${T001_HOLD_MORE:-900}; HOLD_CAP=${T001_HOLD_CAP:-2040}      # overrides exist for the dry run only
KIT=$(dirname "$(readlink -f "$0")")
say() { echo "$(date '+%F %T') $*"; }
gq() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

exec 8> "$P/locks/t001.lock"
flock -n 8 || { say "another t001 queue holds locks/t001.lock: exiting"; exit 1; }
mkdir -p "$T/out"
cd "$KIT" || exit 2
sha256sum -c t001_code.sha256 || { say "t001_code.tgz does not match its sha256: exiting"; exit 2; }
SHA=$(cut -c1-12 t001_code.sha256); CODE=$T/code_$SHA
if [ ! -d "$CODE" ]; then
    mkdir -p "$CODE.tmp" && tar -xzf t001_code.tgz -C "$CODE.tmp" && mv "$CODE.tmp" "$CODE" || { say "unpack failed"; exit 2; }
fi
say "T001 queue start, code $SHA"

ready() {   # 0 = go; 1 = the queue ends
    local said=0 f t
    while [ -e "$P/PAUSE" ]; do [ $said = 1 ] || say "PAUSE: waiting"; said=1; sleep 60; done
    [ -e "$P/STOP" ] && { say "STOP present: queue ends"; return 1; }
    said=0
    while f=$(df -BG --output=avail / | tail -1 | tr -dc '0-9'); [ "${f:-0}" -lt 45 ]; do
        [ $said = 1 ] || say "disk: / ${f} GB available, need 45: waiting"; said=1; sleep 900
    done
    said=0
    while t=$(gq temperature.gpu); [ -z "$t" ] || [ "$t" -gt 75 ]; do
        [ $said = 1 ] || say "GPU at [$t] C, not <= 75: waiting"; said=1; sleep 60
    done
}

while :; do
    ready || exit 0
    NEXT=$(cd "$CODE" && "$PY" runplan.py next --out "$T/out") || { say "runplan failed: queue ends"; exit 1; }
    if [ "$NEXT" = DONE ]; then say "T001 DONE"; touch "$T/DONE"; exit 0; fi
    exec 9> "$LOCK"
    say "waiting for gpu.lock: $NEXT"
    if ! flock -w 7200 9; then exec 9>&-; say "gpu.lock not free after 2 h: waiting again"; continue; fi
    used=$(gq memory.used)
    if [ "${used:-99999}" -gt 3000 ]; then
        say "gpu.lock held but $used MiB in use: releasing, retry in a minute"; flock -u 9; exec 9>&-; sleep 60; continue
    fi
    say "gpu.lock held ($used MiB in use): run"
    h0=$(date +%s); nrun=0
    while :; do
        t0=$(date +%s); tmo=$(( HOLD_CAP - (t0 - h0) )); [ $tmo -gt 1800 ] && tmo=1800
        (cd "$CODE" && exec timeout -k 60 $tmo "$PY" train.py $NEXT --device cuda) &
        child=$!; hot=0; nrun=$((nrun + 1))
        while kill -0 $child 2>/dev/null; do
            sleep 5
            t=$(gq temperature.gpu)
            if [ -n "$t" ] && [ "$t" -ge 87 ]; then hot=$((hot + 1)); else hot=0; fi
            if [ $hot -ge 3 ]; then
                py=$(pgrep -P $child | head -1)
                say "GPU at $t C on 3 samples: stopping this run (timeout $child, python ${py:-none})"
                kill -TERM $child 2>/dev/null
                for _ in $(seq 75); do
                    kill -0 $child 2>/dev/null || { [ -n "$py" ] && kill -0 "$py" 2>/dev/null; } || break; sleep 1
                done
                if [ -n "$py" ] && kill -0 "$py" 2>/dev/null && grep -qa train.py "/proc/$py/cmdline"; then
                    say "python $py still alive 75 s after TERM: KILL"; kill -KILL "$py"
                    for _ in $(seq 10); do kill -0 "$py" 2>/dev/null || break; sleep 1; done
                fi
                kill -KILL $child 2>/dev/null
                break
            fi
        done
        wait $child; rc=$?
        say "run $nrun of this hold: rc $rc after $(( $(date +%s) - t0 )) s (timeout $tmo s)"
        [ $hot -ge 3 ] || [ $rc != 0 ] && break
        age=$(( $(date +%s) - h0 )); t=$(gq temperature.gpu)
        [ $age -lt $HOLD_MORE ] && [ $((HOLD_CAP - age)) -ge 600 ] && [ ! -e "$P/STOP" ] && [ ! -e "$P/PAUSE" ] \
            && [ -n "$t" ] && [ "$t" -le 80 ] || break
        NEXT=$(cd "$CODE" && "$PY" runplan.py next --out "$T/out") || { rc=97; say "runplan failed in a hold"; break; }
        [ "$NEXT" = DONE ] && break
        say "same hold (age $age s, $t C): $NEXT"
    done
    flock -u 9; exec 9>&-
    say "gpu.lock released after $nrun run(s), hold $(( $(date +%s) - h0 )) s"
    [ $hot -ge 3 ] && { say "queue ends after a heat stop"; exit 1; }
    [ $rc = 0 ] || { say "run failed (rc $rc): queue ends, NEEDS A PERSON"; exit 1; }
    sleep 5
done
