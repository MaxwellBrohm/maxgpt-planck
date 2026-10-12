# queue_lib.sh: shared by smoke/smoke.sh, queue_e2.sh and ../E3_seed_noise/queue_e3.sh (sourced, bash, PC WSL).
# Needs CODE (the code root the runs use: harness/, data_prep/, tokenizer/, experiments/, planck_root link).
# Guards follow PLAN and pc/wsl/gpuguard.py: ~/planck/PAUSE holds the queue, ~/planck/STOP stops it (the
# running train.py checkpoints and exits), start only at or below 75 C and with 25 GB free, stop a run at
# 87 C for 3 samples (it checkpoints; the queue waits and resumes it), start only while nothing else holds
# more than 3,000 MiB of GPU memory. One GPU job at a time: each run holds ~/planck/locks/gpu.lock for its
# training plus scoring.
P=${PLANCK_HOME:-$HOME/planck}
PY=${PLANCK_PY:-$P/venv/bin/python}   # PLANCK_PY: tests put a stub here
LOCK=$P/locks/gpu.lock
SMI=${PLANCK_NVIDIA_SMI:-nvidia-smi}
E2DIR=$CODE/experiments/E2_lr_transfer
TOK=$CODE/tokenizer/v0/tok_v0_8k.json
EVAL=$P/data/evalsets/v0
QPOLL=${QUEUE_POLL_S:-10}     # watch interval while train.py runs (tests shorten both)
QWAIT=${QUEUE_WAIT_S:-60}     # wait interval for PAUSE, heat, busy GPU memory and marks
MINFREE=${QUEUE_MIN_FREE_GB:-25}

qlog() { echo "$(date '+%F %T') $*"; }
gpu_q() { $SMI --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_gb() { df -BG --output=avail "$P" | tail -1 | tr -dc '0-9'; }

link_root() {   # the configs reach data and runs through CODE/planck_root
    ln -sfn "$P" "$CODE/planck_root"
}

guards() {      # before each run. 1 = the queue must stop
    local said=0 t
    while [ -e "$P/PAUSE" ]; do
        [ $said = 1 ] || qlog "PAUSE present: waiting"; said=1; sleep "$QWAIT"
    done
    if [ -e "$P/STOP" ]; then qlog "STOP present: queue stops"; return 1; fi
    local f; f=$(free_gb)
    if [ "${f:-0}" -lt "$MINFREE" ]; then qlog "disk: ${f} GB free (< $MINFREE): queue stops"; return 1; fi
    said=0
    while t=$(gpu_q temperature.gpu); [ -z "$t" ] || [ "$t" -gt 75 ]; do
        [ $said = 1 ] || qlog "GPU temperature [$t] not <= 75: waiting"; said=1; sleep "$QWAIT"
    done
    return 0
}

gpu_lock() {    # fd 9 holds the lock until gpu_unlock (children inherit it; the holder is this queue)
    exec 9>"$LOCK"
    qlog "waiting for gpu.lock"
    flock -w 7200 9 || { qlog "gpu.lock not free after 2 h"; return 1; }
    local i used
    for i in $(seq 1 30); do
        used=$(gpu_q memory.used)
        [ -n "$used" ] && [ "$used" -le 3000 ] && { qlog "gpu.lock held; GPU memory in use ${used} MiB"; return 0; }
        qlog "GPU memory in use [${used}] MiB > 3000 with the lock held: waiting"; sleep "$QWAIT"
    done
    flock -u 9; return 1
}
gpu_unlock() { flock -u 9; exec 9>&-; }

# train_run CONFIG OUT_DIR [train.py args...]
#   0 finished (final_*.pt exists)   10 diverged (never resumed)   11 GAP (second crash)
#   12 stopped by ~/planck/STOP      13 prereg refused              14 stopped for heat, retry later
train_run() {
    local cfg=$1 out=$2; shift 2
    mkdir -p "$out"
    [ -e "$out/DIVERGED" ] && return 10
    [ -e "$out/GAP" ] && return 11
    ls "$out"/final_*.pt >/dev/null 2>&1 && return 0
    rm -f "$out/STOP" "$out/HEAT"
    qlog "train $(basename "$cfg") $*"
    ( cd "$CODE/harness" && exec $PY train.py "$cfg" --device cuda "$@" ) >> "$out/train.out" 2>&1 &
    local tp=$! hot=0 t m mx=0
    while kill -0 $tp 2>/dev/null; do
        [ -e "$P/STOP" ] && touch "$out/STOP"
        m=$(gpu_q memory.used); [ -n "$m" ] && [ "$m" -gt "$mx" ] && mx=$m && echo "$mx" > "$out/gpu_mem_max_mib.txt"
        t=$(gpu_q temperature.gpu)
        if [ -n "$t" ] && [ "$t" -ge 87 ]; then hot=$((hot + 1)); else hot=0; fi
        [ $hot -ge 3 ] && { touch "$out/STOP" "$out/HEAT"; qlog "GPU at ${t} C for 3 samples: stopping the run"; }
        sleep "$QPOLL"
    done
    wait $tp; local rc=$?
    rm -f "$out/STOP"
    ls "$out"/final_*.pt >/dev/null 2>&1 && { qlog "done $(basename "$out") rc $rc"; return 0; }
    if grep -qs "non-finite loss" "$out/log.jsonl" "$out/train.out"; then
        touch "$out/DIVERGED"; qlog "DIVERGED $(basename "$out"): scored +inf, never resumed"; return 10
    fi
    [ $rc = 2 ] && { qlog "prereg refused for $(basename "$cfg")"; return 13; }
    if [ $rc = 0 ]; then
        [ -e "$out/HEAT" ] && { rm -f "$out/HEAT"; return 14; }
        qlog "stopped $(basename "$out") at its checkpoint"; return 12
    fi
    echo "$(date '+%F %T') rc $rc" >> "$out/crashes.txt"
    if [ "$(wc -l < "$out/crashes.txt")" -ge 2 ]; then
        touch "$out/GAP"; qlog "GAP $(basename "$out"): second crash (rc $rc)"; return 11
    fi
    qlog "crash $(basename "$out") rc $rc: resuming once from its last checkpoint"
    train_run "$cfg" "$out" "$@"
}

# score_run OUT_DIR: bpb (data_prep/bpb.py, E2 METRIC invocation) of the final checkpoint and the two rolling
# checkpoints before it, one JSON per checkpoint in OUT_DIR/bpb/, flattened into OUT_DIR/bpb.jsonl.
score_run() {
    local out=$1 ck b
    mkdir -p "$out/bpb"
    for ck in $(ls "$out"/final_*.pt 2>/dev/null) $(ls "$out"/ckpt_*.pt 2>/dev/null | sort | tail -2); do
        b=$(basename "$ck" .pt)
        [ -s "$out/bpb/$b.json" ] && continue
        qlog "score $(basename "$out")/$b"
        ( cd "$CODE" && $PY data_prep/bpb.py "$ck" "$EVAL" --tokenizer "$TOK" --device cuda \
            --precision fp32 --batch-tokens 16384 --out "$out/bpb/$b.json" > /dev/null ) \
            2>> "$out/score.err" || { qlog "scoring failed for $b (see score.err)"; return 1; }
    done
    $PY "$E2DIR/bpb_lines.py" "$out" > "$out/bpb.jsonl"
}

# code_hashes DEST: sha256 of every .py and config the runs use (E2 FIXED 2)
code_hashes() {
    ( cd "$CODE" && find harness data_prep experiments/E2_lr_transfer experiments/E3_seed_noise tokenizer corpus \
        \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.txt' -o -name 'tok_v0_8k.json' \) \
        2>/dev/null | grep -v __pycache__ | sort | xargs sha256sum ) > "$1"
}
