#!/bin/bash
# queue_k.sh: run a Track K plan on the PC, one GPU job at a time (K3 round 16, 2026-11-21; nothing here has run on
# the PC yet: the K0-smoke plan is its first run). Start it detached (pcdetach.sh pattern), never tied to SSH.
#
#   bash queue_k.sh --code CODE_ROOT --plan PLAN_FILE --data DATA_ROOT
#
# queue_screens.sh's shape on E2's queue engine (experiments/E2_lr_transfer/queue_lib.sh, sourced, not edited), with
# Track K's pieces in place of the SCREENS ones:
#   1. "train NAME" runs experiments/K/configs/NAME.yaml. Before the GPU lock: kconfig.py check NAME (K's registry:
#      seeds, LR5 x g with g_K from the readings file, the arm's data block, schedule, model, the k block) and
#      kpreflight.py (E2's preflight --strict, the K data root's gen.py check passed with this code). A refusal stops
#      the queue.
#   2. Training is ktrain.py (code/notes.txt SPEC 14 BUILD: harness train.py with the k: block installed), with
#      --require-committed; queue_lib's train_run is redefined for that line and K's STOP, its exit codes kept (0 done,
#      10 DIVERGED, 11 GAP, 12 STOP, 13 prereg refused, 14 heat).
#   3. After a finished run (smoke runs aside: never read), still holding gpu.lock: probes.py on the final
#      checkpoint over K-EVAL-ID, K-EVAL-S, K-EVAL-OOD, FB-PROBE, CONFLICT, K-DEV (and K5's sets for an L arm) into
#      out_dir/k_eval.jsonl (SPEC 9, 12), then queue_lib's score_run (bpb, E2 METRIC). SCORED is written last.
#      A K0-g run scores K-DEV alone (2026-11-22, REVIEW 17b: SPEC 8's K-DEV is for g picks; a selection run puts
#      no K-EVAL, FB-PROBE or CONFLICT number on disk).
#   4. The dirty check and the code hash list also cover experiments/K.
#   5. The disk guard's floor is 40 GB (K OPEN step 3: keep 40 GB free), not queue_lib's 25 (2026-11-23, K review);
#      QUEUE_MIN_FREE_GB still sets it (the tests).
# Plan lines: train NAME | mark TEXT | wait_mark EXP TEXT | stop. Outputs under ~/planck/runs/K/: <run>/,
# queue_k.txt (this log), status.jsonl, preflight/, check/, marks/, code_sha256_*.txt. A queue cut mid-run (power)
# leaves the run without final_*.pt; the next start resumes it from its checkpoint (train_run's one-crash rule).
# Exit: 0 plan finished or "stop"; 1 the queue stopped (see the log); 2 usage.
# Stopping K alone (2026-11-22, port review note 6): ~/planck/runs/K/STOP. ~/planck/STOP stops every Planck queue
# (queue_lib.sh guards), Planck's own training included, so K's plans never ask for it. K's STOP ends a queue waiting
# at a mark, stops a running K run at its checkpoint (as ~/planck/STOP does) and stops the queue before its next train
# line; ~/planck/STOP still does all of that too. A person removes it before the next launch.
set -u
EXP=K; CODE=; PLAN=; DATA=
while [ $# -gt 0 ]; do
    case $1 in
        --code) CODE=$2; shift 2;; --plan) PLAN=$2; shift 2;; --data) DATA=$2; shift 2;;
        *) echo "unknown argument $1"; exit 2;;
    esac
done
[ -n "$CODE" ] && [ -f "$PLAN" ] && [ -n "$DATA" ] || {
    echo "usage: queue_k.sh --code CODE_ROOT --plan PLAN_FILE --data DATA_ROOT"; exit 2; }
CODE=$(cd "$CODE" && pwd); PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
. "$CODE/experiments/E2_lr_transfer/queue_lib.sh"
MINFREE=${QUEUE_MIN_FREE_GB:-40}       # header item 5
KX=$CODE/experiments/K/code
OUT=$P/runs/$EXP
KSTOP=$OUT/STOP
mkdir -p "$OUT/marks" "$OUT/preflight" "$OUT/check" "$P/logs"
exec 8>"$OUT/queue.lock"
flock -n 8 || { echo "another $EXP queue is running"; exit 2; }
exec >> "$OUT/queue_k.txt" 2>&1
slug() { echo "$*" | tr -c 'A-Za-z0-9\n' '_'; }
status() { echo "{\"time\": \"$(date '+%FT%T')\", \"exp\": \"$EXP\", \"run\": \"$1\", \"outcome\": \"$2\"}" >> "$OUT/status.jsonl"; }

train_run() {   # queue_lib.sh's train_run with ktrain.py as the entry (the start line) and K's STOP watched beside
                # ~/planck/STOP (2026-11-22); every other line as queue_lib.sh's
    local cfg=$1 out=$2; shift 2
    mkdir -p "$out"
    [ -e "$out/DIVERGED" ] && return 10
    [ -e "$out/GAP" ] && return 11
    ls "$out"/final_*.pt >/dev/null 2>&1 && return 0
    rm -f "$out/STOP" "$out/HEAT"
    qlog "train $(basename "$cfg") $*"
    ( cd "$KX" && exec $PY ktrain.py "$cfg" --device cuda "$@" ) >> "$out/train.out" 2>&1 &
    local tp=$! hot=0 t m mx=0
    while kill -0 $tp 2>/dev/null; do
        { [ -e "$P/STOP" ] || [ -e "$KSTOP" ]; } && touch "$out/STOP"
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
    [ $rc = 2 ] && { qlog "prereg or k block refused for $(basename "$cfg")"; return 13; }
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

score_k() {     # score_k NAME OUT_DIR: SPEC 9's sets on the final checkpoint, then bpb; SCORED last
    local name=$1 out=$2 ck sets
    ck=$(ls "$out"/final_*.pt 2>/dev/null | tail -1)
    sets="k_eval_id k_eval_s k_eval_ood fb_probe conflict k_dev"
    case $name in
        k0g_*) sets="k_dev" ;;     # K0-g: K-DEV only (SPEC 8: K-DEV is for g picks; no selection run scores a K-EVAL set)
        k5_*) sets="$sets k5" ;;
    esac
    if [ ! -s "$out/k_eval.jsonl" ]; then
        qlog "score $name: $sets"
        ( cd "$KX" && $PY probes.py --ckpt "$ck" --device cuda --out "$out/k_eval.jsonl.tmp" \
            --items $(for s in $sets; do echo "$DATA/evals/$s.jsonl"; done) > /dev/null ) 2>> "$out/score.err" || {
            qlog "K scoring failed for $name (see score.err)"; return 1; }
        mv "$out/k_eval.jsonl.tmp.summary.json" "$out/k_eval.summary.json" && mv "$out/k_eval.jsonl.tmp" "$out/k_eval.jsonl"
    fi
    score_run "$out" || return 1
    sync; touch "$out/SCORED"
}

REF=$(git -C "$CODE" rev-parse HEAD 2>/dev/null) || { qlog "refusing: $CODE is not a git checkout"; exit 1; }
DIRTY=$(cd "$CODE" && git status --porcelain -- harness data_prep tokenizer corpus experiments/E2_lr_transfer \
    experiments/K 2>/dev/null)
[ -z "$DIRTY" ] || { qlog "refusing: uncommitted changes in the code checkout:"; echo "$DIRTY" | head; exit 1; }
link_root
( code_hashes /dev/stdout; cd "$CODE" && find experiments/K -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \
    -o -name '*.txt' \) | grep -v __pycache__ | sort | xargs sha256sum ) > "$OUT/code_sha256_$(date +%Y%m%d_%H%M%S).txt"
qlog "queue start: exp $EXP, plan $(basename "$PLAN") (sha256 $(sha256sum "$PLAN" | cut -c1-16)), code commit $REF, data $DATA"

do_train() {   # 0 = go on with the plan, 1 = the queue stops
    local name=$1 cfg=$CODE/experiments/K/configs/$1.yaml out=$OUT/$1 rc heat=0
    [ -f "$cfg" ] || { qlog "no config for $name"; return 1; }
    if [ -e "$out/DIVERGED" ] || [ -e "$out/GAP" ] || [ -e "$out/SCORED" ]; then
        qlog "skip $name (already $(ls "$out" | grep -E '^(DIVERGED|GAP|SCORED)$' | head -1))"; return 0
    fi
    [ -e "$KSTOP" ] && { qlog "K STOP present ($KSTOP): queue stops before $name"; return 1; }
    ( cd "$KX" && $PY kconfig.py check "$name" ) > "$OUT/check/$name.txt" 2>&1 || {
        qlog "kconfig check refused $name: $(head -3 "$OUT/check/$name.txt" | tr '\n' ' ')"; return 1; }
    ( cd "$KX" && $PY kpreflight.py "$cfg" --code "$CODE" --data "$DATA" --out "$OUT/preflight/$name.json" ) \
        > "$OUT/check/$name.preflight.txt" 2>&1 || {
        qlog "kpreflight refused $name: $(tail -3 "$OUT/check/$name.preflight.txt" | tr '\n' ' ')"; return 1; }
    while :; do
        guards || return 1
        gpu_lock || return 1
        train_run "$cfg" "$out" --require-committed; rc=$?
        if [ $rc = 14 ]; then
            gpu_unlock; heat=$((heat + 1))
            [ $heat -le 5 ] || { qlog "$name stopped for heat 6 times: queue stops"; return 1; }
            continue
        fi
        break
    done
    case $rc in
        0)  case $name in
                k0_smoke*) touch "$out/SCORED"; status "$name" "done (smoke: never scored)" ;;
                *) if score_k "$name" "$out"; then status "$name" "done and scored"; else gpu_unlock; return 1; fi ;;
            esac ;;
        10) status "$name" diverged ;;
        11) status "$name" gap ;;
        *)  gpu_unlock; return 1 ;;          # 12 STOP, 13 refused
    esac
    gpu_unlock
    return 0
}

mapfile -t LINES < "$PLAN"
for line in "${LINES[@]}"; do
    read -r cmd rest <<< "$line"
    case $cmd in
        ''|'#'*) continue ;;
        train) do_train "$rest" || { qlog "queue stops at: train $rest"; exit 1; } ;;
        mark) qlog "MARK $rest"; touch "$OUT/marks/$(slug "$rest")" ;;
        wait_mark)
            read -r ex text <<< "$rest"; said=0
            while [ ! -e "$P/runs/$ex/marks/$(slug "$text")" ]; do
                if [ -e "$P/STOP" ] || [ -e "$KSTOP" ]; then
                    qlog "STOP while waiting for $ex: $text"; qlog "queue stops at: wait_mark $rest"; exit 1
                fi
                [ $said = 1 ] || qlog "waiting for $ex to record: $text"; said=1; sleep "$QWAIT"
            done
            qlog "seen $ex: $text" ;;
        stop) qlog "plan says stop"; exit 0 ;;
        *) qlog "bad plan line: $line"; exit 2 ;;
    esac
done
qlog "plan finished: $(basename "$PLAN")"
exit 0
