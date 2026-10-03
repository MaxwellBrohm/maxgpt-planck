#!/bin/bash
# queue_screens.sh: run a SCREENS plan on the PC, one GPU job at a time. Normally started detached by
# launch_screens.sh; never tied to an SSH session.
#
#   bash queue_screens.sh --code CODE_ROOT --plan PLAN_FILE
#
# E2's queue engine (experiments/E2_lr_transfer/queue_e2.sh) with three changes, because queue_e2.sh knows only
# --exp E2 | E3 and the SCREENS configs live in several directories (it is not edited here):
#   1. "train NAME" finds the config with qlib_screens.sh cfg_for (exactly one of experiments/screens/configs and
#      experiments/S00?_*/configs), and runs screens.py check on it before preflight: C2 (only the registered
#      keys differ from its BASE), the RC-12 GUARD, C1-b's engine values, the C6 hook and the 2% rule. A refusal stops the queue.
#   2. After a run, an out_dir holding rc12_eval.jsonl stops the queue (C2 RC-12 GUARD, the run-directory half).
#   3. The dirty check and the code hash list also cover experiments/screens and experiments/S00?_*.
# Everything else is queue_lib.sh unchanged: PAUSE / STOP / 75 C / 25 GB guards, gpu.lock held per run (flock
# -w 7200; training plus scoring), preflight.py --strict, train.py --require-committed, one resume after a crash,
# a non-finite loss = DIVERGED (never resumed), scoring of every non-trunk run (final + 2 rolling checkpoints).
# Plan lines: train NAME | mark TEXT | wait_mark EXP TEXT | stop (as queue_e2.sh). Outputs under
# ~/planck/runs/SCREENS/: <run>/, queue_screens.txt (this log), status.jsonl, preflight/, marks/, code_sha256_*.txt.
# Exit: 0 plan finished or "stop"; 1 the queue stopped (see the log); 2 usage.
set -u
EXP=SCREENS; CODE=; PLAN=
while [ $# -gt 0 ]; do
    case $1 in
        --code) CODE=$2; shift 2;; --plan) PLAN=$2; shift 2;;
        *) echo "unknown argument $1"; exit 2;;
    esac
done
[ -n "$CODE" ] && [ -f "$PLAN" ] || { echo "usage: queue_screens.sh --code CODE_ROOT --plan PLAN_FILE"; exit 2; }
CODE=$(cd "$CODE" && pwd); PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
. "$CODE/experiments/E2_lr_transfer/queue_lib.sh"
. "$CODE/experiments/screens/qlib_screens.sh"
SX=$CODE/experiments/screens
OUT=$P/runs/$EXP
mkdir -p "$OUT/marks" "$OUT/preflight" "$OUT/check" "$P/logs"
exec 8>"$OUT/queue.lock"
flock -n 8 || { echo "another $EXP queue is running"; exit 2; }
exec >> "$OUT/queue_screens.txt" 2>&1
slug() { echo "$*" | tr -c 'A-Za-z0-9\n' '_'; }
status() { echo "{\"time\": \"$(date '+%FT%T')\", \"exp\": \"$EXP\", \"run\": \"$1\", \"outcome\": \"$2\"}" >> "$OUT/status.jsonl"; }
pyfield() {   # pyfield JSON_FILE EXPR: evaluate EXPR on the first run of a preflight report
    $PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["runs"][0]; print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null
}

REF=$(git -C "$CODE" rev-parse HEAD 2>/dev/null) || { qlog "refusing: $CODE is not a git checkout"; exit 1; }
DIRTY=$(cd "$CODE" && git status --porcelain -- harness data_prep tokenizer corpus experiments/E2_lr_transfer \
    experiments/screens experiments/S00?_* 2>/dev/null)
[ -z "$DIRTY" ] || { qlog "refusing: uncommitted changes in the code checkout:"; echo "$DIRTY" | head; exit 1; }
link_root
code_hashes_screens "$OUT/code_sha256_$(date +%Y%m%d_%H%M%S).txt"
qlog "queue start: exp $EXP, plan $(basename "$PLAN") (sha256 $(sha256sum "$PLAN" | cut -c1-16)), code commit $REF"

inherited() {  # a branch refused only for a missing init_from whose trunk is DIVERGED or a GAP -> that outcome
    local rep=$1 msg init
    msg=$(pyfield "$rep" '"|".join(r["refusals"])')
    case $msg in "branch init_from "*" does not exist") ;; *) return 1;; esac
    [ "$(pyfield "$rep" 'len(r["refusals"])')" = 1 ] || return 1
    init=$(pyfield "$rep" 'r.get("init_from") or ""')
    [ -n "$init" ] || return 1
    [ -e "$CODE/$(dirname "$init")/DIVERGED" ] && { echo DIVERGED; return 0; }
    [ -e "$CODE/$(dirname "$init")/GAP" ] && { echo GAP; return 0; }
    return 1
}

do_train() {   # 0 = go on with the plan, 1 = the queue stops
    local name=$1 cfg out=$OUT/$1 rc heat=0 mode inh
    cfg=$(cfg_for "$name") || { qlog "no single config for $name (cfg_for)"; return 1; }
    if [ -e "$out/DIVERGED" ] || [ -e "$out/GAP" ] || [ -e "$out/SCORED" ] || [ -e "$out/TRUNK_DONE" ]; then
        qlog "skip $name (already $(ls "$out" | grep -E '^(DIVERGED|GAP|SCORED|TRUNK_DONE)$' | head -1))"; return 0
    fi
    ( cd "$SX" && $PY screens.py check "$name" ) > "$OUT/check/$name.txt" 2>&1 || {
        qlog "screens check refused $name: $(grep -v '^ok' "$OUT/check/$name.txt" | head -3 | tr '\n' ' ')"; return 1; }
    while :; do
        guards || return 1
        gpu_lock || return 1
        if ! $PY "$E2DIR/preflight.py" "$cfg" --code "$CODE" --strict --out "$OUT/preflight/$name.json" > /dev/null; then
            if inh=$(inherited "$OUT/preflight/$name.json"); then
                mkdir -p "$out"; touch "$out/$inh"; status "$name" "$(echo "$inh" | tr 'A-Z' 'a-z') (inherited from its trunk)"
                qlog "$name: $inh, inherited (its trunk ended before the branch point)"; gpu_unlock; return 0
            fi
            qlog "preflight refused $name: $(pyfield "$OUT/preflight/$name.json" '"; ".join(r["refusals"])')"
            gpu_unlock; return 1
        fi
        mode=$(pyfield "$OUT/preflight/$name.json" 'r["schedule"]["mode"]')
        train_run "$cfg" "$out" --require-committed; rc=$?
        if [ $rc = 14 ]; then
            gpu_unlock; heat=$((heat + 1))
            [ $heat -le 5 ] || { qlog "$name stopped for heat 6 times: queue stops"; return 1; }
            continue
        fi
        break
    done
    if [ -e "$out/rc12_eval.jsonl" ]; then
        qlog "RC-12 GUARD: $name wrote rc12_eval.jsonl (RC-12 scored before the lock): queue stops"
        status "$name" "rc12 guard"; gpu_unlock; return 1
    fi
    case $rc in
        0)  if [ "$mode" = trunk ]; then touch "$out/TRUNK_DONE"; status "$name" done
            elif score_run "$out"; then touch "$out/SCORED"; status "$name" "done and scored"
            else gpu_unlock; return 1; fi ;;
        10) status "$name" diverged ;;
        11) status "$name" gap ;;
        *)  gpu_unlock; return 1 ;;          # 12 STOP, 13 prereg refused
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
                [ -e "$P/STOP" ] && { qlog "STOP while waiting for $ex: $text"; exit 1; }
                [ $said = 1 ] || qlog "waiting for $ex to record: $text"; said=1; sleep "$QWAIT"
            done
            qlog "seen $ex: $text" ;;
        stop) qlog "plan says stop"; exit 0 ;;
        *) qlog "bad plan line: $line"; exit 2 ;;
    esac
done
qlog "plan finished: $(basename "$PLAN")"
exit 0
