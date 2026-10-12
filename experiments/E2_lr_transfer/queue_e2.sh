#!/bin/bash
# queue_e2.sh: run an E2 plan (or an E3 plan, through ../E3_seed_noise/queue_e3.sh) on the PC, one GPU job at a
# time. Normally started detached by launch_e2.sh / launch_e3.sh; never tied to an SSH session.
#
#   bash queue_e2.sh --code CODE_ROOT --plan PLAN_FILE [--exp E2|E3]
#
# CODE_ROOT is a git checkout of the committed repo in a directory the queue owns (launch_e2.sh clones the
# bundle into ~/planck/qcode/<commit>); the queue refuses uncommitted changes under harness/, data_prep/,
# tokenizer/, corpus/ and experiments/E2_lr_transfer, E3_seed_noise, and every run passes
# train.py --require-committed and preflight.py --strict.
# Plan lines (plans/*.txt; blank lines and '#' ignored), run top to bottom, idempotent on restart:
#   train NAME            configs/NAME.yaml of this experiment: guards, gpu.lock, preflight, train, and for a
#                         branch or full-schedule run the scorer (final + 2 rolling checkpoints); a finished and
#                         scored run, a DIVERGED run and a GAP are skipped
#   mark TEXT             milestone (e.g. "E2 5M DONE"): queue log line + runs/<EXP>/marks/<TEXT>
#   wait_mark EXP TEXT    wait until experiment EXP's queue recorded TEXT (the joint E2/E3 order)
#   stop                  end the queue here (an analyzer or a person writes the next plan)
# A branch whose trunk diverged (or is a GAP) before the branch point inherits DIVERGED (or GAP) without running.
# Outputs under ~/planck/runs/<EXP>/: <run>/ (train.py out_dir, bpb/, bpb.jsonl, train.out), queue_<exp>.txt
# (this log), status.jsonl (one line per run outcome), preflight/<run>.json, marks/, code_sha256_<time>.txt.
# Exit: 0 plan finished or "stop"; 1 the queue stopped (STOP, disk, refusal, failure: see the log); 2 usage.
set -u
EXP=E2; CODE=; PLAN=
while [ $# -gt 0 ]; do
    case $1 in
        --code) CODE=$2; shift 2;; --plan) PLAN=$2; shift 2;; --exp) EXP=$2; shift 2;;
        *) echo "unknown argument $1"; exit 2;;
    esac
done
[ -n "$CODE" ] && [ -f "$PLAN" ] || { echo "usage: queue_e2.sh --code CODE_ROOT --plan PLAN_FILE [--exp E2|E3]"; exit 2; }
CODE=$(cd "$CODE" && pwd); PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
. "$CODE/experiments/E2_lr_transfer/queue_lib.sh"
case $EXP in
    E2) XD=$CODE/experiments/E2_lr_transfer;; E3) XD=$CODE/experiments/E3_seed_noise;;
    *) echo "unknown experiment $EXP"; exit 2;;
esac
OUT=$P/runs/$EXP
mkdir -p "$OUT/marks" "$OUT/preflight" "$P/logs"
exec 8>"$OUT/queue.lock"
flock -n 8 || { echo "another $EXP queue is running"; exit 2; }
exec >> "$OUT/queue_$(echo "$EXP" | tr 'A-Z' 'a-z').txt" 2>&1
slug() { echo "$*" | tr -c 'A-Za-z0-9\n' '_'; }
status() { echo "{\"time\": \"$(date '+%FT%T')\", \"exp\": \"$EXP\", \"run\": \"$1\", \"outcome\": \"$2\"}" >> "$OUT/status.jsonl"; }
pyfield() {   # pyfield JSON_FILE EXPR: evaluate EXPR on the first run of a preflight report
    $PY -c 'import json,sys; r=json.load(open(sys.argv[1]))["runs"][0]; print(eval(sys.argv[2]))' "$1" "$2" 2>/dev/null
}

REF=$(git -C "$CODE" rev-parse HEAD 2>/dev/null) || { qlog "refusing: $CODE is not a git checkout"; exit 1; }
DIRTY=$(git -C "$CODE" status --porcelain -- harness data_prep tokenizer corpus experiments/E2_lr_transfer \
    experiments/E3_seed_noise 2>/dev/null)
[ -z "$DIRTY" ] || { qlog "refusing: uncommitted changes in the code checkout:"; echo "$DIRTY" | head; exit 1; }
link_root
code_hashes "$OUT/code_sha256_$(date +%Y%m%d_%H%M%S).txt"
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
    local name=$1 cfg=$XD/configs/$1.yaml out=$OUT/$1 rc heat=0 mode inh
    [ -f "$cfg" ] || { qlog "no config $cfg"; return 1; }
    if [ -e "$out/DIVERGED" ] || [ -e "$out/GAP" ] || [ -e "$out/SCORED" ] || [ -e "$out/TRUNK_DONE" ]; then
        qlog "skip $name (already $(ls "$out" | grep -E '^(DIVERGED|GAP|SCORED|TRUNK_DONE)$' | head -1))"; return 0
    fi
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
