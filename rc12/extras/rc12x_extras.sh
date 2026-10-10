#!/bin/bash
# rc12x_extras.sh: the RC-12 extras' dev runs on the PC (rc12/notes.txt STEP 13, 13a, 14X; research/
# CUTS_AUDIT_2026-10.md S9 to S12). Runs from the shipped code dir (git archive of a commit's rc12/ into a dir named
# for the commit, e.g. ~/planck/dev/rc12x_<commit8>, never reused; CODE = this file's parent dir), one queue at a
# time, the panel's own dev-baseline queue
# (CODE/queue_dev_baselines.sh with its default log, DONE marker, queue lock, gpu.lock sessions, status and run file):
#   A  P-144: the three Falcon-H1-Tiny-90M siblings that ship a chat template; template and plain      48 runs
#   B  P-144: Falcon-H1-Tiny-90M-Base, plain only (Q_RENDERS=plain: it ships no chat template)          8 runs
#   D  minimind-3 (Max, 2026-10-07; STEP 13a), vLLM at float16 after its own parity (STEP 14X)          16 runs
#   C  cRia-LM-75M-Instruct, split by engines.json "shards" (10 processes a seed, STEP 14X), LAST       16 runs
#   bash rc12x_extras.sh [--no-chain-wait] [PHASE ...]    (default A B D C; e.g. "A B C" leaves minimind-3 out)
# GPU HOLDS UNDER 35 MIN: every phase passes Q_SESSION_S 1500, Q_SESSION_TAIL 600 and Q_TIMEOUT 1020, so a process
# after a session's first starts with at most 1,500 - 600 s gone and ends within 1,020 + 120 s (timeout's kill):
# no hold passes 2,040 s (34 min); the Falcon-H1 processes took 265 to 408 s, a cRia shard 376 s in a smoke.
# WAITS (pgrep -f; an error counts as running; two quiet polls in a row): before every phase, no RC-12 queue or
# dev_batch.py runs (e.g. one boot.d/30_rc12.sh resumed); before C also (long chains go last) the SCREENS chain has
# ended: every file of MARKS exists (W1's chain of 2026-10-10: the replay's SCREENS_BASE_REPLAY_DONE and the C6 queue's
# c6diag/DONE; the chain's older end, SCREENS_BASE_DIAG_DONE, is on the PC since 10-07) and no SCREENS queue,
# launcher, waiter, gate or diag queue runs (--no-chain-wait skips this part; it waits for no mark). At the start: every model of the chosen phases in CODE/engines.json and in the
# HF cache, and cRia split (shards >= 2) if C is chosen. Before each phase: no ~/planck/STOP or PAUSE, no
# rc12_dev_queue.STOP, no CANCEL file, >= MIN_GB free on /, its models present again. A phase counts as ended only by
# a new "queue: start, code CODE," line in rc12_dev_queue.log whose last line is then "queue: end;", with a DONE
# marker whose first line counts the phase's own runs ("complete X of N runs"; X below N is a finished phase with
# failed processes); the marker is copied to logs/rc12x_<phase>.DONE. Any other end stops this script, exit 1.
# A reboot kills this script; boot.d/30_rc12.sh resumes the cut-off queue with its own Q_ settings (the run file),
# so the phase running at the cut finishes; start this script again by hand for the phases after it (it waits for
# that queue, and the queue skips finished runs and parts). Single instance (flock).
# Files: logs/rc12x_extras.log, .DONE (the end), .CANCEL (touch it: stops before the next phase), locks/rc12x_extras.lock.
# RX_* variables exist for the tests and the scratch dry run only; the PC run sets none.
set -u
P=${RX_HOME:-$HOME/planck}
CODE=${RX_CODE:-$(cd "$(dirname "$0")/.." && pwd)}
MARKS=${RX_MARKS:-runs/SCREENS/marks/SCREENS_BASE_REPLAY_DONE c6diag/DONE}   # under $P; all must exist
POLL=${RX_POLL:-300}; MAXPOLLS=${RX_MAXPOLLS:-0}; MIN_GB=${RX_MIN_GB:-45}
PY=${RX_PY:-$P/venv-vllm/bin/python}
SPAT='queue_screens\.sh|launch_screens\.sh|s006_waiter\.sh|chain_waiter\.sh|base_diag_waiter\.sh|replay_waiter\.sh|diag_queue\.sh|diag_start\.sh|diag_runner\.py|c1a_gate\.sh'
RPAT='queue_dev_baselines\.sh|dev_batch\.py'
L=$P/logs; QL=$L/rc12_dev_queue.log; QD=$L/rc12_dev_queue.DONE
SIBS="tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum tiiuae/Falcon-H1-Tiny-90M-Instruct-pre-DPO tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO"
BASE=tiiuae/Falcon-H1-Tiny-90M-Base
CRIA=sz14/cRia-LM-75M-Instruct
MINI=jingyaogong/minimind-3
HOLD="Q_SESSION_S=1500 Q_SESSION_TAIL=600 Q_TIMEOUT=1020"

say() { echo "$(date '+%F %T') $*"; }

models_of() { case $1 in A) echo "$SIBS";; B) echo "$BASE";; C) echo "$CRIA";; D) echo "$MINI";; esac; }
runs_of() { case $1 in A) echo 48;; B) echo 8;; C|D) echo 16;; esac; }
renders_of() { case $1 in B) echo plain;; *) echo "";; esac; }

quiet() {      # quiet ERE...: 0 when no process matches any ERE; 1 when one does or pgrep cannot tell
    local p rc
    for p in "$@"; do
        pgrep -f "$p" > /dev/null 2>&1; rc=$?
        [ $rc -eq 1 ] || return 1
    done
    return 0
}

blocked() {    # 0 (and says which) when a stop file holds this script
    local f
    for f in "$P/STOP" "$P/PAUSE" "$L/rc12_dev_queue.STOP" "$L/rc12x_extras.CANCEL"; do
        [ -e "$f" ] && { say "STOPPED: $f exists"; return 0; }
    done
    return 1
}

wait_for() {   # wait_for chain|rc12: returns after two quiet polls in a row (chain: the mark present too)
    local what=$1 n=0 ok=0 mark f info
    while :; do
        blocked && exit 1
        if [ "$what" = chain ]; then
            mark=present; for f in $MARKS; do [ -e "$P/$f" ] || mark="absent ($f)"; done
            if [ "$mark" = present ] && quiet "$SPAT" "$RPAT"; then ok=$((ok + 1)); else ok=0; fi
            info="marks [$MARKS] $mark, the SCREENS chain"
        else
            if quiet "$RPAT"; then ok=$((ok + 1)); else ok=0; fi
            info="another RC-12 queue"
        fi
        [ $ok -ge 2 ] && { say "START ($what): $info, nothing of it running (2 polls)"; return 0; }
        [ $((n % 12)) -eq 0 ] && say "WAIT ($what): $info, quiet polls $ok"
        n=$((n + 1))
        [ "$MAXPOLLS" -gt 0 ] && [ $n -ge "$MAXPOLLS" ] && { say "GIVEUP after $n polls (test bound)"; exit 3; }
        sleep "$POLL"
    done
}

preflight() {  # preflight MODEL...: each in CODE/engines.json and in the HF cache (the queue runs offline)
    local m
    for m in "$@"; do
        "$PY" -c 'import json, sys; sys.exit(sys.argv[2] not in json.load(open(sys.argv[1]))["models"])' \
            "$CODE/engines.json" "$m" || { say "REFUSED: $m not in $CODE/engines.json"; exit 1; }
        [ -s "$P/hf/hub/models--${m//\//--}/refs/main" ] || { say "REFUSED: $m not in the HF cache $P/hf"; exit 1; }
    done
    say "preflight: $# models in engines.json and the HF cache"
}

phase() {      # phase X: one queue run of phase X; returns only after its clean end
    local name=$1 want models renders n gb rc t
    want=$(runs_of "$1"); models=$(models_of "$1"); renders=$(renders_of "$1")
    wait_for rc12
    [ "$name" = C ] && [ "$CHAIN" = 1 ] && wait_for chain
    blocked && exit 1
    preflight $models
    gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    [ -n "$gb" ] && [ "$gb" -ge "$MIN_GB" ] || { say "REFUSED: phase $name: ${gb:-?} GB free on /, need $MIN_GB"; exit 1; }
    n=0; [ -f "$QL" ] && n=$(wc -l < "$QL" | tr -dc 0-9); n=${n:-0}
    say "phase $name: Q_MODELS=[$models]${renders:+ Q_RENDERS=[$renders]} $HOLD, $gb GB free"
    if [ -n "$renders" ]; then
        env $HOLD Q_CODE="$CODE" Q_MODELS="$models" Q_RENDERS="$renders" bash "$CODE/queue_dev_baselines.sh" 7>&-
    else
        env $HOLD Q_CODE="$CODE" Q_MODELS="$models" bash "$CODE/queue_dev_baselines.sh" 7>&-
    fi
    rc=$?
    t=$(tail -n +$((n + 1)) "$QL" 2>/dev/null | tr -d '\000')
    if printf '%s\n' "$t" | grep -aqF "queue: start, code $CODE," && printf '%s\n' "$t" | tail -1 | grep -aq "queue: end;" \
        && head -1 "$QD" 2>/dev/null | grep -aqE "^complete [0-9]+ of $want runs$"; then
        cp "$QD" "$L/rc12x_$name.DONE"
        say "phase $name ended (exit $rc): $(head -1 "$QD")"; return 0
    fi
    say "STOPPED: phase $name did not end cleanly (exit $rc); last queue line [$(tail -1 "$QL" 2>/dev/null)]"; exit 1
}

main() {
    local x all="" ph
    CHAIN=1
    [ "${1:-}" = --no-chain-wait ] && { CHAIN=0; shift; }
    PHASES=${*:-A B D C}
    for ph in $PHASES; do
        case $ph in A|B|C|D) all="$all $(models_of "$ph")";; *) echo "usage: $0 [--no-chain-wait] [A|B|C|D ...]"; exit 2;; esac
    done
    mkdir -p "$L" "$P/locks" || exit 1
    exec >> "$L/rc12x_extras.log" 2>&1
    exec 7>> "$P/locks/rc12x_extras.lock"
    flock -n 7 || { say "another rc12x_extras holds locks/rc12x_extras.lock: exiting"; exit 1; }
    say "=== rc12x_extras (pid $$): code $CODE, phases [$PHASES], chain wait before C $CHAIN, marks [$MARKS], poll $POLL s"
    [ -f "$CODE/queue_dev_baselines.sh" ] || { say "REFUSED: no $CODE/queue_dev_baselines.sh"; exit 1; }
    ( cd "$CODE" && md5sum queue_dev_baselines.sh engines.json pins.json dev_batch.py shards.py ) 2>/dev/null | sed 's/^/  md5 /'
    preflight $all     # at once, so a missing model shows before any wait (each phase checks its own again)
    case " $PHASES " in *" C "*)
        x=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]].get("shards", 1))' \
            "$CODE/engines.json" "$CRIA" 2>/dev/null)
        [ "${x:-1}" -ge 2 ] 2>/dev/null || { say "REFUSED: $CRIA is not split (engines.json shards ${x:-?})"; exit 1; };;
    esac
    for ph in $PHASES; do phase "$ph"; done
    for ph in $PHASES; do head -1 "$L/rc12x_$ph.DONE"; done > "$L/rc12x_extras.DONE"
    say "=== end: $(tr '\n' ';' < "$L/rc12x_extras.DONE")"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
