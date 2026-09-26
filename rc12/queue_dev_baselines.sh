#!/bin/bash
# RC-12 dev-baseline queue (notes STEP 9 QUEUE; draft s12, s19 item 3). Runs on the PC in WSL, detached (pcdetach.sh).
# Order: models in panel order (the comparator first, the rest of the core panel by size, then the extras by size);
# per model the template render (primary), then plain (diagnostic); per render greedy, then seeds 1, 2, 3.
# Per (model, render, seed): ONE dev_batch.py process under gpu.lock (held for that process only): it loads the
# engine engines.json names for the model, plays the 640 dev conversations in lockstep (<seed>/), then the seed's
# --own-cf twin on the OWN records (<seed>_owncf/), and exits, so other GPU jobs interleave between seeds.
# Idempotent: a seed whose two runs are DONE is skipped (dev_batch.py never overwrites a finished run and redoes a
# stale .partial). One status line per run is appended to <root>/queue_status.txt (queue_status.py). After a failed
# process the rest of that model and render is skipped (a restart retries it). DONE marker at the end.
# Stop between processes: touch ~/planck/logs/rc12_dev_queue.STOP (the running process finishes first).
# The Mac test (test_queue.py, fakes only) overrides Q_CODE Q_ROOT Q_LOGS Q_LOCKS Q_PY Q_ENGINES Q_MODELS Q_RENDERS
# Q_SEEDS Q_LIMIT Q_TIMEOUT Q_LOCK_WAIT.
CODE=${Q_CODE:-$HOME/planck/dev/rc12_q9}
ROOT=${Q_ROOT:-$HOME/planck/runs/rc12_dev}
LOGS=${Q_LOGS:-$HOME/planck/logs}
LOCKS=${Q_LOCKS:-$HOME/planck/locks}
PY=${Q_PY:-$HOME/planck/venv-vllm/bin/python}
ENGINES=${Q_ENGINES:-$CODE/engines.json}
RENDERS=${Q_RENDERS:-template plain}
SEEDS=${Q_SEEDS:-greedy 1 2 3}
TMO=${Q_TIMEOUT:-10800}          # seconds per process (load + dev run + own-cf twin)
LOCK_WAIT=${Q_LOCK_WAIT:-7200}
MODELS=${Q_MODELS:-"Qwen/Qwen2.5-0.5B-Instruct
tiiuae/Falcon-H1-Tiny-90M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct SmallDoge/Doge-160M-Instruct
LiquidAI/LFM2.5-230M LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen3.5-0.8B LiquidAI/LFM2-2.6B
unsloth/gemma-3-270m-it Qwen/Qwen3-0.6B LiquidAI/LFM2-700M LiquidAI/LFM2-1.2B"}
NAME=rc12_dev_queue
ts() { date +%Y-%m-%dT%H:%M:%S; }
mkdir -p "$LOGS/rc12_dev" "$LOCKS" "$ROOT"
exec >> "$LOGS/$NAME.log" 2>&1
exec 8> "$LOCKS/$NAME.lock"
flock -n 8 || { echo "$(ts) queue: another queue holds $LOCKS/$NAME.lock, exiting"; exit 0; }
rm -f "$LOGS/$NAME.DONE"
export PATH="/usr/lib/wsl/lib:$PATH" HF_HOME=${HF_HOME:-$HOME/planck/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false
cd "$CODE" || { echo "$(ts) queue: no code dir $CODE"; exit 1; }
STATUS=$ROOT/queue_status.txt
echo "$(ts) queue: start, code $CODE, root $ROOT, pid $$"

engine_of() {
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"].get(sys.argv[2], {}).get("engine", ""))' \
    "$ENGINES" "$1" 2>/dev/null
}
status() {  # status <model> <render> <seed> <engine> <exit> <t0> <t1> <runs> [note]: one line per run in <runs>
  "$PY" -B queue_status.py --root "$ROOT" --model "$1" --render "$2" --seed "$3" --engine "$4" --exit "$5" \
    --t0 "$6" --t1 "$7" --runs "$8" --note "${9:-}" >> "$STATUS" || echo "$(ts) queue: status failed: $1 $2 $3"
}

for m in $MODELS; do
  s=${m##*/}
  for r in $RENDERS; do
    failed=""
    for seed in $SEEDS; do
      d=$ROOT/$s/$r/$seed
      [ -e "$d/DONE" ] && [ -e "${d}_owncf/DONE" ] && continue
      now=$(date +%s); runs=""; [ -e "$d/DONE" ] || runs=dev; [ -e "${d}_owncf/DONE" ] || runs="$runs owncf"
      if [ -n "$failed" ]; then
        status "$m" "$r" "$seed" "${eng:--}" SKIP "$now" "$now" "$runs" "after_failure_of_$failed"; continue
      fi
      eng=$(engine_of "$m")
      if [ -z "$eng" ]; then status "$m" "$r" "$seed" - NOENGINE "$now" "$now" "$runs" not_in_engines.json; continue; fi
      extra=""; case $eng in hf|hfb) extra=--hf-untested-ok;; esac
      [ -n "$Q_LIMIT" ] && extra="$extra --limit $Q_LIMIT"
      log=$LOGS/rc12_dev/${s}_${r}_${seed}.log; lk=$LOGS/rc12_dev/.${s}_${r}_${seed}.lockstart
      for try in 1 2 3; do
        if [ -e "$LOGS/$NAME.STOP" ]; then echo "$(ts) queue: STOP file, exiting"; exit 0; fi
        echo "$(ts) queue: $s $r $seed ($eng) waiting for the GPU lock (try $try)"
        t0=$(date +%s); rm -f "$lk"
        flock -E 75 -w "$LOCK_WAIT" "$LOCKS/gpu.lock" bash -c 'date +%s > "$0"; exec timeout -k 120 "$@" 8>&-' \
          "$lk" "$TMO" "$PY" -B dev_batch.py --responder "$eng:$m" --render "$r" --seeds "$seed" --root "$ROOT" \
          --engines "$ENGINES" $extra >> "$log" 2>&1
        rc=$?
        [ $rc -ne 75 ] && break
      done
      t1=$(date +%s); [ -s "$lk" ] && t0=$(cat "$lk"); rm -f "$lk"
      echo "$(ts) queue: $s $r $seed ($eng) exit $rc, $((t1 - t0)) s under the lock, log $log"
      status "$m" "$r" "$seed" "$eng" "$rc" "$t0" "$t1" "$runs"
      [ $rc -ne 0 ] && failed=$seed
    done
  done
done
"$PY" -B queue_status.py --root "$ROOT" --progress --models "$MODELS" --renders "$RENDERS" --seeds "$SEEDS" \
  > "$LOGS/$NAME.DONE" 2>&1
echo "$(ts) queue: end; $(head -1 "$LOGS/$NAME.DONE")"
