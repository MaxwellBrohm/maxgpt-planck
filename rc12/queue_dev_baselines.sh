#!/bin/bash
# RC-12 dev-baseline queue (notes STEP 9 QUEUE, STEP 9g; draft s12, s19 item 3). Runs on the PC in WSL, detached.
# Order: models in panel order (the comparator first, the rest of the core panel by size, then the extras by size);
# per model the template render (primary), then plain (diagnostic); per render greedy, then seeds 1, 2, 3.
# Per (model, render, seed): ONE dev_batch.py process: it loads the engine engines.json names for the model, plays the
# 640 dev conversations in lockstep (<seed>/), then the seed's --own-cf twin on the OWN records (<seed>_owncf/), exits.
# GPU LOCK SESSIONS (STEP 9g): the queue takes gpu.lock on its own fd 9 (flock -w LOCK_WAIT; a timeout, exit 75, is
# retried 3 times) and runs consecutive processes under that one hold. Right after each process it releases the lock
# when the session has crossed SESSION_S, or when less than max(SESSION_TAIL, that process's own time) is left of
# it; the next process then waits for the lock again. So a session never ends mid-process, and a process after the
# first of a session starts only with at least SESSION_TAIL (and the last process's time) left. dev_batch.py and
# everything it starts get neither lock fd (8, 9); the per-process timeout keeps fd 9, so if the queue dies
# mid-process the lock goes when that process ends; the flock waiter gets no fd 8, so a queue killed while it
# waits frees the queue lock at once. SESSION_S 0 = one hold per process (the pre-9g behaviour).
# Idempotent: a seed whose two runs are DONE is skipped (dev_batch.py never overwrites a finished run and redoes a
# stale .partial). One status line per run is appended to <root>/queue_status.txt (queue_status.py; note = the
# session and the process's place in it). After a failed process the rest of that model and render is skipped (a
# restart retries it). DONE marker at the end.
# Stop between processes: touch ~/planck/logs/rc12_dev_queue.STOP (the running process finishes first).
# Python per model: engines.json's "python" for the model (~ expanded; Doge runs in ~/planck/venv-doge with
# transformers 4.55, notes STEP 9c), else $PY; a model whose python is not executable gets NOPYTHON lines.
# The Mac test (test_queue.py, fakes only) overrides Q_CODE Q_ROOT Q_LOGS Q_LOCKS Q_PY Q_ENGINES Q_MODELS Q_RENDERS
# Q_SEEDS Q_LIMIT Q_TIMEOUT Q_LOCK_WAIT Q_SESSION_S Q_SESSION_TAIL and Q_CLOCK (a file holding a fake epoch time).
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
SESSION_S=${Q_SESSION_S:-1800}   # one gpu.lock hold runs processes back to back for about this long
SESSION_TAIL=${Q_SESSION_TAIL:-300}
MODELS=${Q_MODELS:-"Qwen/Qwen2.5-0.5B-Instruct
tiiuae/Falcon-H1-Tiny-90M-Instruct HuggingFaceTB/SmolLM2-135M-Instruct SmallDoge/Doge-160M-Instruct
LiquidAI/LFM2.5-230M LiquidAI/LFM2.5-350M HuggingFaceTB/SmolLM2-360M-Instruct Qwen/Qwen3.5-0.8B LiquidAI/LFM2-2.6B
unsloth/gemma-3-270m-it Qwen/Qwen3-0.6B LiquidAI/LFM2-700M LiquidAI/LFM2-1.2B"}
NAME=rc12_dev_queue
ts() { date +%Y-%m-%dT%H:%M:%S; }
now() { if [ -n "$Q_CLOCK" ]; then cat "$Q_CLOCK"; else date +%s; fi; }
mkdir -p "$LOGS/rc12_dev" "$LOCKS" "$ROOT"
exec >> "$LOGS/$NAME.log" 2>&1
exec 8> "$LOCKS/$NAME.lock"
flock -n 8 || { echo "$(ts) queue: another queue holds $LOCKS/$NAME.lock, exiting"; exit 0; }
rm -f "$LOGS/$NAME.DONE"
export PATH="/usr/lib/wsl/lib:$PATH" HF_HOME=${HF_HOME:-$HOME/planck/hf} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false
cd "$CODE" || { echo "$(ts) queue: no code dir $CODE"; exit 1; }
STATUS=$ROOT/queue_status.txt
echo "$(ts) queue: start, code $CODE, root $ROOT, pid $$, gpu.lock sessions of $SESSION_S s (tail $SESSION_TAIL s)"

engine_of() {
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"].get(sys.argv[2], {}).get("engine", ""))' \
    "$ENGINES" "$1" 2>/dev/null
}
python_of() {  # the model's python: engines.json "python" (~ expanded), else $PY
  local p
  p=$("$PY" -c 'import json, os, sys
print(os.path.expanduser(json.load(open(sys.argv[1]))["models"].get(sys.argv[2], {}).get("python", "")))' \
    "$ENGINES" "$1" 2>/dev/null)
  echo "${p:-$PY}"
}
status() {  # status <model> <render> <seed> <engine> <exit> <t0> <t1> <runs> [note]: one line per run in <runs>
  "$PY" -B queue_status.py --root "$ROOT" --model "$1" --render "$2" --seed "$3" --engine "$4" --exit "$5" \
    --t0 "$6" --t1 "$7" --runs "$8" --note "${9:-}" >> "$STATUS" || echo "$(ts) queue: status failed: $1 $2 $3"
}
held=0; sn=0; sk=0; st0=0     # session: gpu.lock held on fd 9, session number, processes in it, its start
take_lock() {  # take_lock <what>: gpu.lock on fd 9, up to 3 waits of LOCK_WAIT s; sets rc and t0 (the last wait's)
  local try
  for try in 1 2 3; do
    if [ -e "$LOGS/$NAME.STOP" ]; then echo "$(ts) queue: STOP file, exiting"; exit 0; fi
    echo "$(ts) queue: $1 waiting for the GPU lock (try $try)"
    t0=$(now)
    exec 9>> "$LOCKS/gpu.lock"
    flock -E 75 -w "$LOCK_WAIT" 9 8>&-   # the waiter gets no fd 8: a queue killed mid-wait frees its lock at once
    rc=$?
    if [ $rc -eq 0 ]; then
      held=1; sn=$((sn + 1)); sk=0; st0=$(now); echo "$st0" > "$lk"
      echo "$(ts) queue: session $$.$sn: GPU lock taken after $((st0 - t0)) s"; return 0
    fi
    exec 9>&-
    [ $rc -ne 75 ] && break
  done
}
release() {  # release <why>: end the session; fd 9 is the lock's only holder once the process has exited
  [ $held -eq 1 ] || return 0
  exec 9>&-
  held=0
  echo "$(ts) queue: session $$.$sn: GPU lock released after $(($(now) - st0)) s, $sk processes ($1)"
}

for m in $MODELS; do
  s=${m##*/}
  for r in $RENDERS; do
    failed=""
    for seed in $SEEDS; do
      d=$ROOT/$s/$r/$seed
      [ -e "$d/DONE" ] && [ -e "${d}_owncf/DONE" ] && continue
      t=$(now); runs=""; [ -e "$d/DONE" ] || runs=dev; [ -e "${d}_owncf/DONE" ] || runs="$runs owncf"
      if [ -n "$failed" ]; then
        status "$m" "$r" "$seed" "${eng:--}" SKIP "$t" "$t" "$runs" "after_failure_of_$failed"; continue
      fi
      eng=$(engine_of "$m")
      if [ -z "$eng" ]; then status "$m" "$r" "$seed" - NOENGINE "$t" "$t" "$runs" not_in_engines.json; continue; fi
      mpy=$(python_of "$m"); pnote=""; [ "$mpy" != "$PY" ] && pnote=", $mpy"
      if [ ! -x "$mpy" ]; then
        status "$m" "$r" "$seed" "$eng" NOPYTHON "$t" "$t" "$runs" "no_python_$mpy"; continue
      fi
      extra=""; case $eng in hf|hfb) extra=--hf-untested-ok;; esac
      [ -n "$Q_LIMIT" ] && extra="$extra --limit $Q_LIMIT"
      log=$LOGS/rc12_dev/${s}_${r}_${seed}.log; lk=$LOGS/rc12_dev/.${s}_${r}_${seed}.lockstart
      rm -f "$lk"; note=""
      if [ $held -eq 0 ]; then take_lock "$s $r $seed ($eng$pnote)"
      elif [ -e "$LOGS/$NAME.STOP" ]; then release "STOP file"; echo "$(ts) queue: STOP file, exiting"; exit 0
      fi
      if [ $held -eq 1 ]; then
        sk=$((sk + 1)); note="session $$.$sn process $sk"; [ -s "$lk" ] || now > "$lk"
        echo "$(ts) queue: $s $r $seed ($eng$pnote) session $$.$sn process $sk"
        timeout -k 120 "$TMO" bash -c 'exec 9>&-; exec "$@"' proc "$mpy" -B dev_batch.py --responder "$eng:$m" \
          --render "$r" --seeds "$seed" --root "$ROOT" --engines "$ENGINES" $extra >> "$log" 2>&1 8>&-
        rc=$?
      fi
      t1=$(now); [ -s "$lk" ] && t0=$(cat "$lk"); rm -f "$lk"
      if [ $held -eq 1 ]; then
        need=$((t1 - t0)); [ $need -lt $SESSION_TAIL ] && need=$SESSION_TAIL
        if [ $t1 -ge $((st0 + SESSION_S)) ] || [ $((st0 + SESSION_S - t1)) -lt $need ]; then
          release "$((st0 + SESSION_S - t1)) s of $SESSION_S left, next needs $need"
        fi
      fi
      echo "$(ts) queue: $s $r $seed ($eng) exit $rc, $((t1 - t0)) s under the lock, log $log"
      status "$m" "$r" "$seed" "$eng" "$rc" "$t0" "$t1" "$runs" "$note"
      [ $rc -ne 0 ] && failed=$seed
    done
  done
done
release "end of the queue"
"$PY" -B queue_status.py --root "$ROOT" --progress --models "$MODELS" --renders "$RENDERS" --seeds "$SEEDS" \
  > "$LOGS/$NAME.DONE" 2>&1
echo "$(ts) queue: end; $(head -1 "$LOGS/$NAME.DONE")"
