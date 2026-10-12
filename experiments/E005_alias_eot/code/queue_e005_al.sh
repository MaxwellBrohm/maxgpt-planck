#!/bin/zsh
# E005 AL queue (notes.txt STEP 5). queue_e005.sh's logic (wait_idle, guarded, check_step, stop rules, log format)
# with the AL jobs. Every model job goes through code/guard.py (E005's byte copy of E004's guard).
#   W. wait until ../logs/queue.txt holds "QUEUE E005 DONE" or a "QUEUE STOPPED" line, AND no zsh runs
#      queue_e005.sh, AND guard.other_model_jobs() is empty; then 60 s; then all three again
#   0. no-model checks (a failure stops): the AL items match al/al_items.sha256 and the builder still reproduces
#      them byte for byte; test_al.py (mutation test of the AL checks and tables)
#   A. dry run on the untouched model (4 items, both renders, scorer self-test); a failure stops the queue
#   B. the untouched SmolLM2-135M-Instruct (HF id, offline)
#   C. each E005 seed's saved weights, ../weights/<slug>__s1..s5 (saved by queue_e005.sh with --save 1)
#   D. E004's saved seeds s1-s5 (read-only; the ADDITION lists them)
#   E. analyze_al.py -> ../al/tables_al.txt, ../al/al_results.json
# A failed scoring job is logged as a GAP and the queue goes on; memory and preflight verdicts stop it.
# Status: one line per job in ../logs/queue_al.txt; the last line is "QUEUE AL DONE" or "QUEUE STOPPED: <reason>".
# Test hooks (simulation only; unset in the real run): AL_PY (python), AL_WAIT (the 60 s wait), AL_E5NAME (the
# script name counted as E005's queue).
PY=${AL_PY:-/Users/brohm/Documents/Projects/video-editor/.venv/bin/python}
W=${AL_WAIT:-60}
E5NAME=${AL_E5NAME:-queue_e005}
CODE=${0:A:h}
cd $CODE || exit 1
Q=../logs/queue_al.txt
E5Q=../logs/queue.txt
E4W=${CODE:h:h}/E004_general_updating/weights
mkdir -p ../logs ../al/out
export HF_HUB_OFFLINE=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }
tables() { $PY -B analyze_al.py > ../logs/analyze_al_stdout.txt 2>&1 || log "analyze_al.py failed (see logs/analyze_al_stdout.txt)" }
stop() { log "QUEUE STOPPED: $*"; tables; exit 2 }

n_other_jobs() {  # number of real model jobs running (guard.py's own rule); prints NaN on a failed check
  local n
  n=$($PY -B -c "import guard; print(len(guard.other_model_jobs()))" 2>/dev/null)
  [[ $n == <-> ]] && echo $n || echo NaN
}

n_e005_queues() {  # zsh processes running queue_e005.sh itself (not this script)
  ps -axo pid=,command= | awk -v s="$E5NAME.sh" '$2 ~ /(^|\/)zsh$/ && ($3 == s || substr($3, length($3) - length(s)) == "/" s)' | wc -l | tr -d ' '
}

e005_ended() { grep -qE "QUEUE E005 DONE|QUEUE STOPPED" $E5Q 2>/dev/null }

wait_idle() {  # waits (never stops) while another model job runs
  local n waited=0
  while true; do
    n=$(n_other_jobs)
    [[ $n == NaN ]] && stop "guard.other_model_jobs() check failed"
    [[ $n == 0 ]] && break
    if [[ $waited == 0 ]]; then
      log "waiting: $n other model job(s) running ($($PY -B -c 'import guard; print(guard.other_model_jobs()[0][:120])' 2>/dev/null))"
      waited=1
    fi
    sleep $((W / 2 + 1))
  done
  [[ $waited == 1 ]] && log "other model jobs ended; continuing"
  return 0
}

guarded() {  # name ceiling cmd...   (one guarded model process; returns its exit code)
  local name=$1 ceil=$2; shift 2
  local rc killed extra
  while true; do
    wait_idle
    rm -f ../logs/$name.guard.json
    $PY guard.py --name $name --ceiling $ceil -- "$@"
    rc=$?
    killed=$($PY -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
    if [[ $killed == preflight_refused ]] && tail -1 ../logs/$name.guard.log | grep -q "REFUSE: other model jobs running"; then
      log "$name preflight refused: another model job started first; waiting, then retrying"
      sleep $W
      continue
    fi
    break
  done
  extra=""
  grep -q "SELFTEST FAIL" ../logs/$name.log 2>/dev/null && extra="$extra SELFTEST_FAIL"
  grep -q "FATAL load check" ../logs/$name.log 2>/dev/null && extra="$extra FATAL_LOAD_CHECK"
  grep -q "^refusing" ../logs/$name.log 2>/dev/null && extra="$extra REFUSED"
  local el=$($PY -c "import json;d=json.load(open('../logs/$name.guard.json'));print(d.get('elapsed_s'),d.get('peak_rss_mb'),d.get('min_free'))" 2>/dev/null)
  local res=$(grep -E "^(scored|generated) al/" ../logs/$name.log 2>/dev/null | sed -E 's/ in [0-9]+s.*//' | tr '\n' ' ')
  log "$name exit=$rc killed=$killed$extra (elapsed_s peak_rss_mb min_free: $el) $res"
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    stop "$killed in $name"
  fi
  return $rc
}

score() {  # name tag model   (one AL scoring job)
  guarded $1 1800 $PY -B e005_al_ft_test.py $3 --tag $2
}

check_step() {  # name cmd...   (no-model check; a failure stops the queue)
  local name=$1; shift
  "$@" > ../logs/$name.stdout 2>&1
  local rc=$?
  log "$name exit=$rc ($(tail -1 ../logs/$name.stdout | cut -c1-200))"
  [[ $rc -ne 0 ]] && stop "$name failed (see logs/$name.stdout)"
  return 0
}

# ---------------- W. wait for E005's queue and for no other model job ----------------
log "QUEUE AL WAITING (pid $$): until logs/queue.txt has QUEUE E005 DONE / QUEUE STOPPED, no queue_e005.sh runs and no other model job, then 60 s"
noted=0
while true; do
  if e005_ended && [[ $(n_e005_queues) == 0 ]]; then
    wait_idle
    sleep $W
    e005_ended && [[ $(n_e005_queues) == 0 && $(n_other_jobs) == 0 ]] && break
    continue
  fi
  if [[ $noted == 0 ]] && ! e005_ended && [[ $(n_e005_queues) == 0 ]]; then
    log "waiting: no queue_e005.sh runs but logs/queue.txt has no end line (the queue waits)"
    noted=1
  fi
  sleep $W
done
log "QUEUE AL START (E005's last line: $(tail -1 $E5Q | cut -c1-160))"

# ---------------- 0. no-model checks ----------------
check_step al_items_check $PY -B -c "import items_al as A; its = A.load(); assert A.dumps(A.build()).encode() == open(A.ITEMS_PATH, 'rb').read(), 'rebuild differs'; print('AL items: sha256 matches al_items.sha256, rebuilt byte-identical,', len(its), 'items')"
TMPD=$(mktemp -d -t e005al) || stop "mktemp failed"
check_step test_al $PY -B test_al.py $TMPD
rm -rf $TMPD

M135=HuggingFaceTB/SmolLM2-135M-Instruct
S135=HuggingFaceTB__SmolLM2-135M-Instruct

# ---------------- A. dry run ----------------
guarded al_dry 900 $PY -B e005_al_ft_test.py $M135 --tag dry --dry || stop "AL dry run failed (see logs/al_dry.log)"

# ---------------- B. untouched model ----------------
score al_base base $M135 || log "GAP: AL on the untouched model failed (see logs/al_base.log)"

# ---------------- C. E005 seeds ----------------
for s in 1 2 3 4 5; do
  w=${CODE:h}/weights/${S135}__s$s
  if [[ -f $w/model.safetensors ]]; then
    score al_e005_s$s e005_s$s $w || log "GAP: AL on E005 s$s failed (see logs/al_e005_s$s.log)"
  else
    log "GAP: no E005 weights for s$s ($w/model.safetensors missing)"
  fi
done

# ---------------- D. E004 seeds (read-only) ----------------
for s in 1 2 3 4 5; do
  w=$E4W/${S135}__s$s
  if [[ -f $w/model.safetensors ]]; then
    score al_e004_s$s e004_s$s $w || log "GAP: AL on E004 s$s failed (see logs/al_e004_s$s.log)"
  else
    log "GAP: no E004 weights for s$s ($w/model.safetensors missing)"
  fi
done

# ---------------- E. tables ----------------
tables
log "tables: $(tail -1 ../al/tables_al.txt 2>/dev/null | cut -c1-400) (al/tables_al.txt)"
log "QUEUE AL DONE"
