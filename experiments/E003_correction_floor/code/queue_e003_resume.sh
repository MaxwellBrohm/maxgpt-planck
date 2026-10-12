#!/bin/zsh
# E003 RESUME queue (Claude, 2026-09-26; notes.txt RESUME entry). queue_e003.sh stopped at 2026-09-24 14:14:35 on
# base_ts33m (preflight_refused: a guard false positive, fixed in the guard now in code/guard.py); E003 was then
# deferred until E004's tiny ladder was read. This script runs queue_e003.sh's steps in the same order, with the
# same job commands, arguments, ceilings, LR grids, pick_lr.py calls and analysis (pre-registration in ../notes.txt):
#   W. wait until no other model job runs (guard.other_model_jobs(), guard.py's own rule), then 60 s, and again
#   A. 8 dry runs, smallest body first; a model whose dry run fails gets no further jobs
#   B. 8 baselines (eval sets + dev draw), then analyze_e003.py
#   C. per model, smallest body first: LR search (seed 0, dev only), pick_lr.py (one extension at most), seeds
#      1 2 3 at the chosen LR on the eval sets, analyze_e003.py
#   E. analyze_e003.py
# A job whose ../logs/<name>.guard.json records exit 0 and no kill is NOT re-run; it is logged as already finished.
# Infrastructure (as queue_e004_resume.sh): wait_idle before every job; a preflight refusal because another model
# job started first is retried after 60 s. Every other preflight refusal and every memory verdict (free_memory*,
# swap_grew*, preflight*, memory_never*) stops the queue, as in queue_e003.sh.
# Status: ../logs/queue.txt; the last line is "QUEUE E003-RESUME DONE" or "QUEUE STOPPED: <reason>".
# Test hooks (simulation only; unset in the real run): R_PY (python), R_Q (this queue's log; required by R_SIM and
# R_PLAN), R_WAIT (the 60 s wait), R_SIM (exit after W), R_PLAN (walk A-E, log each job command as PLAN instead of
# running it; no guard, no model, no analysis; pick_lr.py is replaced by R_PICK / R_PICK_FINAL, default CHOSEN 1e-03).
PY=${R_PY:-~/Documents/Projects/video-editor/.venv/bin/python}
CODE=REPO/experiments/E003_correction_floor/code
cd $CODE || exit 1
if [[ ( -n $R_SIM || -n $R_PLAN ) && -z $R_Q ]]; then echo "R_SIM / R_PLAN need R_Q (a scratch queue log)"; exit 1; fi
Q=${R_Q:-../logs/queue.txt}
W=${R_WAIT:-60}
export HF_HUB_OFFLINE=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }
analyze() {
  if [[ -n $R_PLAN ]]; then log "PLAN analyze_e003.py"; return 0; fi
  $PY -B analyze_e003.py > ../logs/analyze_stdout.txt 2>&1 || log "analyze_e003.py failed (see logs/analyze_stdout.txt)"
}
stop() { log "QUEUE STOPPED: $*"; analyze; exit 2 }

n_other_jobs() {  # number of real model jobs running (guard.py's own rule); prints NaN on a failed check
  local n
  n=$($PY -B -c "import guard; print(len(guard.other_model_jobs()))" 2>/dev/null)
  [[ $n == <-> ]] && echo $n || echo NaN
}

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
    sleep 30
  done
  [[ $waited == 1 ]] && log "other model jobs ended; continuing"
  return 0
}

guarded() {  # name ceiling run.json cmd...   (one guarded model process; returns its exit code)
  local name=$1 ceil=$2 rj=$3; shift 3
  local rc killed extra el sps
  if [[ -n $R_PLAN ]]; then log "PLAN $name: guard.py --name $name --ceiling $ceil -- $*"; return 0; fi
  while true; do
    wait_idle
    rm -f ../logs/$name.guard.json
    $PY guard.py --name $name --ceiling $ceil -- "$@"
    rc=$?
    killed=$($PY -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
    if [[ $killed == preflight_refused ]] && tail -1 ../logs/$name.guard.log | grep -q "REFUSE: other model jobs running"; then
      log "$name preflight refused: another model job started first; waiting, then retrying"
      sleep 60
      continue
    fi
    break
  done
  extra=""
  grep -q "PARAM MISMATCH" ../logs/$name.log 2>/dev/null && extra="$extra PARAM_MISMATCH"
  grep -q "SELFTEST FAIL" ../logs/$name.log 2>/dev/null && extra="$extra SELFTEST_FAIL"
  grep -q "FATAL load check" ../logs/$name.log 2>/dev/null && extra="$extra FATAL_LOAD_CHECK"
  grep -q "loss self-check failed" ../logs/$name.log 2>/dev/null && extra="$extra LOSS_SELFCHECK_FAIL"
  sps=$($PY -c "import json;print(json.load(open('$rj')).get('s_per_step') or '')" 2>/dev/null)
  el=$($PY -c "import json;d=json.load(open('../logs/$name.guard.json'));print(d.get('elapsed_s'),d.get('peak_rss_mb'),d.get('min_free'))" 2>/dev/null)
  log "$name exit=$rc killed=$killed$extra${sps:+ s_per_step=$sps} (elapsed_s peak_rss_mb min_free: $el)"
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    stop "$killed in $name"
  fi
  return $rc
}

run() {  # name ceiling tag model args...   (an e003_ft_test.py job; appends --tag <tag>, as queue_e003.sh)
  local name=$1 ceil=$2 tag=$3 mid=$4; shift 4
  guarded $name $ceil ../out/${mid//\//__}__${tag}__run.json $PY -B e003_ft_test.py $mid "$@" --tag $tag
}

# ---------------- resume helpers ----------------
done_ok() {  # name: true when ../logs/<name>.guard.json records exit 0 and no kill (a finished job)
  [[ $($PY -c "import json;d=json.load(open('../logs/$1.guard.json'));print(d.get('exit')==0 and d.get('killed') is None)" 2>/dev/null) == True ]]
}

rjob() {  # run's arguments; a finished job is logged and not re-run
  local name=$1 el
  if done_ok $name; then
    el=$($PY -c "import json;d=json.load(open('../logs/$name.guard.json'));print(d.get('ended'),d.get('elapsed_s'),d.get('peak_rss_mb'),d.get('min_free'))" 2>/dev/null)
    log "$name already finished, not re-run (exit=0 killed=None; ended elapsed_s peak_rss_mb min_free: $el)"
    return 0
  fi
  run "$@"
}

pick_lr() {  # pick_lr.py's arguments; prints its one line (R_PLAN: R_PICK / R_PICK_FINAL instead)
  if [[ -n $R_PLAN ]]; then
    if [[ ${argv[-1]} == --final ]]; then echo "${R_PICK_FINAL:-CHOSEN 1e-03}"; else echo "${R_PICK:-CHOSEN 1e-03}"; fi
    return 0
  fi
  $PY -B pick_lr.py "$@" 2>>../logs/pick_stderr.txt
}

n_queue_e003() {  # zsh processes running queue_e003.sh or queue_e003_resume.sh, not this one or its subshells
  ps -axo pid=,ppid=,command= | awk -v me=$$ '
    { pp[$1] = $2; if ($3 ~ /(^|\/)zsh$/ && $4 ~ /(^|\/)queue_e003(_resume)?\.sh$/) q[$1] = 1 }
    END { n = 0
          for (p in q) { x = p; mine = 0
                         for (i = 0; i < 8 && x != "" && x > 1; i++) { if (x == me) { mine = 1; break }; x = pp[x] }
                         if (!mine) n++ }
          print n }'
}

# ---------------- W. wait for no other model job ----------------
if [[ $(n_queue_e003) != 0 ]]; then
  log "QUEUE STOPPED: resume (pid $$) found another queue_e003.sh / queue_e003_resume.sh running; not started"
  exit 2
fi
log "QUEUE E003-RESUME WAITING (pid $$): until no other model job runs (guard.py's rule), then $W s with the condition still true"
while true; do
  wait_idle
  sleep $W
  [[ $(n_other_jobs) == 0 ]] && break
done
log "QUEUE E003-RESUME START${R_PLAN:+ (PLAN: simulation, nothing runs)}"
[[ -n $R_SIM ]] && exit 0

# ---------------- settings (identical to queue_e003.sh) ----------------
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
DRY_ARGS=(--seed 0 --dry --sets eval,dev)
BASE_ARGS=(--seed 0 --steps 0 --sets eval,dev)
LRSEARCH_ARGS=(--seed 0 --sets dev --save 0)
SCORED_ARGS=(--sets eval --save 1)
GRID3="5e-05 3e-04 1e-03"
GRID4="5e-05 3e-04 1e-03 3e-03"
# short  model id  LR grid  ceilings in active seconds: dry base lr scored   (smallest body first)
MODELS=(
  "ts1m   roneneldan/TinyStories-1M  G4 900 1800 3600 3600"
  "p14m   EleutherAI/pythia-14m      G4 900 1800 3600 3600"
  "ts3m   roneneldan/TinyStories-3M  G4 900 1800 3600 3600"
  "p31m   EleutherAI/pythia-31m      G3 900 1800 3600 3600"
  "ts8m   roneneldan/TinyStories-8M  G3 900 2400 3600 4800"
  "p70m   EleutherAI/pythia-70m      G3 900 2400 4800 4800"
  "ts33m  roneneldan/TinyStories-33M G3 900 2400 4800 4800"
  "p160m  EleutherAI/pythia-160m     G3 900 3000 5400 5400"
)
typeset -A SKIP

# ---------------- A. dry runs ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  rjob dry_$short $parts[4] dry $mid $COMMON $DRY_ARGS
  if [[ $? -ne 0 ]]; then SKIP[$short]=1; log "SKIP $short: dry run failed (no further jobs for this model)"; fi
done

# ---------------- B. baselines ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  rjob base_$short $parts[5] base $mid $COMMON $BASE_ARGS
done
analyze

# ---------------- C. LR search, pick, scored seeds ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  if [[ $parts[3] == G4 ]]; then grid=$GRID4; else grid=$GRID3; fi
  for lr in ${=grid}; do
    rjob lr_${short}_$lr $parts[6] lr${lr}_s0 $mid $COMMON $LRSEARCH_ARGS --lr $lr
  done
  pick=$(pick_lr $mid --grid ${=grid})
  log "pick $short: ${pick:-pick failed} (grid $grid; logs/lr_pick_*.json)${R_PLAN:+ SIMULATED}"
  if [[ $pick == EXTEND* ]]; then
    ext=${pick#EXTEND }
    rjob lr_${short}_$ext $parts[6] lr${ext}_s0 $mid $COMMON $LRSEARCH_ARGS --lr $ext
    pick=$(pick_lr $mid --grid ${=grid} $ext --final)
    log "pick $short (final, after extension): ${pick:-pick failed}${R_PLAN:+ SIMULATED}"
  fi
  if [[ $pick != CHOSEN* ]]; then
    log "SKIP seeds $short: ${pick:-pick failed}"
    continue
  fi
  lr=${pick#CHOSEN }
  for s in 1 2 3; do
    rjob ft_${short}_s$s $parts[7] s$s $mid $COMMON $SCORED_ARGS --seed $s --lr $lr
  done
  analyze
  log "$short block done (tables in logs/tables.txt)"
done

# ---------------- E. analysis ----------------
analyze
log "QUEUE E003-RESUME DONE"
