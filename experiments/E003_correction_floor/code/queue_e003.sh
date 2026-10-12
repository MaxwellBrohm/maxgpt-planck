#!/bin/zsh
# E003 queue (pre-registration in ../notes.txt). Every model job goes through guard.py (E002's guard, copied
# byte-for-byte: one model process at a time, LM Studio unloaded, waits for >= 35% free memory, MPS watermarks
# 0.7/0.6, kills on the active-time ceiling, 3x wall clock, < 15% free or > 768 MB swap growth, pauses while the
# lid is closed on battery). Jobs run strictly one after another; a memory-related kill stops the queue.
#   0. wait until E002 is finished: no process matches queue_e002b.sh, E002's ft_test.py, gen_probe.py or
#      run_chat.py (or any other guard model script), then 60 s more with the condition still true
#   A. dry runs, smallest body first (5 steps padded to 768, 2 items per set, loss-mutant and scorer self-tests);
#      a model whose dry run fails gets no further jobs
#   B. baselines: every untouched model scored first (eval sets + the dev draw)
#   C. per model, smallest body first: LR search (seed 0, dev draw only), pick_lr.py (pre-registered rule, one
#      possible extension), then seeds 1 2 3 at the chosen LR on the eval sets, then analyze_e003.py
# Status: one line per job in ../logs/queue.txt; the last line is "QUEUE E003 DONE" or "QUEUE STOPPED: <reason>".
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
CODE=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E003_correction_floor/code
cd $CODE || exit 1
Q=../logs/queue.txt
export HF_HUB_OFFLINE=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }

# ---------------- 0. wait for E002 ----------------
e002_busy() {
  pgrep -f 'queue_e002b\.sh' >/dev/null && return 0
  pgrep -f '(^|[ /])(ft_test|gen_probe|run_chat|rescore|run_battery|check_shared|run_capacity|run_model|uprobe|khard|copysplit|exp_fixed_history)\.py' >/dev/null && return 0
  return 1
}
log "QUEUE E003 WAITING for E002 (pid $$): until no queue_e002b.sh / ft_test.py / gen_probe.py / run_chat.py (or other guard model job) runs, then 60 s"
while true; do
  while e002_busy; do sleep 30; done
  sleep 60
  e002_busy || break
done
log "QUEUE E003 START (E002 idle for 60 s)"

# ---------------- settings (identical to E002 135M except the LR, which is searched) ----------------
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

analyze() { $PY -B analyze_e003.py > ../logs/analyze_stdout.txt 2>&1 || log "analyze_e003.py failed (see logs/analyze_stdout.txt)" }

run() {  # name ceiling tag model args...   (appends --tag <tag>)
  local name=$1 ceil=$2 tag=$3 mid=$4; shift 4
  rm -f ../logs/$name.guard.json
  $PY guard.py --name $name --ceiling $ceil -- $PY -B e003_ft_test.py $mid "$@" --tag $tag
  local rc=$?
  local killed=$($PY -c "import json;print(json.load(open('../logs/$name.guard.json')).get('killed'))" 2>/dev/null)
  local extra=""
  grep -q "PARAM MISMATCH" ../logs/$name.log 2>/dev/null && extra="$extra PARAM_MISMATCH"
  grep -q "SELFTEST FAIL" ../logs/$name.log 2>/dev/null && extra="$extra SELFTEST_FAIL"
  grep -q "FATAL load check" ../logs/$name.log 2>/dev/null && extra="$extra FATAL_LOAD_CHECK"
  local sps=$($PY -c "import json;print(json.load(open('../out/${mid//\//__}__${tag}__run.json')).get('s_per_step') or '')" 2>/dev/null)
  log "$name exit=$rc killed=$killed$extra${sps:+ s_per_step=$sps}"
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    log "QUEUE STOPPED: $killed in $name"
    analyze
    exit 2
  fi
  return $rc
}

# ---------------- A. dry runs ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  run dry_$short $parts[4] dry $mid $COMMON $DRY_ARGS
  if [[ $? -ne 0 ]]; then SKIP[$short]=1; log "SKIP $short: dry run failed (no further jobs for this model)"; fi
done

# ---------------- B. baselines ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  run base_$short $parts[5] base $mid $COMMON $BASE_ARGS
done
analyze

# ---------------- C. LR search, pick, scored seeds ----------------
for spec in $MODELS; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  if [[ $parts[3] == G4 ]]; then grid=$GRID4; else grid=$GRID3; fi
  for lr in ${=grid}; do
    run lr_${short}_$lr $parts[6] lr${lr}_s0 $mid $COMMON $LRSEARCH_ARGS --lr $lr
  done
  pick=$($PY -B pick_lr.py $mid --grid ${=grid} 2>>../logs/pick_stderr.txt)
  log "pick $short: $pick (grid $grid; logs/lr_pick_*.json)"
  if [[ $pick == EXTEND* ]]; then
    ext=${pick#EXTEND }
    run lr_${short}_$ext $parts[6] lr${ext}_s0 $mid $COMMON $LRSEARCH_ARGS --lr $ext
    pick=$($PY -B pick_lr.py $mid --grid ${=grid} $ext --final 2>>../logs/pick_stderr.txt)
    log "pick $short (final, after extension): $pick"
  fi
  if [[ $pick != CHOSEN* ]]; then
    log "SKIP seeds $short: $pick"
    continue
  fi
  lr=${pick#CHOSEN }
  for s in 1 2 3; do
    run ft_${short}_s$s $parts[7] s$s $mid $COMMON $SCORED_ARGS --seed $s --lr $lr
  done
  analyze
  log "$short block done (tables in logs/tables.txt)"
done
analyze
log "QUEUE E003 DONE"
