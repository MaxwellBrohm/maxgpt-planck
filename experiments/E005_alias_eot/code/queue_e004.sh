#!/bin/zsh
# E004 queue (pre-registration in ../notes.txt; step 5). Every model job goes through guard.py (E002's fixed guard,
# copied byte-for-byte into this directory, sha256 in ../logs/copied_sha256.txt: one model process at a time, LM
# Studio unloaded, waits for >= 35% free memory, MPS watermarks 0.7/0.6, kills on the active-time ceiling, 3x wall
# clock, < 15% free, or swap growth with < 35% free (or > 4 GB); pauses while the lid is closed on battery).
#   W. wait until E002's queue.txt holds "QUEUE C DONE" or "QUEUE C STOPPED", then until guard.other_model_jobs()
#      is empty (real Python processes only), then 60 s, then check again
#   0. no-model acceptance: validate_e004.py (stream acceptance, every seed x tokenizer), test_rules_e004.py
#      (rules + mutants), test_analyze_e004.py (analyzer and LR pick on synthetic files); any failure stops the queue
#   A. dry runs (5 steps padded to 768, 2 items per set, loss mutants, scorer self-test; SmolLM2 also the dev path
#      and one chat-probe conversation); a model whose dry run fails gets no further jobs
#   B. untouched baselines of every model (eval, cont, know; SmolLM2 also its chat render); an untouched model that
#      passes the rule gets no fine-tune (notes (c))
#   C. SmolLM2-135M-Instruct: LR check (seed 0, dev draw only, 5e-5 vs 1.5e-4), pick, seeds 1-5 (eval, cont, know,
#      chat render, free generation inside each run, weights saved), then the 53-conversation chat probe on the
#      untouched model and each seed, then its reading
#   D. tiny ladder, largest body first: LR search (dev draw), pick (one extension at most), seeds 1-3 (eval, cont,
#      know, free generation inside each run). If the SmolLM2 reading is FAIL (or it was not scored), the ladder
#      runs its LR search only (notes (d))
#   E. analyze_e004.py -> ../results.json, ../logs/tables.txt
# Before EVERY job the queue waits (never stops) while another model job runs; a guard preflight refusal caused by
# another model job is waited out and retried. A memory-related verdict (free_memory*, swap_grew*, memory_never*)
# or any other preflight refusal (LM Studio / ollama) stops the queue, as in E002 and E003.
# Status: one line per job in ../logs/queue.txt; the last line is "QUEUE E004 DONE" or "QUEUE STOPPED: <reason>".
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
CODE=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E004_general_updating/code
E2Q=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E002_ft_test/logs/queue.txt
cd $CODE || exit 1
Q=../logs/queue.txt
mkdir -p ../logs ../out ../transcripts ../weights
export HF_HUB_OFFLINE=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }
analyze() { $PY -B analyze_e004.py > ../logs/analyze_stdout.txt 2>&1 || log "analyze_e004.py failed (see logs/analyze_stdout.txt)" }
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
      sleep 60
      continue
    fi
    break
  done
  extra=""
  grep -q "SELFTEST FAIL" ../logs/$name.log 2>/dev/null && extra="$extra SELFTEST_FAIL"
  grep -q "FATAL load check" ../logs/$name.log 2>/dev/null && extra="$extra FATAL_LOAD_CHECK"
  grep -q "loss self-check failed" ../logs/$name.log 2>/dev/null && extra="$extra LOSS_SELFCHECK_FAIL"
  local el=$($PY -c "import json;d=json.load(open('../logs/$name.guard.json'));print(d.get('elapsed_s'),d.get('peak_rss_mb'),d.get('min_free'))" 2>/dev/null)
  log "$name exit=$rc killed=$killed$extra (elapsed_s peak_rss_mb min_free: $el)"
  if [[ "$killed" == free_memory* || "$killed" == swap_grew* || "$killed" == preflight* || "$killed" == memory_never* ]]; then
    stop "$killed in $name"
  fi
  return $rc
}

run() {  # name ceiling tag model args...   (an e004_ft_test.py job; appends --tag <tag>)
  local name=$1 ceil=$2 tag=$3 mid=$4; shift 4
  guarded $name $ceil $PY -B e004_ft_test.py $mid "$@" --tag $tag
}

check_step() {  # name cmd...   (no-model check; a failure stops the queue)
  local name=$1; shift
  "$@" > ../logs/$name.stdout 2>&1
  local rc=$?
  log "$name exit=$rc ($(tail -1 ../logs/$name.stdout))"
  [[ $rc -ne 0 ]] && stop "$name failed (see logs/$name.stdout)"
  return 0
}

# ---------------- W. wait for E002 queue C and for no other model job ----------------
log "QUEUE E004 WAITING (pid $$): until E002 queue.txt has QUEUE C DONE / QUEUE C STOPPED, then no other model job, then 60 s"
until grep -qE "QUEUE C DONE|QUEUE C STOPPED" $E2Q; do sleep 60; done
while true; do
  wait_idle
  sleep 60
  [[ $(n_other_jobs) == 0 ]] && break
done
log "QUEUE E004 START"

# ---------------- 0. no-model acceptance ----------------
check_step validate_e004 $PY -B validate_e004.py
check_step test_rules_e004 $PY -B test_rules_e004.py
check_step test_analyze_e004 $PY -B test_analyze_e004.py

# ---------------- settings ----------------
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
M135=HuggingFaceTB/SmolLM2-135M-Instruct
S135=HuggingFaceTB__SmolLM2-135M-Instruct
GRID3="5e-05 3e-04 1e-03"
GRID4="5e-05 3e-04 1e-03 3e-03"
# short  model id  grid  ceilings in active seconds: dry base lr scored   (largest body first, notes (d))
TINY=(
  "ts8m   roneneldan/TinyStories-8M  G3 900 2400 3600 5400"
  "p31m   EleutherAI/pythia-31m      G3 900 2400 3600 5400"
  "ts3m   roneneldan/TinyStories-3M  G4 900 2400 3600 5400"
  "p14m   EleutherAI/pythia-14m      G4 900 2400 3600 5400"
  "ts1m   roneneldan/TinyStories-1M  G4 900 2400 3600 5400"
)
typeset -A SKIP NOFT

# ---------------- A. dry runs ----------------
run dry_135m 900 dry $M135 $COMMON --seed 0 --dry --sets eval,cont,know --chat || { SKIP[135m]=1; log "SKIP 135m: dry run failed"; }
if [[ -z $SKIP[135m] ]]; then
  run drydev_135m 900 drydev $M135 $COMMON --seed 0 --dry --sets dev || { SKIP[135m]=1; log "SKIP 135m: dev-path dry run failed"; }
fi
if [[ -z $SKIP[135m] ]]; then
  guarded drychat_135m 900 $PY -B run_chat.py $M135 --mode greedy --only R_d2_name --out ../transcripts/dry_chat.jsonl \
    || log "drychat_135m failed: the chat probe will be attempted anyway and reported as a gap if it fails"
fi
for spec in $TINY; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  run dry_$short $parts[4] dry $mid $COMMON --seed 0 --dry --sets eval,cont,know || { SKIP[$short]=1; log "SKIP $short: dry run failed"; }
done

# ---------------- B. untouched baselines ----------------
[[ -z $SKIP[135m] ]] && run base_135m 5400 base $M135 $COMMON --seed 0 --steps 0 --sets eval,cont,know --chat
for spec in $TINY; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  run base_$short $parts[5] base $mid $COMMON --seed 0 --steps 0 --sets eval,cont,know
done
analyze
for spec in "135m $M135" $TINY; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] ]] && continue
  g=$($PY -B analyze_e004.py --gate-base $mid 2>/dev/null)
  log "untouched $short: ${g:-gate failed}"
  [[ $g == UNTOUCHED_PASS ]] && { NOFT[$short]=1; log "NO FINE-TUNE $short: the untouched model passes (notes (c))"; }
done

# ---------------- C. SmolLM2-135M-Instruct ----------------
if [[ -z $SKIP[135m] && -z $NOFT[135m] ]]; then
  for lr in 5e-05 1.5e-04; do
    run lr_135m_$lr 5400 lr${lr}_s0 $M135 $COMMON --seed 0 --sets dev --save 0 --lr $lr
  done
  pick=$($PY -B pick_lr_e004.py $M135 --mode 135m --lrs 5e-05 1.5e-04 2>>../logs/pick_stderr.txt)
  log "pick 135m: ${pick:-pick failed} (logs/lr_pick_${S135}.json)"
  if [[ $pick == CHOSEN* ]]; then
    lr=${pick#CHOSEN }
    for s in 1 2 3 4 5; do
      run ft_135m_s$s 9000 s$s $M135 $COMMON --seed $s --lr $lr --sets eval,cont,know --chat --save 1
    done
  else
    log "SKIP seeds 135m: ${pick:-pick failed}"
  fi
  analyze
fi
if [[ -z $SKIP[135m] ]]; then
  guarded chat_135m_base 1800 $PY -B run_chat.py $M135 --mode greedy --out ../transcripts/${S135}__base__greedy.jsonl \
    || log "GAP: chat probe of the untouched 135M failed (reported, not skipped silently)"
  for s in 1 2 3 4 5; do
    w=../weights/${S135}__s$s
    if [[ -d $w ]]; then
      guarded chat_135m_s$s 1800 $PY -B run_chat.py $w --mode greedy --out ../transcripts/${S135}__s${s}__greedy.jsonl \
        || log "GAP: chat probe of 135M s$s failed"
    else
      log "GAP: no weights for 135M s$s, chat probe not run"
    fi
  done
fi
analyze
reading=$($PY -B analyze_e004.py --gate $M135 2>/dev/null)
log "reading 135m: ${reading:-gate failed} (logs/tables.txt)"
if [[ $reading == PASS || $reading == PARTIAL-G || $reading == PARTIAL-X || -n $NOFT[135m] ]]; then
  TINY_MODE=full
else
  TINY_MODE=lronly
  log "tiny ladder: LR search only (SmolLM2 reading ${reading:-unknown}; notes (d): a FAIL means the tiny ladder runs its LR search only)"
fi

# ---------------- D. tiny ladder ----------------
for spec in $TINY; do
  parts=(${=spec}); short=$parts[1]; mid=$parts[2]
  [[ -n $SKIP[$short] || -n $NOFT[$short] ]] && continue
  if [[ $parts[3] == G4 ]]; then grid=$GRID4; else grid=$GRID3; fi
  for lr in ${=grid}; do
    run lr_${short}_$lr $parts[6] lr${lr}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $lr
  done
  pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=grid} 2>>../logs/pick_stderr.txt)
  log "pick $short: ${pick:-pick failed} (grid $grid)"
  if [[ $pick == EXTEND* ]]; then
    ext=${pick#EXTEND }
    run lr_${short}_$ext $parts[6] lr${ext}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $ext
    pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=grid} --ext $ext 2>>../logs/pick_stderr.txt)
    log "pick $short (final, after the extension): ${pick:-pick failed}"
  fi
  if [[ $pick != CHOSEN* ]]; then
    log "SKIP seeds $short: ${pick:-pick failed} (no usable LR)"
  elif [[ $TINY_MODE != full ]]; then
    log "SKIP seeds $short: LR search only (would use ${pick#CHOSEN })"
  else
    lr=${pick#CHOSEN }
    for s in 1 2 3; do
      run ft_${short}_s$s $parts[7] s$s $mid $COMMON --seed $s --lr $lr --sets eval,cont,know --save 0
    done
  fi
  analyze
  log "$short block done (logs/tables.txt)"
done

# ---------------- E. analysis ----------------
analyze
log "QUEUE E004 DONE"
