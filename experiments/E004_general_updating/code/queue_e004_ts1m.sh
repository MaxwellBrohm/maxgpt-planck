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


# ======== follow-up: TinyStories-1M re-added (Claude, 2026-09-24 22:15) ========
# Its dry run was killed at 20:56 by the OLD guard wall-clock rule (paused time counted while the lid was
# closed on battery), so the main queue skipped it. guard.py now excludes paused time. Same steps and
# settings as the main queue's tiny ladder; runs only after the main queue ends with QUEUE E004 DONE.
until grep -qE "QUEUE E004 DONE|QUEUE STOPPED" $Q; do sleep 60; done
if ! grep -q "QUEUE E004 DONE" $Q; then log "E004-TS1M not started: the main queue stopped"; exit 0; fi
while true; do wait_idle; sleep 60; [[ $(n_other_jobs) == 0 ]] && break; done
log "QUEUE E004-TS1M START (TinyStories-1M re-added)"
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
GRID4="5e-05 3e-04 1e-03 3e-03"
mid=roneneldan/TinyStories-1M; short=ts1m
run dry_ts1m_r 900 dry $mid $COMMON --seed 0 --dry --sets eval,cont,know || { log "SKIP ts1m (rerun): dry run failed"; analyze; log "QUEUE E004-TS1M DONE"; exit 0; }
run base_ts1m 2400 base $mid $COMMON --seed 0 --steps 0 --sets eval,cont,know
analyze
g=$($PY -B analyze_e004.py --gate-base $mid 2>/dev/null)
log "untouched ts1m: ${g:-gate failed}"
if [[ $g == UNTOUCHED_PASS ]]; then log "NO FINE-TUNE ts1m: the untouched model passes"; analyze; log "QUEUE E004-TS1M DONE"; exit 0; fi
if grep -q "tiny ladder: LR search only" $Q; then TINY_MODE=lronly; else TINY_MODE=full; fi
for lr in ${=GRID4}; do
  run lr_ts1m_$lr 3600 lr${lr}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $lr
done
pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=GRID4} 2>>../logs/pick_stderr.txt)
log "pick ts1m: ${pick:-pick failed} (grid $GRID4)"
if [[ $pick == EXTEND* ]]; then
  ext=${pick#EXTEND }
  run lr_ts1m_$ext 3600 lr${ext}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $ext
  pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=GRID4} --ext $ext 2>>../logs/pick_stderr.txt)
  log "pick ts1m (final, after the extension): ${pick:-pick failed}"
fi
if [[ $pick != CHOSEN* ]]; then
  log "SKIP seeds ts1m: ${pick:-pick failed} (no usable LR)"
elif [[ $TINY_MODE != full ]]; then
  log "SKIP seeds ts1m: LR search only (would use ${pick#CHOSEN })"
else
  lr=${pick#CHOSEN }
  for s in 1 2 3; do run ft_ts1m_s$s 5400 s$s $mid $COMMON --seed $s --lr $lr --sets eval,cont,know --save 0; done
fi
analyze
log "QUEUE E004-TS1M DONE"
