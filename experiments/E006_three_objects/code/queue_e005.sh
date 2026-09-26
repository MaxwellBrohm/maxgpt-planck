#!/bin/zsh
# E005 queue (notes.txt QUEUE; step 4). E004's queue logic (queue_e004.sh: wait_idle, guarded, check_step, stop rules,
# log format) with E005's jobs. Every model job goes through code/guard.py, E005's byte copy of E004's guard (sha256
# in ../logs/copied_from_e004_sha256.txt): one model process at a time, LM Studio unloaded, >= 35% free memory to
# start, MPS watermarks 0.7/0.6, kills on the active-time ceiling, 3x wall clock, < 15% free or swap growth; pauses
# while the lid is closed on battery.
#   W. wait until E004's logs/queue.txt holds "QUEUE E004-TS1M DONE", "E004-TS1M not started" or a "QUEUE STOPPED"
#      line, AND no queue_e004*.sh runs, AND guard.other_model_jobs() is empty; then 60 s; then all three again
#   0. no-model checks (any failure stops the queue): a sha256 list of code/ at start (logs/code_sha256_at_start.txt),
#      eval_identity_e005 (copies intact, 1,120/1,120 eval/dev/probe items), validate_e005 (kept examples: oracles,
#      drawn-vs-kept shift <= 2 points, purity), test_train_e005, test_render_e005, selfcheck_e005, test_analyze_e005,
#      test_queue_tools_e005
#   A. dry run (5 steps padded to 768; loss self-check on the first, a plain and a chat batch with both wrong-loss
#      mutants caught; both renders trained; 2 items per set, plain and chat eval renders; scorer self-test), then one
#      chat-probe conversation on the untouched model. A failed dry run stops the queue (E005 has one model).
#   B. untouched baseline: reuse_base_e005.py copies E004's untouched-model records (item, record, code and weight
#      checks in its header; logs/base_reuse_sha256.txt). Only if it refuses is the untouched model scored here
#      (e005_ft_test.py --steps 0, the same scoring calls) and its chat probe run.
#   C. seeds 1-5, lr 1.5e-4, 400 steps: train (probe every 50), eval LIK + GEN plain and chat (free generation inside
#      the run), continuity, knowledge, weights saved; right after each seed its 53-conversation chat probe
#   D. analyze_e005.py -> ../results.json, ../logs/tables.txt; the readings go into queue.txt
# A seed that fails (exit != 0, not a memory verdict) is logged and the queue goes on, as E004's did.
# Status: one line per job in ../logs/queue.txt; the last line is "QUEUE E005 DONE" or "QUEUE STOPPED: <reason>".
PY=/Users/brohm/Documents/Projects/video-editor/.venv/bin/python
CODE=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E005_alias_eot/code
E4Q=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E004_general_updating/logs/queue.txt
cd $CODE || exit 1
Q=../logs/queue.txt
mkdir -p ../logs ../out ../transcripts ../weights
export HF_HUB_OFFLINE=1
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }
analyze() { $PY -B analyze_e005.py > ../logs/analyze_stdout.txt 2>&1 || log "analyze_e005.py failed (see logs/analyze_stdout.txt)" }
stop() { log "QUEUE STOPPED: $*"; analyze; exit 2 }

n_other_jobs() {  # number of real model jobs running (guard.py's own rule); prints NaN on a failed check
  local n
  n=$($PY -B -c "import guard; print(len(guard.other_model_jobs()))" 2>/dev/null)
  [[ $n == <-> ]] && echo $n || echo NaN
}

n_e004_queues() {  # zsh processes running an E004 queue script (the interpreter and script name, not any text)
  ps -axo pid=,command= | awk '$2 ~ /(^|\/)zsh$/ && $3 ~ /(^|\/)queue_e004[a-z0-9_]*\.sh$/' | wc -l | tr -d ' '
}

e004_ended() { grep -qE "QUEUE E004-TS1M DONE|E004-TS1M not started|QUEUE STOPPED" $E4Q 2>/dev/null }

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

run() {  # name ceiling tag model args...   (an e005_ft_test.py job; appends --tag <tag>)
  local name=$1 ceil=$2 tag=$3 mid=$4; shift 4
  guarded $name $ceil $PY -B e005_ft_test.py $mid "$@" --tag $tag
}

check_step() {  # name cmd...   (no-model check; a failure stops the queue)
  local name=$1; shift
  "$@" > ../logs/$name.stdout 2>&1
  local rc=$?
  log "$name exit=$rc ($(tail -1 ../logs/$name.stdout | cut -c1-200))"
  [[ $rc -ne 0 ]] && stop "$name failed (see logs/$name.stdout)"
  return 0
}

# ---------------- W. wait for E004's queues and for no other model job ----------------
log "QUEUE E005 WAITING (pid $$): until E004 queue.txt has QUEUE E004-TS1M DONE / E004-TS1M not started / QUEUE STOPPED, no queue_e004 script runs and no other model job, then 60 s"
noted=0
while true; do
  if e004_ended && [[ $(n_e004_queues) == 0 ]]; then
    wait_idle
    sleep 60
    e004_ended && [[ $(n_e004_queues) == 0 && $(n_other_jobs) == 0 ]] && break
    continue
  fi
  if [[ $noted == 0 ]] && ! e004_ended && [[ $(n_e004_queues) == 0 ]]; then
    log "waiting: no queue_e004 script runs but E004 queue.txt has no end line (the queue waits; see notes.txt)"
    noted=1
  fi
  sleep 60
done
log "QUEUE E005 START"

# ---------------- 0. no-model checks ----------------
shasum -a 256 *.py *.sh > ../logs/code_sha256_at_start.txt 2>&1
log "code sha256 at start: $(wc -l < ../logs/code_sha256_at_start.txt | tr -d ' ') files, digest $(shasum -a 256 < ../logs/code_sha256_at_start.txt | cut -c1-16) (logs/code_sha256_at_start.txt)"
TMPD=$(mktemp -d -t e005q) || stop "mktemp failed"
check_step eval_identity_e005 $PY -B eval_identity_e005.py
check_step validate_e005 $PY -B validate_e005.py
check_step test_train_e005 $PY -B test_train_e005.py
check_step test_render_e005 $PY -B test_render_e005.py
check_step selfcheck_e005 $PY -B selfcheck_e005.py
check_step test_analyze_e005 $PY -B test_analyze_e005.py
check_step test_queue_tools_e005 $PY -B test_queue_tools_e005.py $TMPD
rm -rf $TMPD

# ---------------- settings ----------------
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
M135=HuggingFaceTB/SmolLM2-135M-Instruct
S135=HuggingFaceTB__SmolLM2-135M-Instruct
LR=1.5e-04

# ---------------- A. dry runs ----------------
run dry_135m 900 dry $M135 $COMMON --seed 0 --dry --sets eval,cont,know --chat || stop "dry run failed (see logs/dry_135m.log)"
guarded drychat_135m 900 $PY -B run_chat.py $M135 --mode greedy --only R_d2_name --out ../transcripts/dry_chat.jsonl \
  || log "drychat_135m failed: the chat probes will be attempted anyway and reported as gaps if they fail"

# ---------------- B. untouched baseline ----------------
if $PY -B reuse_base_e005.py > ../logs/reuse_base_e005.stdout 2>&1; then
  log "base: REUSED E004's untouched-model records, not rescored ($(tail -1 ../logs/reuse_base_e005.stdout | cut -c1-160); checks in logs/reuse_base_e005.stdout)"
else
  log "base: reuse refused ($(grep -m3 '^FAIL\|REFUSED' ../logs/reuse_base_e005.stdout | tr '\n' ' ' | cut -c1-240)); scoring the untouched model"
  run base_135m 5400 base $M135 $COMMON --seed 0 --steps 0 --sets eval,cont,know --chat
  guarded chat_135m_base 1800 $PY -B run_chat.py $M135 --mode greedy --out ../transcripts/${S135}__base__greedy.jsonl \
    || log "GAP: chat probe of the untouched 135M failed"
fi
analyze

# ---------------- C. seeds 1-5 ----------------
for s in 1 2 3 4 5; do
  run ft_135m_s$s 9000 s$s $M135 $COMMON --seed $s --lr $LR --sets eval,cont,know --chat --save 1 \
    || log "seed $s failed (exit != 0; see logs/ft_135m_s$s.log); it counts as failing, the queue goes on"
  w=../weights/${S135}__s$s
  if [[ -d $w ]]; then
    guarded chat_135m_s$s 1800 $PY -B run_chat.py $w --mode greedy --out ../transcripts/${S135}__s${s}__greedy.jsonl \
      || log "GAP: chat probe of 135M s$s failed"
  else
    log "GAP: no weights for 135M s$s, chat probe not run"
  fi
  analyze
done

# ---------------- D. analysis ----------------
analyze
rd=$($PY -B -c "
import json; r = json.load(open('../results.json'))['e005']['reading']
a = r['alias']
print(f\"rule {r['rule']['label']} | alias {a['label']} (HIGH {a['n_high']}, LOW {a['n_low']}) | stopping learned {r['stopping']['learned']}\")" 2>/dev/null)
log "readings: ${rd:-could not read results.json} (logs/tables.txt)"
log "QUEUE E005 DONE"
