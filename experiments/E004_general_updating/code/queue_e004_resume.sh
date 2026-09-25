#!/bin/zsh
# E004 RESUME queue (Claude, 2026-09-25). The main queue (queue_e004.sh) was paused on purpose at 09:32:15 after
# lr_ts3m_1e-03 so E005 could run first (notes.txt deviation line). The tiny ladder is LR-search-only (the 135M
# reading is FAIL, notes (d)). This script runs what is left, with queue_e004.sh's helpers (lines 25-98, copied;
# the only edits are the R_* test hooks below) and the exact job commands and ceilings of queue_e004.sh (tiny
# ladder D) and queue_e004_ts1m.sh:
#   W. wait until E005's logs/queue_al.txt has, after its last "QUEUE AL WAITING" line, a line that starts with a
#      timestamp and "QUEUE AL DONE", "QUEUE AL STOPPED" or "QUEUE STOPPED:"; OR until no queue_e005_al.sh runs and
#      E005's logs/queue.txt has, after its last "QUEUE E005 WAITING" line, a timestamped "QUEUE STOPPED:" line.
#      Either way no queue_e005.sh / queue_e005_al.sh may run. Then wait_idle, 60 s, and all of it again.
#   D. ts3m: G4 grid (5e-05 3e-04 1e-03 3e-03), pick, one extension at most; p14m: the same; no seeds (LR only)
#   T. ts1m (queue_e004_ts1m.sh steps): dry run, baseline, untouched gate, G4 grid, pick, extension; no seeds
#   E. analyze_e004.py -> ../results.json, ../logs/tables.txt
# A job whose ../logs/<name>.guard.json records exit 0 and no kill is NOT re-run; it is logged as already finished.
# Status: ../logs/queue.txt; the last line is "QUEUE E004-RESUME DONE" or "QUEUE STOPPED: <reason>".
# Test hooks (simulation only; unset in the real run): R_PY (python), R_Q (this queue's log), R_ALQ, R_E5Q (E005's
# logs), R_WAIT (the 60 s wait), R_ALNAME, R_E5NAME (script names counted as E005's queues), R_SIM (exit after W).
PY=${R_PY:-/Users/brohm/Documents/Projects/video-editor/.venv/bin/python}
CODE=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E004_general_updating/code
E5=/Users/brohm/Documents/Projects/maxgpt-planck/experiments/E005_alias_eot/logs
cd $CODE || exit 1
Q=${R_Q:-../logs/queue.txt}
ALQ=${R_ALQ:-$E5/queue_al.txt}
E5Q=${R_E5Q:-$E5/queue.txt}
W=${R_WAIT:-60}
ALNAME=${R_ALNAME:-queue_e005_al}
E5NAME=${R_E5NAME:-queue_e005}
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

# ---------------- resume helpers ----------------
TS='^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9] '

n_zsh() {  # zsh processes running <name>.sh (the interpreter and script name, as E005's queues count them)
  ps -axo pid=,command= | awk -v s="$1.sh" '$2 ~ /(^|\/)zsh$/ && ($3 == s || substr($3, length($3) - length(s)) == "/" s)' | wc -l | tr -d ' '
}

al_ended() {  # the AL queue's current run (after its last WAITING line) has a timestamped end line
  awk -v ts="$TS" '$0 ~ (ts "QUEUE AL WAITING") {f=0} $0 ~ (ts "(QUEUE AL DONE|QUEUE AL STOPPED|QUEUE STOPPED:)") {f=1} END {exit !f}' $ALQ 2>/dev/null
}

e005_stopped() {  # E005's current run (after its last WAITING line) has a timestamped "QUEUE STOPPED:" line
  awk -v ts="$TS" '$0 ~ (ts "QUEUE E005 WAITING") {f=0} $0 ~ (ts "QUEUE STOPPED:") {f=1} END {exit !f}' $E5Q 2>/dev/null
}

ready() {  # no E005 queue script runs, and the AL queue ended (or E005 stopped with the AL queue gone)
  [[ $(n_zsh $ALNAME) == 0 && $(n_zsh $E5NAME) == 0 ]] || return 1
  al_ended || e005_stopped
}

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

# ---------------- W. wait for E005's AL queue and for no other model job ----------------
log "QUEUE E004-RESUME WAITING (pid $$): until E005 queue_al.txt has QUEUE AL DONE / QUEUE STOPPED (or no queue_e005_al.sh runs and E005 queue.txt has QUEUE STOPPED:), no queue_e005*.sh runs and no other model job, then 60 s"
noted=0
while true; do
  if ready; then
    wait_idle
    sleep $W
    ready && [[ $(n_other_jobs) == 0 ]] && break
    continue
  fi
  if [[ $noted == 0 && $(n_zsh $ALNAME) == 0 ]]; then
    log "waiting: no $ALNAME.sh runs but its log has no end line and E005 has no QUEUE STOPPED: line, or $E5NAME.sh runs (the queue waits)"
    noted=1
  fi
  sleep $W
done
log "QUEUE E004-RESUME START (AL's last line: $(tail -1 $ALQ 2>/dev/null | cut -c1-160))"
[[ -n $R_SIM ]] && exit 0

grep -qE "${TS}tiny ladder: LR search only" $Q || stop "resume: no 'tiny ladder: LR search only' line in queue.txt; this queue runs LR searches only"

# ---------------- settings (queue_e004.sh / queue_e004_ts1m.sh) ----------------
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
GRID4="5e-05 3e-04 1e-03 3e-03"

lr_block() {  # short model ceiling   (tiny ladder D in LR-only mode: G4 grid, pick, one extension at most)
  local short=$1 mid=$2 ceil=$3 lr pick ext
  for lr in ${=GRID4}; do
    rjob lr_${short}_$lr $ceil lr${lr}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $lr
  done
  pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=GRID4} 2>>../logs/pick_stderr.txt)
  log "pick $short: ${pick:-pick failed} (grid $GRID4)"
  if [[ $pick == EXTEND* ]]; then
    ext=${pick#EXTEND }
    rjob lr_${short}_$ext $ceil lr${ext}_s0 $mid $COMMON --seed 0 --sets dev --save 0 --lr $ext
    pick=$($PY -B pick_lr_e004.py $mid --mode grid --lrs ${=GRID4} --ext $ext 2>>../logs/pick_stderr.txt)
    log "pick $short (final, after the extension): ${pick:-pick failed}"
  fi
  if [[ $pick != CHOSEN* ]]; then
    log "SKIP seeds $short: ${pick:-pick failed} (no usable LR)"
  else
    log "SKIP seeds $short: LR search only (would use ${pick#CHOSEN })"
  fi
  analyze
  log "$short block done (logs/tables.txt)"
}

# ---------------- D. rest of the tiny ladder (main queue order: ts3m, then p14m) ----------------
lr_block ts3m roneneldan/TinyStories-3M 3600
lr_block p14m EleutherAI/pythia-14m 3600

# ---------------- T. TinyStories-1M (queue_e004_ts1m.sh steps) ----------------
mid=roneneldan/TinyStories-1M
log "ts1m block start (TinyStories-1M re-added, queue_e004_ts1m.sh steps)"
if ! rjob dry_ts1m_r 900 dry $mid $COMMON --seed 0 --dry --sets eval,cont,know; then
  log "SKIP ts1m (rerun): dry run failed"
else
  rjob base_ts1m 2400 base $mid $COMMON --seed 0 --steps 0 --sets eval,cont,know
  analyze
  g=$($PY -B analyze_e004.py --gate-base $mid 2>/dev/null)
  log "untouched ts1m: ${g:-gate failed}"
  if [[ $g == UNTOUCHED_PASS ]]; then
    log "NO FINE-TUNE ts1m: the untouched model passes"
  else
    lr_block ts1m $mid 3600
  fi
fi

# ---------------- E. analysis ----------------
analyze
log "QUEUE E004-RESUME DONE"
