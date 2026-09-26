#!/bin/bash
# E006 queue on the PC (notes.txt PC QUEUE). Start it detached: a wrapper whose first line is
#   exec > ~/planck/logs/e006_queue.out 2>&1
# then runs this file with bash; pcdetach.sh starts the wrapper. Working copy ~/planck/dev/e006 (sync_e006.sh).
# Outputs are persistent in ~/planck/e006_run (out/ logs/ transcripts/ weights/ replay/, linked into E006's folder).
# Every GPU job: flock -w 7200 gpu.lock + guard_e006_pc.py (one model process at a time; temp, gpu_mem, host_mem, smi
# verdicts; wall ceiling). Logs are never overwritten: a job whose guard.json exists is not run again (exit 0: skipped
# as done; otherwise a GAP line; reruns are by hand, as <tag>_r2, logged as deviations). One line per job in
# logs/queue.txt; the last line is "QUEUE E006 DONE" or "QUEUE STOPPED: <reason>".
# Stop rules: a step-0 check, a dry-run or dry-comparison failure, or a temp/gpu_mem/host_mem/smi verdict stops the
# queue; a failed scored job gives a GAP line and the queue goes on (E004's rule).
set -u
PY=${E006_PY:-$HOME/planck/venv-hf/bin/python}
ROOT=${E006_ROOT:-$HOME/planck/dev/e006}
RUN=${E006_RUN:-$HOME/planck/e006_run}
LOCK=${E006_LOCK:-$HOME/planck/locks/gpu.lock}
FLOCK=${E006_FLOCK:-flock}
LOCK_TRIES=${E006_LOCK_TRIES:-12}
DATA=${E006_DATA:-$HOME/planck/data}
export HF_HOME=${HF_HOME:-$HOME/planck/hf} HF_HUB_OFFLINE=1
EXP=$ROOT/experiments/E006_three_objects
cd $EXP/code || exit 1
mkdir -p $RUN/out $RUN/logs $RUN/transcripts $RUN/weights $RUN/replay
for d in out logs transcripts weights replay; do [ -e $EXP/$d ] || ln -s $RUN/$d $EXP/$d; done
for e in E005_alias_eot:e005 E004_general_updating:e004; do
  w=$ROOT/experiments/${e%%:*}/weights; [ -e $w ] || ln -s $HOME/planck/dev/e006_refs_${e##*:} $w
done
RUNID=$(date '+%Y%m%d-%H%M%S')
Q=../logs/queue.txt
S0=../logs/step0_$RUNID
M=HuggingFaceTB/SmolLM2-135M-Instruct
S=HuggingFaceTB__SmolLM2-135M-Instruct
COMMON=(--bs 4 --accum 4 --max-len 768 --grad-ckpt)
LR=1.5e-04
log() { echo "$(date '+%F %T') $*" >> $Q; }
analyze() { $PY -B analyze_e006.py --trigger "$1" >> ../logs/analyze_stdout.txt 2>&1 || log "analyze_e006 failed (trigger $1)"; }
stop() { log "QUEUE STOPPED: $*"; analyze stopped; exit 2; }
gj() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2]))" "$1" "$2" 2>/dev/null; }

check_step() {  # name cmd...: a no-GPU check; any failure stops the queue
  local name=$1; shift
  "$@" > $S0/$name.stdout 2>&1
  local rc=$?
  log "check $name exit=$rc ($(tail -1 $S0/$name.stdout | cut -c1-200))"
  [ $rc -ne 0 ] && stop "$name failed (see logs/step0_$RUNID/$name.stdout)"
  return 0
}

gpu_job() {  # name ceiling [guard options] -- cmd...
  local name=$1 ceil=$2 rc tries=0 killed extra; shift 2
  if [ -e ../logs/$name.guard.json ]; then
    if [ "$(gj ../logs/$name.guard.json exit)" == 0 ] && [ "$(gj ../logs/$name.guard.json killed)" == None ]; then
      log "$name: done earlier (exit 0), skipped"; return 0
    fi
    log "GAP: $name failed earlier (see logs/$name.*); the queue does not rerun it"; return 1
  fi
  while :; do
    $FLOCK -w 7200 $LOCK $PY -B guard_e006_pc.py --name $name --ceiling $ceil "$@"
    rc=$?
    [ -e ../logs/$name.guard.json ] && break
    tries=$((tries + 1))
    log "$name: not started (exit $rc: gpu.lock busy for 7200 s or preflight wait over; try $tries of $LOCK_TRIES)"
    [ $tries -ge $LOCK_TRIES ] && stop "$name could not start after $tries tries"
  done
  killed=$(gj ../logs/$name.guard.json killed)
  extra=""
  grep -q "SELFTEST FAIL" ../logs/$name.log 2>/dev/null && extra="$extra SELFTEST_FAIL"
  grep -q "FATAL load check" ../logs/$name.log 2>/dev/null && extra="$extra FATAL_LOAD_CHECK"
  grep -q "loss self-check failed" ../logs/$name.log 2>/dev/null && extra="$extra LOSS_SELFCHECK_FAIL"
  log "$name exit=$rc killed=$killed$extra (elapsed_s $(gj ../logs/$name.guard.json elapsed_s), peak_gpu_mem_mib" \
      "$(gj ../logs/$name.guard.json peak_gpu_mem_mib), weights_sha256 $(gj ../logs/$name.guard.json weights_sha256))"
  case "$killed" in temp|gpu_mem|host_mem|smi) stop "$killed verdict in $name" ;; esac
  return $rc
}

ft() {  # name ceiling tag args...: one e006_ft_test.py job
  local name=$1 ceil=$2 tag=$3; shift 3
  gpu_job $name $ceil --sha-from ../out/${S}__${tag}__run.json -- $PY -B e006_ft_test.py $M "${COMMON[@]}" "$@" --tag $tag
}

chat() {  # name model tag
  gpu_job $1 3600 -- $PY -B chat_e006.py $2 --tag $3 || log "GAP: chat probe $3"
}

# ---------------- 0. no-GPU checks ----------------
mkdir -p $S0
log "QUEUE E006 START (run $RUNID)"
sha256sum *.py *.sh > ../logs/code_sha256_at_start_$RUNID.txt
log "code sha256 at start: $(wc -l < ../logs/code_sha256_at_start_$RUNID.txt) files, digest" \
    "$(sha256sum < ../logs/code_sha256_at_start_$RUNID.txt | cut -c1-16)"
if [ ! -e ../replay/replay_pool.jsonl ]; then
  check_step replay_build python3 -B replay_e006.py --tok-sample $DATA/tok_sample_v0c \
    --raw $DATA/raw/starter/OpenAssistant__oasst2/2023-11-05_oasst2_all.trees.jsonl.gz --out ../replay
fi
check_step identity_e006 $PY -B identity_e006.py
check_step eval_identity_e006 python3 -B eval_identity_e006.py
check_step items_big_check python3 -B items_big_e006.py --check
check_step validate_e006 $PY -B validate_e006.py
check_step test_train_e006p python3 -B test_train_e006p.py
check_step test_replay_e006 $PY -B test_replay_e006.py
check_step test_rules_e006 $PY -B test_rules_e006.py
check_step test_analyze_e006 $PY -B test_analyze_e006.py
check_step mutation_e006 $PY -B mutation_e006.py
check_step test_queue_e006 bash test_queue_e006.sh

# ---------------- 1. dry runs ----------------
gpu_job dryC5 1800 -- $PY -B numerics_e006.py e005_ft_test.py $M "${COMMON[@]}" --seed 0 --dry --device cuda \
  --sets eval,cont,know --chat --tag dryC5 || stop "dryC5 failed"
for t in dryC6a dryC6b dryP dryG; do
  arm=${t:3:1}
  ft $t 1800 $t --arm $arm --seed 0 --dry --chat || stop "$t failed"
done
gpu_job drychat 900 -- $PY -B chat_e006.py $M --tag drychat --only R_d2_name || stop "drychat failed"
check_step compare_dry $PY -B compare_dry_e006.py

# ---------------- 2. untouched model; 3. references ----------------
ft base 9000 base --seed 0 --steps 0 --chat || log "GAP: base"
chat chat_base $M base
analyze after_base
for s in 1 2 3 4 5; do
  w5=../../E005_alias_eot/weights/${S}__s$s
  w4=../../E004_general_updating/weights/${S}__s$s
  ft e005w$s 9000 e005w$s --seed $s --steps 0 --weights $w5 --chat || log "GAP: e005w$s"
  chat chat_e005w$s $w5 e005w$s
  ft e004w$s 9000 e004w$s --seed $s --steps 0 --weights $w4 --sets eval,big,h5l,al --chat || log "GAP: e004w$s"
  analyze after_refs_s$s
done

# ---------------- 4. the three arms, seed by seed ----------------
for s in 1 2 3 4 5; do
  for arm in C P G; do
    ft $arm$s 14400 $arm$s --arm $arm --seed $s --lr $LR --chat --save 1 || log "GAP: $arm$s"
    if [ -d ../weights/${S}__$arm$s ]; then chat chat_$arm$s ../weights/${S}__$arm$s $arm$s
    else log "GAP: no weights for $arm$s, chat probe not run"; fi
    analyze after_$arm$s
  done
done

# ---------------- 5. numerics twins (Q-device noise floor) ----------------
for s in 1 2; do
  ft C${s}t 14400 C${s}t --arm C --seed $s --lr $LR --tf32 --sets eval,big,h5l --chat --save 1 || log "GAP: C${s}t"
done

# ---------------- 6. final reading ----------------
analyze final
log "readings: $(tail -1 ../logs/readings/index.txt 2>/dev/null | cut -c1-300)"
log "QUEUE E006 DONE"
