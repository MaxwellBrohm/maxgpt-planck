#!/bin/bash
# E006 queue test (notes.txt CODE TO WRITE, test_queue_e006.sh): runs queue_e006.sh in a temporary tree with a fake
# interpreter (fake_py_e006.py), a fake flock and stub scripts: no model, no GPU, nothing outside the temp tree.
# Scenarios: 1 full run (every GPU job under the lock and the guard, in the pre-registered order, ends DONE);
# 2 a restart skips every finished job; 3 a failed scored job gives a GAP line, its chat probe is not run, the queue
# goes on; 4 a temp verdict stops the queue and no later GPU job starts; 5 a step-0 check failure stops before any
# GPU job; 6 a failed dry run stops; 7 a busy lock is retried; 8 the retry limit stops; 9 a job that failed in an
# earlier run is not rerun (GAP). Exit 0 = every scenario passed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REAL_PY=$(command -v python3)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p $T/bin
cat > $T/bin/python3 <<EOF
#!/bin/bash
exec $REAL_PY $HERE/fake_py_e006.py "\$@"
EOF
cat > $T/bin/fakeflock <<'EOF'
#!/bin/bash
shift 3
name=$(for ((i=1; i<=$#; i++)); do [ "${!i}" == "--name" ] && j=$((i+1)) && echo ${!j}; done)
echo "LOCK $name" >> $E006T_CALLS
for kv in ${E006T_BUSY:-}; do
  if [ "${kv%%:*}" == "$name" ]; then
    f=$E006T_CALLS.busy_$name; n=$(cat $f 2>/dev/null || echo 0)
    [ $n -lt ${kv##*:} ] && { echo $((n + 1)) > $f; exit 1; }
  fi
done
exec "$@"
EOF
if ! command -v sha256sum >/dev/null; then printf '#!/bin/bash\nexec shasum -a 256 "$@"\n' > $T/bin/sha256sum; fi
chmod +x $T/bin/*
FAILS=0
EXPECT_GPU="dryC5 dryC6a dryC6b dryP dryG drychat base chat_base"
for s in 1 2 3 4 5; do EXPECT_GPU="$EXPECT_GPU e005w$s chat_e005w$s e004w$s"; done
for s in 1 2 3 4 5; do for a in C P G; do EXPECT_GPU="$EXPECT_GPU $a$s chat_$a$s"; done; done
EXPECT_GPU="$EXPECT_GPU C1t C2t"

setup() {  # a fresh tree; $1 = scenario name
  S=$T/$1; rm -rf $S; mkdir -p $S/root/experiments/E006_three_objects/code $S/run
  C=$S/root/experiments/E006_three_objects/code
  cp $HERE/queue_e006.sh $C/
  for f in replay_e006 identity_e006 eval_identity_e006 items_big_e006 validate_e006 test_train_e006p test_replay_e006 \
      test_rules_e006 test_analyze_e006 mutation_e006 compare_dry_e006 analyze_e006 guard_e006_pc e006_ft_test \
      chat_e006 numerics_e006 e005_ft_test; do echo "# stub" > $C/$f.py; done
  printf '#!/bin/bash\necho ok\n' > $C/test_queue_e006.sh
  mkdir -p $S/run/replay && touch $S/run/replay/replay_pool.jsonl
  export E006T_CALLS=$S/calls E006T_REAL_PY=$REAL_PY E006_PY=$T/bin/python3 E006_ROOT=$S/root E006_RUN=$S/run
  export E006_LOCK=$S/gpu.lock E006_FLOCK=$T/bin/fakeflock E006_LOCK_TRIES=3 E006_DATA=$S/data PATH=$T/bin:$PATH
  export E006T_FAIL="${2:-}" E006T_VERDICT="${3:-}" E006T_BUSY="${4:-}" E006T_REFUSE=""
  : > $S/calls
}
runq() { (cd $C && bash queue_e006.sh > $S/stdout 2>&1); }
qtail() { tail -1 $S/run/logs/queue.txt; }
gpus() { grep "^GUARD" $S/calls | awk '{print $2}' | tr '\n' ' ' | sed 's/ $//'; }
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; FAILS=$((FAILS + 1)); fi; }

setup full; runq
check "1 full run ends DONE" '[[ "$(qtail)" == *"QUEUE E006 DONE"* ]]'
check "1 GPU jobs in the pre-registered order" '[ "$(gpus)" == "$EXPECT_GPU" ]'
check "1 every GPU job took the lock first" '[ $(grep -c "^LOCK" $S/calls) == $(grep -c "^GUARD" $S/calls) ]'
check "1 step-0 checks ran before the first GPU job" \
  '[ $(grep -n "RUN validate_e006.py" $S/calls | cut -d: -f1) -lt $(grep -n "^GUARD" $S/calls | head -1 | cut -d: -f1) ]'
check "1 compare_dry ran after the dry runs, before base" \
  '[ $(grep -n "RUN compare_dry_e006.py" $S/calls | cut -d: -f1) -gt $(grep -n "GUARD drychat" $S/calls | cut -d: -f1) ]'
check "1 an analysis after every scored arm job and a final one" '[ $(grep -c "RUN analyze_e006.py" $S/calls) -ge 22 ]'
runq
check "2 a restart skips every finished job" '[ "$(grep -c "^GUARD" $S/calls)" == "$(echo $EXPECT_GPU | wc -w | tr -d " ")" ]'
check "2 the restart logs 'done earlier' and ends DONE" \
  '[ $(grep -c "done earlier" $S/run/logs/queue.txt) -ge 50 ] && [[ "$(qtail)" == *DONE* ]]'

setup gap "P3"; runq
check "3 a failed scored job gives a GAP line" 'grep -q "GAP: P3" $S/run/logs/queue.txt'
check "3 its chat probe is not run" '! grep -q "GUARD chat_P3" $S/calls && grep -q "GAP: no weights for P3" $S/run/logs/queue.txt'
check "3 the queue goes on to G3 and ends DONE" 'grep -q "GUARD G3" $S/calls && [[ "$(qtail)" == *DONE* ]]'

setup verdict "" "G2"; runq
check "4 a temp verdict stops the queue" '[[ "$(qtail)" == *"QUEUE STOPPED: temp verdict in G2"* ]]'
check "4 no GPU job after the verdict" '[ "$(gpus | awk "{print \$NF}")" == "G2" ]'

setup check0 "validate_e006"; runq
check "5 a step-0 failure stops before any GPU job" '[[ "$(qtail)" == *"QUEUE STOPPED: validate_e006 failed"* ]] && ! grep -q "^GUARD" $S/calls'

setup dry "dryP"; runq
check "6 a failed dry run stops" '[[ "$(qtail)" == *"QUEUE STOPPED: dryP failed"* ]] && [ "$(gpus | awk "{print \$NF}")" == "dryP" ]'

setup busy "" "" "base:2"; runq
check "7 a busy lock is retried and the job then runs" \
  '[ $(grep -c "base: not started" $S/run/logs/queue.txt) == 2 ] && grep -q "GUARD base" $S/calls && [[ "$(qtail)" == *DONE* ]]'

setup busy3 "" "" "e005w1:5"; runq
check "8 the retry limit stops the queue" '[[ "$(qtail)" == *"QUEUE STOPPED: e005w1 could not start after 3 tries"* ]]'

setup earlier; mkdir -p $S/run/logs; echo '{"exit": 1, "killed": null}' > $S/run/logs/C4.guard.json; runq
check "9 a job that failed earlier is not rerun" '! grep -q "GUARD C4" $S/calls && grep -q "GAP: C4 failed earlier" $S/run/logs/queue.txt'

echo "RESULT: $([ $FAILS == 0 ] && echo 'ALL PASS' || echo "$FAILS FAILED")"
[ $FAILS == 0 ]
