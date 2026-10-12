#!/bin/bash
echo "$*" >> "$E3W_STUBLOG"
Q=$(dirname "$(dirname "$(dirname "$0")")")
LOG=$E3W_HOME/runs/E3/queue_e3.txt
o=$(head -1 "$E3W_OUTCOMES"); sed -i.bak 1d "$E3W_OUTCOMES"
t() { echo "$(date '+%F %T') $*" >> "$LOG"; }
[ -n "$E3W_SLEEP" ] && sleep "$E3W_SLEEP"
[ "$o" = refuse ] && { t "refusing: uncommitted changes in the code checkout:"; exit 1; }
sha=$(shasum -a 256 "$Q/experiments/E3_seed_noise/plans/part3.txt" | cut -c1-16)
[ "$o" = othersha ] && sha=0123456789abcdef
t "queue start: exp E3, plan part3.txt (sha256 $sha), code commit $(git -C "$Q" rev-parse HEAD)"
t "seen E3: E3 PART 2 DONE"
case $o in
  clean|othersha) t "MARK E3 PART 3 DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_DONE"; t "plan finished: part3.txt"; exit 0 ;;
  nomark) t "MARK E3 PART 3 DONE"; t "plan finished: part3.txt"; exit 0 ;;
  lockwait) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_init2"; exit 1 ;;
  memwait) t "GPU memory in use [8328] MiB > 3000 with the lock held: waiting"; t "queue stops at: train e3_5m_A_s1_P3"; exit 1 ;;
  fail) t "scoring failed for final_00007630 (see score.err)"; t "queue stops at: train e3_5m_A_init2"; exit 1 ;;
  kill) exit 137 ;;
  clean_rc1) t "MARK E3 PART 3 DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_DONE"; t "plan finished: part3.txt"; exit 1 ;;
  lockwait_rc0) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_init2"; exit 0 ;;
esac
