#!/bin/zsh
# queue_e003_retry.sh: restarts queue_e003_resume.sh after a MEMORY stop (the guard's swap or free-memory kill),
# once free memory has stayed >= 50% for 5 minutes. Any other stop ends the wrapper. The resume queue skips every
# finished job, so a restart only re-runs the killed job. The guard and its thresholds are unchanged.
# Added 2026-09-26 01:40 by Claude (a sustained dip to 31% free while other work ran killed lr_ts1m_3e-04 at 00:57).
cd ${0:A:h} || exit 1
Q=../logs/queue.txt
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> $Q }
for i in {1..400}; do
  ok=0
  while (( ok < 10 )); do
    f=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
    if (( f >= 50 )); then ok=$((ok+1)); else ok=0; fi
    sleep 30
  done
  log "RETRY WRAPPER: start $i (free memory >= 50% for 5 min)"
  /bin/zsh queue_e003_resume.sh >> ../logs/queue_resume.out 2>&1
  last=$(tail -1 $Q)
  if [[ $last == *"QUEUE STOPPED: swap_grew"* || $last == *"QUEUE STOPPED: free_memory"* ]]; then
    log "RETRY WRAPPER: memory stop, will retry after memory recovers"
    continue
  fi
  log "RETRY WRAPPER: finished (last line: ${last:0:120})"; exit 0
done
log "RETRY WRAPPER: gave up after 400 memory stops"
