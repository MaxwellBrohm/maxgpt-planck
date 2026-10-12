#!/bin/bash
# run_core_v0.sh: the strict-open core v0 job on the PC, tier by tier in the plan's order
# (1 small human sets and StackExchange dumps, 2 Gutenberg, 3 YouTube, 4 LoC, 5 CCCC):
#   pass A  stream_core.py: download (resume, checksum), gates, exact dedup, MinHash sidecars,
#           stage1 shards, raw file deleted once committed (LEDGER.jsonl is the record);
#   seal    stream_core.py --seal-open for the tier's sources (a file that failed for good);
#   B-D     core_finalize.py: near-dedup across this and earlier tiers, boilerplate, final
#           shards, exact dedup of the cleaned text (FINAL/_tiers/tier<K>.json is the record).
# While a tier runs, core_prefetch.py downloads the later tiers' files (no lock; up to $PF_GB GB
# on disk), and it is stopped before the next stream_core.py run so one file never has two
# downloaders. Resumable: run it again; done files and up-to-date tiers are skipped. Stop cleanly: touch
# $S1/STOP. Heavy work takes ~/planck/locks/pc_heavy.lock per chunk (nice 15); a lock timeout
# (exit 4) or a code change under the run (exit 6, a synced fix) queues the step again.
set -u
PY=${PY:-$HOME/planck/venv/bin/python}
CODE=$(cd "$(dirname "$0")" && pwd)
S1=${S1:-$HOME/planck/data/core_v0_work/stage1}
RAW=${RAW:-$HOME/planck/data/raw/core_v0}
FIN=${FIN:-$HOME/planck/data/core_v0}
STARTER=${STARTER:-$HOME/planck/data/raw/starter}
TIERS=${TIERS:-"1 2 3 4 5"}
WORKERS=${WORKERS:-16}
PF_GB=${PF_GB:-100}
PF_LOG=${PF_LOG:-$HOME/planck/logs/core_v0_prefetch.log}
MIRROR="https://archive.org/download/=https://ia601509.us.archive.org/14/items/"
COMMON=(--out "$S1" --raw "$RAW" --starter "$STARTER" --sidecar mh=stream_work:mh_row
        --allow-unrecorded-license cccc --url-rewrite "$MIRROR")
declare -A SRC=([1]="dolly foodista irc news oasst2 oercommons pdr pressbooks stackexchange wikimedia"
                [2]="gutenberg" [3]="youtube" [4]="loc" [5]="cccc")
declare -A DL=([1]=6 [2]=4 [3]=4 [4]=4 [5]=4)
say() { echo "$(date '+%F %T') run_core_v0: $*"; }
stop_prefetch() {
  pkill -TERM -f "core_prefetch.py" 2>/dev/null
  for _ in $(seq 90); do pgrep -f "core_prefetch.py" >/dev/null || break; sleep 1; done
  pkill -KILL -f "core_prefetch.py" 2>/dev/null
  pkill -TERM -f "curl .*$RAW/" 2>/dev/null     # no curl of a killed prefetcher may outlive it
  sleep 2
}
start_prefetch() {                              # the tiers after $1
  local later=() x
  for x in $TIERS; do [ "$x" -gt "$1" ] && later+=("$x"); done
  [ ${#later[@]} -eq 0 ] && return 0
  "$PY" "$CODE/core_prefetch.py" --out "$S1" --raw "$RAW" --starter "$STARTER" \
    --tiers "${later[@]}" --window-gb "$PF_GB" --dl-threads 4 --url-rewrite "$MIRROR" >> "$PF_LOG" 2>&1 &
  say "prefetch of tiers ${later[*]} started (pid $!)"
}
trap 'stop_prefetch' EXIT
stopped() { [ -e "$S1/STOP" ] && { say "STOP file, exiting"; exit 5; }; return 0; }
say "start, code $CODE, tiers $TIERS"
for t in $TIERS; do
  tries=0
  while :; do
    stopped
    say "tier $t pass A (failed-file retries so far: $tries)"
    stop_prefetch
    start_prefetch "$t"
    "$PY" "$CODE/stream_core.py" "${COMMON[@]}" --tiers "$t" --workers "$WORKERS" \
      --dl-threads "${DL[$t]}"
    rc=$?
    say "tier $t stream_core exit $rc"
    case $rc in
      0) break ;;
      2) tries=$((tries + 1)); [ "$tries" -ge 3 ] && break; sleep 300 ;;
      4) sleep 30 ;;
      6) sleep 90 ;;
      *) say "stopping on stream_core exit $rc"; exit "$rc" ;;
    esac
  done
  "$PY" "$CODE/stream_core.py" "${COMMON[@]}" --seal-open --sources ${SRC[$t]}
  while :; do
    stopped
    "$PY" "$CODE/core_finalize.py" --stage1 "$S1" --final "$FIN" --tier "$t" --workers "$WORKERS"
    rc=$?
    say "tier $t core_finalize exit $rc"
    case $rc in
      0) break ;;
      4) sleep 30 ;;
      *) say "stopping on core_finalize exit $rc"; exit "$rc" ;;
    esac
  done
  "$PY" "$CODE/core_status.py" --stage1 "$S1" --final "$FIN" | head -40
done
say "all tiers done"
