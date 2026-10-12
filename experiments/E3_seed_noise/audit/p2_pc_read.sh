#!/bin/bash
# E3 Part 2 audit, read-only, run on the PC by pcwsl.sh. Prints hashes and records of the six 20M runs, the Part 2
# queue files and the runs.jsonl events; writes nothing. Paths are printed relative to ~/planck/runs.
R=~/planck/runs; E=$R/E3
echo "LS_E3 $(ls $E | tr '\n' ' ')"
echo "MARKS $(ls $E/marks 2>/dev/null | tr '\n' ' ')"
echo "CODELISTS $(ls $E | grep code_sha256 | tr '\n' ' ')"
for f in queue_e3.txt status.jsonl code_sha256_20261003_004304.txt; do echo "SHA $f $(sha256sum $E/$f | cut -c1-64)"; done
echo "QUEUE_LINES $(wc -l < $E/queue_e3.txt)"
for d in $E/e3_20m_*; do
  n=$(basename $d)
  echo "DIR $n $(ls $d | tr '\n' ' ')"
  for f in bpb.jsonl log.jsonl latest.json; do echo "SHA $n/$f $(sha256sum $d/$f | cut -c1-64)"; done
  echo "SHA preflight/$n.json $(sha256sum $E/preflight/$n.json | cut -c1-64)"
  echo "LATEST $n $(tr -d '\n' < $d/latest.json | sed "s#$HOME#~#g")"
  echo "CKPT $n $(for c in $d/*.pt; do echo -n "$(basename $c):$(stat -c %s $c) "; done)"
  python3 - "$d/log.jsonl" "$n" <<'PY'
import json, math, sys
n = sys.argv[2]; steps = []; nonfinite = 0; other = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if 'loss' in r:
        steps.append(r['step'])
        if not math.isfinite(r['loss']): nonfinite += 1
        if r['step'] > 14953: print('TAIL', n, json.dumps(r, sort_keys=True))
    else:
        other += 1
print('LOGSTAT', n, len(steps), steps[0], steps[-1], 'grid_ok', steps == list(range(10, 15251, 10)) + [15259],
      'nonfinite', nonfinite, 'nonloss_records', other)
PY
done
python3 - "$R/runs.jsonl" <<'PY'
import json, sys, collections
c = collections.Counter()
for l in open(sys.argv[1]):
    r = json.loads(l)
    run = str(r.get('run', ''))
    if run.startswith('e3_'):
        c[(run[:6], r.get('event'))] += 1
    if run.startswith('e3_20m'):
        print('RUNS', json.dumps({k: r.get(k) for k in ('event', 'run', 'step', 'resumed_from', 'config_sha256', 'total_steps',
                                                       'prereg_sha256', 'n_params', 'schedule')}, sort_keys=True))
print('RUNS_E3_COUNTS', json.dumps(sorted((list(k), v) for k, v in c.items())))
PY
