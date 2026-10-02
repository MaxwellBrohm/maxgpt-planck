#!/bin/bash
# read-only, run on the PC by pcwsl.sh: print step and loss of every logged step > 7477 for each E3 run and
# the E2 seed-1 pick branch, plus the sha256 of each bpb.jsonl. Prints nothing else; writes nothing.
for d in ~/planck/runs/E3/e3_5m_* ~/planck/runs/E2/5m_e3_r1_b250M; do
  n=$(basename $d)
  echo "BPB $n $(sha256sum $d/bpb.jsonl | cut -c1-64)"
  grep -o '"loss": [0-9.]*\|"step": [0-9]*' $d/log.jsonl | paste -d' ' - - 2>/dev/null | awk -v n=$n '{print}' >/dev/null
  python3 - "$d/log.jsonl" "$n" <<'PY'
import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get('step',0)>7477 and 'loss' in r:
        print('LOSS',sys.argv[2],r['step'],r['loss'],r.get('drawn_oasst2'),r.get('drawn_cccc'))
PY
done
