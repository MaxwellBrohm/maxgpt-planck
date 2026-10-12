# E3 audit: the copied out/<run>/train_tail.jsonl and bpb.jsonl against the PC originals (audit/pc_tail_losses.out,
# printed on the PC by audit/pc_tail_losses.sh, read-only). Run with Planck's venv from the E3 folder.
import json, os, hashlib
E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pc = {}; bpb = {}
for l in open(os.path.join(E3, 'audit', 'pc_tail_losses.out')):
    p = l.split()
    if p[0] == 'LOSS': pc.setdefault(p[1], {})[int(p[2])] = (float(p[3]), int(p[4]), int(p[5]))
    if p[0] == 'BPB': bpb[p[1]] = p[2]
bad = 0
for run, rows in pc.items():
    d = 'e2_' + run if run.startswith('5m_') else run
    tail = {r['step']: (r['loss'], r['drawn_oasst2'], r['drawn_cccc']) for r in map(json.loads, open(os.path.join(E3, 'out', d, 'train_tail.jsonl')))}
    h = hashlib.sha256(open(os.path.join(E3, 'out', d, 'bpb.jsonl'), 'rb').read()).hexdigest()
    ok = tail == rows and h == bpb[run]
    bad += not ok
    print(('OK  ' if ok else 'BAD ') + run, len(rows), 'steps', min(rows), '..', max(rows), 'bpb sha', h[:12])
print('runs compared', len(pc), 'mismatches', bad)
