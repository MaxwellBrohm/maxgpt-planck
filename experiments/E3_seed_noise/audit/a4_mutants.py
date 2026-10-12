# E3 audit: mutation test of the audit's own checks (a1_recompute.py, a2_checks.py, a3_pc_compare.py).
# Each mutant copies the raw records to a scratch mirror, breaks one thing, and runs the audit scripts on the
# mirror; the mutant is KILLED if the relevant check reports the break. The unmutated mirror must pass.
# Run: ~/.venvs/planck/bin/python audit/a4_mutants.py SCRATCH_DIR (from the E3 folder). Writes only in SCRATCH_DIR.
import json, os, shutil, subprocess, sys
E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.abspath(sys.argv[1])
PY = sys.executable

def mirror():
    root = os.path.join(SCR, 'mut')
    shutil.rmtree(root, ignore_errors=True)
    m = os.path.join(root, 'E3')
    for d in ('out', 'configs', 'logs'):
        shutil.copytree(os.path.join(E3, d), os.path.join(m, d))
    os.makedirs(os.path.join(m, 'audit'))
    for f in ('a1_recompute.py', 'a2_checks.py', 'a3_pc_compare.py', 'pc_tail_losses.out'):
        shutil.copy(os.path.join(E3, 'audit', f), os.path.join(m, 'audit', f))
    os.makedirs(os.path.join(root, 'E2_lr_transfer'))
    shutil.copy(os.path.join(E3, '..', 'E2_lr_transfer', 'results.json'), os.path.join(root, 'E2_lr_transfer'))
    return m

def run(m, script):
    p = subprocess.run([PY, os.path.join(m, 'audit', script)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr

def edit_jsonl(path, fn):
    rows = [json.loads(l) for l in open(path)]
    rows = [fn(i, r) for i, r in enumerate(rows)]
    open(path, 'w').write(''.join(json.dumps(r) + '\n' for r in rows))

def edit_json(path, fn):
    o = json.load(open(path)); fn(o); json.dump(o, open(path, 'w'))

def a1_crn(m):
    rc, out = run(m, 'a1_recompute.py')
    if rc: return 'crash', out[-200:]
    return json.load(open(os.path.join(m, 'audit', 'a1_out.json')))['crn'], ''

MUTANTS = []
def mutant(f): MUTANTS.append(f); return f

@mutant
def crn_pair_drawn(m):  # B_s3 draws one more oasst2 token at step 7550
    edit_jsonl(os.path.join(m, 'out/e3_5m_B_s3/train_tail.jsonl'),
               lambda i, r: {**r, 'drawn_oasst2': r['drawn_oasst2'] + (1 if r['step'] == 7550 else 0)})
    c, _ = a1_crn(m); return c != 'crash' and c['pairs_equal']['3'] is False

@mutant
def crn_replay_end(m):  # R2's end record draws a different cccc count
    edit_json(os.path.join(m, 'out/e3_5m_A_s1_R2/run_records.json'), lambda o: o[1].update(drawn_cccc=o[1]['drawn_cccc'] + 7))
    c, _ = a1_crn(m); return c != 'crash' and c['replays_equal_A1'][0] is False

@mutant
def crn_seed_collision(m):  # seed 6's stream equal to seed 5's at the end
    s5 = json.load(open(os.path.join(m, 'out/e3_5m_A_s5/run_records.json')))[1]
    edit_json(os.path.join(m, 'out/e3_5m_A_s6/run_records.json'),
              lambda o: o[1].update({k: v for k, v in s5.items() if k.startswith(('drawn_', 'dropped_', 'tokens', 'sup_'))}))
    c, _ = a1_crn(m); return c != 'crash' and c['distinct_seed_streams'] == 7

@mutant
def pooled_prose_row(m):  # the stored PROSE row of A_s2 is not the pool of its sources
    edit_jsonl(os.path.join(m, 'out/e3_5m_A_s2/bpb.jsonl'),
               lambda i, r: {**r, 'bits': r['bits'] * 1.0001, 'bpb': r['bpb'] * 1.0001} if (r['set'], r['step']) == ('PROSE', 7630) else r)
    rc, out = run(m, 'a1_recompute.py')
    return rc == 0 and 'max |diff|: 0.0\n' not in out

@mutant
def wrong_run_in_folder(m):  # A_s5's folder holds B_s5's scores
    shutil.copy(os.path.join(m, 'out/e3_5m_B_s5/bpb.jsonl'), os.path.join(m, 'out/e3_5m_A_s5/bpb.jsonl'))
    rc, out = run(m, 'a1_recompute.py'); rc2, out2 = run(m, 'a2_checks.py')
    return rc != 0 and 'FAIL e3_5m_A_s5 bpb records' in out2

@mutant
def config_lr_swapped(m):  # B_s2's config says 3e-3
    p = os.path.join(m, 'configs/e3_5m_B_s2.yaml')
    t = open(p).read().replace('0.0015', '0.003'); open(p, 'w').write(t)
    rc, out = run(m, 'a2_checks.py')
    return 'FAIL e3_5m_B_s2 config lr' in out and 'FAIL e3_5m_B_s2 preflight' in out

@mutant
def engine_compile_on(m):
    edit_json(os.path.join(m, 'out/e3_5m_A_s4/preflight.json'), lambda o: o['runs'][0]['engine'].update({'train.compile': True}))
    rc, out = run(m, 'a2_checks.py'); return 'FAIL e3_5m_A_s4 preflight' in out

@mutant
def resumed_run(m):
    edit_json(os.path.join(m, 'out/e3_5m_B_s7/run_records.json'), lambda o: o[0].update(resumed_from='x/ckpt_00003060.pt'))
    rc, out = run(m, 'a2_checks.py'); return 'FAIL e3_5m_B_s7 run start/end' in out

@mutant
def evalset_changed(m):
    edit_jsonl(os.path.join(m, 'out/e3_5m_A_s8/bpb.jsonl'), lambda i, r: {**r, 'evalset_sha256': '0' * 64} if i == 5 else r)
    rc, out = run(m, 'a2_checks.py'); return 'FAIL e3_5m_A_s8 bpb records' in out

@mutant
def max_windows_set(m):
    edit_jsonl(os.path.join(m, 'out/e3_5m_B_s1/bpb.jsonl'), lambda i, r: {**r, 'max_windows': 500} if i == 0 else r)
    rc, out = run(m, 'a2_checks.py'); return 'FAIL e3_5m_B_s1 bpb records' in out

@mutant
def prereg_sha(m):
    edit_json(os.path.join(m, 'out/e3_5m_A_s3/run_records.json'), lambda o: o[0].update(prereg_sha256='f' * 64))
    rc, out = run(m, 'a2_checks.py'); return 'FAIL e3_5m_A_s3 run start/end' in out

@mutant
def tail_loss_copied_wrong(m):  # the copied tail differs from the PC log in one loss
    edit_jsonl(os.path.join(m, 'out/e3_5m_A_s6/train_tail.jsonl'), lambda i, r: {**r, 'loss': r['loss'] + 1e-5} if i == 3 else r)
    rc, out = run(m, 'a3_pc_compare.py'); return 'BAD e3_5m_A_s6' in out

@mutant
def tail_step_missing(m):  # one logged tail step missing
    p = os.path.join(m, 'out/e3_5m_B_s4/train_tail.jsonl')
    t = ''.join(open(p).readlines()[1:]); open(p, 'w').write(t)
    rc, out = run(m, 'a1_recompute.py'); return rc != 0 and 'AssertionError' in out and '7490' in out

m = mirror()
rc1, o1 = run(m, 'a1_recompute.py'); rc2, o2 = run(m, 'a2_checks.py'); rc3, o3 = run(m, 'a3_pc_compare.py')
base_ok = rc1 == 0 and 'FAILS: []' in o2 and 'mismatches 0' in o3
c = json.load(open(os.path.join(m, 'audit', 'a1_out.json')))['crn']
base_ok = base_ok and all(c['pairs_equal'].values()) and all(c['replays_equal_A1']) and c['distinct_seed_streams'] == 8
print('unmutated mirror passes every check:', base_ok)
killed = 0
for f in MUTANTS:
    m = mirror()
    k = bool(f(m)); killed += k
    print(('KILLED  ' if k else 'SURVIVED') + ' ' + f.__name__)
print(f'{killed}/{len(MUTANTS)} killed')
