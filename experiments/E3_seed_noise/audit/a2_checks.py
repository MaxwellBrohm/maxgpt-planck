# E3 audit: provenance checks from the raw records, written independently of e3analyze.py's check list.
# Run: ~/.venvs/planck/bin/python audit/a2_checks.py (from the E3 folder). Prints PASS/FAIL lines.
import json, os, hashlib, re
E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(E3, 'out')
FIX = dict(tok='078b24c4b0755d81985ebc122e912d7c70ff204d335519721235a405756fcfb5',
           evalset='fe1b55bbe3612d0c968934cbad831a17fd3d73d84aa8f1b7cb2f4523cf4d505b',
           prereg=hashlib.sha256(open(os.path.join(E3, 'configs/prereg.yaml'), 'rb').read()).hexdigest(),
           shares={'cccc': 0.414241, 'stackexchange': 0.221922, 'gutenberg': 0.147915, 'wikimedia': 0.147915,
                   'irc': 0.040004, 'dolly': 0.016002, 'oasst2': 0.012001},
           n_files={'cccc': 31, 'stackexchange': 11, 'gutenberg': 6, 'irc': 6, 'wikimedia': 3, 'dolly': 1, 'oasst2': 1},
           engine={'train.compile': False, 'train.doc_attn': 'varlen', 'train.ce_chunk_rows': 0, 'optim.batched': True,
                   'train.lazy_metrics': False, 'train.cuda_mem_cap_gib': None, 'train.auto_micro_on_oom': False,
                   'train.precision': 'bf16', 'data.mode': 'pack', 'data.window_tokens': 131072,
                   'data.max_item_len': 2048})
runs = [f'e3_5m_{a}_s{s}' for s in range(1, 9) for a in 'AB'] + ['e3_5m_A_s1_R2', 'e3_5m_A_s1_R3']
codelist = {l.split()[1]: l.split()[0] for l in open(os.path.join(E3, 'logs/code_sha256_20260928_162455.txt')) if l.strip()}
fails = []
def check(name, ok, info=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not ok: fails.append(name)

print('prereg.yaml sha256', FIX['prereg'][:16])
truncs = {}
for r in runs:
    d = os.path.join(OUT, r)
    cfg = os.path.join(E3, 'configs', r + '.yaml')
    fsha = hashlib.sha256(open(cfg, 'rb').read()).hexdigest()
    pf = json.load(open(os.path.join(d, 'preflight.json')))
    rr = json.load(open(os.path.join(d, 'run_records.json')))
    st, en = rr[0], rr[1]
    p = pf['runs'][0]
    ok = (pf['ok'] and pf['strict'] and p['ok'] and not p['refusals'] and p['name'] == r and p['file_sha256'] == fsha
          and codelist.get(f'experiments/E3_seed_noise/configs/{r}.yaml') == fsha
          and p['config_sha256'] == st['config_sha256'] and p['engine'] == {**p['engine'], **FIX['engine']}
          and all(p['engine'][k] == v for k, v in FIX['engine'].items()) and p['engine_fixed'] == '2026-09-26 93ea41c'
          and p['shares'] == FIX['shares'] and p['n_files'] == FIX['n_files'] and p['n_params'] == 5010133
          and p['prereg']['sha256'] == FIX['prereg'] and p['prereg']['committed'])
    check(f'{r} preflight/config/engine/shares/prereg', ok)
    ok = (st['event'] == 'start' and st['run'] == r and st['resumed_from'] is None and st['step'] == 0
          and st['prereg_id'] == 'E3' and st['prereg_sha256'] == FIX['prereg'] and st['precision'] == 'bf16'
          and st['doc_attn'] == 'varlen' and st['optim_batched'] and st['n_params'] == 5010133
          and st['schedule'] == {'decay_start': 6104, 'mode': 'full', 'total_steps': 7630, 'warmup_steps': 76}
          and en['event'] == 'end' and en['step'] == 7630 and en['total_steps'] == 7630 and en['run'] == r
          and 'compile' not in st and 'ce_chunk_rows' not in st)
    check(f'{r} run start/end records', ok, f"loss_end {en['loss']:.4f} s {en['seconds']}")
    recs = [json.loads(l) for l in open(os.path.join(d, 'bpb.jsonl'))]
    ok = (len(recs) == 39 and all(x['run'] == r and x['tokenizer_sha256'] == FIX['tok'] and x['evalset_sha256'] == FIX['evalset']
                                  and x['precision'] == 'fp32' and x['max_windows'] is None for x in recs)
          and {x['step'] for x in recs} == {7344, 7497, 7630}
          and all(x['ckpt'] == ('final_00007630.pt' if x['step'] == 7630 else f"ckpt_{x['step']:08d}.pt") for x in recs))
    check(f'{r} bpb records (tok, evalset, fp32, all windows, 3 ckpts)', ok)
    for x in recs:
        if x['step'] == 7630 and 'truncated' in x:
            truncs.setdefault(r, {})[x['set']] = x['truncated']
    # windows and bytes per set must be identical across runs (same eval set)
    sig = sorted((x['set'], x['split'], x['bytes'], x['windows']) for x in recs if x['step'] == 7630)
    truncs.setdefault('_sig', set()).add(json.dumps(sig))
check('every run scored the same windows and bytes per set', len(truncs.pop('_sig')) == 1)
print('truncated windows per set (final):', {k: v for k, v in list(truncs.items())[:1]},
      'identical across runs:', len({json.dumps(v, sort_keys=True) for v in truncs.values()}) == 1)
# E2 extra pair: identify the run and its harness
e2 = json.load(open(os.path.join(OUT, 'e2_5m_e3_r1_b250M', 'run_records.json')))
print('E2 pair start:', {k: e2[0][k] for k in ('run', 'prereg_id', 'resumed_from', 'schedule', 'step')})
# the E2 re-check uses E2's own seed-1 values: stage A/C, 250M
e2r = json.load(open(os.path.join(E3, '..', 'E2_lr_transfer', 'results.json')))
fb = e2r['stage_A']['final_bpb']
print('E2 250M F(CHAT): 1.5e-3', fb['0.0015']['b250M']['chat'], '3e-3', fb['0.003']['b250M']['chat'], '6e-3', fb['0.006']['b250M']['chat'],
      '-> runner-up', 'B=1.5e-3' if fb['0.0015']['b250M']['chat'] < fb['0.006']['b250M']['chat'] else 'B=6e-3',
      'stage_C lrs', e2r['stage_C']['lrs'])
# configs: A arms 3e-3, B arms 1.5e-3, seed = s
for r in runs:
    t = open(os.path.join(E3, 'configs', r + '.yaml')).read()
    lr = re.search(r'optim: \{lr: ([0-9.e-]+), embed_lr: ([0-9.e-]+), scalar_lr: ([0-9.e-]+)\}', t).groups()
    seed = int(re.search(r'^seed: (\d+)', t, re.M).group(1))
    want_lr = '0.003' if '_A_' in r else '0.0015'
    want_seed = int(re.search(r'_s(\d)', r).group(1))
    check(f'{r} config lr {lr[0]} seed {seed}', set(lr) == {want_lr} and seed == want_seed and 'total_steps: 7630' in t
          and 'ckpt_every: 153' in t and 'mode: full' in t)
print('FAILS:', fails)
