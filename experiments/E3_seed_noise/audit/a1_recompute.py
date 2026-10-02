# E3 audit, independent recomputation from the raw records in out/ (not e3analyze.py, not results.json values).
# Quantiles: my own numeric integration of the t and chi-square densities (not the analyzer's incomplete beta);
# a1_scipy_quantiles.py cross-checks them with scipy on the system python.
# Run: ~/.venvs/planck/bin/python audit/a1_recompute.py  (from the E3 folder). Writes audit/a1_out.json.
import json, math, os, itertools, sys
E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(E3, 'out')

def tpdf(x, n):
    c = math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2)) / math.sqrt(n * math.pi)
    return c * (1 + x * x / n) ** (-(n + 1) / 2)

def simpson(f, a, b, m=20000):
    h = (b - a) / m
    s = f(a) + f(b)
    for i in range(1, m):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3

def tcdf(x, n):  # P(T <= x), x >= 0 via 0.5 + integral(0..x)
    if x < 0:
        return 1 - tcdf(-x, n)
    return 0.5 + simpson(lambda u: tpdf(u, n), 0.0, x)

def bisect(f, lo, hi, target, it=80):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def tq(p, n): return bisect(lambda x: tcdf(x, n), 0.0, 50.0, p)

def chi2pdf(x, k):
    if x <= 0: return 0.0
    return math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))

def chi2cdf(x, k): return simpson(lambda u: chi2pdf(u, k), 1e-12, x)

def chi2q(p, k): return bisect(lambda x: chi2cdf(x, k), 1e-9, 60.0, p)

def p2(t, n): return 2 * (1 - tcdf(abs(t), n))

def mean(v): return sum(v) / len(v)

def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

def corr(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))

# ---------- raw reading ----------
def load(run):
    d = os.path.join(OUT, run)
    recs = [json.loads(l) for l in open(os.path.join(d, 'bpb.jsonl'))]
    tail = [json.loads(l) for l in open(os.path.join(d, 'train_tail.jsonl'))]
    rr = json.load(open(os.path.join(d, 'run_records.json')))
    return recs, tail, rr

def bpb_at(recs, step, sets, split='all'):
    rows = [r for r in recs if r['step'] == step and r['set'] in sets and r['split'] == split]
    assert len(rows) == len(sets), (step, sets, split, len(rows))
    for r in rows:  # stored bpb must equal bits/bytes
        assert abs(r['bpb'] - r['bits'] / r['bytes']) < 1e-12
    return sum(r['bits'] for r in rows) / sum(r['bytes'] for r in rows)

PROSE = ['cccc', 'gutenberg', 'wikimedia']
CKS = [7344, 7497, 7630]

def metrics(run):
    recs, tail, rr = load(run)
    assert {r['run'] for r in recs} == {run.replace('e2_', '')}, run
    fin = [r for r in recs if r['step'] == 7630]
    assert all(r['ckpt'] == 'final_00007630.pt' for r in fin) and len(fin) == 13
    m = {}
    m['F(CHAT)'] = bpb_at(recs, 7630, ['oasst2'])
    m['F(PROSE)'] = bpb_at(recs, 7630, PROSE)
    # the stored pooled rows must equal my pooling
    m['_chk_CHAT_row'] = bpb_at(recs, 7630, ['CHAT']) - m['F(CHAT)']
    m['_chk_PROSE_row'] = bpb_at(recs, 7630, ['PROSE']) - m['F(PROSE)']
    m['M(CHAT)'] = mean([bpb_at(recs, s, ['oasst2']) for s in CKS])
    m['M(PROSE)'] = mean([bpb_at(recs, s, PROSE) for s in CKS])
    m['CHAT user'] = bpb_at(recs, 7630, ['oasst2'], 'user')
    m['CHAT assistant'] = bpb_at(recs, 7630, ['oasst2'], 'assistant')
    for s in PROSE: m['PROSE ' + s] = bpb_at(recs, 7630, [s])
    for s in ['stackexchange', 'irc', 'dolly']: m[s] = bpb_at(recs, 7630, [s])
    steps = sorted(r['step'] for r in tail)
    assert steps == list(range(7480, 7631, 10)), steps
    m['train loss last 2%'] = mean([r['loss'] for r in tail if r['step'] > 7630 - 0.02 * 7630])
    m['_tail'] = {r['step']: r for r in tail}
    m['_end'] = [x for x in rr if x['event'] == 'end'][0]
    m['_start'] = [x for x in rr if x['event'] == 'start'][0]
    return m

A = {s: metrics(f'e3_5m_A_s{s}') for s in range(1, 9)}
B = {s: metrics(f'e3_5m_B_s{s}') for s in range(1, 9)}
R = [A[1], metrics('e3_5m_A_s1_R2'), metrics('e3_5m_A_s1_R3')]
E2 = metrics('e2_5m_e3_r1_b250M')
NAMES = ['F(CHAT)', 'F(PROSE)', 'M(CHAT)', 'M(PROSE)', 'CHAT user', 'CHAT assistant', 'PROSE cccc',
         'PROSE gutenberg', 'PROSE wikimedia', 'stackexchange', 'irc', 'dolly', 'train loss last 2%']
out = {'pooled_row_max_abs_diff': max(abs(x[k]) for x in list(A.values()) + list(B.values()) + R[1:] + [E2]
                                      for k in ('_chk_CHAT_row', '_chk_PROSE_row'))}

# ---------- quantiles ----------
Q = {'t975_7': tq(.975, 7), 't80_7': tq(.80, 7), 't975_6': tq(.975, 6), 't80_6': tq(.80, 6),
     't975_2': tq(.975, 2), 't80_2': tq(.80, 2)}
Q['factor7'] = Q['t975_7'] + Q['t80_7']; Q['factor6'] = Q['t975_6'] + Q['t80_6']; Q['factor2'] = Q['t975_2'] + Q['t80_2']
Q['ub80_7'] = math.sqrt(7 / chi2q(0.2, 7)); Q['ub80_2'] = math.sqrt(2 / chi2q(0.2, 2))
Q['chi2_95_2_sdbound'] = math.sqrt(chi2q(0.95, 2) / 2)
out['quantiles'] = Q

# ---------- per-metric statistics ----------
def stats(name, seeds):
    a = [A[s][name] for s in seeds]; b = [B[s][name] for s in seeds]
    d = [y - x for x, y in zip(a, b)]
    n = len(d); df = n - 1
    sdd = sd(d); md = mean(d)
    sseed = math.sqrt((sd(a) ** 2 + sd(b) ** 2) / 2)
    t = md / (sdd / math.sqrt(n))
    t975 = tq(.975, df)
    o = dict(n=n, mean_a=mean(a), mean_d=md, SD_d=sdd, SD_d_rel=sdd / mean(a), sigma_seed=sseed,
             sigma_seed_rel=sseed / mean(a), sd_A_only=sd(a), rho=corr(a, b), t=t, p=p2(t, df),
             ci95=[md - t975 * sdd / math.sqrt(n), md + t975 * sdd / math.sqrt(n)],
             crn_ratio=math.sqrt(2) * sseed / sdd, n_B_worse=sum(x > 0 for x in d), d=d)
    return o

res = {}
for nm in NAMES:
    o = stats(nm, range(1, 9))
    rep = [r[nm] for r in R]
    o['replays'] = rep; o['sigma_rep'] = sd(rep); o['rep_share'] = 2 * sd(rep) ** 2 / o['SD_d'] ** 2
    o['E2_pair'] = E2[nm]; o['E2_minus_repmean_in_sigma_rep'] = (E2[nm] - mean(rep)) / sd(rep)
    f = Q['factor7']; ma = o['mean_a']
    o['MDE_paired_pct'] = {k: 100 * f * o['SD_d'] / math.sqrt(k) / ma for k in (1, 2, 3, 5)}
    o['MDE_unpaired_pct'] = {k: 100 * f * o['sigma_seed'] * math.sqrt(2 / k) / ma for k in (1, 2, 3, 5)}
    o['MDE_paired_ub_pct'] = {k: v * Q['ub80_7'] for k, v in o['MDE_paired_pct'].items()}
    o['MDE_unpaired_ub_pct'] = {k: v * Q['ub80_7'] for k, v in o['MDE_unpaired_pct'].items()}
    def kneed(sdref, unp):
        out_ = {}
        for delta in (0.5, 1.0, 2.0):
            k = 1
            while 100 * f * sdref * (math.sqrt(2 / k) if unp else 1 / math.sqrt(k)) / ma > delta and k < 500:
                k += 1
            out_[delta] = k
        return out_
    o['k_needed_paired'] = kneed(o['SD_d'], False); o['k_needed_unpaired'] = kneed(o['sigma_seed'], True)
    full_sign = 1 if o['mean_d'] > 0 else -1
    o['two_seed_agree'] = sum(1 for i, j in itertools.combinations(range(8), 2)
                              if o['d'][i] * full_sign > 0 and o['d'][j] * full_sign > 0)
    half = Q['t975_7'] * o['SD_d'] * math.sqrt(1 / 3 + 1 / 8)
    o['refresh_pi'] = [o['mean_d'] - half, o['mean_d'] + half]
    o['refresh_sd_bound'] = o['SD_d'] * Q['chi2_95_2_sdbound']
    o['s2_8'] = {k: v for k, v in stats(nm, range(2, 9)).items() if k != 'd'}
    res[nm] = o
out['metrics'] = res
out['averaging'] = {c: {'SD_d_F': res[f'F({c})']['SD_d'], 'SD_d_M': res[f'M({c})']['SD_d'],
                        'reduction': 1 - res[f'M({c})']['SD_d'] / res[f'F({c})']['SD_d']} for c in ('CHAT', 'PROSE')}

# ---------- CRN: drawn token counts per source at every logged step ----------
def stream(m):
    return {st: tuple(sorted((k, v) for k, v in r.items() if k.startswith(('drawn_', 'dropped_', 'sup_tokens', 'tokens'))))
            for st, r in m['_tail'].items()}, tuple(sorted((k, v) for k, v in m['_end'].items()
                                                        if k.startswith(('drawn_', 'dropped_', 'sup_tokens', 'tokens'))))
crn = {'pairs_equal': {s: stream(A[s]) == stream(B[s]) for s in range(1, 9)},
       'replays_equal_A1': [stream(r) == stream(A[1]) for r in R[1:]], 'E2_equal_A1': stream(E2) == stream(A[1]),
       'distinct_seed_streams': len({stream(A[s])[1] for s in range(1, 9)})}
out['crn'] = crn
out['starts'] = {f'{k}{s}': (m['_start']['resumed_from'], m['_start']['schedule'], m['_start']['config_sha256'][:16],
                             m['_start']['prereg_sha256'][:16]) for k, D in (('A', A), ('B', B)) for s, m in D.items()}
out['ends'] = {f'{k}{s}': (m['_end']['step'], m['_end']['total_steps']) for k, D in (('A', A), ('B', B)) for s, m in D.items()}
for m in list(A.values()) + list(B.values()) + R + [E2]:
    for k in ('_tail', '_end', '_start'): m.pop(k, None)
json.dump(out, open(os.path.join(E3, 'audit', 'a1_out.json'), 'w'), indent=1, default=str)

# ---------- print ----------
print('pooled CHAT/PROSE row vs my pooling, max |diff|:', out['pooled_row_max_abs_diff'])
print('quantiles:', {k: round(v, 4) for k, v in Q.items()})
print(f"{'metric':20s} {'meanA':>7s} {'mean_d':>9s} {'sseed':>8s} {'SD_d':>8s} {'rel':>7s} {'rho':>6s} {'CRN':>5s} {'p':>7s} {'srep':>8s} {'shr':>5s}")
for nm in NAMES:
    o = res[nm]
    print(f"{nm:20s} {o['mean_a']:7.4f} {o['mean_d']:+9.5f} {o['sigma_seed']:8.5f} {o['SD_d']:8.5f} {100*o['SD_d_rel']:6.3f}% "
          f"{o['rho']:6.3f} {o['crn_ratio']:5.2f} {o['p']:7.4f} {o['sigma_rep']:8.5f} {o['rep_share']:5.2f}")
for nm in ('F(CHAT)', 'F(PROSE)'):
    o = res[nm]; s = o['s2_8']
    print(nm, 'd per seed', [round(x, 5) for x in o['d']], 'B worse', o['n_B_worse'], 'CI', [round(x, 5) for x in o['ci95']])
    print('  s2..8: mean_d %+.5f SD_d %.5f (%.3f%%) t %+.3f p %.4f' % (s['mean_d'], s['SD_d'], 100 * s['SD_d_rel'], s['t'], s['p']))
    print('  replays', [round(x, 5) for x in o['replays']], 'E2', round(o['E2_pair'], 5), 'z_rep %+.2f' % o['E2_minus_repmean_in_sigma_rep'])
    print('  MDE p', {k: round(v, 3) for k, v in o['MDE_paired_pct'].items()}, 'u', {k: round(v, 3) for k, v in o['MDE_unpaired_pct'].items()})
    print('  MDE k2 ub p %.3f u %.3f' % (o['MDE_paired_ub_pct'][2], o['MDE_unpaired_ub_pct'][2]))
    print('  k_needed p', o['k_needed_paired'], 'u', o['k_needed_unpaired'], 'two-seed agree', o['two_seed_agree'])
    print('  refresh PI [%+.5f, %+.5f] SD bound %.5f' % (*o['refresh_pi'], o['refresh_sd_bound']), 'sd_A_only %.5f' % o['sd_A_only'])
print('averaging', {c: round(v['reduction'], 4) for c, v in out['averaging'].items()})
print('CRN', crn)
print('two-seed agreement all metrics', {nm: res[nm]['two_seed_agree'] for nm in NAMES})
print('starts unique', set((v[0], json.dumps(v[1]), v[3]) for v in out['starts'].values()), set(out['ends'].values()))
