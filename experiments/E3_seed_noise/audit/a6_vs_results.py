# E3 audit: my a1_out.json against the analyst's results.json, every shared statistic (Planck's venv, from E3 folder).
import json
mine = json.load(open('audit/a1_out.json'))['metrics']; res = json.load(open('results.json'))
nz = res['noise_5M_250M']; worst = (0, '')
pairs = [('mean_a', 'mean_a'), ('mean_d', 'mean_d'), ('SD_d', 'SD_d'), ('sigma_seed', 'sigma_seed'), ('rho', 'rho'),
         ('p', 'p_two_sided'), ('t', 't'), ('crn_ratio', 'crn_ratio'), ('refresh_sd_bound', 'refresh_sd_bound')]
n = 0
for nm, m in mine.items():
    r = nz[nm]
    for a, b in pairs:
        dlt = abs(m[a] - r[b]); n += 1
        if dlt > worst[0]: worst = (dlt, f'{nm} {a}')
    for i in range(2):
        for a, b in (('ci95', 'ci95_d'), ('refresh_pi', 'refresh_pi95_3pair_mean')):
            dlt = abs(m[a][i] - r[b][i]); n += 1
            if dlt > worst[0]: worst = (dlt, f'{nm} {a}')
    for k in ('1', '2', '3', '5'):
        for a, b in (('MDE_paired_pct', 'paired_rel'), ('MDE_unpaired_pct', 'unpaired_rel'), ('MDE_paired_ub_pct', 'paired_rel_ub80'), ('MDE_unpaired_ub_pct', 'unpaired_rel_ub80')):
            dlt = abs(m[a][k] / 100 - r['MDE'][k][b]); n += 1
            if dlt > worst[0]: worst = (dlt, f'{nm} {a} k{k}')
    kp = {float(k.rstrip('%')): v for k, v in r['k_needed_paired'].items()}; ku = {float(k.rstrip('%')): v for k, v in r['k_needed_unpaired'].items()}
    mk = {float(k): v for k, v in m['k_needed_paired'].items()}; mu = {float(k): v for k, v in m['k_needed_unpaired'].items()}
    print(nm, 'k_needed paired', 'same' if kp == mk else f'DIFF mine {mk} theirs {kp}', '| unpaired', 'same' if ku == mu else f'DIFF mine {mu} theirs {ku}')
print(f'{n} shared statistics compared; largest |difference| {worst[0]:.2e} ({worst[1]})')
rc = res['e2_recheck']; m = mine['F(CHAT)']['s2_8']
print('E2 recheck mean_d/t/p diffs:', abs(rc['mean_d'] - m['mean_d']), abs(rc['t'] - m['t']), abs(rc['p_two_sided'] - m['p']), rc['verdict'])
