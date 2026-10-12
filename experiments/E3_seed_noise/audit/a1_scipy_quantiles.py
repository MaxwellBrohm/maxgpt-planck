# E3 audit: cross-check a1_recompute.py's own numeric quantiles and p-values with scipy (system python 3.9,
# scipy 1.13.1; Planck's venv has no scipy). Also: share CI for sigma_rep, and A1-vs-replay sensitivity of d_1.
# Run: /usr/bin/python3 audit/a1_scipy_quantiles.py (from the E3 folder).
import json, math
from scipy import stats
o = json.load(open('audit/a1_out.json'))
Q = o['quantiles']
ref = {'t975_7': stats.t.ppf(.975, 7), 't80_7': stats.t.ppf(.8, 7), 't975_6': stats.t.ppf(.975, 6), 't80_6': stats.t.ppf(.8, 6),
       't975_2': stats.t.ppf(.975, 2), 't80_2': stats.t.ppf(.8, 2), 'ub80_7': math.sqrt(7 / stats.chi2.ppf(.2, 7)),
       'ub80_2': math.sqrt(2 / stats.chi2.ppf(.2, 2)), 'chi2_95_2_sdbound': math.sqrt(stats.chi2.ppf(.95, 2) / 2)}
print('max |quantile diff| vs scipy:', max(abs(Q[k] - v) for k, v in ref.items()))
pd = 0
for nm, m in o['metrics'].items():
    for key, df in (('', 7), ('s2_8', 6)):
        mm = m if not key else m[key]
        pd = max(pd, abs(mm['p'] - 2 * stats.t.sf(abs(mm['t']), df)))
print('max |p diff| vs scipy (all metrics, 8 and 7 seeds):', pd)
for nm in ('F(CHAT)', 'F(PROSE)'):
    m = o['metrics'][nm]
    a = [m['replays'][0]]  # A1
    d = m['d']
    # scipy paired t from the raw d's
    t8 = stats.ttest_1samp(d, 0); t7 = stats.ttest_1samp(d[1:], 0)
    lo, hi = 2 / stats.chi2.ppf(.975, 2), 2 / stats.chi2.ppf(.025, 2)
    print(nm, 'scipy t8 %.3f p %.4f | t7 %.3f p %.4f' % (t8.statistic, t8.pvalue, t7.statistic, t7.pvalue),
          '| rep share %.2f, 95%% CI from sigma_rep df 2 alone [%.2f, %.1f]' % (m['rep_share'], m['rep_share'] * lo, m['rep_share'] * hi))
    # sensitivity: pair 1 with R2, R3 or the replay mean in place of A1
    b1 = m['replays'][0] + d[0]
    for lab, a1 in (('R2', m['replays'][1]), ('R3', m['replays'][2]), ('rep mean', sum(m['replays']) / 3)):
        dd = [b1 - a1] + d[1:]
        r = stats.ttest_1samp(dd, 0)
        print('   pair 1 uses %-8s d1 %+.5f mean_d %+.5f SD_d %.5f p %.4f' % (lab, dd[0], sum(dd) / 8, stats.tstd(dd), r.pvalue))
