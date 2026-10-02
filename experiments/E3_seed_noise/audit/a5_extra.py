# E3 audit extras (system python, scipy): uncertainty of the CRN ratio via rho, and the untested E2 neighbour.
# Run: /usr/bin/python3 audit/a5_extra.py (from the E3 folder).
import json, math
from scipy import stats
o = json.load(open('audit/a1_out.json'))['metrics']
for nm in ('F(CHAT)', 'F(PROSE)'):
    m = o[nm]; n = 8; z = math.atanh(m['rho']); se = 1 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    ratio = lambda r: 1 / math.sqrt(1 - r) if r < 1 else float('inf')   # sqrt2 sigma / SD_d with equal arm SDs
    print(f"{nm}: rho {m['rho']:.3f}, Fisher 95% CI [{lo:.2f}, {hi:.2f}] -> CRN ratio about [{ratio(lo):.2f}, {ratio(hi):.2f}] (point {m['crn_ratio']:.2f})")
    # 95% CI on SD_d (chi-square, df 7)
    print(f"   SD_d {m['SD_d']:.5f}, 95% CI [{m['SD_d']*math.sqrt(7/stats.chi2.ppf(.975,7)):.5f}, {m['SD_d']*math.sqrt(7/stats.chi2.ppf(.025,7)):.5f}]")
e2 = json.load(open('../E2_lr_transfer/results.json'))['stage_A']['final_bpb']
g = e2['0.006']['b250M']['chat'] - e2['0.003']['b250M']['chat']
print(f"E2 seed-1 F(CHAT) gap 6e-3 minus 3e-3 at 250M: {g:+.4f} = {g/o['F(CHAT)']['SD_d']:.2f} SD_d; paired k=1 MDE {o['F(CHAT)']['MDE_paired_pct']['1']:.3f}% = {o['F(CHAT)']['MDE_paired_pct']['1']/100*o['F(CHAT)']['mean_a']:.4f} bpb")
# seeds 2..8 MDE (notes lines 92-93: printed beside the noise table with factor 3.353, df 6)
f6 = stats.t.ppf(.975, 6) + stats.t.ppf(.8, 6)
for nm in ('F(CHAT)', 'F(PROSE)'):
    s = o[nm]['s2_8']
    print(nm, 'seeds 2..8 MDE paired %:', {k: round(100 * f6 * s['SD_d'] / math.sqrt(k) / s['mean_a'], 3) for k in (1, 2, 3, 5)},
          'unpaired %:', {k: round(100 * f6 * s['sigma_seed'] * math.sqrt(2 / k) / s['mean_a'], 3) for k in (1, 2, 3, 5)})
