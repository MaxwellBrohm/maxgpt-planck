"""E3 Part 2 audit: scipy cross-check of the auditor's own numbers (audit/p2_a1_out.json). Reads the raw values from
out/e3_20m_* again (final-checkpoint CHAT and pooled PROSE), recomputes with scipy/numpy, compares.
  python3 audit/p2_a5_scipy.py      (system Python with scipy; Planck's venv has none)"""
import json, math, os, sys
import numpy as np, scipy
from scipy import stats

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mine = json.load(open(os.path.join(E3, "audit", "p2_a1_out.json")))
print("scipy", scipy.__version__, "python", sys.version.split()[0])
worst = [0.0, ""]


def cmp(name, a, b):
    r = abs(a - b) / (abs(b) if b else 1.0)
    if r > worst[0]: worst[:] = [r, name]
    print(f"  {name:38s} scipy {a:.10g}  own {b:.10g}  rel diff {r:.1e}")


q = mine["quantiles"]
cmp("factor df1", stats.t.ppf(0.975, 1) + stats.t.ppf(0.8, 1), q["factor_df1"])
cmp("factor df2", stats.t.ppf(0.975, 2) + stats.t.ppf(0.8, 2), q["factor_df2"])
cmp("factor df7", stats.t.ppf(0.975, 7) + stats.t.ppf(0.8, 7), q["factor_df7"])
cmp("ub80 df2", math.sqrt(2 / stats.chi2.ppf(0.2, 2)), q["ub80_df2"])
cmp("chi2 0.025 df2", stats.chi2.ppf(0.025, 2), q["chi2_0.025_df2"])
cmp("chi2 0.975 df2", stats.chi2.ppf(0.975, 2), q["chi2_0.975_df2"])
cmp("F 0.025 (2,7)", stats.f.ppf(0.025, 2, 7), q["F_0.025_df2_7"])
cmp("F 0.975 (2,7)", stats.f.ppf(0.975, 2, 7), q["F_0.975_df2_7"])


def final(run, kind):
    rows = [json.loads(x) for x in open(os.path.join(E3, "out", run, "bpb.jsonl"))]
    sel = [x for x in rows if x["step"] == 15259 and x["split"] == "all" and (x["set"] in ("cccc", "gutenberg", "wikimedia") if kind == "PROSE" else x["set"] == "oasst2")]
    return sum(x["bits"] for x in sel) / sum(x["bytes"] for x in sel)


for m in ("CHAT", "PROSE"):
    a = np.array([final(f"e3_20m_A_s{s}", m) for s in (1, 2, 3)]); b = np.array([final(f"e3_20m_B_s{s}", m) for s in (1, 2, 3)])
    z = mine["noise20"][f"F({m})"]; z2 = mine["noise20_s23"][f"F({m})"]
    print(f"F({m})")
    sd_d = np.std(b - a, ddof=1); s_seed = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    tt = stats.ttest_rel(b, a); tt2 = stats.ttest_rel(b[1:], a[1:]); pr = stats.pearsonr(a, b)
    cmp("SD_d", sd_d, z["SD_d"]); cmp("sigma_seed", s_seed, z["sigma_seed"]); cmp("rho", pr.statistic, z["rho"])
    cmp("rho = 0 p (pearsonr, exact)", pr.pvalue, z["rho_zero_p_two_sided"])
    cmp("t (3 seeds)", tt.statistic, z["t"]); cmp("p (3 seeds)", tt.pvalue, z["p_two_sided"])
    ci = tt.confidence_interval(0.95); cmp("CI low (3 seeds)", ci.low, z["ci95_d"][0]); cmp("CI high (3 seeds)", ci.high, z["ci95_d"][1])
    cmp("t (seeds 2..3)", tt2.statistic, z2["t"]); cmp("p (seeds 2..3)", tt2.pvalue, z2["p_two_sided"])
    cmp("CRN ratio", math.sqrt(2) * s_seed / sd_d, z["crn_ratio"])
    f2 = stats.t.ppf(0.975, 2) + stats.t.ppf(0.8, 2)
    cmp("MDE paired k1 (bpb)", f2 * sd_d, z["MDE"]["1"]["paired"]); cmp("MDE unpaired k1 (bpb)", f2 * s_seed * math.sqrt(2), z["MDE"]["1"]["unpaired"])
print(f"largest relative difference {worst[0]:.1e} ({worst[1]})")
