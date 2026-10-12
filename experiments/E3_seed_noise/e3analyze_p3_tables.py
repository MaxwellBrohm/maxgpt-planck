"""tables_part3.txt from results_part3.json's dict (e3analyze_p3.py). Formatting only: no statistic is computed here."""
from __future__ import annotations
import json


def tables(r: dict) -> str:
    fails = [c for c in r["checks"] if not c["ok"]]
    L = ["E3 PART 3 TABLES (e3analyze_p3.py, importing e3analyze.py; 5M arm A 3e-3 r 1, 250M slots, full WSD; "
         "INIT = seeds 1..5 at data_seed 1, DATA = data_seeds 1..5 at seed 1; e3_5m_A_s1_P3 is member 1 of both)", "",
         "CHECKS: " + ", ".join(f"{c['check']} FAILED ({c['detail']})" for c in fails) +
         f"  [{len(r['checks']) - len(fails)}/{len(r['checks'])} pass; expected failures {len(r['expected_failures'])}; "
         f"unexpected {len(r['unexpected_failures'])}: {r['unexpected_failures']}]",
         f"GAPS: {r['gaps'] or 'none'}", "",
         "FAMILY SDs (df n - 1); rel = fraction of the family's mean; [95% chi-square interval on rel]; ub80 = 80% upper bound"]
    L.append(f"{'metric':20s} {'fam':4s} {'n':>2s} {'mean':>8s} {'SD':>9s} {'rel':>7s} {'rel 95% CI':>18s} {'ub80':>7s}")
    for m, fz in r["families"].items():
        for fam, z in fz.items():
            if z is None:
                L.append(f"{m:20s} {fam:4s} fewer than 4 runs: no SD"); continue
            L.append(f"{m:20s} {fam:4s} {z['n']:2d} {z['mean']:8.5f} {z['sd']:9.6f} {z['rel']:7.3%} "
                     f"[{z['rel_ci95'][0]:6.3%}, {z['rel_ci95'][1]:6.3%}] {z['rel_ub80']:7.3%}")
    L += ["", "RULE 1 REFERENCE SDs (sqrt2 x family SD, df 4 unless a GAP): MDE_u(k) as % of the family mean, k = 1, 2, 3, 5 "
          "[k2 with the 80% upper-bound SD]; k_needed 0.5% / 1% / 2%; rule 2 k = max(2, k_needed); >5 underpowered"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)"):
        for cls, v in r["rule1"][m].items():
            if isinstance(v, str):
                L.append(f"{m:9s} {cls:22s} {v}"); continue
            L.append(f"{m:9s} {cls:22s} ref {v['ref_rel']:.3%} ({v['ref_bpb']:.5f} bpb), df {v['df']}, factor {v['factor']:.3f}; MDE "
                     + " ".join(f"k{k} {x:.3%}" for k, x in v["MDE_rel"].items()) + f" [k2 {v['MDE_rel_ub80']['2']:.3%}]; k_needed "
                     + " ".join(f"{dl}:{k}" for dl, k in v["k_needed"].items()) + "; k " + " ".join(f"{dl}:{k}" for dl, k in v["k_rule2"].items())
                     + "; underpowered " + (",".join(dl for dl, u in v["underpowered"].items() if u) or "none"))
    L += ["", "RULE 1, OTHER METRICS: reference rel (sqrt2 x family SD) and k_needed 0.5% / 1% / 2%"]
    for m, cl in r["rule1"].items():
        if m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)"):
            continue
        L.append(f"{m:20s} " + "; ".join(f"{c}: {v['ref_rel']:.3%} k " + "/".join(str(k) for k in v["k_needed"].values())
                                         if not isinstance(v, str) else f"{c}: {v}" for c, v in cl.items()))
    L += ["", "AGAINST PART 1's sigma_seed (descriptive; relative SDs; another commit): ratio [95% F interval, df 4 and 7]; "
          "sqrt(sigma_init^2 + sigma_data^2) / sigma_seed"]
    for m, c in r["compare_part1_sigma_seed"].items():
        part = [f"{fam} {c[fam]['ratio']:.2f} [{c[fam]['ratio_95ci_F'][0]:.2f}, {c[fam]['ratio_95ci_F'][1]:.2f}]" for fam in ("INIT", "DATA") if fam in c]
        tot = f"; sum {c['sqrt_init2_plus_data2_over_seed']:.2f}" if "sqrt_init2_plus_data2_over_seed" in c else ""
        L.append(f"{m:20s} sigma_seed {c['sigma_seed_rel_part1_df7']:.3%}; " + "; ".join(part) + tot)
    L += ["", "PER-RUN VALUES"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)", "train loss last 2%"):
        for fam, v in r["values"][m].items():
            L.append(f"{m:20s} {fam:4s} " + "  ".join(f"{n.replace('e3_5m_A_', '')} {x:.5f}" for n, x in v.items()))
    L += ["", "e3_5m_A_s1_P3 BESIDE PART 1's A1, R2, R3 (descriptive, a replay across commits; no rule): value, minus A1, "
          "minus the replay mean, in Part 1 sigma_rep"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)", "train loss last 2%"):
        x = r["replay_vs_part1_A1"].get(m)
        if x:
            L.append(f"{m:20s} {x['s1_P3']:.5f}  A1/R2/R3 {' '.join(f'{y:.5f}' for y in x['A1_R2_R3'])}  {x['minus_A1']:+.5f}  "
                     f"{x['minus_rep_mean']:+.5f}  " + (f"{x['in_sigma_rep']:+.2f}" if x["in_sigma_rep"] is not None else "n/a"))
    L += [f"s1_P3 drew Part 1 A1's stream (end-record counts): {r['replay_stream_equal_A1']}",
          "", f"Quantiles: {json.dumps({k: round(v, 4) for k, v in r['quantiles'].items()})}", f"Not run: {json.dumps(r['not_run'])}", ""]
    return "\n".join(L)
