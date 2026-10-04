"""E3 Part 2 audit: the auditor's own numbers (audit/p2_a1_out.json) against the analyst's results_part2.json, every
shared statistic. Prints the count compared and the largest absolute and relative differences.
  ~/.venvs/planck/bin/python audit/p2_a6_vs_results.py"""
import json, os

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mine = json.load(open(os.path.join(E3, "audit", "p2_a1_out.json")))
theirs = json.load(open(os.path.join(E3, "results_part2.json")))
pairs = []


def add(name, a, b):
    if isinstance(a, dict):
        for k in a:
            if k in b:
                add(f"{name}/{k}", a[k], b[k])
            else:
                pairs.append((f"{name}/{k}", a[k], "MISSING"))
    elif isinstance(a, list):
        assert len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            add(f"{name}[{i}]", x, y)
    else:
        pairs.append((name, a, b))


KEYS = ("n", "df", "mean_a", "mean_d", "sigma_seed", "sigma_seed_rel", "SD_d", "SD_d_rel", "rho", "crn_ratio", "t", "p_two_sided",
        "ci95_d", "MDE", "k_needed_paired", "k_needed_unpaired")
for m, z in mine["noise20"].items():
    add(f"noise20/{m}", {k: z[k] for k in KEYS}, theirs["noise_20M_500M"][m])
    add(f"values/{m}", {"A": mine["values"][m]["A"], "B": mine["values"][m]["B"], "d": z["d"]}, theirs["values"][m])
for m, z in mine["noise20_s23"].items():
    add(f"seeds2to3/{m}", {k: z[k] for k in KEYS if k in ("n", "df", "mean_a", "mean_d", "sigma_seed", "SD_d", "SD_d_rel", "rho", "crn_ratio", "t", "p_two_sided", "ci95_d")},
        theirs["noise_20M_500M_seeds2to3"][m])
for m, cl in mine["rule4"].items():
    for c, v in cl.items():
        tc = {"paired (SD_d)": "same init, same stream (SD_d)"}.get(c, c)
        t = theirs["rule4_20M_confirmations"][m][tc]; cmp = theirs["compare_20M_vs_5M_descriptive"][m][tc]
        add(f"rule4/{m}/{c}", {"rel_20M_df2": v["rel20"], "rel_5M_df7": v["rel5"], "used": v["used"], "df_used": v["df"], "MDE_rel": v["MDE_rel"],
                               "k_needed": v["k_needed"], "k_rule2": v["k"]}, t)
        add(f"rule4/{m}/{c}/mde_bpb_k1", v["MDE_bpb_k1_at_A20_mean"], t["MDE_bpb_at_A20_mean"]["1"])
        add(f"rule4/{m}/{c}/underpowered", sorted(v["underpowered"]), sorted(d for d, u in t["underpowered"].items() if u))
        add(f"compare/{m}/{c}", {"rel_ratio_20M_over_5M": v["ratio_20_over_5"], "sd_ratio_95ci_F_df2_7": v["ratio_95ci_F"]}, cmp)
for m in ("F(CHAT)", "F(PROSE)"):
    z = mine["noise20"][m]; key = {"SD_d": "same init, same stream (SD_d)", "sigma_seed": "other classes (sqrt2 sigma_seed)"}
    add(f"chi2ci/{m}/SD_d", z["SD_d_95ci_chi2"], theirs["compare_20M_vs_5M_descriptive"][m][key["SD_d"]]["sd_20M_95ci_chi2_df2"])
    s2 = [x * 2 ** 0.5 for x in z["sigma_seed_95ci_chi2"]]
    add(f"chi2ci/{m}/sqrt2sigma_seed", s2, theirs["compare_20M_vs_5M_descriptive"][m][key["sigma_seed"]]["sd_20M_95ci_chi2_df2"])
e = mine["e2_label"]; g = theirs["e2_20m_one_seed_label_input"]
add("e2_label", {"difference_bpb": e["gap"], "primary_20M_paired_MDE_k1_bpb": e["mde20_k1"], "rule4_paired_MDE_k1_bpb": e["mde_rule4_k1"],
                 "label_primary": "resolved at one seed" if e["resolved_20"] else "not resolved",
                 "label_rule4": "resolved at one seed" if e["resolved_rule4"] else "not resolved"}, g)
for k, v in mine["averaging"].items():
    add(f"averaging/{k}", v, theirs["averaging_20M_descriptive"][k]["reduction"])
for m in ("F(CHAT)", "F(PROSE)"):
    add(f"crn/{m}", mine["noise20"][m]["crn_ratio"], theirs["crn_20M"][m]["ratio"])
rc = theirs["e2_recheck_20M"]; s = mine["noise20_s23"]["F(CHAT)"]
add("recheck", {"mean_d": s["mean_d"], "t": s["t"], "p_two_sided": s["p_two_sided"], "ci95_d": s["ci95_d"], "d": s["d"],
                "B_lower_and_p_lt_0.05": s["mean_d"] < 0 and s["p_two_sided"] < 0.05}, rc)
for k in ("factor_df1", "factor_df2", "factor_df7", "ub80_df2", "F_0.025_df2_7", "F_0.975_df2_7", "chi2_0.025_df2", "chi2_0.975_df2"):
    add(f"quantiles/{k}", mine["quantiles"][k], theirs["quantiles"][k])

bad, amax, rmax, worst = [], 0.0, 0.0, None
for name, a, b in pairs:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        da = abs(a - b); dr = da / abs(b) if b else da
        if (isinstance(a, int) and isinstance(b, int) and a != b) or dr > 1e-8:
            bad.append((name, a, b))
        if da > amax: amax = da
        if dr > rmax: rmax, worst = dr, name
    elif a != b:
        bad.append((name, a, b))
print(f"compared {len(pairs)} values; mismatches {len(bad)}; largest abs diff {amax:.2e}; largest rel diff {rmax:.2e} ({worst})")
for x in bad:
    print("MISMATCH", x)
