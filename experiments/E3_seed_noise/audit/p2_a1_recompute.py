"""E3 Part 2 audit (independent): every Part 2 statistic from the raw records, with this file's own code.
Reads out/e3_20m_*/{bpb.jsonl,train_tail.jsonl} and, for rule 4, Part 1's raw out/e3_5m_{A,B}_s{1..8}/bpb.jsonl
(not results.json). Does not import e3analyze.py or e3analyze_p2.py. Quantiles: closed forms where they exist (t at
df 1 and 2, chi-square at df 2, F with numerator df 2), Simpson integration of the t density otherwise (df 7).
Writes audit/p2_a1_out.json; prints a summary (audit/p2_a1_stdout.txt).
  ~/.venvs/planck/bin/python audit/p2_a1_recompute.py"""
import json, math, os, statistics as st

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(E3, "out")
CK20, T20 = (14945, 15250, 15259), 15259
CK5, T5 = (7344, 7497, 7630), 7630
KS, DELTAS = (1, 2, 3, 5), (0.005, 0.01, 0.02)
PROSE_SETS = ("cccc", "gutenberg", "wikimedia")
METRICS = {"F(CHAT)": ("CHAT", "all", "F"), "F(PROSE)": ("PROSE", "all", "F"), "M(CHAT)": ("CHAT", "all", "M"),
           "M(PROSE)": ("PROSE", "all", "M"), "CHAT user": ("oasst2", "user", "F"),
           "CHAT assistant": ("oasst2", "assistant", "F"), "PROSE cccc": ("cccc", "all", "F"),
           "PROSE gutenberg": ("gutenberg", "all", "F"), "PROSE wikimedia": ("wikimedia", "all", "F"),
           "stackexchange": ("stackexchange", "all", "F"), "irc": ("irc", "all", "F"), "dolly": ("dolly", "all", "F"),
           "train loss last 2%": (None, None, "T")}


# ---------- quantiles (own) ----------
def t_pdf(x, v):
    return math.exp(math.lgamma((v + 1) / 2) - math.lgamma(v / 2)) / math.sqrt(v * math.pi) * (1 + x * x / v) ** (-(v + 1) / 2)


def t_cdf(x, v):
    if v == 1:
        return 0.5 + math.atan(x) / math.pi
    if v == 2:
        return 0.5 + x / (2 * math.sqrt(2 + x * x))
    n = 20000; h = abs(x) / n
    s = t_pdf(0, v) + t_pdf(abs(x), v) + sum((4 if i % 2 else 2) * t_pdf(i * h, v) for i in range(1, n))
    area = s * h / 3
    return 0.5 + area if x >= 0 else 0.5 - area


def t_ppf(q, v):
    if v == 1:
        return math.tan(math.pi * (q - 0.5))
    if v == 2:
        return (2 * q - 1) / math.sqrt(2 * q * (1 - q))
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, v) < q else (lo, mid)
    return (lo + hi) / 2


def chi2_ppf_df2(q): return -2 * math.log(1 - q)
def f2_ppf(q, d2): return d2 / 2 * ((1 - q) ** (-2 / d2) - 1)          # F(2, d2): CDF 1 - (1 + 2x/d2)^(-d2/2)
def p_two(t, v): return 2 * (1 - t_cdf(abs(t), v))
def factor(v): return t_ppf(0.975, v) + t_ppf(0.80, v)


def ub80(v):   # one-sided 80% upper bound multiplier for an SD: sqrt(df / chi2_0.20(df)); df 2 closed form only
    assert v == 2
    return math.sqrt(2 / chi2_ppf_df2(0.20))


# ---------- records ----------
def rows(run, f): return [json.loads(x) for x in open(os.path.join(OUT, run, f))]


def value(run, metric, ckpts, total):
    s, sp, kind = METRICS[metric]
    if kind == "T":
        r = [x["loss"] for x in rows(run, "train_tail.jsonl") if x["step"] > total - 0.02 * total]
        return st.fmean(r), len(r)
    b = rows(run, "bpb.jsonl")
    if s == "PROSE":   # own pooling: bits over bytes of cccc + gutenberg + wikimedia at each checkpoint
        v = {c: sum(x["bits"] for x in b if x["step"] == c and x["set"] in PROSE_SETS and x["split"] == "all")
             / sum(x["bytes"] for x in b if x["step"] == c and x["set"] in PROSE_SETS and x["split"] == "all") for c in ckpts}
    elif s == "CHAT":
        v = {c: next(x["bits"] / x["bytes"] for x in b if x["step"] == c and x["set"] == "oasst2" and x["split"] == "all") for c in ckpts}
    else:
        v = {c: next(x["bits"] / x["bytes"] for x in b if x["step"] == c and x["set"] == s and x["split"] == sp) for c in ckpts}
    return (v[total] if kind == "F" else st.fmean(v[c] for c in ckpts)), len(v)


def sd(x): return st.stdev(x)


def corr(a, b):
    ma, mb = st.fmean(a), st.fmean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def stats(a, b):
    n = len(a); v = n - 1; d = [y - x for x, y in zip(a, b)]
    ma = st.fmean(a); s_seed = math.sqrt((st.variance(a) + st.variance(b)) / 2); s_d = sd(d)
    md = st.fmean(d); se = s_d / math.sqrt(n); t = md / se; f = factor(v); tq = t_ppf(0.975, v)
    r = corr(a, b)
    out = {"n": n, "df": v, "mean_a": ma, "mean_d": md, "sigma_seed": s_seed, "sigma_seed_rel": s_seed / ma, "SD_d": s_d,
           "SD_d_rel": s_d / ma, "rho": r, "crn_ratio": math.sqrt(2) * s_seed / s_d, "t": t, "p_two_sided": p_two(t, v),
           "ci95_d": [md - tq * se, md + tq * se], "d": d, "B_lower_all": all(x < 0 for x in d), "B_higher_all": all(x > 0 for x in d)}
    if n >= 3:
        tr = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)          # exact test of rho = 0 under bivariate normality
        out["rho_zero_p_two_sided"] = p_two(tr, n - 2)
    if v == 2:
        out["MDE"] = {str(k): {"paired": f * s_d / math.sqrt(k), "unpaired": f * s_seed * math.sqrt(2 / k)} for k in KS}
        for k, m in out["MDE"].items():
            m.update(paired_rel=m["paired"] / ma, unpaired_rel=m["unpaired"] / ma, paired_rel_ub80=m["paired"] * ub80(2) / ma,
                     unpaired_rel_ub80=m["unpaired"] * ub80(2) / ma)
        out["k_needed_paired"] = {f"{dl:.3%}": kneed(f, s_d / ma, dl) for dl in DELTAS}
        out["k_needed_unpaired"] = {f"{dl:.3%}": kneed(f, math.sqrt(2) * s_seed / ma, dl) for dl in DELTAS}
        lo, hi = math.sqrt(2 / chi2_ppf_df2(0.975)), math.sqrt(2 / chi2_ppf_df2(0.025))
        out["SD_d_95ci_chi2"] = [s_d * lo, s_d * hi]; out["sigma_seed_95ci_chi2"] = [s_seed * lo, s_seed * hi]
    return out


def kneed(f, rel, dl):
    k = 1
    while f * rel / math.sqrt(k) > dl:
        k += 1
    return k


def main():
    res = {"quantiles": {"factor_df1": factor(1), "factor_df2": factor(2), "factor_df7": factor(7), "ub80_df2": ub80(2),
                         "t975_df7": t_ppf(0.975, 7), "F_0.025_df2_7": f2_ppf(0.025, 7), "F_0.975_df2_7": f2_ppf(0.975, 7),
                         "chi2_0.025_df2": chi2_ppf_df2(0.025), "chi2_0.975_df2": chi2_ppf_df2(0.975)},
           "values": {}, "n_records": {}, "noise20": {}, "noise20_s23": {}, "noise5_FCHAT_FPROSE": {}}
    for m in METRICS:
        a = [value(f"e3_20m_A_s{s}", m, CK20, T20) for s in (1, 2, 3)]; b = [value(f"e3_20m_B_s{s}", m, CK20, T20) for s in (1, 2, 3)]
        res["n_records"][m] = sorted({x[1] for x in a + b})
        a, b = [x[0] for x in a], [x[0] for x in b]
        res["values"][m] = {"A": a, "B": b}
        res["noise20"][m] = stats(a, b)
        res["noise20_s23"][m] = stats(a[1:], b[1:])
    for m in ("F(CHAT)", "F(PROSE)"):   # Part 1's 5M SDs from Part 1's raw records (8 pairs, df 7)
        a5 = [value(f"e3_5m_A_s{s}", m, CK5, T5)[0] for s in range(1, 9)]; b5 = [value(f"e3_5m_B_s{s}", m, CK5, T5)[0] for s in range(1, 9)]
        res["noise5_FCHAT_FPROSE"][m] = stats(a5, b5)
    Z, Z5, q = res["noise20"], res["noise5_FCHAT_FPROSE"], res["quantiles"]
    # rule 4 (notes HOW LATER SCREENS USE IT, 4) with rule 2's k = max(2, k_needed)
    r4 = {}
    for m in ("F(CHAT)", "F(PROSE)"):
        for cls, key, mult in (("paired (SD_d)", "SD_d_rel", 1.0), ("other classes (sqrt2 sigma_seed)", "sigma_seed_rel", math.sqrt(2))):
            r20, r5 = Z[m][key] * mult, Z5[m][key] * mult
            used, v = (r20, 2) if r20 >= r5 else (r5, 7)
            f = factor(v); kn = {f"{dl:.3%}": kneed(f, used, dl) for dl in DELTAS}
            vr = (r20 / r5) ** 2
            r4.setdefault(m, {})[cls] = {"rel20": r20, "rel5": r5, "used": "20M" if v == 2 else "5M", "df": v,
                                        "MDE_rel": {str(k): f * used / math.sqrt(k) for k in KS},
                                        "MDE_bpb_k1_at_A20_mean": f * used * Z[m]["mean_a"],
                                        "k_needed": kn, "k": {d: max(2, k) for d, k in kn.items()},
                                        "underpowered": [d for d, k in kn.items() if k > 5],
                                        "ratio_20_over_5": r20 / r5,
                                        "ratio_95ci_F": [math.sqrt(vr / q["F_0.975_df2_7"]), math.sqrt(vr / q["F_0.025_df2_7"])]}
    res["rule4"] = r4
    # E2's 20M one-seed label input (E2 RULES, "After E3"); E2 results.json q2, 4 decimals as E2's rules compare
    e2 = json.load(open(os.path.join(E3, "..", "E2_lr_transfer", "results.json")))["q2"]["final_bpb_20m"]
    gap = round(e2["0.0015"]["b500M"]["chat"] - e2["0.003"]["b500M"]["chat"], 4)
    res["e2_label"] = {"g1_chat": e2["0.003"]["b500M"]["chat"], "g0.5_chat": e2["0.0015"]["b500M"]["chat"], "g2_chat": e2["0.006"]["b500M"]["chat"],
                       "g1_prose": e2["0.003"]["b500M"]["prose"], "g0.5_prose": e2["0.0015"]["b500M"]["prose"], "gap": gap,
                       "mde20_k1": Z["F(CHAT)"]["MDE"]["1"]["paired"], "mde_rule4_k1": r4["F(CHAT)"]["paired (SD_d)"]["MDE_bpb_k1_at_A20_mean"],
                       "resolved_20": gap > Z["F(CHAT)"]["MDE"]["1"]["paired"],
                       "resolved_rule4": gap > r4["F(CHAT)"]["paired (SD_d)"]["MDE_bpb_k1_at_A20_mean"]}
    res["averaging"] = {k: 1 - Z[f"M({k})"]["SD_d"] / Z[f"F({k})"]["SD_d"] for k in ("CHAT", "PROSE")}
    res["crn_ge2"] = {m: z["crn_ratio"] for m, z in Z.items() if z["crn_ratio"] >= 2}
    # rule 3 applied to F(PROSE) A20 vs B20 as if it were a 20M confirmation (B improves = d < 0), SD_ref from rule 4
    z = Z["F(PROSE)"]; sref = r4["F(PROSE)"]["paired (SD_d)"]["rel5"] * z["mean_a"]; k = 3
    thr = t_ppf(0.975, 7) * sref / math.sqrt(k); imp = -z["mean_d"]
    res["rule3_prose_if_screen"] = {"improvement": imp, "win_threshold": thr, "every_pair_improves": all(x < 0 for x in z["d"]),
                                    "win": imp > thr and all(x < 0 for x in z["d"]),
                                    "ci95_improvement": [imp - thr, imp + thr], "delta_1pct_bpb": 0.01 * z["mean_a"]}
    json.dump(res, open(os.path.join(E3, "audit", "p2_a1_out.json"), "w"), indent=1, sort_keys=True)

    P = print
    P("quantiles:", json.dumps({k: round(v, 6) for k, v in q.items()}))
    P("records per value (checkpoints or tail steps):", json.dumps(res["n_records"]))
    P(f"\n{'metric':20s} {'meanA':>8s} {'mean_d':>9s} {'sig_seed':>8s} {'rel':>7s} {'SD_d':>8s} {'rel':>7s} {'rho':>7s} {'CRN':>6s} {'t':>7s} {'p':>6s}  CI95")
    for m, z in Z.items():
        P(f"{m:20s} {z['mean_a']:8.4f} {z['mean_d']:+9.5f} {z['sigma_seed']:8.5f} {z['sigma_seed_rel']:7.3%} {z['SD_d']:8.5f} {z['SD_d_rel']:7.3%} "
          f"{z['rho']:7.3f} {z['crn_ratio']:6.2f} {z['t']:+7.2f} {z['p_two_sided']:6.3f}  [{z['ci95_d'][0]:+.5f}, {z['ci95_d'][1]:+.5f}]")
    for m in ("F(CHAT)", "F(PROSE)"):
        z = Z[m]
        P(f"\n{m}: A {[round(x, 5) for x in res['values'][m]['A']]} B {[round(x, 5) for x in res['values'][m]['B']]} d {[round(x, 5) for x in z['d']]}"
          f"; B lower at all 3: {z['B_lower_all']}; rho = 0 test p {z['rho_zero_p_two_sided']:.3f}")
        P(f"  SD_d 95% chi2 [{z['SD_d_95ci_chi2'][0]:.5f}, {z['SD_d_95ci_chi2'][1]:.5f}]; sigma_seed [{z['sigma_seed_95ci_chi2'][0]:.5f}, {z['sigma_seed_95ci_chi2'][1]:.5f}]")
        P("  MDE % paired|unpaired [ub80 k2]: " + "  ".join(f"k{k} {v['paired_rel']:.3%}|{v['unpaired_rel']:.3%}" for k, v in z["MDE"].items())
          + f"  [{z['MDE']['2']['paired_rel_ub80']:.3%}|{z['MDE']['2']['unpaired_rel_ub80']:.3%}]")
        P(f"  MDE bpb k1 {z['MDE']['1']['paired']:.5f}|{z['MDE']['1']['unpaired']:.5f}, k2 {z['MDE']['2']['paired']:.5f}|{z['MDE']['2']['unpaired']:.5f}")
        P(f"  k_needed paired {z['k_needed_paired']} unpaired {z['k_needed_unpaired']}")
        s = res["noise20_s23"][m]
        P(f"  seeds 2..3 (df 1): d {[round(x, 5) for x in s['d']]} mean {s['mean_d']:+.5f} SD_d {s['SD_d']:.5f} ({s['SD_d_rel']:.3%}) t {s['t']:+.3f} p {s['p_two_sided']:.4f} "
          f"CI [{s['ci95_d'][0]:+.4f}, {s['ci95_d'][1]:+.4f}]")
        z5 = Z5[m]
        P(f"  5M from raw (8 pairs): SD_d {z5['SD_d']:.5f} ({z5['SD_d_rel']:.3%}), sqrt2 sigma_seed {math.sqrt(2) * z5['sigma_seed_rel']:.3%}, rho {z5['rho']:.3f}")
    P("\nRULE 4:")
    for m, cl in r4.items():
        for c, v in cl.items():
            P(f"  {m:9s} {c:33s} 20M {v['rel20']:.3%} 5M {v['rel5']:.3%} -> {v['used']} df {v['df']}; MDE " + " ".join(f"k{k} {x:.3%}" for k, x in v["MDE_rel"].items())
              + f" (k1 {v['MDE_bpb_k1_at_A20_mean']:.4f} bpb); k_needed {v['k_needed']} k {v['k']} underpowered {v['underpowered']}; "
              f"ratio 20/5 {v['ratio_20_over_5']:.2f} [{v['ratio_95ci_F'][0]:.2f}, {v['ratio_95ci_F'][1]:.2f}]")
    P("\nE2 label input:", json.dumps(res["e2_label"]))
    P("averaging (1 - SD_d(M)/SD_d(F)):", json.dumps({k: round(v, 4) for k, v in res["averaging"].items()}))
    P("CRN ratio >= 2:", json.dumps({k: round(v, 2) for k, v in res["crn_ge2"].items()}))
    P("secondary rel SD_d: " + ", ".join(f"{m} {Z[m]['SD_d_rel']:.3%}" for m in ("CHAT user", "dolly", "CHAT assistant", "PROSE wikimedia")))
    P("rule 3 on F(PROSE) if it were a 20M screen:", json.dumps(res["rule3_prose_if_screen"]))


if __name__ == "__main__":
    main()
