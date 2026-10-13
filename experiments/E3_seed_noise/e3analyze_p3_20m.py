"""E3 Part 3 at 20M statistics (notes.txt ADDENDUM 2026-10-12, PART 3 AT 20M) on the copied records
out/e3_20m_A_{s1_P3,init2..5,data2..5} and logs/part3_20m/. Stdlib only. Written and committed before any value exists.
  python e3analyze_p3_20m.py [--checks-only]   writes results_part3_20m.json and tables_part3_20m.txt beside this file
e3analyze_p3.py's method at 20M (its family() and rule1(), imported unchanged; values by e3analyze_p2.py's value(): final
15,259, M over 14,945, 15,250, 15,259, the D6 tail): INIT = s1_P3 and init2..5 (seeds 1..5, data_seed 1), DATA = s1_P3
and data2..5 (seed 1, data_seeds 1..5); sigma_init, sigma_data (df n - 1) as fractions of the family mean, rule 1's
references sqrt(2) x each with MDE_u, k_needed and rule 2's k at this size. RULE 4 AT 20M (registered in the addendum):
per class, the larger of the 20M reference and the 5M one (results_part3.json), relative, with the df of the one used.
Descriptive, no rule: each 20M family SD against its 5M one (95% F interval, df n - 1 and 4) and against Part 2's 20M
sigma_seed (df n - 1 and 2); sqrt(sigma_init^2 + sigma_data^2) / that sigma_seed; s1_P3 beside Part 2's e3_20m_A_s1.
A GAP or DIVERGED run leaves its family; fewer than 4 runs: no SD, and this part sets no 20M reference for that class."""
from __future__ import annotations
import json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3add1_lib as L  # noqa: E402
import e3analyze_p2 as P2  # noqa: E402  (registered, unchanged: value() at 20M)
E, P3 = L.E, L.P3

SUB, RUNS, SHARED, P2S1 = "part3_20m", L.P20, "e3_20m_A_s1_P3", "e3_20m_A_s1"
FAM = {"INIT": [SHARED] + [f"e3_20m_A_init{k}" for k in range(2, 6)],
       "DATA": [SHARED] + [f"e3_20m_A_data{k}" for k in range(2, 6)]}
SEEDS = {SHARED: (1, 1), **{f"e3_20m_A_init{k}": (k, 1) for k in range(2, 6)}, **{f"e3_20m_A_data{k}": (1, k) for k in range(2, 6)}}
CLASS = P3.CLASS
KS, DELTAS = (1, 2, 3, 5), (0.005, 0.01, 0.02)


def checks(runs, p2s1):
    out, add = [], lambda name, ok, det="": out.append({"check": name, "ok": bool(ok), "detail": det})
    ok, q = L.queue_new(SUB)
    add("queue log starts with Part 3's copy, byte for byte", ok)
    pb = os.path.join(HERE, "logs", "part1b", "queue_e3.txt")
    add("queue log continues logs/part1b/'s copy byte for byte", os.path.exists(pb) and open(
        os.path.join(HERE, "logs", SUB, "queue_e3.txt")).read().startswith(open(pb).read()))
    L.code_checks(add, SUB, q, L.P1B + RUNS)
    tail = [ln for ln in q if ln.strip()][-2:]
    add("registered order (Part 1b, then init2 data2 init3 data3 s1_P3 init4 data4 init5 data5), each mark once, a clean end",
        L.order(q, L.P1B + RUNS) == L.P1B + RUNS and sum(ln.endswith(" MARK E3 PART 1B DONE") for ln in q) == 1
        and sum(ln.endswith(" MARK E3 PART 3 AT 20M DONE") for ln in q) == 1 and len(tail) == 2
        and tail[0].endswith(" MARK E3 PART 3 AT 20M DONE") and tail[1].endswith(f" plan finished: {L.PLAN}"))
    prereg = L.sha256(os.path.join(HERE, "configs", "prereg.yaml"))
    for n in RUNS:
        bad = ["GAP or DIVERGED"] if n not in runs else L.run_checks(runs[n], n, *SEEDS[n], "20m", q, prereg)
        add(f"{n}: FIXED hashes, settings, one clean run", not bad, "; ".join(bad))
    tails = {n: [P3.stream(t) for t in runs[n]["tail"]] for n in runs}
    add("split, INIT: each init run drew s1_P3's stream (drawn, dropped, tokens, sup_tokens equal at all 31 tail log steps "
        "and the end)", SHARED in runs and all(tails[n] == tails[SHARED] and P3.drawn(runs[n]["recs"][-1]) == P3.drawn(
            runs[SHARED]["recs"][-1]) for n in FAM["INIT"] if n in runs))
    e = L.ends(runs, FAM["DATA"])
    add("split, DATA: every data run drew another stream (end records pairwise different)",
        len(set(e)) == len(e) == sum(n in runs for n in FAM["DATA"]))
    add("s1_P3 drew Part 2's e3_20m_A_s1 stream (all 31 tail steps and the end)", SHARED in runs and tails[SHARED] == [
        P3.stream(t) for t in p2s1["tail"]] and P3.drawn(runs[SHARED]["recs"][-1]) == P3.drawn(p2s1["recs"][-1]))
    return out


def ref_from(rel, df):
    f = E.factor(df)
    kn = {f"{dl:.3%}": next(k for k in range(1, 10**6) if f * rel / math.sqrt(k) <= dl) for dl in DELTAS}
    return {"df": df, "factor": f, "ref_rel": rel, "MDE_rel": {str(k): f * rel / math.sqrt(k) for k in KS},
            "k_needed": kn, "k_rule2": {dl: max(2, k) for dl, k in kn.items()}, "underpowered": {dl: k > 5 for dl, k in kn.items()}}


def fci(ratio, d1, d2):
    vr = ratio ** 2
    return [math.sqrt(vr / P3.f_ppf(0.975, d1, d2)), math.sqrt(vr / P3.f_ppf(0.025, d1, d2))]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = {n: E.load(n) for n in RUNS if not L.gapped(n)}
    p2s1 = E.load(P2S1)
    ck = checks(runs, p2s1)
    if "--checks-only" in argv:
        print(json.dumps(ck, indent=1)); return 0
    r3 = json.load(open(os.path.join(HERE, "results_part3.json")))
    r2 = json.load(open(os.path.join(HERE, "results_part2.json")))
    res = {"part": "E3 Part 3 at 20M, arm A20 (3e-3, r 1), 500M slots, full WSD: INIT (seeds 1..5, data_seed 1), DATA (seed 1, data_seeds 1..5)",
           "analyzer": {"python": sys.version.split()[0], **{a: L.sha256(os.path.join(HERE, a)) for a in L.ANALYZERS}},
           "checks": ck, "expected_failures": L.EXPECTED_FAIL, "unexpected_failures": L.fails(ck),
           "gaps": [n for n in RUNS if n not in runs], "commit": L.code_checks(lambda *a: None, SUB, L.queue_new(SUB)[1], L.P1B + RUNS),
           "quantiles": {"factor_df4": E.factor(4), "ub80_df4": E.ub(4), "F_0.025_df4_4": P3.f_ppf(0.025, 4, 4),
                         "F_0.975_df4_4": P3.f_ppf(0.975, 4, 4), "F_0.025_df4_2": P3.f_ppf(0.025, 4, 2), "F_0.975_df4_2": P3.f_ppf(0.975, 4, 2)},
           "values": {}, "families": {}, "rule1_20M": {}, "rule4_20M": {}, "compare_5M": {}, "compare_part2_sigma_seed": {},
           "replay_vs_part2_A_s1": {}}
    for m in E.METRICS:
        vals = {fam: {n: P2.value(runs[n], m) for n in names if n in runs} for fam, names in FAM.items()}
        res["values"][m] = vals
        fz = {fam: P3.family(list(v.values())) if len(v) >= 4 else None for fam, v in vals.items()}
        res["families"][m] = fz
        res["rule1_20M"][m] = {CLASS[f]: P3.rule1(z) if z else "no SD (fewer than 4 runs)" for f, z in fz.items()}
        s2 = r2["noise_20M_500M"][m]["sigma_seed_rel"]
        r4, c5, c2 = {}, {}, {"sigma_seed_rel_part2_df2": s2}
        for fam, z in fz.items():
            z5 = r3["families"][m][fam]
            if not z:
                r4[CLASS[fam]] = "no 20M SD (fewer than 4 runs): this part sets no 20M reference for the class"; continue
            rel20, rel5 = math.sqrt(2) * z["rel"], math.sqrt(2) * z5["rel"]
            use20 = rel20 >= rel5
            r4[CLASS[fam]] = {"rel_20M": rel20, "df_20M": z["df"], "rel_5M": rel5, "df_5M": z5["df"], "used": "20M" if use20 else "5M",
                              **ref_from(rel20 if use20 else rel5, z["df"] if use20 else z5["df"])}
            c5[fam] = {"ratio": z["rel"] / z5["rel"], "ratio_95ci_F": fci(z["rel"] / z5["rel"], z["df"], z5["df"])}
            c2[fam] = {"ratio": z["rel"] / s2, "ratio_95ci_F": fci(z["rel"] / s2, z["df"], 2)}
        if fz["INIT"] and fz["DATA"]:
            c2["sqrt_init2_plus_data2_over_seed"] = math.hypot(fz["INIT"]["rel"], fz["DATA"]["rel"]) / s2
        res["rule4_20M"][m], res["compare_5M"][m], res["compare_part2_sigma_seed"][m] = r4, c5, c2
        if SHARED in runs:
            v, w = P2.value(runs[SHARED], m), P2.value(p2s1, m)
            res["replay_vs_part2_A_s1"][m] = {"s1_P3": v, "part2_A_s1": w, "diff": v - w, "rel": (v - w) / w}
    res["not_run"] = {"20M replays (sigma_rep at 20M)": "not in the design"}
    json.dump(res, open(os.path.join(HERE, "results_part3_20m.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(HERE, "tables_part3_20m.txt"), "w").write(tables(res))
    print(tables(res)); return 0


def tables(r):
    f = [c for c in r["checks"] if not c["ok"]]
    T = ["E3 PART 3 AT 20M TABLES (e3analyze_p3_20m.py; 20M arm A20 3e-3 r 1, 500M slots, full WSD; INIT = seeds 1..5 at "
         f"data_seed 1, DATA = data_seeds 1..5 at seed 1; e3_20m_A_s1_P3 is member 1 of both; commit {(r['commit'] or 'unknown')[:12]})", "",
         "CHECKS: " + ", ".join(f"{c['check']} FAILED ({c['detail']})" for c in f) +
         f"  [{len(r['checks']) - len(f)}/{len(r['checks'])} pass; unexpected {len(r['unexpected_failures'])}: {r['unexpected_failures']}]",
         f"GAPS: {r['gaps'] or 'none'}", "", "FAMILY SDs at 20M (df n - 1), rel = fraction of the family mean [95% chi-square interval]"]
    for m, fz in r["families"].items():
        T.append(f"{m:20s} " + "  ".join(f"{fam} " + (f"{z['rel']:.3%} [{z['rel_ci95'][0]:.3%}, {z['rel_ci95'][1]:.3%}] n {z['n']}"
                                                    if z else "no SD") for fam, z in fz.items()))
    T += ["", "RULE 4 AT 20M: reference = larger of sqrt2 x the 20M and the 5M family SD (rel), df of the one used; MDE k = 1, 2, 3, 5; "
          "k_needed 0.5% / 1% / 2%; rule 2's k"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)"):
        for cls, v in r["rule4_20M"][m].items():
            T.append(f"{m:9s} {cls:22s} " + (v if isinstance(v, str) else
                     f"20M {v['rel_20M']:.3%} 5M {v['rel_5M']:.3%} -> {v['used']} (df {v['df']}, factor {v['factor']:.3f}); MDE "
                     + " ".join(f"k{k} {x:.3%}" for k, x in v["MDE_rel"].items()) + "; k_needed "
                     + "/".join(str(k) for k in v["k_needed"].values()) + "; k " + "/".join(str(k) for k in v["k_rule2"].values())))
    T += ["", "AGAINST 5M (Part 3) AND PART 2's 20M sigma_seed (descriptive): ratio of relative SDs [95% F interval]"]
    for m in r["families"]:
        c5, c2 = r["compare_5M"][m], r["compare_part2_sigma_seed"][m]
        T.append(f"{m:20s} " + "; ".join(f"{fam} /5M {c5[fam]['ratio']:.2f} [{c5[fam]['ratio_95ci_F'][0]:.2f}, {c5[fam]['ratio_95ci_F'][1]:.2f}]"
                                         f" /seed20 {c2[fam]['ratio']:.2f}" for fam in ("INIT", "DATA") if fam in c5)
                 + (f"; sum/seed20 {c2['sqrt_init2_plus_data2_over_seed']:.2f}" if "sqrt_init2_plus_data2_over_seed" in c2 else ""))
    T += ["", "PER-RUN VALUES"]
    for m in ("F(CHAT)", "F(PROSE)"):
        for fam, v in r["values"][m].items():
            T.append(f"{m:9s} {fam:4s} " + "  ".join(f"{n.replace('e3_20m_A_', '')} {x:.5f}" for n, x in v.items()))
    for m in ("F(CHAT)", "F(PROSE)"):
        x = r["replay_vs_part2_A_s1"].get(m)
        if x:
            T.append(f"s1_P3 beside Part 2's A_s1 (another commit, descriptive) {m}: {x['s1_P3']:.5f} vs {x['part2_A_s1']:.5f}, {x['diff']:+.5f}")
    T += ["", f"Quantiles: {json.dumps({k: round(v, 4) for k, v in r['quantiles'].items()})}", f"Not run: {json.dumps(r['not_run'])}", ""]
    return "\n".join(T)


if __name__ == "__main__":
    sys.exit(main())
