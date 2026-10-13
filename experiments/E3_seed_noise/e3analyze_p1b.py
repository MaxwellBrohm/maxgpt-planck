"""E3 Part 1b statistics (notes.txt ADDENDUM 2026-10-12, PART 1b) on the copied records out/e3_5m_A_s12..s21 and
logs/part1b/. Stdlib only. Written and committed before any Part 1b value exists.
  python e3analyze_p1b.py [--checks-only]    writes results_part1b.json and tables_part1b.txt beside this file
Per metric of e3analyze.METRICS (value() unchanged): Part 1b alone (n 10, df 9) with e3analyze_p3.family(): mean, SD as
a fraction of the mean, 95% chi-square interval, 80% upper bound. POOLED with Part 1's arm A (seeds 1..8, df 7):
sigma_A = sqrt((SS_1 + SS_1b) / (n_1 + n_1b - 2)), each part's squares about its own mean, df 16, as a fraction of the
mean of all 18; beside it, unchanged, Part 1's sigma_seed (arms A and B, df 7; results.json) and Part 1's arm A alone.
Unpaired MDE from sqrt(2) x sigma_A with factor(16), k = 1, 2, 3, 5, and k_needed for 0.5, 1, 2% (rule 2's k).
LEVEL (descriptive): Part 1b mean minus Part 1 arm A mean, t on sigma_A (df 16); Part 1b mean minus the screens' BASE
(seeds 101..105, eval.induction on; the five bpb.jsonl files pinned below), t on the pooled SD of Part 1b and BASE (df 13).
A GAP or DIVERGED run leaves Part 1b; with fewer than 6 runs left Part 1b gives no SD (Part 1's numbers stand alone)."""
from __future__ import annotations
import json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3add1_lib as L  # noqa: E402
E, P3 = L.E, L.P3

SUB, RUNS, MIN_RUNS = "part1b", L.P1B, 6
SEED = {n: int(n.rsplit("_s", 1)[1]) for n in RUNS}
P1A = [f"e3_5m_A_s{s}" for s in range(1, 9)]
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = {  # SCREENS BASE at seeds 101..105 (hook on), read only: path under the repository root, sha256
    "base_s101": ("experiments/S001_attn_gate/out/base_s101/bpb.jsonl", "19f54e630fe30f67b87ae4e75aca96392922659593f7996a1408c2ca1b7808e6"),
    "base_s102": ("experiments/S001_attn_gate/out/base_s102/bpb.jsonl", "b5f9df39a045385a38e5034249f6ebeee397378de37e06c97771a2e08ff7b6ed"),
    "base_s103": ("experiments/screens/base_out/base_s103/bpb.jsonl", "241a38ba9cb3a6bab46553cb3f2ad66bfb1642c90bab2e9bf07c5ff41a38d5da"),
    "base_s104": ("experiments/screens/base_out/base_s104/bpb.jsonl", "b9c10ff0a415a6fb71947a4d6589e39ab6b23fcbec05753d5ab35c6572ca42c8"),
    "base_s105": ("experiments/screens/base_out/base_s105/bpb.jsonl", "ec0ca006ad3ef3e254bcf9220ed19654b85a6e5aa6e5475c8f8da7810e25549b")}
KS, DELTAS = (1, 2, 3, 5), (0.005, 0.01, 0.02)


def checks(runs, p1):
    out, add = [], lambda name, ok, det="": out.append({"check": name, "ok": bool(ok), "detail": det})
    ok, q = L.queue_new(SUB)
    add("Part 1b queue log starts with Part 3's copy, byte for byte", ok)
    L.code_checks(add, SUB, q, RUNS)
    seq = L.order(q, RUNS)
    mk = [i for i, ln in enumerate(q) if ln.endswith(" MARK E3 PART 1B DONE")]
    dn = [i for i, ln in enumerate(q) if any(ln.endswith(f" done {n} rc 0") for n in RUNS)]
    add("registered order s12 .. s21, then MARK E3 PART 1B DONE once, after the last of them",
        seq == RUNS and len(mk) == 1 and bool(dn) and mk[0] > max(dn), " ".join(seq))
    prereg = L.sha256(os.path.join(HERE, "configs", "prereg.yaml"))
    for n in RUNS:
        bad = ["GAP or DIVERGED"] if n not in runs else L.run_checks(runs[n], n, SEED[n], None, "5m", q, prereg)
        add(f"{n}: FIXED hashes, settings, one clean run", not bad, "; ".join(bad))
    e1b, e1 = L.ends(runs, RUNS), L.ends(p1, P1A)
    add("streams: the Part 1b runs drew pairwise different streams, none equal to a Part 1 arm A seed's (end counts)",
        len(set(e1b)) == len(e1b) and not set(e1b) & set(e1) and len(set(e1)) == 8)
    add("the screens' BASE records (seeds 101..105) as pinned", all(
        L.sha256(os.path.join(ROOT, p)) == h for p, h in BASE.values()))
    return out


def pooled(groups):
    ss = sum(sum((x - st.fmean(g)) ** 2 for x in g) for g in groups)
    df = sum(len(g) - 1 for g in groups); sd = math.sqrt(ss / df); mean = st.fmean([x for g in groups for x in g])
    lo, hi = math.sqrt(df / E.chi2_ppf(0.975, df)), math.sqrt(df / E.chi2_ppf(0.025, df))
    return {"n": sum(len(g) for g in groups), "df": df, "mean": mean, "sd": sd, "rel": sd / mean,
            "rel_ci95": [sd * lo / mean, sd * hi / mean], "rel_ub80": sd * E.ub(df) / mean}


def unpaired(rel, df):
    f, ref = E.factor(df), math.sqrt(2) * rel
    kn = {f"{dl:.3%}": next(k for k in range(1, 10**6) if f * ref / math.sqrt(k) <= dl) for dl in DELTAS}
    return {"df": df, "factor": f, "ref_rel": ref, "MDE_rel": {str(k): f * ref / math.sqrt(k) for k in KS},
            "k_needed": kn, "k_rule2": {dl: max(2, k) for dl, k in kn.items()}, "underpowered": {dl: k > 5 for dl, k in kn.items()}}


def level(a, b, sd, df):
    """mean(b) - mean(a), t on a pooled SD with df, two-sided p, 95% CI (descriptive)."""
    diff, se = st.fmean(b) - st.fmean(a), sd * math.sqrt(1 / len(a) + 1 / len(b))
    t, tq = diff / se, E.t_ppf(0.975, df)
    return {"diff": diff, "t": t, "df": df, "p_two_sided": 2 * (1 - E.t_cdf(abs(t), df)), "ci95": [diff - tq * se, diff + tq * se],
            "n_a": len(a), "n_b": len(b), "mean_a": st.fmean(a), "mean_b": st.fmean(b)}


def base_value(name, m):
    s, sp, kind = E.METRICS[m]
    rows = [json.loads(x) for x in open(os.path.join(ROOT, BASE[name][0]))]
    return E.value({"bpb": rows, "tail": []}, m) if kind != "T" else None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = {n: E.load(n) for n in RUNS if not L.gapped(n)}
    p1 = {n: E.load(n) for n in P1A}
    ck = checks(runs, p1)
    if "--checks-only" in argv:
        print(json.dumps(ck, indent=1)); return 0
    r1 = json.load(open(os.path.join(HERE, "results.json")))
    res = {"part": "E3 Part 1b, 5M arm A (3e-3, r 1), 250M slots, full WSD, seeds 12..21, pooled with Part 1's arm A (seeds 1..8)",
           "analyzer": {"python": sys.version.split()[0], **{a: L.sha256(os.path.join(HERE, a)) for a in L.ANALYZERS}},
           "checks": ck, "expected_failures": L.EXPECTED_FAIL, "unexpected_failures": L.fails(ck),
           "gaps": [n for n in RUNS if n not in runs], "commit": L.code_checks(lambda *a: None, SUB, L.queue_new(SUB)[1], RUNS),
           "quantiles": {f"factor_df{d}": E.factor(d) for d in (7, 9, 13, 16)} | {f"ub80_df{d}": E.ub(d) for d in (9, 16)}
           | {f"t975_df{d}": E.t_ppf(0.975, d) for d in (13, 16)},
           "values": {}, "part1b": {}, "part1_A": {}, "pooled": {}, "part1_sigma_seed": {}, "pooled_over_part1_sigma_seed": {},
           "unpaired_pooled": {}, "unpaired_part1": {}, "level_vs_part1": {}, "level_vs_base": {}}
    for m in E.METRICS:
        v1b = {n: E.value(runs[n], m) for n in RUNS if n in runs}
        v1 = [E.value(p1[n], m) for n in P1A]
        vb = {n: base_value(n, m) for n in BASE} if E.METRICS[m][2] != "T" else {}
        res["values"][m] = {"part1b": v1b, "part1_A": dict(zip(P1A, v1)), "base": vb}
        res["part1_A"][m] = P3.family(v1)
        s1 = r1["noise_5M_250M"][m]
        res["part1_sigma_seed"][m] = {"sigma_seed": s1["sigma_seed"], "sigma_seed_rel": s1["sigma_seed_rel"], "df": s1["df"]}
        res["unpaired_part1"][m] = unpaired(s1["sigma_seed_rel"], s1["df"])
        if len(v1b) < MIN_RUNS:
            for k in ("part1b", "pooled", "pooled_over_part1_sigma_seed", "unpaired_pooled", "level_vs_part1", "level_vs_base"):
                res[k][m] = f"not computed: {len(v1b)} Part 1b runs (fewer than {MIN_RUNS})"
            continue
        b = list(v1b.values())
        res["part1b"][m] = P3.family(b)
        pz = pooled([v1, b])
        res["pooled"][m] = pz
        res["pooled_over_part1_sigma_seed"][m] = pz["rel"] / s1["sigma_seed_rel"]
        res["unpaired_pooled"][m] = unpaired(pz["rel"], pz["df"])
        res["level_vs_part1"][m] = level(v1, b, pz["sd"], pz["df"])
        if vb:
            bz = pooled([b, list(vb.values())])
            res["level_vs_base"][m] = level(list(vb.values()), b, bz["sd"], bz["df"])
    res["not_run"] = {"Part 1b arm B": "not in the design (arm A only): SD_d is not re-estimated"}
    json.dump(res, open(os.path.join(HERE, "results_part1b.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(HERE, "tables_part1b.txt"), "w").write(tables(res))
    print(tables(res)); return 0


def tables(r):
    f = [c for c in r["checks"] if not c["ok"]]
    L_ = ["E3 PART 1b TABLES (e3analyze_p1b.py; 5M arm A 3e-3 r 1, 250M slots, full WSD; Part 1b = seeds 12..21 at commit "
          f"{(r['commit'] or 'unknown')[:12]}; Part 1 arm A = seeds 1..8 at e39112a)", "",
          "CHECKS: " + ", ".join(f"{c['check']} FAILED ({c['detail']})" for c in f) +
          f"  [{len(r['checks']) - len(f)}/{len(r['checks'])} pass; unexpected {len(r['unexpected_failures'])}: {r['unexpected_failures']}]",
          f"GAPS: {r['gaps'] or 'none'}", "",
          "SDs as fractions of the mean [95% chi-square interval]: Part 1b alone (df 9), Part 1 arm A alone (df 7), POOLED arm A "
          "(df 16), Part 1's sigma_seed (A and B, df 7), pooled / Part 1 sigma_seed"]
    for m in r["part1_A"]:
        z, a, s = r["part1b"][m], r["part1_A"][m], r["part1_sigma_seed"][m]
        if isinstance(z, str):
            L_.append(f"{m:20s} {z}"); continue
        p = r["pooled"][m]
        L_.append(f"{m:20s} 1b {z['rel']:.3%} [{z['rel_ci95'][0]:.3%}, {z['rel_ci95'][1]:.3%}]  1A {a['rel']:.3%}  pooled "
                  f"{p['rel']:.3%} [{p['rel_ci95'][0]:.3%}, {p['rel_ci95'][1]:.3%}] ub80 {p['rel_ub80']:.3%}  sigma_seed "
                  f"{s['sigma_seed_rel']:.3%}  ratio {r['pooled_over_part1_sigma_seed'][m]:.2f}")
    L_ += ["", "UNPAIRED MDE (sqrt2 x SD), % of the mean, k = 1, 2, 3, 5; k_needed 0.5% / 1% / 2%: pooled (df 16) | Part 1 sigma_seed (df 7)"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)"):
        u, w = r["unpaired_pooled"][m], r["unpaired_part1"][m]
        if isinstance(u, str):
            continue
        L_.append(f"{m:9s} pooled " + " ".join(f"k{k} {x:.3%}" for k, x in u["MDE_rel"].items()) + " k_needed " +
                  "/".join(str(k) for k in u["k_needed"].values()) + " | part1 " + " ".join(f"k{k} {x:.3%}" for k, x in w["MDE_rel"].items())
                  + " k_needed " + "/".join(str(k) for k in w["k_needed"].values()))
    L_ += ["", "LEVEL (descriptive, no rule): mean(Part 1b) - mean(other), t, p, 95% CI"]
    for m in r["part1_A"]:
        for k, lab in (("level_vs_part1", "Part 1 arm A"), ("level_vs_base", "BASE 101-105")):
            x = r[k].get(m)
            if isinstance(x, dict):
                L_.append(f"{m:20s} vs {lab:13s} {x['mean_b']:.5f} - {x['mean_a']:.5f} = {x['diff']:+.5f}  t {x['t']:+.2f} (df {x['df']})  "
                          f"p {x['p_two_sided']:.4f}  [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]")
    L_ += ["", "PER-RUN VALUES F(CHAT), F(PROSE)"]
    for m in ("F(CHAT)", "F(PROSE)"):
        L_.append(f"{m:9s} " + "  ".join(f"{n.replace('e3_5m_A_', '')} {x:.5f}" for n, x in r["values"][m]["part1b"].items()))
    L_ += ["", f"Quantiles: {json.dumps({k: round(v, 4) for k, v in r['quantiles'].items()})}", f"Not run: {json.dumps(r['not_run'])}", ""]
    return "\n".join(L_)


if __name__ == "__main__":
    sys.exit(main())
