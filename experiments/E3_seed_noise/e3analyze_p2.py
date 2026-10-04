"""E3 Part 2 statistics (20M; notes.txt STATISTICS, fixed 2026-09-26, and PART 2 ANALYSIS, DEVIATIONS, 2026-10-04)
on the copied records in out/e3_20m_*. Stdlib only.
  python e3analyze_p2.py [--checks-only]   writes results_part2.json and tables_part2.txt beside this file (D5)
Imports e3analyze.py unchanged (noise, factor, ub, the t and chi-square quantiles, load, METRICS, FIXED, FIXED_D4) and
changes only what the size changes: 15,259 steps, checkpoints 14,945 / 15,250 / 15,259, decay from 12,207, 20,001,511
params, seeds 1..3, the Part 2 queue start (3c102f7) and its code list in logs/part2/. Part 1's results.json and
tables.txt are read, never written."""
from __future__ import annotations
import hashlib, json, math, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3analyze as E  # noqa: E402  (the registered Part 1 analyzer, unchanged)

SEEDS, TOTAL, CKPTS, KS, DELTAS = range(1, 4), 15259, [14945, 15250, 15259], (1, 2, 3, 5), (0.005, 0.01, 0.02)
SCHED = {"decay_start": 12207, "mode": "full", "total_steps": TOTAL, "warmup_steps": 76}
N_PARAMS, LOGS = 20001511, os.path.join(HERE, "logs", "part2")
QUEUE_COMMIT, PLAN_SHA16, CODE_LIST = "3c102f75dc68a21c1e7a4f36a187850f4b07a1cd", "6b9962074b7e996d", "code_sha256_20261003_004304.txt"
D4_DIGEST16 = "56b4a68f38d43cc8"   # notes D4: harness/ + data_prep/ .py digest at the Part 2 commit (100 files)
CONFIG_SHA = {  # notes PART 2 QUEUED
    "e3_20m_A_s1": "83e28d5f80ddf646fdbf593e1949aa32664f3adb688667fdd063a211d78c580f",
    "e3_20m_A_s2": "aa1be48b280708985625ee2efcdfaa74037eb31c6d555a81da26dd89a5507475",
    "e3_20m_A_s3": "a6e329e86163d580940df1bad7379e42eaf2b9d02e46e7fd9c9a3bd9d7e0c2d7",
    "e3_20m_B_s1": "a2843208b338f67cf352264330c326092198dde65414da28c9ac6572e6a4ac4b",
    "e3_20m_B_s2": "a952ce4b1fcc9b0d3b1b1878d0346f6b2c7174b651879243ae236ed4586cdb62",
    "e3_20m_B_s3": "55908528a0f86c38a9f0a7563401be0d4a84c31bf101e50951b94e5698764b7f"}
LR = {"A": 0.003, "B": 0.0015}
P1_RESULTS_SHA = "1f017ecb1e6d14dd4f676c90ca703c20e9395b217fc841f83fffcb93e57994a2"
P1_TABLES_SHA = "100a55ab58bd8b40129f23d659963869128aad254f4b08cad570c08147d5e6ec"
E2_20M_500M = {"g1 (3e-3)": 0.9726, "g0.5 (1.5e-3)": 0.9749}   # E2 results.json q2 final_bpb_20m b500M chat, 4 decimals
TAIL_STEPS = list(range(14960, 15251, 10)) + [15259]           # D6
RUNS = [f"e3_20m_{a}_s{s}" for s in SEEDS for a in "AB"]


def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def value(run, metric):
    s, sp, kind = E.METRICS[metric]
    if kind == "T":   # D6: every logged step inside the last 2% (31 records)
        return st.fmean(r["loss"] for r in run["tail"] if r["step"] > TOTAL - 0.02 * TOTAL)
    v = {r["step"]: r["bpb"] for r in run["bpb"] if r["set"] == s and r["split"] == sp}
    assert sorted(v) == CKPTS, sorted(v)
    return v[TOTAL] if kind == "F" else st.fmean(v.values())


def stream(rec):
    return {k: v for k, v in rec.items() if k.startswith(("drawn_", "dropped_")) or k in ("tokens", "sup_tokens", "step")}


def checks(runs):
    out, add = [], lambda name, ok, det="": out.append({"check": name, "ok": bool(ok), "detail": det})
    code = open(os.path.join(LOGS, CODE_LIST)).read().split("\n")
    rows = sorted((ln.split("  ")[1], ln.split("  ")[0]) for ln in code if ln.strip())
    hp = [f"{h}  {p}\n" for p, h in rows if p.startswith(("harness/", "data_prep/")) and p.endswith(".py")]
    dig = hashlib.sha256("".join(sorted(hp, key=lambda s: s.split("  ")[1].encode())).encode()).hexdigest()
    add("code digest harness+data_prep .py == FIXED 2 (92deaef0)", dig == E.FIXED["code_digest"], f"{dig[:16]} over {len(hp)} files")
    add("code digest == D4's Part 2 commit digest (56b4a68f, 100 files)", dig[:16] == D4_DIGEST16 and len(hp) == 100, f"{dig[:16]} over {len(hp)} files")
    fh = dict(rows)
    for p, h in E.FIXED["files"].items():
        got, d4 = fh.get(p, "")[:16], E.FIXED_D4.get(p)
        add(f"{p} == FIXED 2", got == h or got == d4, fh.get(p, "missing")[:16] + (" (D4)" if got == d4 else ""))
    add("code list has the registered analyzer (e3analyze.py) and the 6 Part 2 configs as on disk",
        fh.get("experiments/E3_seed_noise/e3analyze.py") == sha256(os.path.join(HERE, "e3analyze.py"))
        and all(fh.get(f"experiments/E3_seed_noise/configs/{n}.yaml") == CONFIG_SHA[n] for n in RUNS))
    qt = open(os.path.join(LOGS, "queue_e3.txt")).read(); q = qt.split("\n")
    add("Part 2 queue log starts with Part 1's copy, byte for byte", qt.startswith(open(os.path.join(HERE, "logs", "queue_e3.txt")).read()))
    i0 = next(i for i, ln in enumerate(q) if "plan finished: part1.txt" in ln) + 1
    q2 = q[i0:]
    starts = [ln for ln in q2 if "queue start" in ln]
    add("one Part 2 queue start, plan part2.txt 6b996207, commit 3c102f7, code list from that start",
        len(starts) == 1 and f"plan part2.txt (sha256 {PLAN_SHA16})" in starts[0] and QUEUE_COMMIT in starts[0]
        and starts[0][:19].replace("-", "").replace(":", "").replace(" ", "_") == CODE_LIST[12:27], f"{len(starts)} start(s)")
    add("plans/part2.txt on disk has the queued sha", sha256(os.path.join(HERE, "plans", "part2.txt")).startswith(PLAN_SHA16))
    trains = [ln.split("train ")[1].split(".yaml")[0] for ln in q2 if " train " in ln]
    add("registered order A1 B1 A2 B2 A3 B3, then MARK E3 PART 2 DONE", trains == RUNS and sum("MARK E3 PART 2 DONE" in ln for ln in q2) == 1, " ".join(trains))
    prereg = sha256(os.path.join(HERE, "configs", "prereg.yaml"))
    for name in RUNS:
        r, bad = runs[name], []
        arm, seed = name.split("_")[2], int(name[-1])
        ev = {(x["evalset_sha256"], x["tokenizer_sha256"], x["precision"], x["max_windows"]) for x in r["bpb"]}
        if ev != {(E.FIXED["evalset"], E.FIXED["tokenizer"], "fp32", None)}: bad.append(f"scorer/eval/tokenizer {ev}")
        if {x["run"] for x in r["bpb"]} != {name}: bad.append("bpb run field")
        per = {}
        for x in r["bpb"]: per.setdefault((x["set"], x["split"]), []).append(x["step"])
        if any(sorted(v) != CKPTS for v in per.values()) or len(per) < 13: bad.append(f"bpb steps {sorted({tuple(sorted(v)) for v in per.values()})}")
        pf = r["pf"]; cfg = os.path.join(HERE, "configs", name + ".yaml"); txt = open(cfg).read()
        if sha256(cfg) != pf["file_sha256"] or sha256(cfg) != CONFIG_SHA[name]: bad.append("config file sha")
        if not re.search(rf"lr: {LR[arm]}, embed_lr: {LR[arm]}, scalar_lr: {LR[arm]}\}}", txt) or f"\nseed: {seed}\n" not in txt: bad.append("config LR/seed")
        if not (pf["ok"] and pf["refusals"] == [] and pf["prereg"]["committed"] and pf["prereg"]["sha256"] == prereg): bad.append("preflight ok/strict/prereg")
        if pf["engine"] != E.FIXED["engine"] or pf["engine_fixed"] != E.FIXED["engine_fixed"]: bad.append("engine")
        if pf["shares"] != E.FIXED["shares"] or pf["n_files"] != E.FIXED["n_files"]: bad.append("shares/files")
        s, e = r["recs"][0], r["recs"][-1]
        if (s["event"], e["event"]) != ("start", "end") or len(r["recs"]) != 2: bad.append("records")
        if s["config_sha256"] != pf["config_sha256"] or s["prereg_sha256"] != prereg or not s["prereg_committed"]: bad.append("config/prereg sha")
        if s["schedule"] != SCHED: bad.append("schedule")
        if (s["batch_tokens"], s["precision"], s["doc_attn"], s["optim_batched"], s["n_params"], s["resumed_from"]) != (32768, "bf16", "varlen", True, N_PARAMS, None): bad.append("run setting")
        if e["step"] != TOTAL or not math.isfinite(e["loss"]): bad.append("end step/loss")
        if [t["step"] for t in r["tail"]] != TAIL_STEPS or not all(math.isfinite(t["loss"]) for t in r["tail"]): bad.append("tail steps/finite")
        if stream(r["tail"][-1]) != stream(e): bad.append("tail end != end record stream")
        if sum(ln.endswith(f"train {name}.yaml --require-committed") for ln in q2) != 1 or sum(ln.endswith(f"done {name} rc 0") for ln in q2) != 1: bad.append("queue: not exactly one clean train")
        add(f"{name}: FIXED hashes, settings, one clean run", not bad, "; ".join(bad))
    drawn = lambda n: {k: v for k, v in runs[n]["recs"][-1].items() if k.startswith(("drawn_", "dropped_"))}  # noqa: E731
    add("CRN: A_s and B_s drew identical per-source tokens at every seed (end record)", all(drawn(f"e3_20m_A_s{s}") == drawn(f"e3_20m_B_s{s}") for s in SEEDS))
    add("CRN: A_s and B_s equal drawn, dropped, tokens, sup_tokens at all 31 tail log steps",
        all([stream(t) for t in runs[f"e3_20m_A_s{s}"]["tail"]] == [stream(t) for t in runs[f"e3_20m_B_s{s}"]["tail"]] for s in SEEDS))
    add("seeds differ: 3 distinct streams", len({json.dumps(drawn(f"e3_20m_A_s{s}"), sort_keys=True) for s in SEEDS}) == 3)
    add("Part 1 results.json and tables.txt unchanged (D5)", sha256(os.path.join(HERE, "results.json")) == P1_RESULTS_SHA
        and sha256(os.path.join(HERE, "tables.txt")) == P1_TABLES_SHA)
    return out


def f_ppf(q, d1, d2): return E.ppf(lambda x: E.betai(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2)), q, 0.0, 1e4)


def k_needed(f, rel, dl): return next(k for k in range(1, 10**6) if f * rel / math.sqrt(k) <= dl)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = {n: E.load(n) for n in RUNS}
    ck = checks(runs)
    if "--checks-only" in argv:
        print(json.dumps(ck, indent=1)); return 0
    p1 = json.load(open(os.path.join(HERE, "results.json")))
    N5 = p1["noise_5M_250M"]
    res = {"part": "E3 Part 2, 20M, 500M slots, full WSD, A20 (3e-3, r 1) vs B20 (1.5e-3, r 1), seeds 1..3, d = B - A",
           "analyzer": {"e3analyze_p2.py": "this file", "e3analyze.py sha256 (imported, unchanged)": sha256(os.path.join(HERE, "e3analyze.py")),
                        "python": sys.version.split()[0]},
           "part1_results_sha256": sha256(os.path.join(HERE, "results.json")), "checks": ck,
           "quantiles": {"factor_df2": E.factor(2), "factor_df1": E.factor(1), "factor_df7": E.factor(7), "ub80_df2": E.ub(2),
                         "ub80_df1": E.ub(1), "F_0.025_df2_7": f_ppf(0.025, 2, 7), "F_0.975_df2_7": f_ppf(0.975, 2, 7),
                         "chi2_0.025_df2": E.chi2_ppf(0.025, 2), "chi2_0.975_df2": E.chi2_ppf(0.975, 2)},
           "values": {}, "noise_20M_500M": {}, "noise_20M_500M_seeds2to3": {}}
    for m in E.METRICS:
        a = [value(runs[f"e3_20m_A_s{s}"], m) for s in SEEDS]; b = [value(runs[f"e3_20m_B_s{s}"], m) for s in SEEDS]
        res["values"][m] = {"A": a, "B": b, "d": [y - x for x, y in zip(a, b)]}
        res["noise_20M_500M"][m] = E.noise(a, b, st.fmean(a))
        res["noise_20M_500M_seeds2to3"][m] = E.noise(a[1:], b[1:], st.fmean(a[1:]))
    Z = res["noise_20M_500M"]
    res["averaging_20M_descriptive"] = {k: {"SD_d_F": Z[f"F({k})"]["SD_d"], "SD_d_M": Z[f"M({k})"]["SD_d"],
                                            "reduction": 1 - Z[f"M({k})"]["SD_d"] / Z[f"F({k})"]["SD_d"]} for k in ("CHAT", "PROSE")}
    rc = res["noise_20M_500M_seeds2to3"]["F(CHAT)"]
    res["e2_recheck_20M"] = {"test": "paired t on F(CHAT) d_s = B - A, seeds 2..3, df 1 (reported, moves nothing)", "d": res["values"]["F(CHAT)"]["d"][1:],
                             "mean_d": rc["mean_d"], "t": rc["t"], "p_two_sided": rc["p_two_sided"], "ci95_d": rc["ci95_d"],
                             "B_lower_and_p_lt_0.05": rc["mean_d"] < 0 and rc["p_two_sided"] < 0.05, "moves": "nothing (notes STATISTICS)"}
    res["crn_20M"] = {m: {"ratio": Z[m]["crn_ratio"], "ge_2": Z[m]["crn_ratio"] >= 2} for m in ("F(CHAT)", "F(PROSE)")}
    res["crn_20M"]["reading"] = ("ratio >= 2 on F at 20M: CRN mandatory for every screen (reading d)" if any(res["crn_20M"][m]["ge_2"] for m in ("F(CHAT)", "F(PROSE)"))
                                 else "both 20M ratios below 2: the 5M reading stands (CRN default, not mandatory)")
    r4, cmp_ = {}, {}
    for m in ("F(CHAT)", "F(PROSE)"):
        mean20 = Z[m]["mean_a"]
        for cls, key in (("same init, same stream (SD_d)", "SD_d_rel"), ("other classes (sqrt2 sigma_seed)", "sigma_seed_rel")):
            mult = 1.0 if key == "SD_d_rel" else math.sqrt(2)
            rel20, rel5 = Z[m][key] * mult, N5[m][key] * mult
            used, df = (rel20, 2) if rel20 >= rel5 else (rel5, 7)
            f = E.factor(df)
            kn = {f"{dl:.3%}": k_needed(f, used, dl) for dl in DELTAS}
            r4.setdefault(m, {})[cls] = {
                "rel_20M_df2": rel20, "rel_5M_df7": rel5, "used": "20M" if df == 2 else "5M", "rel_used": used, "df_used": df, "factor": f,
                "MDE_rel": {str(k): f * used / math.sqrt(k) for k in KS}, "MDE_bpb_at_A20_mean": {str(k): f * used / math.sqrt(k) * mean20 for k in KS},
                "k_needed": kn, "k_rule2": {dl: max(2, k) for dl, k in kn.items()}, "underpowered": {dl: k > 5 for dl, k in kn.items()},
                "own_20M_k_needed": Z[m]["k_needed_paired" if key == "SD_d_rel" else "k_needed_unpaired"]}
            ratio = rel20 / rel5
            vr = ratio ** 2
            lo, hi = vr / f_ppf(0.975, 2, 7), vr / f_ppf(0.025, 2, 7)
            s20 = Z[m]["SD_d" if key == "SD_d_rel" else "sigma_seed"] * mult
            cmp_.setdefault(m, {})[cls] = {"rel_ratio_20M_over_5M": ratio, "sd_ratio_95ci_F_df2_7": [math.sqrt(lo), math.sqrt(hi)],
                                           "sd_20M_bpb": s20, "sd_20M_95ci_chi2_df2": [s20 * math.sqrt(2 / E.chi2_ppf(0.975, 2)), s20 * math.sqrt(2 / E.chi2_ppf(0.025, 2))]}
    res["rule4_20M_confirmations"] = r4
    res["compare_20M_vs_5M_descriptive"] = cmp_
    diff = round(E2_20M_500M["g0.5 (1.5e-3)"] - E2_20M_500M["g1 (3e-3)"], 4)
    mde20 = Z["F(CHAT)"]["MDE"]["1"]["paired"]
    mde_r4 = r4["F(CHAT)"]["same init, same stream (SD_d)"]["MDE_bpb_at_A20_mean"]["1"]
    res["e2_20m_one_seed_label_input"] = {
        "rule": "E2 RULES: 'resolved at one seed' when best-vs-second exceeds E3's paired MDE at k = 1 for F(CHAT)",
        "decision": "E2 20M grid at 500M: g 1 (3e-3) 0.9726 vs runner-up g 0.5 (1.5e-3) 0.9749", "difference_bpb": diff,
        "primary_20M_paired_MDE_k1_bpb": mde20, "label_primary": "resolved at one seed" if diff > mde20 else "not resolved",
        "rule4_paired_MDE_k1_bpb": mde_r4, "label_rule4": "resolved at one seed" if diff > mde_r4 else "not resolved",
        "where": "E2's notes carry the label; E3 reports the input"}
    res["plan_curve"] = {"PLAN continuous confirmations": 3,
                         "rule2_k_at_1pct_20M": {m: {c: v["k_rule2"]["1.000%"] for c, v in r4[m].items()} for m in r4},
                         "lock_in": "not read: lock-in turns on pass-rate spread (Part 4b), not run"}
    res["not_run"] = {"Part 3 (init/data split)": "not run (data_seed patch not committed)", "Part 4 (incl. 4b pass-rate spread at 20M)": "not run",
                      "20M replays, sigma_rep, two-seed sign agreement": "not defined for Part 2"}
    json.dump(res, open(os.path.join(HERE, "results_part2.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(HERE, "tables_part2.txt"), "w").write(tables(res, N5))
    print(tables(res, N5)); return 0


def tables(r, N5):
    Z = r["noise_20M_500M"]
    L = ["E3 PART 2 TABLES (e3analyze_p2.py, importing e3analyze.py; 20M, 500M slots, full WSD; A20 3e-3 r 1, B20 1.5e-3 r 1; d = B - A)", "",
         "CHECKS: " + ", ".join(f"{c['check']}{'' if c['ok'] else ' FAILED (' + c['detail'] + ')'}" for c in r["checks"] if not c["ok"]) +
         f"  [{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])} pass]", "",
         "NOISE, 3 seeds (df 2, factor 5.363); rel = fraction of A's mean; CRN ratio = sqrt2 sigma_seed / SD_d; p(d) is the 3-seed paired t (descriptive)"]
    L.append(f"{'metric':20s} {'mean A':>8s} {'mean d':>9s} {'sig_seed':>9s} {'rel':>7s} {'SD_d':>9s} {'rel':>7s} {'rho':>6s} {'CRN':>5s} {'p(d)':>6s}  {'5M SD_d rel':>11s}")
    for m, z in Z.items():
        L.append(f"{m:20s} {z['mean_a']:8.4f} {z['mean_d']:+9.5f} {z['sigma_seed']:9.5f} {z['sigma_seed_rel']:7.3%} {z['SD_d']:9.5f} {z['SD_d_rel']:7.3%} "
                 f"{z['rho']:6.3f} {z['crn_ratio']:5.2f} {z['p_two_sided']:6.3f}  {N5[m]['SD_d_rel']:11.3%}")
    L += ["", f"SEEDS 2..3 (df 1, factor {r['quantiles']['factor_df1']:.3f}; seed 1 selected A20 in E2's 20M grid)"]
    for m, z in r["noise_20M_500M_seeds2to3"].items():
        L.append(f"{m:20s} mean d {z['mean_d']:+9.5f}  SD_d {z['SD_d']:9.5f} ({z['SD_d_rel']:.3%})  sigma_seed {z['sigma_seed_rel']:.3%}  t {z['t']:+7.2f}  p {z['p_two_sided']:.4f}")
    L += ["", "MDE at 20M (80% power, two-sided 0.05), % of A20's mean: paired (SD_d) | unpaired (sqrt2 sigma_seed); [80% upper-bound SD, x2.117]"]
    for m, z in Z.items():
        L.append(f"{m:20s} " + "  ".join(f"k{k}: {v['paired_rel']:.3%}|{v['unpaired_rel']:.3%} [{v['paired_rel_ub80']:.3%}|{v['unpaired_rel_ub80']:.3%}]" for k, v in z["MDE"].items()))
    L += ["", "MDE at 20M in bpb, paired | unpaired"]
    for m in ("F(CHAT)", "F(PROSE)"):
        L.append(f"{m:20s} " + "  ".join(f"k{k}: {v['paired']:.5f}|{v['unpaired']:.5f}" for k, v in Z[m]["MDE"].items()))
    L += ["", "k_needed at 20M from its own SDs (df 2) for delta = 0.5% / 1% / 2% of A20's mean: paired | unpaired"]
    for m, z in Z.items():
        L.append(f"{m:20s} " + "  ".join(f"{dl}: {z['k_needed_paired'][dl]}|{z['k_needed_unpaired'][dl]}" for dl in z["k_needed_paired"]))
    L += ["", "PER-SEED VALUES: A, B, d"]
    for m in ("F(CHAT)", "F(PROSE)", "M(CHAT)", "M(PROSE)", "train loss last 2%"):
        v = r["values"][m]
        L.append(f"{m:20s} " + "  ".join(f"s{i + 1} {a:.5f} {b:.5f} {d:+.5f}" for i, (a, b, d) in enumerate(zip(v["A"], v["B"], v["d"]))))
    e = r["e2_recheck_20M"]
    L += ["", f"E2 RE-CHECK AT 20M ({e['test']}): d {', '.join(f'{x:+.5f}' for x in e['d'])}; mean d {e['mean_d']:+.5f}, t {e['t']:+.3f}, p {e['p_two_sided']:.4f}, "
          f"95% CI [{e['ci95_d'][0]:+.5f}, {e['ci95_d'][1]:+.5f}]; B lower with p < 0.05: {e['B_lower_and_p_lt_0.05']}; moves {e['moves']}",
          "", "CRN ratio at 20M: " + ", ".join(f"{m} {r['crn_20M'][m]['ratio']:.2f}" for m in ("F(CHAT)", "F(PROSE)")) + f" -> {r['crn_20M']['reading']}",
          "", "AVERAGING at 20M (descriptive; the rule reads 5M): " + "; ".join(f"{k}: F {v['SD_d_F']:.5f}, M {v['SD_d_M']:.5f}, reduction {v['reduction']:+.1%}" for k, v in r["averaging_20M_descriptive"].items()),
          "", "RULE 4, 20M CONFIRMATIONS: reference = larger of the 20M (df 2) and 5M (df 7) relative SDs; k = max(2, k_needed); >5 underpowered"]
    for m, cl in r["rule4_20M_confirmations"].items():
        for c, v in cl.items():
            L.append(f"{m:9s} {c:34s} 20M {v['rel_20M_df2']:.3%} 5M {v['rel_5M_df7']:.3%} -> {v['used']} (df {v['df_used']}, factor {v['factor']:.3f}); "
                     f"MDE k1 {v['MDE_rel']['1']:.3%} k2 {v['MDE_rel']['2']:.3%} k3 {v['MDE_rel']['3']:.3%} k5 {v['MDE_rel']['5']:.3%}; "
                     "k_needed " + " ".join(f"{dl}:{k}" for dl, k in v["k_needed"].items()) + "; k " + " ".join(f"{dl}:{k}" for dl, k in v["k_rule2"].items())
                     + "; underpowered " + ",".join(dl for dl, u in v["underpowered"].items() if u))
    L += ["", "20M vs 5M (descriptive): ratio of relative SDs 20M/5M [95% F interval, df 2 and 7]; 20M SD in bpb [95% chi-square interval, df 2]"]
    for m, cl in r["compare_20M_vs_5M_descriptive"].items():
        for c, v in cl.items():
            L.append(f"{m:9s} {c:34s} ratio {v['rel_ratio_20M_over_5M']:.2f} [{v['sd_ratio_95ci_F_df2_7'][0]:.2f}, {v['sd_ratio_95ci_F_df2_7'][1]:.2f}]; "
                     f"SD {v['sd_20M_bpb']:.5f} [{v['sd_20M_95ci_chi2_df2'][0]:.5f}, {v['sd_20M_95ci_chi2_df2'][1]:.5f}]")
    g = r["e2_20m_one_seed_label_input"]
    L += ["", f"E2 20M ONE-SEED LABEL INPUT: {g['decision']}, difference {g['difference_bpb']:.4f}; 20M paired MDE k1 {g['primary_20M_paired_MDE_k1_bpb']:.5f} -> {g['label_primary']}; "
          f"rule-4 paired MDE k1 {g['rule4_paired_MDE_k1_bpb']:.5f} -> {g['label_rule4']} ({g['where']})",
          "", f"PLAN: continuous confirmations use 3 seeds; rule 2 + 4 at delta 1%: {json.dumps(r['plan_curve']['rule2_k_at_1pct_20M'])}; lock-in: {r['plan_curve']['lock_in']}",
          f"Quantiles: {json.dumps({k: round(v, 4) for k, v in r['quantiles'].items()})}", f"Not run: {json.dumps(r['not_run'])}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
