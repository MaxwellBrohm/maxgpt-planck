"""E3 Part 1 statistics (notes.txt STATISTICS, fixed 2026-09-26) on the copied records in out/. Stdlib only.
  python e3analyze.py [--checks-only]      writes results.json and tables.txt beside this file
Reads out/<run>/{bpb.jsonl,train_tail.jsonl,run_records.json,preflight.json}, logs/queue_e3.txt, logs/code_sha256_*.txt,
configs/. Quantiles: t and chi-square from a regularized incomplete beta / gamma (scipy is not in Planck's venv)."""
from __future__ import annotations
import hashlib, itertools, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, TOTAL, KS, DELTAS = range(1, 9), 7630, (1, 2, 3, 5), (0.005, 0.01, 0.02)
REPS = ("e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3")
E2RUN = "e2_5m_e3_r1_b250M"
METRICS = {  # name: (set, split, "F" final | "M" mean of the last 3 checkpoints | "T" train loss)
    "F(CHAT)": ("CHAT", "all", "F"), "F(PROSE)": ("PROSE", "all", "F"), "M(CHAT)": ("CHAT", "all", "M"),
    "M(PROSE)": ("PROSE", "all", "M"), "CHAT user": ("oasst2", "user", "F"), "CHAT assistant": ("oasst2", "assistant", "F"),
    "PROSE cccc": ("cccc", "all", "F"), "PROSE gutenberg": ("gutenberg", "all", "F"), "PROSE wikimedia": ("wikimedia", "all", "F"),
    "stackexchange": ("stackexchange", "all", "F"), "irc": ("irc", "all", "F"), "dolly": ("dolly", "all", "F"),
    "train loss last 2%": (None, None, "T")}
FIXED = {"tokenizer": "078b24c4b0755d81985ebc122e912d7c70ff204d335519721235a405756fcfb5",
         "evalset": "fe1b55bbe3612d0c968934cbad831a17fd3d73d84aa8f1b7cb2f4523cf4d505b",
         "code_digest": "92deaef0f4f99f827080b4a6aef0e192d70c4991cc8dbdc57274eafec108f9e0",
         "files": {"data_prep/bpb.py": "4f73dbda3ea2f3eb", "data_prep/evalwin.py": "e31107725df25771",
                   "harness/train.py": "d6898e7ceb84a3aa", "harness/trainer.py": "c1f3ff03477d8776",
                   "harness/optim.py": "2a070d507a93e869", "harness/optim_batched.py": "128afabc37ffb591",
                   "harness/docattn.py": "4e42363c0d1eb2dd", "harness/data.py": "cc9b0acf0a2665fd",
                   "harness/model.py": "1997098ed2d39b67", "harness/blocks.py": "f1962a5cb7ead3d3",
                   "harness/runio.py": "438d4f5438aeee79"},
         "engine": {"data.max_item_len": 2048, "data.mode": "pack", "data.window_tokens": 131072, "optim.batched": True,
                    "train.auto_micro_on_oom": False, "train.ce_chunk_rows": 0, "train.compile": False,
                    "train.compile_backend": "inductor", "train.compile_dynamic": None, "train.cuda_mem_cap_gib": None,
                    "train.cuda_mem_margin_gib": 0.5, "train.doc_attn": "varlen", "train.lazy_metrics": False,
                    "train.precision": "bf16"}, "engine_fixed": "2026-09-26 93ea41c",
         "shares": {"cccc": 0.414241, "stackexchange": 0.221922, "gutenberg": 0.147915, "wikimedia": 0.147915,
                    "irc": 0.040004, "dolly": 0.016002, "oasst2": 0.012001},
         "n_files": {"cccc": 31, "stackexchange": 11, "gutenberg": 6, "irc": 6, "wikimedia": 3, "dolly": 1, "oasst2": 1}}
# DEVIATION D4 (notes.txt; E2 DEVIATION 1), 2026-10-03: the durable-saves commit changes harness/runio.py's bytes, not
# its computation (fsync before rename). A code list taken at or after that commit carries this hash; it passes the
# runio.py check, labelled D4. FIXED["files"] keeps 438d4f54, which Part 1's code list carries.
FIXED_D4 = {"harness/runio.py": "ffcdf24596ab46cb"}


def _betacf(a, b, x):
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, 400):
        for aa in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300; h *= d * c
    return h


def betai(a, b, x):
    if x <= 0 or x >= 1:
        return max(0.0, min(1.0, x))
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    p = 0.5 * betai(df / 2, 0.5, df / (df + t * t))
    return 1 - p if t > 0 else p


def gammap(a, x):     # regularized lower incomplete gamma
    if x <= 0:
        return 0.0
    if x < a + 1:
        s = term = 1 / a; n = a
        for _ in range(1000):
            n += 1; term *= x / n; s += term
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a); h = d
    for i in range(1, 1000):
        an = -i * (i - a); b += 2; d = 1 / (an * d + b); c = b + an / c; h *= d * c
    return 1 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def ppf(cdf, q, lo, hi):
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < q else (lo, mid)
    return (lo + hi) / 2


def t_ppf(q, df): return ppf(lambda x: t_cdf(x, df), q, -1e4, 1e4)
def chi2_ppf(q, df): return ppf(lambda x: gammap(df / 2, x / 2), q, 0.0, 1e4)
def factor(df): return t_ppf(0.975, df) + t_ppf(0.80, df)
def ub(df): return math.sqrt(df / chi2_ppf(0.20, df))          # one-sided 80% upper bound on an SD


def load(run):
    d = os.path.join(HERE, "out", run)
    bpb = [json.loads(x) for x in open(os.path.join(d, "bpb.jsonl"))]
    tail = [json.loads(x) for x in open(os.path.join(d, "train_tail.jsonl"))]
    recs = json.load(open(os.path.join(d, "run_records.json")))
    pf = json.load(open(os.path.join(d, "preflight.json")))["runs"][0] if os.path.exists(os.path.join(d, "preflight.json")) else None
    return {"bpb": bpb, "tail": tail, "recs": recs, "pf": pf}


def value(run, metric):
    s, sp, kind = METRICS[metric]
    if kind == "T":
        return st.fmean(r["loss"] for r in run["tail"] if r["step"] > TOTAL - 0.02 * TOTAL)
    v = {r["step"]: r["bpb"] for r in run["bpb"] if r["set"] == s and r["split"] == sp}
    assert sorted(v) == [7344, 7497, 7630], sorted(v)
    return v[7630] if kind == "F" else st.fmean(v.values())


def checks(runs):
    out, add = [], lambda name, ok, det="": out.append({"check": name, "ok": bool(ok), "detail": det})
    code = open(os.path.join(HERE, "logs", sorted(f for f in os.listdir(os.path.join(HERE, "logs")) if f.startswith("code_sha256_"))[0])).read().split("\n")
    rows = sorted((ln.split("  ")[1], ln.split("  ")[0]) for ln in code if ln.strip())
    hp = [f"{h}  {p}\n" for p, h in rows if p.startswith(("harness/", "data_prep/")) and p.endswith(".py")]
    dig = hashlib.sha256("".join(sorted(hp, key=lambda s: s.split("  ")[1].encode())).encode()).hexdigest()
    add("code digest harness+data_prep .py == FIXED 2 (92deaef0)", dig == FIXED["code_digest"], f"{dig[:16]} over {len(hp)} files")
    fh = dict(rows)
    for p, h in FIXED["files"].items():
        got, d4 = fh.get(p, "")[:16], FIXED_D4.get(p)
        add(f"{p} == FIXED 2", got == h or got == d4, fh.get(p, "missing")[:16] + (" (D4)" if got == d4 else ""))
    q = open(os.path.join(HERE, "logs", "queue_e3.txt")).read().split("\n")
    starts = [ln for ln in q if "queue start" in ln]
    add("one queue start, commit e39112a", len(starts) == 1 and "e39112adf088" in starts[0], f"{len(starts)} start(s)")
    prereg = hashlib.sha256(open(os.path.join(HERE, "configs", "prereg.yaml"), "rb").read()).hexdigest()
    for name, r in runs.items():
        if name == E2RUN:
            continue
        bad = []
        ev = {(x["evalset_sha256"], x["tokenizer_sha256"], x["precision"], x["max_windows"]) for x in r["bpb"]}
        if ev != {(FIXED["evalset"], FIXED["tokenizer"], "fp32", None)}: bad.append(f"scorer/eval/tokenizer {ev}")
        pf = r["pf"]; cfg = os.path.join(HERE, "configs", name + ".yaml")
        if hashlib.sha256(open(cfg, "rb").read()).hexdigest() != pf["file_sha256"]: bad.append("config file sha")
        if pf["engine"] != FIXED["engine"] or pf["engine_fixed"] != FIXED["engine_fixed"]: bad.append("engine")
        if pf["shares"] != FIXED["shares"] or pf["n_files"] != FIXED["n_files"]: bad.append("shares/files")
        s, e = r["recs"][0], r["recs"][-1]
        if (s["event"], e["event"]) != ("start", "end") or len(r["recs"]) != 2: bad.append("records")
        if s["config_sha256"] != pf["config_sha256"] or s["prereg_sha256"] != prereg or not s["prereg_committed"]: bad.append("config/prereg sha")
        if s["schedule"] != {"decay_start": 6104, "mode": "full", "total_steps": TOTAL, "warmup_steps": 76}: bad.append("schedule")
        if (s["batch_tokens"], s["precision"], s["doc_attn"], s["optim_batched"], s["n_params"], s["resumed_from"]) != (32768, "bf16", "varlen", True, 5010133, None): bad.append("run setting")
        if e["step"] != TOTAL or not math.isfinite(e["loss"]): bad.append("end step/loss")
        if sum(ln.endswith(f"train {name}.yaml --require-committed") for ln in q) != 1 or sum(ln.endswith(f"done {name} rc 0") for ln in q) != 1: bad.append("queue: not exactly one clean train")
        add(f"{name}: FIXED hashes, settings, one clean run", not bad, "; ".join(bad))
    drawn = lambda n: {k: v for k, v in runs[n]["recs"][-1].items() if k.startswith(("drawn_", "dropped_"))}  # noqa: E731
    add("CRN: A_s and B_s drew identical per-source tokens at every seed", all(drawn(f"e3_5m_A_s{s}") == drawn(f"e3_5m_B_s{s}") for s in SEEDS))
    add("seeds differ: 8 distinct streams", len({json.dumps(drawn(f"e3_5m_A_s{s}"), sort_keys=True) for s in SEEDS}) == 8)
    add("replays R2, R3 drew A1's stream", drawn(REPS[1]) == drawn(REPS[0]) == drawn(REPS[2]))
    add("E2 5m_e3_r1 (trunk+branch) drew A1's stream", drawn(E2RUN) == drawn(REPS[0]))
    return out


def sd(x): return st.stdev(x) if len(x) > 1 else float("nan")


def noise(a, b, mean_a):
    n = len(a); df = n - 1; d = [y - x for x, y in zip(a, b)]
    s_seed = math.sqrt((st.variance(a) + st.variance(b)) / 2); s_d = sd(d); f = factor(df)
    md, se = st.fmean(d), s_d / math.sqrt(n); t = md / se; p = 2 * (1 - t_cdf(abs(t), df)); tq = t_ppf(0.975, df)
    mde = {k: {"paired": f * s_d / math.sqrt(k), "unpaired": f * s_seed * math.sqrt(2 / k)} for k in KS}
    knd = lambda s_ref: {f"{dl:.3%}": next(k for k in range(1, 10**6) if f * s_ref / math.sqrt(k) <= dl * mean_a) for dl in DELTAS}  # noqa: E731
    return {"n": n, "df": df, "factor": f, "mean_a": mean_a, "mean_b": st.fmean(b), "sigma_seed": s_seed,
            "sigma_seed_rel": s_seed / mean_a, "SD_d": s_d, "SD_d_rel": s_d / mean_a, "rho": st.correlation(a, b),
            "mean_d": md, "t": t, "p_two_sided": p, "ci95_d": [md - tq * se, md + tq * se],
            "crn_ratio": math.sqrt(2) * s_seed / s_d,
            "MDE": {str(k): {**v, "paired_rel": v["paired"] / mean_a, "unpaired_rel": v["unpaired"] / mean_a,
                             "paired_rel_ub80": v["paired"] * ub(df) / mean_a, "unpaired_rel_ub80": v["unpaired"] * ub(df) / mean_a}
                    for k, v in mde.items()},
            "k_needed_paired": knd(s_d), "k_needed_unpaired": knd(math.sqrt(2) * s_seed),
            "refresh_pi95_3pair_mean": [md - tq * s_d * math.sqrt(1 / 3 + 1 / n), md + tq * s_d * math.sqrt(1 / 3 + 1 / n)],
            "refresh_sd_bound": s_d * math.sqrt(chi2_ppf(0.95, 2) / 2)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    names = [f"e3_5m_{a}_s{s}" for s in SEEDS for a in "AB"] + list(REPS[1:]) + [E2RUN]
    runs = {n: load(n) for n in names}
    ck = checks(runs)
    if "--checks-only" in argv:
        print(json.dumps(ck, indent=1)); return 0
    res = {"checks": ck, "quantiles": {"factor_df7": factor(7), "factor_df6": factor(6), "factor_df2": factor(2),
           "ub80_df7": ub(7), "ub80_df6": ub(6), "ub80_df2": ub(2), "chi2_95_df2_sd_bound": math.sqrt(chi2_ppf(0.95, 2) / 2)},
           "values": {}, "noise_5M_250M": {}, "noise_5M_250M_seeds2to8": {}, "replay": {}, "sign_agreement_2seed": {}}
    for m in METRICS:
        a = [value(runs[f"e3_5m_A_s{s}"], m) for s in SEEDS]; b = [value(runs[f"e3_5m_B_s{s}"], m) for s in SEEDS]
        rep = [value(runs[r], m) for r in REPS]; e2 = value(runs[E2RUN], m)
        res["values"][m] = {"A": a, "B": b, "d": [y - x for x, y in zip(a, b)], "A1_R2_R3": rep, "E2_5m_e3_r1_b250M": e2}
        nz = noise(a, b, st.fmean(a)); res["noise_5M_250M"][m] = nz
        res["noise_5M_250M_seeds2to8"][m] = noise(a[1:], b[1:], st.fmean(a[1:]))
        s_rep = sd(rep)
        res["replay"][m] = {"sigma_rep": s_rep, "sigma_rep_rel": s_rep / nz["mean_a"], "df": 2, "share_of_paired_var": 2 * s_rep ** 2 / nz["SD_d"] ** 2,
                            "max_abs_diff": max(rep) - min(rep), "e2_minus_A1": e2 - rep[0], "e2_minus_rep_mean": e2 - st.fmean(rep),
                            "e2_minus_rep_mean_in_sigma_rep": (e2 - st.fmean(rep)) / s_rep if s_rep else None}
        d = res["values"][m]["d"]; sg = math.copysign(1, nz["mean_d"])
        sub = [all(math.copysign(1, d[i]) == sg and d[i] != 0 for i in c) for c in itertools.combinations(range(8), 2)]
        res["sign_agreement_2seed"][m] = {"subsets": len(sub), "agree": sum(sub), "share": sum(sub) / len(sub)}
    F, M = res["noise_5M_250M"], res["noise_5M_250M"]
    res["averaging_rule"] = {k: {"SD_d_F": F[f"F({k})"]["SD_d"], "SD_d_M": M[f"M({k})"]["SD_d"],
                                 "reduction": 1 - M[f"M({k})"]["SD_d"] / F[f"F({k})"]["SD_d"],
                                 "screens_may_use_M": 1 - M[f"M({k})"]["SD_d"] / F[f"F({k})"]["SD_d"] >= 0.30} for k in ("CHAT", "PROSE")}
    rc = res["noise_5M_250M_seeds2to8"]["F(CHAT)"]
    to_b = rc["mean_d"] < 0 and rc["p_two_sided"] < 0.05
    res["e2_recheck"] = {"test": "paired t on F(CHAT) d_s = B - A, seeds 2..8, df 6", "mean_d": rc["mean_d"], "t": rc["t"],
                         "p_two_sided": rc["p_two_sided"], "ci95_d": rc["ci95_d"], "B_lower_and_p_lt_0.05": to_b,
                         "verdict": "OVERRIDE: 5M LR becomes B (1.5e-3, r 1)" if to_b else "CONFIRM: 5M LR stays A (3e-3, r 1)",
                         "lr5_for_screens": [0.0015, 1.0] if to_b else [0.003, 1.0]}
    res["crn_mandatory"] = {m: F[m]["crn_ratio"] >= 2 for m in ("F(CHAT)", "F(PROSE)")}
    res["not_run"] = {"Part 2 (20M)": "not run yet (waits for E2's 20M picks)", "Part 3 (init/data split)": "not run (data_seed patch not committed)",
                      "Part 4": "not run", "shorter lengths": "not defined for E3 Part 1 (250M full runs only)"}
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, sort_keys=True)
    open(os.path.join(HERE, "tables.txt"), "w").write(tables(res))
    print(tables(res)); return 0


def tables(r):
    L = ["E3 PART 1 TABLES (e3analyze.py; 5M, 250M slots, full WSD; arm A 3e-3 r 1, arm B 1.5e-3 r 1; d = B - A)", "",
         "CHECKS: " + ", ".join(f"{c['check']}{'' if c['ok'] else ' FAILED (' + c['detail'] + ')'}" for c in r["checks"] if not c["ok"]) +
         f"  [{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])} pass]", "",
         "NOISE, 8 seeds (df 7); rel = fraction of A's mean; CRN ratio = sqrt2 sigma_seed / SD_d"]
    L.append(f"{'metric':20s} {'mean A':>8s} {'mean d':>9s} {'sig_seed':>9s} {'rel':>7s} {'SD_d':>9s} {'rel':>7s} {'rho':>6s} {'CRN':>5s} {'p(d)':>6s} {'sig_rep':>8s} {'repshr':>6s}")
    for m, z in r["noise_5M_250M"].items():
        rp = r["replay"][m]
        L.append(f"{m:20s} {z['mean_a']:8.4f} {z['mean_d']:+9.5f} {z['sigma_seed']:9.5f} {z['sigma_seed_rel']:7.3%} {z['SD_d']:9.5f} {z['SD_d_rel']:7.3%} "
                 f"{z['rho']:6.3f} {z['crn_ratio']:5.2f} {z['p_two_sided']:6.3f} {rp['sigma_rep']:8.5f} {rp['share_of_paired_var']:6.2f}")
    L += ["", "SEEDS 2..8 (df 6; seed 1 selected A in E2)"]
    for m, z in r["noise_5M_250M_seeds2to8"].items():
        L.append(f"{m:20s} mean d {z['mean_d']:+9.5f}  SD_d {z['SD_d']:9.5f} ({z['SD_d_rel']:.3%})  sigma_seed {z['sigma_seed_rel']:.3%}  t {z['t']:+6.2f}  p {z['p_two_sided']:.4f}")
    L += ["", "MDE (80% power, two-sided 0.05) as % of A's mean: paired (SD_d) | unpaired (sqrt2 sigma_seed); [80% upper-bound SD]"]
    for m, z in r["noise_5M_250M"].items():
        L.append(f"{m:20s} " + "  ".join(f"k{k}: {v['paired_rel']:.3%}|{v['unpaired_rel']:.3%} [{v['paired_rel_ub80']:.3%}|{v['unpaired_rel_ub80']:.3%}]" for k, v in z["MDE"].items()))
    L += ["", "k_needed for delta = 0.5% / 1% / 2% of A's mean: paired | unpaired"]
    for m, z in r["noise_5M_250M"].items():
        L.append(f"{m:20s} " + "  ".join(f"{dl}: {z['k_needed_paired'][dl]}|{z['k_needed_unpaired'][dl]}" for dl in z["k_needed_paired"]))
    L += ["", "PER-SEED VALUES F(CHAT), F(PROSE): A, B, d"]
    for m in ("F(CHAT)", "F(PROSE)"):
        v = r["values"][m]
        L.append(f"{m}: " + "  ".join(f"s{i + 1} {a:.4f} {b:.4f} {d:+.4f}" for i, (a, b, d) in enumerate(zip(v["A"], v["B"], v["d"]))))
        L.append(f"   A1/R2/R3 {' '.join(f'{x:.5f}' for x in v['A1_R2_R3'])}; E2 5m_e3_r1 (trunk+branch, 93ea41c harness) {v['E2_5m_e3_r1_b250M']:.5f}")
    L += ["", "TWO-SEED SIGN AGREEMENT with the 8-seed mean d (28 subsets): " + ", ".join(f"{m} {s['agree']}/28" for m, s in r["sign_agreement_2seed"].items()),
          "", "AVERAGING (M vs F, SD_d): " + "; ".join(f"{k}: F {v['SD_d_F']:.5f}, M {v['SD_d_M']:.5f}, reduction {v['reduction']:+.1%}, M allowed {v['screens_may_use_M']}" for k, v in r["averaging_rule"].items()),
          "", f"E2 RE-CHECK: {r['e2_recheck']['test']}: mean d {r['e2_recheck']['mean_d']:+.5f}, t {r['e2_recheck']['t']:+.3f}, p {r['e2_recheck']['p_two_sided']:.4f}, "
          f"95% CI [{r['e2_recheck']['ci95_d'][0]:+.5f}, {r['e2_recheck']['ci95_d'][1]:+.5f}] -> {r['e2_recheck']['verdict']}",
          f"CRN mandatory (ratio >= 2): {r['crn_mandatory']}", f"Quantiles: {json.dumps({k: round(v, 4) for k, v in r['quantiles'].items()})}",
          f"Not run: {json.dumps(r['not_run'])}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
