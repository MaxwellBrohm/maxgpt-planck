"""E3 Part 3 statistics (notes.txt PART 3 REOPENED, 2026-10-11: STATISTICS and ANALYZER) on the copied records in
out/e3_5m_A_{s1_P3,init2..5,data2..5}. Stdlib only. Written and committed before any Part 3 value was read.
  python e3analyze_p3.py [--checks-only]   writes results_part3.json and tables_part3.txt beside this file (D5)
Imports e3analyze.py unchanged (load, value, METRICS, FIXED, FIXED_D4, factor, ub, the t / chi-square quantiles).
  INIT family: e3_5m_A_s1_P3 and init2..5 (seeds 1..5, data_seed 1); DATA family: e3_5m_A_s1_P3 and data2..5 (seed 1,
  data_seeds 1..5). sigma_init, sigma_data = SD within each family, df n - 1 (4), each as a fraction of its family's mean,
  with the 95% chi-square interval and the 80% upper bound. Rule 1's reference SD for "same stream, new init" is sqrt(2) x
  sigma_init, for "same init, new stream" sqrt(2) x sigma_data, with the family's df: MDE_u(k) = factor(df) x sqrt(2) x
  sigma / sqrt(k), k = 1, 2, 3, 5; k_needed for 0.5%, 1%, 2%; rule 2's k = max(2, k_needed), above 5 "underpowered".
  A run with a GAP or DIVERGED mark (copied to out/<run>/) leaves its family (df down by one); fewer than 4 runs left:
  no SD, and that class keeps sqrt(2) x sigma_seed. Descriptive, no rule: each family's relative SD against Part 1's
  sigma_seed (relative, df 7) with a 95% F interval (df n - 1 and 7), sqrt(sigma_init^2 + sigma_data^2) against
  sigma_seed (another commit), and e3_5m_A_s1_P3 beside Part 1's A1, R2 and R3 in units of Part 1's sigma_rep.
Expected check failures (D2, and bpb.py's --hd since Part 2): the FIXED 2 digest and train.py, trainer.py, model.py,
bpb.py; each of those files must equal commit 4962426's (one check). Any other failure is handled before a value is read."""
from __future__ import annotations
import hashlib, json, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3analyze as E  # noqa: E402  (the registered Part 1 analyzer, unchanged)

TOTAL, CKPTS, KS, DELTAS = 7630, [7344, 7497, 7630], (1, 2, 3, 5), (0.005, 0.01, 0.02)
SCHED = {"decay_start": 6104, "mode": "full", "total_steps": TOTAL, "warmup_steps": 76}
TAIL_STEPS, N_PARAMS = list(range(7480, 7631, 10)), 5010133
LOGS = os.path.join(HERE, "logs", "part3")
COMMIT, PLAN_SHA = "4962426b8f0d17738051bd499f9141a7cbc18c40", "07018f0fec5729944f648b20d499d8b2d346426c81fec7cbc892fb894d1c2417"
P3_DIGEST = "aced7148757af02a461e75c281009786f591f0ed7804f6e33b817659b4281c26"     # FIXED 2 recipe, 111 files
P3_ROWS = "b2893c82eb73e0ebbe2f905bc3d7bfb1a21895869ca0e381e8c247ad14c2a252"       # every code-list row, 386
P3_FILES = {"harness/train.py": "aa07b64c7a4e3089", "harness/trainer.py": "15d1a9719d972d8b",
            "harness/model.py": "f4e61556ee9ece8a", "data_prep/bpb.py": "7367071e460e0039"}
EXPECTED_FAIL = ["code digest harness+data_prep .py == FIXED 2 (92deaef0)", "harness/train.py == FIXED 2",
                 "harness/trainer.py == FIXED 2", "harness/model.py == FIXED 2", "data_prep/bpb.py == FIXED 2"]
CONFIG_SHA = {  # notes PART 3 REOPENED, RUNS
    "e3_5m_A_init2": "b6d1d1da6025a090bfb9c7dad25dc91772398e57d6b3b2c5e7c96591b957c1cd",
    "e3_5m_A_data2": "f240ce58e92b1a1a9eb2b44081802b93e8c2989cc3c659342f949939c140e61b",
    "e3_5m_A_init3": "b6f5cff839d322599f0ff124db4ad5d7fa1d67ba7f098671f5596951dc09ae38",
    "e3_5m_A_data3": "67acf407be4ea7e4d9a9f2a410097aa6da362e9f735d4636ffd19df5d0f6795a",
    "e3_5m_A_s1_P3": "b364101f884e8dc33116c300950e55140978d784819c7110e29202a4a5fda403",
    "e3_5m_A_init4": "820b4dfa17bb84fbcc830e35f025bf8b1413bd11d72f98ddded4bfb50ead10df",
    "e3_5m_A_data4": "d50175dd620b1812abc67eac80319d07997787ae07a851c10423b0dc1080e836",
    "e3_5m_A_init5": "cdc2ea3c8e49ffef8273fda5a63021167ec480669811ce5f1897f8ed6e790d18",
    "e3_5m_A_data5": "3210030e489662e8f8c96875f1dae1e79e95ec0888e49acc3df30bf66174dca0"}
ORDER = list(CONFIG_SHA)
SHARED = "e3_5m_A_s1_P3"
FAM = {"INIT": [SHARED] + [f"e3_5m_A_init{k}" for k in range(2, 6)], "DATA": [SHARED] + [f"e3_5m_A_data{k}" for k in range(2, 6)]}
SEEDS = {SHARED: (1, 1), **{f"e3_5m_A_init{k}": (k, 1) for k in range(2, 6)}, **{f"e3_5m_A_data{k}": (1, k) for k in range(2, 6)}}
CLASS = {"INIT": "same stream, new init", "DATA": "same init, new stream"}
PINNED = {"results.json": "1f017ecb1e6d14dd4f676c90ca703c20e9395b217fc841f83fffcb93e57994a2",
          "tables.txt": "100a55ab58bd8b40129f23d659963869128aad254f4b08cad570c08147d5e6ec",
          "results_part2.json": "e8256d57c6a73e4abb2f52aac9d042bcf8cf43ff9670adc586dce8c97ec6eaba",
          "tables_part2.txt": "b9377cab87c9f6c4fe2ed65ab3021c8a1ef8a4d83e024a7b61a7ac5a687d5edd",
          "results_part4d.json": "8a2afd7daed170f2ebef4a9723a4dfe42f9e02aebf2b6d4e93b0e2190e3afb57"}
P1_REPS = ("e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3")


def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def stream(rec): return {k: v for k, v in rec.items() if k.startswith(("drawn_", "dropped_")) or k in ("tokens", "sup_tokens", "step")}
def drawn(rec): return {k: v for k, v in rec.items() if k.startswith(("drawn_", "dropped_"))}
def gapped(run): return any(os.path.exists(os.path.join(HERE, "out", run, m)) for m in ("GAP", "DIVERGED"))


def p3_queue():
    qt = open(os.path.join(LOGS, "queue_e3.txt")).read()
    p2 = open(os.path.join(HERE, "logs", "part2", "queue_e3.txt")).read()
    return qt, p2, qt[len(p2):].split("\n") if qt.startswith(p2) else qt.split("\n")


def checks(runs):
    out, add = [], lambda name, ok, det="": out.append({"check": name, "ok": bool(ok), "detail": det})
    qt, p2, q3 = p3_queue()
    starts = [ln for ln in q3 if " queue start: " in ln]
    have = sorted(f for f in os.listdir(LOGS) if f.startswith("code_sha256_"))   # one per queue start, written just before it
    secs = lambda t: int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])   # noqa: E731
    lists = [f for f, s in zip(have, starts) if f[12:20] == s[:10].replace("-", "")
             and 0 <= secs(s[11:19].replace(":", "")) - secs(f[21:27]) <= 5] if len(have) == len(starts) else []
    code = open(os.path.join(LOGS, have[0])).read().split("\n") if have else []
    rows = sorted((ln.split("  ")[1], ln.split("  ")[0]) for ln in code if ln.strip())
    hp = [f"{h}  {p}\n" for p, h in rows if p.startswith(("harness/", "data_prep/")) and p.endswith(".py")]
    dig = hashlib.sha256("".join(sorted(hp, key=lambda s: s.split("  ")[1].encode())).encode()).hexdigest()
    add("code digest harness+data_prep .py == FIXED 2 (92deaef0)", dig == E.FIXED["code_digest"], f"{dig[:16]} over {len(hp)} files")
    fh = dict(rows)
    for p, h in E.FIXED["files"].items():
        got, d4 = fh.get(p, "")[:16], E.FIXED_D4.get(p)
        add(f"{p} == FIXED 2", got == h or got == d4, fh.get(p, "missing")[:16] + (" (D4)" if got == d4 else ""))
    add("code digest == commit 4962426's (aced7148, 111 files)", dig == P3_DIGEST and len(hp) == 111, f"{dig[:16]} over {len(hp)} files")
    add("train.py, trainer.py, model.py, bpb.py == commit 4962426's (data_seed; D2 as in Part 2; --hd)",
        all(fh.get(p, "")[:16] == h for p, h in P3_FILES.items()))
    rsha = lambda f: hashlib.sha256("".join(f"{h}  {p}\n" for p, h in sorted((ln.split("  ")[1], ln.split("  ")[0])  # noqa: E731
                                     for ln in open(os.path.join(LOGS, f)).read().split("\n") if ln.strip())).encode()).hexdigest()
    add("every Part 3 start's code list is copied and equals commit 4962426's files (386 rows, b2893c82)",
        starts and have == lists and all(rsha(f) == P3_ROWS for f in have) and len(rows) == 386, f"{len(have)} of {len(lists)} lists")
    add("code list has the registered analyzer (e3analyze.py) and the 9 Part 3 configs as registered",
        fh.get("experiments/E3_seed_noise/e3analyze.py") == sha256(os.path.join(HERE, "e3analyze.py"))
        and all(fh.get(f"experiments/E3_seed_noise/configs/{n}.yaml") == CONFIG_SHA[n] for n in ORDER))
    add("Part 3 queue log starts with Part 2's copy, byte for byte", qt.startswith(p2))
    add("every Part 3 queue start: plan part3.txt 07018f0f, commit 4962426", starts and all(
        f"plan part3.txt (sha256 {PLAN_SHA[:16]}), code commit {COMMIT}" in s for s in starts), f"{len(starts)} start(s)")
    add("plans/part3.txt on disk has the registered sha", sha256(os.path.join(HERE, "plans", "part3.txt")) == PLAN_SHA)
    trains = [ln.split("train ")[1].split(".yaml")[0] for ln in q3 if " train " in ln and ln.endswith("--require-committed")]
    seq = [t for i, t in enumerate(trains) if i == 0 or t != trains[i - 1]]     # a relaunch retrains the cut run once
    add("registered order init2 data2 init3 data3 s1_P3 init4 data4 init5 data5, then MARK E3 PART 3 DONE once",
        seq == ORDER and sum(ln.endswith(" MARK E3 PART 3 DONE") for ln in q3) == 1, " ".join(seq))
    prereg = sha256(os.path.join(HERE, "configs", "prereg.yaml"))
    for name in ORDER:
        if name not in runs:
            add(f"{name}: FIXED hashes, settings, one clean run", False, "GAP or DIVERGED"); continue
        r, bad = runs[name], []
        seed, dseed = SEEDS[name]
        ev = {(x["evalset_sha256"], x["tokenizer_sha256"], x["precision"], x["max_windows"]) for x in r["bpb"]}
        if ev != {(E.FIXED["evalset"], E.FIXED["tokenizer"], "fp32", None)}: bad.append(f"scorer/eval/tokenizer {ev}")
        if {x["run"] for x in r["bpb"]} != {name}: bad.append("bpb run field")
        per = {}
        for x in r["bpb"]: per.setdefault((x["set"], x["split"]), []).append(x["step"])
        if any(sorted(v) != CKPTS for v in per.values()) or len(per) < 13: bad.append("bpb steps")
        pf = r["pf"]; cfg = os.path.join(HERE, "configs", name + ".yaml"); txt = open(cfg).read()
        if sha256(cfg) != pf["file_sha256"] or sha256(cfg) != CONFIG_SHA[name]: bad.append("config file sha")
        if "lr: 0.003, embed_lr: 0.003, scalar_lr: 0.003}" not in txt or f"\nseed: {seed}\ndata_seed: {dseed}\n" not in txt: bad.append("config LR/seeds")
        if not (pf["ok"] and pf["refusals"] == [] and pf["prereg"]["committed"] and pf["prereg"]["sha256"] == prereg): bad.append("preflight ok/strict/prereg")
        if pf["engine"] != E.FIXED["engine"] or pf["engine_fixed"] != E.FIXED["engine_fixed"]: bad.append("engine")
        if pf["shares"] != E.FIXED["shares"] or pf["n_files"] != E.FIXED["n_files"]: bad.append("shares/files")
        s, e = r["recs"][0], r["recs"][-1]
        if (s["event"], e["event"]) != ("start", "end") or len(r["recs"]) != 2: bad.append("records")
        if s["config_sha256"] != pf["config_sha256"] or s["prereg_sha256"] != prereg or not s["prereg_committed"]: bad.append("config/prereg sha")
        if s.get("data_seed") != dseed: bad.append("start record data_seed")
        if s["schedule"] != SCHED: bad.append("schedule")
        if (s["batch_tokens"], s["precision"], s["doc_attn"], s["optim_batched"], s["n_params"], s["resumed_from"]) != (32768, "bf16", "varlen", True, N_PARAMS, None): bad.append("run setting")
        if e["step"] != TOTAL or not math.isfinite(e["loss"]): bad.append("end step/loss")
        if [t["step"] for t in r["tail"]] != TAIL_STEPS or not all(math.isfinite(t["loss"]) for t in r["tail"]): bad.append("tail steps/finite")
        if stream(r["tail"][-1]) != stream(e): bad.append("tail end != end record stream")
        if sum(ln.endswith(f"train {name}.yaml --require-committed") for ln in q3) != 1 or sum(ln.endswith(f"done {name} rc 0") for ln in q3) != 1: bad.append("queue: not exactly one clean train")
        add(f"{name}: FIXED hashes, settings, one clean run", not bad, "; ".join(bad))
    tails = {n: [stream(t) for t in runs[n]["tail"]] for n in runs}
    ok_init = SHARED in runs and all(tails[n] == tails[SHARED] and drawn(runs[n]["recs"][-1]) == drawn(runs[SHARED]["recs"][-1]) for n in FAM["INIT"] if n in runs)
    add("split, INIT: each init run drew s1_P3's stream (drawn, dropped, tokens, sup_tokens equal at all 16 tail log steps and the end)", ok_init)
    ends = [json.dumps(drawn(runs[n]["recs"][-1]), sort_keys=True) for n in FAM["DATA"] if n in runs]
    add("split, DATA: every data run drew another stream (end records pairwise different, each differs from s1_P3)", len(set(ends)) == len(ends) == sum(n in runs for n in FAM["DATA"]))
    add("Parts 1, 2 and 4d results (and tables) unchanged", all(sha256(os.path.join(HERE, f)) == h for f, h in PINNED.items()))
    return out


def f_ppf(q, d1, d2): return E.ppf(lambda x: E.betai(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2)), q, 0.0, 1e4)


def family(vals):
    n = len(vals); df = n - 1; mean = st.fmean(vals); sd = st.stdev(vals)
    lo, hi = math.sqrt(df / E.chi2_ppf(0.975, df)), math.sqrt(df / E.chi2_ppf(0.025, df))
    return {"n": n, "df": df, "mean": mean, "sd": sd, "rel": sd / mean, "ci95": [sd * lo, sd * hi], "rel_ci95": [sd * lo / mean, sd * hi / mean],
            "ub80": sd * E.ub(df), "rel_ub80": sd * E.ub(df) / mean}


def rule1(fz):
    f, ref = E.factor(fz["df"]), math.sqrt(2) * fz["rel"]
    kn = {f"{dl:.3%}": next(k for k in range(1, 10**6) if f * ref / math.sqrt(k) <= dl) for dl in DELTAS}
    return {"df": fz["df"], "factor": f, "ref_rel": ref, "ref_bpb": ref * fz["mean"], "MDE_rel": {str(k): f * ref / math.sqrt(k) for k in KS},
            "MDE_bpb": {str(k): f * ref / math.sqrt(k) * fz["mean"] for k in KS}, "MDE_rel_ub80": {str(k): f * ref * E.ub(fz["df"]) / math.sqrt(k) for k in KS},
            "k_needed": kn, "k_rule2": {dl: max(2, k) for dl, k in kn.items()}, "underpowered": {dl: k > 5 for dl, k in kn.items()}}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runs = {n: E.load(n) for n in ORDER if not gapped(n)}
    ck = checks(runs)
    if "--checks-only" in argv:
        print(json.dumps(ck, indent=1)); return 0
    p1 = json.load(open(os.path.join(HERE, "results.json")))
    reps = {n: E.load(n) for n in P1_REPS}
    res = {"part": "E3 Part 3, 5M arm A (3e-3, r 1), 250M slots, full WSD: INIT (seeds 1..5, data_seed 1) and DATA (seed 1, data_seeds 1..5)",
           "analyzer": {"e3analyze_p3.py": "this file", "e3analyze.py sha256 (imported, unchanged)": sha256(os.path.join(HERE, "e3analyze.py")),
                        "python": sys.version.split()[0]},
           "checks": ck, "expected_failures": EXPECTED_FAIL, "unexpected_failures": [c["check"] for c in ck if not c["ok"] and c["check"] not in EXPECTED_FAIL],
           "gaps": [n for n in ORDER if n not in runs],
           "quantiles": {"factor_df4": E.factor(4), "ub80_df4": E.ub(4), "chi2_sd_lo_df4": math.sqrt(4 / E.chi2_ppf(0.975, 4)),
                         "chi2_sd_hi_df4": math.sqrt(4 / E.chi2_ppf(0.025, 4)), "F_0.025_df4_7": f_ppf(0.025, 4, 7), "F_0.975_df4_7": f_ppf(0.975, 4, 7)},
           "values": {}, "families": {}, "rule1": {}, "compare_part1_sigma_seed": {}, "replay_vs_part1_A1": {}}
    for m in E.METRICS:
        vals = {fam: {n: E.value(runs[n], m) for n in names if n in runs} for fam, names in FAM.items()}
        res["values"][m] = vals
        fz = {fam: family(list(v.values())) if len(v) >= 4 else None for fam, v in vals.items()}
        res["families"][m] = fz
        res["rule1"][m] = {CLASS[fam]: rule1(z) if z else "no SD (fewer than 4 runs): keeps sqrt(2) x sigma_seed" for fam, z in fz.items()}
        s1 = p1["noise_5M_250M"][m]; cmp_ = {"sigma_seed_rel_part1_df7": s1["sigma_seed_rel"]}
        for fam, z in fz.items():
            if z:
                vr = (z["rel"] / s1["sigma_seed_rel"]) ** 2
                cmp_[fam] = {"ratio": math.sqrt(vr), "ratio_95ci_F": [math.sqrt(vr / f_ppf(0.975, z["df"], 7)), math.sqrt(vr / f_ppf(0.025, z["df"], 7))]}
        if fz["INIT"] and fz["DATA"]:
            cmp_["sqrt_init2_plus_data2_over_seed"] = math.hypot(fz["INIT"]["rel"], fz["DATA"]["rel"]) / s1["sigma_seed_rel"]
        res["compare_part1_sigma_seed"][m] = cmp_
        if SHARED in runs:
            rp = [E.value(reps[n], m) for n in P1_REPS]; v = E.value(runs[SHARED], m); srep = p1["replay"][m]["sigma_rep"]
            res["replay_vs_part1_A1"][m] = {"s1_P3": v, "A1_R2_R3": rp, "minus_A1": v - rp[0], "minus_rep_mean": v - st.fmean(rp),
                                            "in_sigma_rep": (v - st.fmean(rp)) / srep if srep else None, "sigma_rep_part1": srep}
    res["replay_stream_equal_A1"] = SHARED in runs and drawn(runs[SHARED]["recs"][-1]) == drawn(reps["e3_5m_A_s1"]["recs"][-1])
    res["not_run"] = {"Part 4 (a, b, c)": "not run", "Part 3 CUDA flag-off parity (default_parity.py --device cuda)": "not run"}
    json.dump(res, open(os.path.join(HERE, "results_part3.json"), "w"), indent=1, sort_keys=True)
    from e3analyze_p3_tables import tables
    open(os.path.join(HERE, "tables_part3.txt"), "w").write(tables(res))
    print(tables(res)); return 0


if __name__ == "__main__":
    sys.exit(main())
