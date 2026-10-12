"""Independent re-check of E3 Part 3 (notes.txt PART 3 ANALYZER, INDEPENDENT CHECK). Written 2026-10-11 before any Part 3
value was read. A fresh reading of the raw copied files: it imports nothing of e3analyze*, takes its quantiles from scipy
(not e3analyze's incomplete beta and gamma), and compares every family statistic, rule 1 reference, MDE, k_needed, the F
comparison with Part 1 and the replay lines with results_part3.json. Also the split and the queue order from the raw logs.
  python3 audit/p3_recheck.py      (a Python with scipy; run from experiments/E3_seed_noise or anywhere)
Exit 0 and "RECHECK PASS" only if every compared number agrees to 1e-9 relative (k counts and booleans exactly)."""
import json, math, os, sys
from scipy import stats

E3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH = "e3_5m_A_s1_P3"
FAM = {"INIT": [SH] + [f"e3_5m_A_init{k}" for k in range(2, 6)], "DATA": [SH] + [f"e3_5m_A_data{k}" for k in range(2, 6)]}
CLS = {"INIT": "same stream, new init", "DATA": "same init, new stream"}
MET = {"F(CHAT)": ("CHAT", "all", "F"), "F(PROSE)": ("PROSE", "all", "F"), "M(CHAT)": ("CHAT", "all", "M"),
       "M(PROSE)": ("PROSE", "all", "M"), "CHAT user": ("oasst2", "user", "F"), "CHAT assistant": ("oasst2", "assistant", "F"),
       "PROSE cccc": ("cccc", "all", "F"), "PROSE gutenberg": ("gutenberg", "all", "F"), "PROSE wikimedia": ("wikimedia", "all", "F"),
       "stackexchange": ("stackexchange", "all", "F"), "irc": ("irc", "all", "F"), "dolly": ("dolly", "all", "F"),
       "train loss last 2%": (None, None, "T")}
bad, n_cmp = [], 0


def lines(p): return [json.loads(x) for x in open(p) if x.strip()]


def val(run, m):
    s, sp, kind = MET[m]
    d = os.path.join(E3, "out", run)
    if kind == "T":
        xs = [r["loss"] for r in lines(os.path.join(d, "train_tail.jsonl")) if r["step"] > 7630 - 152.6]
        return math.fsum(xs) / len(xs)
    v = {r["step"]: r["bpb"] for r in lines(os.path.join(d, "bpb.jsonl")) if r["set"] == s and r["split"] == sp}
    return v[7630] if kind == "F" else math.fsum(v[k] for k in (7344, 7497, 7630)) / 3


def sd(xs):
    mu = math.fsum(xs) / len(xs)
    return math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1)), mu


def same(name, a, b, tol=1e-9):
    global n_cmp
    n_cmp += 1
    ok = a == b if isinstance(a, (bool, int)) and isinstance(b, (bool, int)) else abs(a - b) <= tol * max(1e-12, abs(b))
    if not ok:
        bad.append(f"{name}: recheck {a!r} vs results {b!r}")


R = json.load(open(os.path.join(E3, "results_part3.json")))
P1 = json.load(open(os.path.join(E3, "results.json")))
gaps = [n for n in sorted({*FAM["INIT"], *FAM["DATA"]}) if any(os.path.exists(os.path.join(E3, "out", n, g)) for g in ("GAP", "DIVERGED"))]
same("gap count", len(gaps), len(R["gaps"]))
for m in MET:
    for fam, names in FAM.items():
        names = [n for n in names if n not in gaps]
        xs = [val(n, m) for n in names]
        z = R["families"][m][fam]
        if len(xs) < 4:
            same(f"{m} {fam} no SD", z is None, True); continue
        s, mu = sd(xs); df = len(xs) - 1; rel = s / mu
        for k, a in (("n", len(xs)), ("df", df)): same(f"{m} {fam} {k}", a, z[k])
        for k, a in (("mean", mu), ("sd", s), ("rel", rel), ("ub80", s * math.sqrt(df / stats.chi2.ppf(0.2, df)))): same(f"{m} {fam} {k}", a, z[k])
        same(f"{m} {fam} ci lo", s * math.sqrt(df / stats.chi2.ppf(0.975, df)), z["ci95"][0])
        same(f"{m} {fam} ci hi", s * math.sqrt(df / stats.chi2.ppf(0.025, df)), z["ci95"][1])
        f = stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df); ref = math.sqrt(2) * rel; u = R["rule1"][m][CLS[fam]]
        same(f"{m} {fam} factor", f, u["factor"], 1e-7); same(f"{m} {fam} ref", ref, u["ref_rel"])
        for k in (1, 2, 3, 5):
            same(f"{m} {fam} MDE k{k}", f * ref / math.sqrt(k), u["MDE_rel"][str(k)], 1e-7)
            same(f"{m} {fam} MDE bpb k{k}", f * ref / math.sqrt(k) * mu, u["MDE_bpb"][str(k)], 1e-7)
        for dl, key in ((0.005, "0.500%"), (0.01, "1.000%"), (0.02, "2.000%")):
            kn = 1
            while f * ref / math.sqrt(kn) > dl: kn += 1
            same(f"{m} {fam} k_needed {key}", kn, u["k_needed"][key]); same(f"{m} {fam} k {key}", max(2, kn), u["k_rule2"][key])
            same(f"{m} {fam} underpowered {key}", kn > 5, u["underpowered"][key])
        s1 = P1["noise_5M_250M"][m]["sigma_seed_rel"]; c = R["compare_part1_sigma_seed"][m][fam]; ratio = rel / s1
        same(f"{m} {fam} ratio", ratio, c["ratio"])
        same(f"{m} {fam} ratio lo", ratio / math.sqrt(stats.f.ppf(0.975, df, 7)), c["ratio_95ci_F"][0], 1e-7)
        same(f"{m} {fam} ratio hi", ratio / math.sqrt(stats.f.ppf(0.025, df, 7)), c["ratio_95ci_F"][1], 1e-7)
    if SH not in gaps:
        v = val(SH, m); rp = [val(n, m) for n in ("e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3")]
        x = R["replay_vs_part1_A1"][m]
        same(f"{m} replay minus A1", v - rp[0], x["minus_A1"], 1e-9); same(f"{m} replay minus mean", v - math.fsum(rp) / 3, x["minus_rep_mean"], 1e-9)
strm = lambda r: {k: v for k, v in r.items() if k.startswith(("drawn_", "dropped_")) or k in ("tokens", "sup_tokens", "step")}  # noqa: E731
tail = {n: [strm(r) for r in lines(os.path.join(E3, "out", n, "train_tail.jsonl"))] for n in FAM["INIT"] + FAM["DATA"] if n not in gaps}
end = {n: strm(json.load(open(os.path.join(E3, "out", n, "run_records.json")))[-1]) for n in tail}
same("split INIT same stream", all(tail[n] == tail[SH] and end[n] == end[SH] for n in FAM["INIT"] if n in tail), True)
same("split DATA distinct", len({json.dumps({k: v for k, v in end[n].items() if k != "step"}, sort_keys=True) for n in FAM["DATA"] if n in tail}),
     len([n for n in FAM["DATA"] if n in tail]))
q = open(os.path.join(E3, "logs", "part3", "queue_e3.txt")).read().split("\n")
i0 = max(i for i, ln in enumerate(q) if "plan finished: part2.txt" in ln)
done = [ln.split("done ")[1].split(" rc")[0] for ln in q[i0:] if " done e3_5m_A_" in ln and ln.endswith(" rc 0")]
same("queue order", done == ["e3_5m_A_init2", "e3_5m_A_data2", "e3_5m_A_init3", "e3_5m_A_data3", SH, "e3_5m_A_init4",
                             "e3_5m_A_data4", "e3_5m_A_init5", "e3_5m_A_data5"], True)
print(f"{n_cmp} comparisons, {len(bad)} disagree" + "".join("\n  " + b for b in bad[:40]))
print("RECHECK PASS" if not bad else "RECHECK FAIL")
sys.exit(1 if bad else 0)
