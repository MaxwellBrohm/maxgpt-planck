from common import *
from collections import Counter, defaultdict

flat = items_flat()
assert len(flat) == 640, len(flat)
spec = []
for it in flat:
    spec.append((it["gold"], list(it["values"]), obj_words(it)))


def check_join(R):
    bad = 0
    for it, r in zip(flat, R):
        if (r["family"], r["idx"], r["vtype"], r["k"], r["d"]) != (it["family"] if "family" in it else r["family"], it["idx"], it["vtype"], it["k"], it["d"]):
            bad += 1
    return bad


def grade_all(R):
    out = []
    for (gold, pool, obj), r in zip(spec, R):
        s, f = my_strict(r["reply"], r["stop"], gold, pool, obj)
        out.append(s)
    return out


def first_line(R):
    out = []
    for (gold, pool, obj), r in zip(spec, R):
        fl = r["reply"].split("\n")[0].strip()
        s, f = my_strict(fl, "eos", gold, pool, obj)
        out.append(s)
    return out


def per_fam(R, vals):
    d = defaultdict(list)
    for r, v in zip(R, vals):
        d[r["family"]].append(v)
    return {f: sum(d[f]) / len(d[f]) for f in FAMS}


print("family check of the join (GEN record family/idx/vtype/k/d vs built item)")
for exp in ["E005", "E004"]:
    tags = ["base"] + SEEDS if exp == "E005" else SEEDS
    res = {}
    for t in tags:
        P = recs(exp, t, "gen_e004", "plain")
        C = recs(exp, t, "gen_e004", "chat")
        badj = sum((r["idx"], r["vtype"], r["k"], r["d"]) != (it["idx"], it["vtype"], it["k"], it["d"]) for it, r in zip(flat, P)) \
            + sum((r["idx"], r["vtype"], r["k"], r["d"]) != (it["idx"], it["vtype"], it["k"], it["d"]) for it, r in zip(flat, C))
        fams_ok = all(r["family"] == f for r, f in zip(P, [f for f in FAMS for _ in range(64)]))
        ps, cs, cf = grade_all(P), grade_all(C), first_line(C)
        dis_p = sum(a != r["strict"] for a, r in zip(ps, P))
        dis_c = sum(a != r["strict"] for a, r in zip(cs, C))
        pf, cfm, ff = per_fam(P, ps), per_fam(C, cs), per_fam(C, cf)
        stops = Counter(r["stop"] for r in C)
        cap48 = sum(r["n_new"] >= 48 and r["stop"] == "cap" for r in C)
        gaps = [cfm[f] - pf[f] for f in PASS]
        within = all(abs(g) <= 0.10 + 1e-9 for g in gaps)
        res[t] = within
        print(f"\n{exp} {t}: join mismatches {badj}, family order ok {fams_ok}; my strict vs stored strict disagree: plain {dis_p}/640, chat {dis_c}/640")
        print(f"  chat stops {dict(stops)}; capped at 48: {cap48}/640; n_new max {max(r['n_new'] for r in C)}")
        print("  fam        " + " ".join(f"{f:>9}" for f in FAMS))
        print("  plain str  " + " ".join(f"{pf[f]:9.3f}" for f in FAMS))
        print("  chat str   " + " ".join(f"{cfm[f]:9.3f}" for f in FAMS))
        print("  chat 1line " + " ".join(f"{ff[f]:9.3f}" for f in FAMS))
        print(f"  chat-plain over pass families: min {min(gaps):+.3f} max {max(gaps):+.3f}; within 0.10 on all 9: {within}")
        # pooled capped by family
        capf = defaultdict(int)
        for r in C:
            capf[r["family"]] += r["stop"] == "cap"
        print("  capped/fam " + " ".join(f"{capf[f]:9d}" for f in FAMS))
    n_ok = sum(res[t] for t in SEEDS)
    print(f"\n{exp}: seeds within 0.10 on every pass family: {n_ok}/5 -> stopping learned = {n_ok >= 3}")
