import sys, json
sys.path.insert(0, ".")
import load as L, mygrade as G
from collections import Counter, defaultdict
D = L.items()
order = [it for f in L.FAMS for it in D[f]]
H5 = [(i, it) for i, it in enumerate(order) if it["family"] == "H5"]
assert len(H5) == 64

def info(it):
    st = it["stmts"]; a = it["asked"]
    about = [s for s in st if s["obj"] == a]
    lat = about[-1]
    after = [s for s in st if s["obj"] != a and s["turn"] > lat["turn"]]
    after_ind = any(s["ref"] in ("pron", "ell") for s in after)
    form = "never" if lat["role"] == "orig" else lat["ref"]
    return dict(lat=lat, about=about, after=after, after_ind=after_ind, form=form, n_after=len(after),
                last=st[-1])

def classify(it, v):
    I = info(it)
    if v is None: return "none"
    if v == it["gold"]: return "gold"
    hold = [s for s in it["stmts"] if s["value"] == v]
    if not hold: return "not_in_context"
    s = hold[-1]
    if s["obj"] == it["asked"]:
        return "A_prev" if s is I["about"][-2] else "A_earlier(orig)" if s["role"] == "orig" else "A_earlier"
    if s["turn"] > I["lat"]["turn"]:
        return "other_after"
    return "other_before"

def detail_other_after(it, v):
    s = [x for x in it["stmts"] if x["value"] == v][-1]
    own = [x for x in it["stmts"] if x["obj"] == s["obj"]]
    return dict(current=s is own[-1], indirect=s["ref"] in ("pron", "ell"), last_in_dialogue=s is it["stmts"][-1], ref=s["ref"])

# sanity: my after_ind vs plan meta
mm = sum(info(it)["after_ind"] != it["meta"]["after_ind"] for _, it in H5)
print("after_ind mismatches vs item meta:", mm, "| after_ind items:", sum(info(it)["after_ind"] for _, it in H5),
      "| latest forms:", dict(Counter(info(it)["form"] for _, it in H5)))
print("asked_k:", dict(Counter(it["meta"]["asked_k"] for _, it in H5)), " ntok>768 not checked here")

def gen_value(it, reply):
    # first in-pool value mentioned (lenient-style)
    firsts = []
    for v in it["values"]:
        h = G.mentions((reply or "").replace("’", "'"), v)
        if h: firsts.append((h[0], v))
    return min(firsts)[1] if firsts else None

res = {}
for exp, root in (("E005", L.E5), ("E004", L.E4)):
    for tag in ["base", "s1", "s2", "s3", "s4", "s5"]:
        if exp == "E004" and tag == "base": continue
        lik = L.recs(tag, "e004", "plain", root); gen = L.recs(tag, "gen_e004", "plain", root)
        rows = []
        for i, it in H5:
            r, g = lik[i], gen[i]
            assert r["h"] == L.phash(L.plain_prompt(it, True)) and r["cand_vals"]["gold"] == L.my_gold(it) and r["family"] == "H5"
            assert g["family"] == "H5" and g["idx"] == it["idx"]
            lr = L.lik_right(r["scores"])
            pick = L.lik_pick(r)
            phrase, head = it["objects"][it["asked"]]
            ok, _ = G.grade(g["reply"], g["stop"], L.my_gold(it), it["values"], G.objwords(phrase, head, None))
            assert ok == g["strict"]
            gv = gen_value(it, g["reply"])
            rows.append(dict(i=i, lik=lr, gen=ok, lcls=("tie" if (not lr and pick == it["gold"]) else classify(it, pick) if not lr else "gold"),
                             gcls=("gold_fmt" if (not ok and gv == it["gold"]) else classify(it, gv) if not ok else "gold"),
                             lpick=pick, gpick=gv, **{k: v for k, v in info(it).items() if k in ("after_ind", "form", "n_after")}))
        res[(exp, tag)] = rows
json.dump({f"{e}|{t}": v for (e, t), v in res.items()}, open("q5_rows.json", "w"))

def frac(rows, key, cond=lambda r: True):
    rr = [r for r in rows if cond(r)]
    return f"{sum(r[key] for r in rr)}/{len(rr)}"

print("\nH5 per seed: LIK, GEN | split after_ind yes/no (LIK) | latest form (LIK): never, full, head, pron, ell")
for (exp, tag), rows in res.items():
    forms = ["never", "full", "head", "pron", "ell"]
    print(f"{exp} {tag:4s} LIK {frac(rows,'lik')} GEN {frac(rows,'gen')} | ind-after {frac(rows,'lik',lambda r:r['after_ind'])} none {frac(rows,'lik',lambda r:not r['after_ind'])}"
          f" | GEN ind-after {frac(rows,'gen',lambda r:r['after_ind'])} none {frac(rows,'gen',lambda r:not r['after_ind'])} | " +
          " ".join(f"{f} {frac(rows,'lik',lambda r,f=f:r['form']==f)}" for f in forms))

print("\nPooled over s1-s5 (LIK / GEN):")
for exp in ("E005", "E004"):
    rows = [r for t in ["s1","s2","s3","s4","s5"] for r in res[(exp, t)]]
    def pf(cond, key="lik"):
        rr = [r for r in rows if cond(r)]; return f"{sum(r[key] for r in rr)}/{len(rr)}={sum(r[key] for r in rr)/len(rr):.3f}"
    print(exp, "all", pf(lambda r: True), pf(lambda r: True, "gen"))
    print("  after_ind yes", pf(lambda r: r["after_ind"]), pf(lambda r: r["after_ind"], "gen"), " no", pf(lambda r: not r["after_ind"]), pf(lambda r: not r["after_ind"], "gen"))
    for f in ["never", "full", "head", "pron", "ell"]:
        print(f"  form {f:5s}", pf(lambda r, f=f: r["form"] == f), pf(lambda r, f=f: r["form"] == f, "gen"),
              "| with ind-after", pf(lambda r, f=f: r["form"] == f and r["after_ind"]), "| without", pf(lambda r, f=f: r["form"] == f and not r["after_ind"]))
    for n in range(0, 7):
        rr = [r for r in rows if r["n_after"] == n]
        if rr: print(f"  n other stmts after asked latest = {n}: LIK {sum(r['lik'] for r in rr)}/{len(rr)}")
    print("  LIK wrong classes:", dict(Counter(r["lcls"] for r in rows if r["lcls"] != "gold")))
    print("  GEN wrong classes:", dict(Counter(r["gcls"] for r in rows if r["gcls"] != "gold")))
    print("  LIK wrong per seed:", {t: dict(Counter(r["lcls"] for r in res[(exp, t)] if r["lcls"] != "gold")) for t in ["s1","s2","s3","s4","s5"]})
    # other_after details
    det = Counter()
    for t in ["s1","s2","s3","s4","s5"]:
        for r in res[(exp, t)]:
            if r["lcls"] == "other_after":
                it = order[r["i"]]; dd = detail_other_after(it, r["lpick"])
                det["n"] += 1; det["current"] += dd["current"]; det["indirect"] += dd["indirect"]; det["last_in_dialogue"] += dd["last_in_dialogue"]; det["ref_" + dd["ref"]] += 1
    print("  LIK other_after detail:", dict(det))
