import sys, json
sys.path.insert(0, ".")
import load as L, mygrade as G
from collections import defaultdict
D = L.items()
order = [it for f in L.FAMS for it in D[f]]
assert len(order) == 640
out = {}
mism = defaultdict(int)
for tag in L.TAGS:
    lik = L.recs(tag, "e004", "plain"); gen = L.recs(tag, "gen_e004", "plain")
    assert len(lik) == 640 and len(gen) == 640
    cell = defaultdict(lambda: [0, 0, 0, 0])  # likR, n, genR, n
    for i, (it, r, g) in enumerate(zip(order, lik, gen)):
        gold = L.my_gold(it)
        # LIK matching
        if not (r["id"] == i and r["family"] == it["family"] and r["idx"] == it["idx"]): mism[(tag, "lik_id")] += 1
        if r["h"] != L.phash(L.plain_prompt(it, True)): mism[(tag, "lik_hash")] += 1
        if r["cand_vals"]["gold"] != gold: mism[(tag, "lik_gold")] += 1
        if sorted(r["cand_vals"].values()) != sorted(L.my_cands(it)): mism[(tag, "lik_cands")] += 1
        lr = L.lik_right(r["scores"])
        if lr != r["right"]: mism[(tag, "lik_right_flag")] += 1
        # GEN matching
        if not (g["id"] == i and g["family"] == it["family"] and g["idx"] == it["idx"] and g["vtype"] == it["vtype"] and g["k"] == it["k"]): mism[(tag, "gen_id")] += 1
        if g.get("latest_ref") != it.get("meta", {}).get("latest_ref"): mism[(tag, "gen_ref")] += 1
        phrase, head = it["objects"][it["asked"]]
        alias = it["alias"] if it.get("alias_obj") == it["asked"] else None
        ok, fails = G.grade(g["reply"], g["stop"], gold, it["values"], G.objwords(phrase, head, alias))
        if ok != g["strict"] or sorted(fails) != sorted(g["fails"]): mism[(tag, "gen_strict_flag")] += 1; print("GEN DIFF", tag, i, g["reply"], fails, g["fails"])
        c = cell[it["family"]]
        c[0] += lr; c[1] += 1; c[2] += ok; c[3] += 1
    out[tag] = {f: tuple(v) for f, v in cell.items()}
print("mismatches:", dict(mism) if mism else "none")
json.dump(out, open("q1_cells.json", "w"))
print("%-10s" % "fam" + "".join("%16s" % t for t in L.TAGS))
for f in L.FAMS:
    print("%-10s" % f + "".join("%16s" % ("%2d/%2d=.%03d" % (out[t][f][0], out[t][f][2], 0) if False else "%d|%d" % (out[t][f][0], out[t][f][2])) for t in L.TAGS))
