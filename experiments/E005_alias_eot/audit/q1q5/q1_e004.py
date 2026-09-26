import sys, json
sys.path.insert(0, ".")
import load as L, mygrade as G
D = L.items(); order = [it for f in L.FAMS for it in D[f]]
E5 = json.load(open("q1_cells.json"))
E4 = {}
bad = 0
for tag in ["s1","s2","s3","s4","s5"]:
    lik = L.recs(tag, "e004", "plain", L.E4); gen = L.recs(tag, "gen_e004", "plain", L.E4)
    c = {f: [0, 0] for f in L.FAMS}
    for i, (it, r, g) in enumerate(zip(order, lik, gen)):
        bad += r["h"] != L.phash(L.plain_prompt(it, True)) or r["cand_vals"]["gold"] != L.my_gold(it) or g["idx"] != it["idx"] or g["family"] != it["family"]
        ph, hd = it["objects"][it["asked"]]; al = it["alias"] if it.get("alias_obj") == it["asked"] else None
        ok, _ = G.grade(g["reply"], g["stop"], L.my_gold(it), it["values"], G.objwords(ph, hd, al))
        bad += ok != g["strict"]
        c[it["family"]][0] += L.lik_right(r["scores"]); c[it["family"]][1] += ok
    E4[tag] = c
print("E004 mismatches:", bad)
for f in L.FAMS:
    a5 = [E5[t][f][0] for t in ["s1","s2","s3","s4","s5"]]; g5 = [E5[t][f][2] for t in ["s1","s2","s3","s4","s5"]]
    a4 = [E4[t][f][0] for t in ["s1","s2","s3","s4","s5"]]; g4 = [E4[t][f][1] for t in ["s1","s2","s3","s4","s5"]]
    print(f"{f:10s} E004 LIK {a4} GEN {g4} | E005 LIK {a5} GEN {g5} | diff LIK {(sum(a5)-sum(a4))/320:+.4f} GEN {(sum(g5)-sum(g4))/320:+.4f}")
