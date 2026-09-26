import sys, json
sys.path.insert(0, ".")
import load as L
from collections import Counter
D = L.items(); order = [it for f in L.FAMS for it in D[f]]
rows = json.load(open("q5_rows.json"))
def lastval(it): return it["stmts"][-1]["value"]
def pos_of_asked(it):
    firsts = []
    for s in it["stmts"]:
        if s["obj"] not in firsts: firsts.append(s["obj"])
    return firsts.index(it["asked"])
for exp in ("E005", "E004"):
    rr = [r for t in ["s1","s2","s3","s4","s5"] for r in rows[f"{exp}|{t}"]]
    lastgold = [r for r in rr if lastval(order[r["i"]]) == order[r["i"]]["gold"]]
    notlast = [r for r in rr if lastval(order[r["i"]]) != order[r["i"]]["gold"]]
    agree_last = sum(r["lpick"] == lastval(order[r["i"]]) for r in rr)
    wrong = [r for r in rr if not r["lik"]]
    print(exp, f"gold is the dialogue's last statement: LIK {sum(r['lik'] for r in lastgold)}/{len(lastgold)}; not last: {sum(r['lik'] for r in notlast)}/{len(notlast)}")
    print(f"   LIK pick = dialogue's last statement value: {agree_last}/{len(rr)} all; wrong picks {sum(r['lpick']==lastval(order[r['i']]) for r in wrong)}/{len(wrong)}")
    for p in range(3):
        q = [r for r in rr if pos_of_asked(order[r["i"]]) == p]
        print(f"   asked object introduced {p+1}{'st' if p==0 else 'nd' if p==1 else 'rd'}: LIK {sum(r['lik'] for r in q)}/{len(q)} = {sum(r['lik'] for r in q)/len(q):.3f}")
# per item: how many seeds right (E005 vs E004), items consistently wrong
per = {}
for exp in ("E005", "E004"):
    c = Counter()
    for t in ["s1","s2","s3","s4","s5"]:
        for r in rows[f"{exp}|{t}"]: c[r["i"]] += r["lik"]
    per[exp] = c
print("items right on k of 5 seeds (E005):", dict(sorted(Counter(per["E005"].values()).items())))
print("items right on k of 5 seeds (E004):", dict(sorted(Counter(per["E004"].values()).items())))
diff = Counter(per["E005"][i] - per["E004"][i] for i in per["E005"])
print("per-item E005 minus E004 seeds-right:", dict(sorted(diff.items())))
# seed-paired
for t in ["s1","s2","s3","s4","s5"]:
    a = rows[f"E005|{t}"]; b = rows[f"E004|{t}"]
    print(t, "E005 LIK", sum(r["lik"] for r in a), "E004 LIK", sum(r["lik"] for r in b), "| E005 GEN", sum(r["gen"] for r in a), "E004 GEN", sum(r["gen"] for r in b),
          "| both right", sum(x["lik"] and y["lik"] for x, y in zip(a, b)), "E005 only", sum(x["lik"] and not y["lik"] for x, y in zip(a, b)), "E004 only", sum(y["lik"] and not x["lik"] for x, y in zip(a, b)))
