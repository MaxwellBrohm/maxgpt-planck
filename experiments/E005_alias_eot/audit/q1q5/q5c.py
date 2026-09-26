import sys, json
sys.path.insert(0, ".")
import load as L
from collections import Counter, defaultdict
D = L.items(); order = [it for f in L.FAMS for it in D[f]]
rows = json.load(open("q5_rows.json"))
def intro(it):
    f = []
    for s in it["stmts"]:
        if s["obj"] not in f: f.append(s["obj"])
    return f.index(it["asked"])
def n_objs_after(it):
    about = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    return len({s["obj"] for s in it["stmts"] if s["obj"] != it["asked"] and s["turn"] > about[-1]["turn"]})
def asked_first_stmt_rank(it):
    return [s["obj"] for s in it["stmts"]].index(it["asked"])
for exp in ("E005", "E004"):
    rr = [r for t in ["s1","s2","s3","s4","s5"] for r in rows[f"{exp}|{t}"]]
    def acc(cond):
        q = [r for r in rr if cond(order[r["i"]], r)]
        return f"{sum(r['lik'] for r in q)}/{len(q)}={sum(r['lik'] for r in q)/max(1,len(q)):.2f}"
    print(exp, "by asked index:", [acc(lambda it, r, a=a: it["asked"] == a) for a in range(3)])
    print("   by #other objects stated after asked latest:", [acc(lambda it, r, n=n: n_objs_after(it) == n) for n in (1, 2)])
    print("   intro order x corrected: ", {(p, c): acc(lambda it, r, p=p, c=c: intro(it) == p and ((it['k'] > 0) == c)) for p in range(3) for c in (False, True)})
# item counts
print(Counter((intro(it), n_objs_after(it)) for it in D["H5"]))
