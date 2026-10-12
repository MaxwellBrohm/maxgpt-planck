import sys, collections, re
sys.dont_write_bytecode = True
import load, mygrade as G
d = load.eval_items()
NAME = re.compile(r"\b(Mr|Mrs|Ms|Dr)\. ([A-Z][a-z]+)")
alias = [it for f in ("H1", "H2") for it in d[f] if it["meta"].get("latest_ref") == "alias"]
# LN oracle on the 42 E004-eval alias items: latest value-bearing user turn that holds a title+name
ln = 0; nnames = collections.Counter()
for it in alias:
    st = []
    for ti, (u, a) in enumerate(it["turns"]):
        vals = [v for v in it["values"] if G.mentions(u, v)]
        if vals: st.append((ti, vals, NAME.findall(u)))
    named = [s for s in st if s[2]]
    ln += len(named[-1][1]) == 1 and named[-1][1][0] == it["gold"]
    nnames[len({n for s in st for n in s[2]})] += 1
print("E004 eval alias items:", len(alias), "| latest-statement-with-a-name oracle right:", ln, "| distinct names per item:", dict(nnames))
def cls(it, v):
    about = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    last = about[-1]
    if v == last["value"]: return "gold"
    if v == about[-2]["value"]: return "A_prev"
    if any(s["value"] == v for s in about[:-2]): return "A_earlier"
    oth = [s for s in it["stmts"] if s["obj"] != it["asked"] and s["value"] == v]
    if oth: return "other_after" if oth[-1]["turn"] > last["turn"] else "other_before"
    return "not_in_context"
for exp in ("e005", "e004"):
    c = collections.Counter(); cg = collections.Counter()
    for s in range(1, 6):
        L = {(r["family"], r["idx"]): r for r in load.recs(exp, f"s{s}", "e004", "plain")}
        for it in alias:
            r = L[(it["family"], it["idx"])]
            if not load.my_right(r["scores"]): c[cls(it, load.lik_pick(r))] += 1
    print(exp, "wrong LIK picks on the 42 alias items, pooled over 5 seeds:", sum(c.values()), dict(c))
