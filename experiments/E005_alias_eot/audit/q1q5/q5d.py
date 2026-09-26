import sys, json
sys.path.insert(0, ".")
import load as L
from collections import Counter
D = L.items(); order = [it for f in L.FAMS for it in D[f]]
rows = json.load(open("q5_rows.json"))
def intro(it):
    f = []
    for s in it["stmts"]:
        if s["obj"] not in f: f.append(s["obj"])
    return f
def pick_desc(it, v):
    s = [x for x in it["stmts"] if x["value"] == v][-1]
    fo = intro(it)
    own = [x for x in it["stmts"] if x["obj"] == s["obj"]]
    return ("obj%d" % (fo.index(s["obj"]) + 1)) + ("_latest" if s is own[-1] else "_stale") + ("_LASTSTMT" if s is it["stmts"][-1] else "") + ("_FIRSTSTMT" if s is it["stmts"][0] else "")
for exp in ("E005", "E004"):
    rr = [r for t in ["s1","s2","s3","s4","s5"] for r in rows[f"{exp}|{t}"]]
    for p in range(3):
        w = [r for r in rr if not r["lik"] and intro(order[r["i"]]).index(order[r["i"]]["asked"]) == p]
        print(exp, "asked intro", p + 1, "wrong", len(w), dict(Counter(pick_desc(order[r["i"]], r["lpick"]) for r in w).most_common(8)))
    # pick = first statement of dialogue?
    w = [r for r in rr if not r["lik"]]
    print(exp, "wrong pick = the dialogue's first statement:", sum(order[r["i"]]["stmts"][0]["value"] == r["lpick"] for r in w), "of", len(w))
# render one E005 wrong example: asked intro 2, uncorrected
shown = 0
for r in rows["E005|s3"]:
    it = order[r["i"]]
    if not r["lik"] and intro(it).index(it["asked"]) == 1 and it["k"] == 0 and shown < 1:
        shown += 1
        for t, (u, a) in enumerate(it["turns"]):
            if any(s["turn"] == t for s in it["stmts"]): print("  T%d U: %s | A: %s" % (t, u, a))
        print("  Q:", it["question"], "| gold", it["gold"], "| E005 s3 pick", r["lpick"], "| E005 picks all seeds:", [rows[f"E005|{t}"][[x['i'] for x in rows[f'E005|{t}']].index(r['i'])]["lpick"] for t in ["s1","s2","s3","s4","s5"]],
              "| E004:", [rows[f"E004|{t}"][[x['i'] for x in rows[f'E004|{t}']].index(r['i'])]["lpick"] for t in ["s1","s2","s3","s4","s5"]])
