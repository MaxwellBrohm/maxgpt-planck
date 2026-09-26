import sys, json, collections, statistics
sys.dont_write_bytecode = True
import load, alparse as AP, mygrade as G
import q3b_oracles as O
items = {it["idx"]: it for it in load.al_items()}
P = {i: AP.parse(it) for i, it in items.items()}
tags = ["base"] + [f"e005_s{i}" for i in range(1, 6)] + [f"e004_s{i}" for i in range(1, 6)]
S = json.load(open("sets.json"))
both = {i for i, it in items.items() if len([a for a in it["aliases"] if a]) == 2}
same = {i for i in both if items[i]["aliases"][0].split()[0] == items[i]["aliases"][1].split()[0]}
pred = {name: {i: fn(items[i], P[i]) for i in items} for name, fn in O.ORACLES.items()}
print("both-aliased items:", len(both), "same honorific:", len(same))
print("\n(1) LIK plain: share of the model's picks equal to each oracle's value, ON ITEMS WHERE THE ORACLE IS WRONG (n in header)")
names = [n for n in O.ORACLES if n != "LINK(full name)"]
wrongsets = {n: [i for i in items if pred[n][i] is not None and pred[n][i] != items[i]["gold"]] for n in names}
print(f"{'tag':9s} " + " ".join(f"{n[:14]:>14s}" for n in names))
print(f"{'n':9s} " + " ".join(f"{len(wrongsets[n]):14d}" for n in names))
for tag in tags:
    L = {r["idx"]: r for r in load.al_recs(tag, "al", "plain")}
    row = []
    for n in names:
        ids = wrongsets[n]
        row.append(sum(load.lik_pick(L[i]) == pred[n][i] for i in ids) / len(ids) if ids else float("nan"))
    print(f"{tag:9s} " + " ".join(f"{x:14.2f}" for x in row))
print("\n(2) pooled over 5 seeds x plain+chat, LIK and GEN right: both-aliased same-title (15) vs different-title (33) vs one alias (16)")
for grp in ("e005", "e004"):
    for meas in ("al", "gen_al"):
        acc = collections.Counter(); n = collections.Counter()
        for s in range(1, 6):
            for render in ("plain", "chat"):
                for r in load.al_recs(f"{grp}_s{s}", meas, render):
                    i = r["idx"]
                    ok = load.my_right(r["scores"]) if meas == "al" else r["strict"]
                    k = "same_title" if i in same else ("diff_title" if i in both else "one_alias")
                    acc[k] += ok; n[k] += 1
        print(grp, "LIK" if meas == "al" else "GEN(stored strict; my grader agreed 100%)", {k: f"{acc[k]}/{n[k]}={acc[k]/n[k]:.3f}" for k in ("same_title", "diff_title", "one_alias")})
print("\n(3) E005 wrong LIK picks (plain and chat), classified")
for tag in [f"e005_s{i}" for i in range(1, 6)]:
    for render in ("plain", "chat"):
        for r in load.al_recs(tag, "al", render):
            if load.my_right(r["scores"]): continue
            i = r["idx"]; it = items[i]; pk = load.lik_pick(r)
            cls = [n for n in ("alias_recency(AR)", "latest_any_name(LN)", "latest_any_object", "asked_first_mention", "asked_latest_ignoring_alias", "topic_tracker(T)", "strict_adjacency(T1)") if pred[n][i] == pk]
            src = [s for s in it["stmts"] if s["value"] == pk][-1]
            print(f"  {tag} {render} idx {i} {it['cell']} {it['meta']['placement']:9s} gold {it['gold']:9s} pick {pk:9s} (= obj {src['obj']} {src['role']}/{src['ref']}; asked {it['asked']}) margin {r['scores']['gold'] - max(v for k, v in r['scores'].items() if k != 'gold'):+.2f} oracles agreeing: {cls}")
print("\n(4) LIK plain margin (gold minus best other, nats), median [min] per group")
for tag in tags:
    L = {r["idx"]: r for r in load.al_recs(tag, "al", "plain")}
    m = {i: L[i]["scores"]["gold"] - max(v for k, v in L[i]["scores"].items() if k != "gold") for i in L}
    def st(ids): 
        xs = [m[i] for i in ids]; return f"{statistics.median(xs):+6.2f} [{min(xs):+6.2f}]"
    print(f"  {tag:9s} needs-linking14 {st(S['nl'])}  other50 {st([i for i in items if i not in S['nl']])}  same_title15 {st(same)}")
