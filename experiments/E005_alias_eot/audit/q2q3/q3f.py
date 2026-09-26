import sys, json, collections
sys.dont_write_bytecode = True
import load, alparse as AP
import q3b_oracles as O
items = {it["idx"]: it for it in load.al_items()}
P = {i: AP.parse(it) for i, it in items.items()}
my = json.load(open("al_my_scores.json"))
for nm, fn in O.ORACLES.items():
    if nm in ("title_only_link", "LINK(full name)"): continue
    def hyb(i):
        v = O.o_TITLE(items[i], P[i])
        return v if v is not None else fn(items[i], P[i])
    wrong = [i for i in items if hyb(i) != items[i]["gold"]]
    tot = sum(1 for i in items if hyb(i) == items[i]["gold"])
    row = []
    for grp in ("e005", "e004"):
        ok = n = 0
        for s in range(1, 6):
            for render in ("plain", "chat"):
                for meas in ("lik", "gen"):
                    d = my[f"{grp}_s{s}"][render][meas]
                    ok += sum(d[str(i)] for i in wrong); n += len(wrong)
        row.append(f"{grp} {ok}/{n}={ok/n:.3f}" if n else f"{grp} -")
    print(f"hybrid title-link else {nm:28s}: scores {tot}/64; items it gets wrong {len(wrong):2d} ({dict(collections.Counter(items[i]['cell'] for i in wrong))}); model right there (5 seeds x 2 renders x LIK+GEN): {row}")
