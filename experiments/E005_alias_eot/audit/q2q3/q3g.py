"""structural (no-name) heuristics that use the COUNT and ORDER of alias corrections, never the name string."""
import sys, json, collections
sys.dont_write_bytecode = True
import load, alparse as AP
items = {it["idx"]: it for it in load.al_items()}
P = {i: AP.parse(it) for i, it in items.items()}
my = json.load(open("al_my_scores.json"))
def named(it, Pi): return [p for p in Pi if p["obj"] == it["asked"]]
def HA(it, Pi):
    """one alias correction -> ignore it; two -> the first alias correction after the asked object's original,
    if it is later than the asked object's latest named statement"""
    al = [p for p in Pi if p["obj"] is None]
    nm = named(it, Pi)
    if len(al) < 2: return nm[-1]["value"]
    after = [p for p in al if p["turn"] > nm[0]["turn"]]
    cand = after[0] if after else None
    return cand["value"] if cand and cand["turn"] > nm[-1]["turn"] else nm[-1]["value"]
def HB(it, Pi):
    """each alias correction claims the most recent not-yet-claimed ORIGINAL that holds a name (LIFO)"""
    stack, att = [], {}
    for p in Pi:
        if p["obj"] is not None and p["names"]: stack.append(p["obj"])
        elif p["obj"] is None and stack: att[p["turn"]] = stack.pop()
    a = [p for p in Pi if p["obj"] == it["asked"] or att.get(p["turn"]) == it["asked"]]
    return a[-1]["value"]
def HC(it, Pi):
    """FIFO: each alias correction claims the earliest not-yet-claimed original that holds a name"""
    q, att = [], {}
    for p in Pi:
        if p["obj"] is not None and p["names"]: q.append(p["obj"])
        elif p["obj"] is None and q: att[p["turn"]] = q.pop(0)
    a = [p for p in Pi if p["obj"] == it["asked"] or att.get(p["turn"]) == it["asked"]]
    return a[-1]["value"]
def HD(it, Pi):
    """one alias correction -> ignore it; two -> FIFO claim"""
    al = [p for p in Pi if p["obj"] is None]
    return named(it, Pi)[-1]["value"] if len(al) < 2 else HC(it, Pi)
def HE(it, Pi):
    """one alias correction -> ignore it; two -> LIFO claim"""
    al = [p for p in Pi if p["obj"] is None]
    return named(it, Pi)[-1]["value"] if len(al) < 2 else HB(it, Pi)
for nm, fn in (("HA one->ignore, two->first alias after asked original", HA), ("HB LIFO claim", HB), ("HC FIFO claim", HC),
               ("HD one->ignore, two->FIFO", HD), ("HE one->ignore, two->LIFO", HE)):
    per = collections.Counter(); wrong = []
    for i, it in items.items():
        ok = fn(it, P[i]) == it["gold"]; per[it["cell"]] += ok
        if not ok: wrong.append(i)
    row = []
    for grp in ("e005", "e004"):
        v = []
        for s in range(1, 6):
            v.append(sum(my[f"{grp}_s{s}"]["plain"]["lik"][str(i)] for i in wrong))
        row.append(f"{grp} LIK plain right per seed {v} of {len(wrong)}")
    print(f"{nm:52s} AL1 {per['AL1']:2d} AL2 {per['AL2']:2d} AL3 {per['AL3']:2d} AL4 {per['AL4']:2d} = {sum(per.values())}/64 | fails: {dict(collections.Counter((items[i]['cell'], items[i]['meta']['placement']) for i in wrong))}")
    print(f"{'':52s} {row}")
# combined: items that EVERY heuristic (these five + the earlier oracles, positional ones excluded) gets wrong
import q3b_oracles as O
allfns = [HA, HB, HC, HD, HE] + [fn for n, fn in O.ORACLES.items() if n not in ("LINK(full name)", "title_only_link") and not n.startswith("stmt_last")]
hard = [i for i, it in items.items() if all(fn(it, P[i]) != it["gold"] for fn in allfns)]
print("items every non-positional, non-name heuristic gets wrong:", hard, [(items[i]["cell"], items[i]["meta"]["placement"]) for i in hard])
