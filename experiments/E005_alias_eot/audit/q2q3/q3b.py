import sys, json, collections
sys.dont_write_bytecode = True
import load, alparse as AP
items = load.al_items()
assert len(items) == 64
cells = ["AL1", "AL2", "AL3", "AL4"]
# ---- oracles (text-level) ----
def asked_names(it, P):
    """names introduced in the asked object's original (join), from text"""
    return [n for p in P if p["obj"] == it["asked"] for n in p["names"]]
def o_AR(it, P):
    al = [p for p in P if p["obj"] is None and p["names"]]
    return al[-1]["value"] if al else None
def o_LN(it, P):
    n = [p for p in P if p["names"]]
    return n[-1]["value"] if n else None
def o_LAST(it, P):
    return P[-1]["value"]
def o_FIRST(it, P):
    a = [p for p in P if p["obj"] == it["asked"]]
    return a[0]["value"]
def o_NOALIAS(it, P):
    a = [p for p in P if p["obj"] == it["asked"]]
    return a[-1]["value"]
def o_T(it, P):   # topic tracker: an unnamed-object statement belongs to the object of the most recent statement
    cur, att = None, []
    for p in P:
        o = p["obj"] if p["obj"] is not None else cur
        att.append(o); cur = o
    a = [p for p, o in zip(P, att) if o == it["asked"]]
    return a[-1]["value"] if a else None
def o_T1(it, P):  # strict adjacency: unnamed statement belongs to the object of the statement in the previous TURN, else dropped
    att, prev_turn, prev_obj = [], None, None
    for p in P:
        if p["obj"] is not None: o = p["obj"]
        elif prev_turn is not None and p["turn"] == prev_turn + 1: o = prev_obj
        else: o = None
        att.append(o); prev_turn, prev_obj = p["turn"], o
    a = [p for p, o in zip(P, att) if o == it["asked"]]
    return a[-1]["value"] if a else None
def o_LINK(it, P):  # text-level linking by full name (the intended skill)
    names = set(asked_names(it, P))
    a = [p for p in P if p["obj"] == it["asked"] or (p["obj"] is None and set(p["names"]) & names)]
    return a[-1]["value"]
def o_TITLE(it, P):  # linking by honorific only; ambiguous if both objects' aliases share the title -> None
    own = {n.split()[0] for n in asked_names(it, P)}
    other = {n.split()[0] for p in P if p["obj"] not in (None, it["asked"]) for n in p["names"]}
    a = []
    for p in P:
        if p["obj"] == it["asked"]: a.append(p)
        elif p["obj"] is None:
            t = p["names"][0].split()[0]
            if t in own and t in other: return None
            if t in own: a.append(p)
    return a[-1]["value"]
def o_ORDER(it, P):  # alias corrections assigned to aliased objects in the order their originals appeared
    origs = [p["obj"] for p in P if p["obj"] is not None and p["names"]]
    seen = []
    for o in origs:
        if o not in seen: seen.append(o)
    als = [p for p in P if p["obj"] is None]
    att = {}
    if len(als) == 1:
        # one alias correction: give it to the only object that has a name ... if both have one, the first-defined
        att[als[0]["turn"]] = seen[0] if len(seen) else None
    else:
        for p, o in zip(als, seen): att[p["turn"]] = o
    a = [p for p in P if p["obj"] == it["asked"] or att.get(p["turn"]) == it["asked"]]
    return a[-1]["value"] if a else None
def pos_from_end(k):
    return lambda it, P: P[-k]["value"] if len(P) >= k else None
ORACLES = {"alias_recency(AR)": o_AR, "latest_any_name(LN)": o_LN, "latest_any_object": o_LAST,
           "asked_first_mention": o_FIRST, "asked_latest_ignoring_alias": o_NOALIAS,
           "topic_tracker(T)": o_T, "strict_adjacency(T1)": o_T1, "order_match": o_ORDER,
           "title_only_link": o_TITLE, "LINK(full name)": o_LINK,
           "stmt_last-1": pos_from_end(2), "stmt_last-2": pos_from_end(3), "stmt_last-3": pos_from_end(4)}
parsed = {}
for it in items:
    P = AP.parse(it); AP.check_vs_meta(it, P); parsed[it["idx"]] = P
print("parsed 64 items from text; statement turns, values, names and object mentions agree with stmts metadata")
res = {}
print(f"{'oracle':30s} " + " ".join(f"{c:>6s}" for c in cells) + "    all  AL1+2+4")
for name, fn in ORACLES.items():
    per = collections.Counter(); amb = 0
    right = {}
    for it in items:
        v = fn(it, parsed[it["idx"]])
        if v is None: amb += 1
        ok = v == it["gold"]; right[it["idx"]] = ok
        per[it["cell"]] += ok
    res[name] = right
    tot = sum(per.values()); t3 = per["AL1"] + per["AL2"] + per["AL4"]
    print(f"{name:30s} " + " ".join(f"{per[c]/16:6.2f}" for c in cells) + f"  {tot/64:5.2f}  {t3/48:5.2f}" + (f"   (undecided {amb})" if amb else ""))
json.dump(res, open("oracle_right.json", "w"))
# gold position from the end, per cell
pc = collections.Counter()
for it in items:
    P = parsed[it["idx"]]
    k = len(P) - [i for i, p in enumerate(P) if p["value"] == it["gold"]][-1]
    pc[(it["cell"], k)] += 1
print("gold position from the end of the statement list (1 = last):", sorted(pc.items()))
# needs linking set: T and LN both wrong
nl = [it["idx"] for it in items if not res["topic_tracker(T)"][it["idx"]] and not res["latest_any_name(LN)"][it["idx"]]]
print("needs-linking (T and LN wrong):", len(nl), collections.Counter(items[i]["cell"] for i in nl))
nl2 = [it["idx"] for it in items if not any(res[o][it["idx"]] for o in res if o not in ("LINK(full name)",))]
print("items where EVERY non-linking oracle above is wrong:", len(nl2), collections.Counter(items[i]["cell"] for i in nl2))
json.dump(dict(nl=nl, nl2=nl2), open("sets.json", "w"))
# honorific sharing among two-alias items
sh = collections.Counter()
for it in items:
    al = [a for a in it["aliases"] if a]
    if len(al) == 2: sh[(it["cell"], al[0].split()[0] == al[1].split()[0])] += 1
print("two-alias items, same honorific (True) or not:", sorted(sh.items()))
