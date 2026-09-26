import sys, collections
sys.dont_write_bytecode = True
import alparse as AP
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
