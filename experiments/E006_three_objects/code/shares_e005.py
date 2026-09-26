"""E005 step 2: stream shares vs notes.txt, as a function so the mutation test can run it on mutated streams.
share_checks(X) -> [(ok, message)] over a pooled list of examples (test_train_e005.py: seeds 1-5, 6,500 draws each).
The targets are copied from notes.txt (a), (b), (c) here, not imported from the generator."""
from collections import Counter

VTYPES = ("weekday", "colour", "month", "city")
HONORIFICS = ("Mr.", "Mrs.", "Ms.", "Dr.")
PLACES = ("adjacent", "filler", "other_obj")


def near(x, target, tol):
    return abs(x - target) <= tol


def share(xs, pred):
    return sum(1 for x in xs if pred(x)) / max(1, len(xs))


def _alias_gold(x):
    return any(st["ref"] == "alias" and st["value"] == x["gold"] for st in x["stmts"])


def share_checks(X):
    out = []
    ok = lambda c, m: out.append((bool(c), m))
    AL = [x for x in X if x["block"] == "alias"]
    IN = [x for x in X if x["block"] == "ind"]
    for blk, p in (("e004", 0.70), ("alias", 0.20), ("ind", 0.10)):
        v = share(X, lambda x: x["block"] == blk)
        ok(near(v, p, 0.01), f"block {blk:5s} {v:.3f} (target {p:.2f})")
    for blk in ("e004", "alias", "ind"):
        r = share([x for x in X if x["block"] == blk], lambda x: x["render"] == "chat")
        ok(near(r, 0.50, 0.02), f"chat render in block {blk}: {r:.3f} (target 0.50)")
    for case, p in (("latest", 0.60), ("earlier", 0.15), ("other", 0.25)):
        xs = [x for x in AL if x["case"] == case]
        ok(near(len(xs) / max(1, len(AL)), p, 0.025), f"alias case {case:7s} {len(xs) / max(1, len(AL)):.3f} "
                                                      f"(target {p:.2f})")
        pc = Counter(x["placement"] for x in xs)
        ok(all(near(pc[q] / max(1, len(xs)), 1 / 3, 0.05) for q in PLACES), f"  placements {dict(pc)} (1/3 each)")
        ba = share(xs, lambda x: x["both_aliased"])
        ok(near(ba, 1 / 3, 0.05), f"  both aliased {ba:.3f} (target 0.333)")
    kb = Counter(sum(1 for st in x["stmts"] if st["obj"] != x["asked"] and st["role"] == "corr")
                 for x in AL if x["case"] != "other")
    nk = max(1, sum(kb.values()))
    ok(all(near(kb[k] / nk, p, 0.03) for k, p in ((0, .25), (1, .5), (2, .25))),
       f"B's corrections when A is aliased {dict(sorted(kb.items()))} (1/4, 1/2, 1/4)")
    AB = [x for x in AL if x["case"] != "other" and x["placement"] != "other_obj"]
    bb = share(AB, lambda x: min(st["turn"] for st in x["stmts"] if st["obj"] != x["asked"]) <
               min(st["turn"] for st in x["stmts"] if st["obj"] == x["asked"]))
    ok(near(bb, 0.5, 0.04), f"adjacent/filler, A aliased: B before A's original {bb:.3f} (target 0.50)")
    titles = Counter(a.split(" ")[0] for x in AL for a in x["aliases"].values())
    hon = sum(titles[t] for t in HONORIFICS) / max(1, sum(titles.values()))
    ok(near(hon, 2 / 3, 0.03) and len(titles) == 8, f"honorific titles {hon:.3f} (target 0.667) {dict(titles)}")
    acs = [st for x in AL for st in x["stmts"] if st["ref"] == "alias"]
    nm = share(acs, lambda st: st["marker"] is None)
    ok(near(nm, 0.35, 0.03), f"alias corrections without a marker {nm:.3f} (target 0.35)")
    g = share(X, lambda x: x["block"] == "alias" and _alias_gold(x))
    ok(near(g, 0.12, 0.01), f"the alias correction is the gold in {g:.3f} of all examples (target 0.12)")
    lure = share(X, lambda x: x["block"] == "alias" and not _alias_gold(x))
    ok(near(lure, 0.08, 0.01), f"an alias lure (alias correction not the answer) in {lure:.3f} (target 0.08)")
    vt = Counter(x["vtype"] for x in AL + IN)
    ok(all(near(vt[v] / max(1, len(AL + IN)), 0.25, 0.02) for v in VTYPES), f"value types in the new blocks {dict(vt)}")
    ax = share(IN, lambda x: x["asked"] == 0)
    ok(near(ax, 0.75, 0.03), f"IND asks X {ax:.3f} (target 0.75)")
    xl = Counter(max((st for st in x["stmts"] if st["obj"] == 0), key=lambda st: st["turn"])["ref"] for x in IN)
    ok(near(xl["ell"] / max(1, len(IN)), 2 / 3, 0.04), f"IND X's last correction {dict(xl)} (ellipsis 2/3, pronoun 1/3)")
    yl = Counter(x["stmts"][-1]["ref"] for x in IN)
    yi = (yl["ell"] + yl["pron"]) / max(1, len(IN))
    ok(near(yi, 2 / 3, 0.04), f"IND Y's last correction indirect {yi:.3f} (target 0.667) {dict(yl)}")
    ok(near(yl["full"] / max(1, yl["full"] + yl["head"]), 0.6, 0.07), "IND Y's direct last: full/head 3:2")
    qf = share(AL + IN, lambda x: x["q_form"] == "full")
    ok(near(qf, 0.5, 0.03), f"question by full phrase in the new blocks {qf:.3f}")
    ok(set(Counter(x["d"] for x in AL + IN)) == set(range(11)), "d covers 0-10 in the new blocks")
    nobj = Counter(x["n_obj"] for x in AL + IN)
    ok(set(nobj) == {2}, f"new blocks have exactly 2 objects {dict(nobj)}")
    first = share(X, lambda x: x["stmts"][0]["value"] == x["gold"])
    last = share(X, lambda x: x["stmts"][-1]["value"] == x["gold"])
    ok(first <= 0.50 and last <= 0.50, f"pre-check of the stream rule (full oracles in oracles_e005.py): "
                                       f"first-statement gold {first:.3f}, last-statement gold {last:.3f} (<= 0.50)")
    return out


def e004_identity(xs, seed):
    """-> (the E004-block examples of xs, minus block/render, are train_e004.stream(seed)'s first ones in order, n)"""
    import train_e004 as T4
    e4 = [{k: v for k, v in x.items() if k not in ("block", "render")} for x in xs if x["block"] == "e004"]
    return e4 == T4.take(seed, len(e4)), len(e4)
