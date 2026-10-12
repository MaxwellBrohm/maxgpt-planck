"""Scoring rules for the E001 likelihood items (the "item scorers").

An item is right when the gold continuation's summed log-prob is STRICTLY greater than every
other candidate's. Ties fail, so a model that cannot tell the candidates apart (or an empty
scorer that returns the same number for everything) scores 0, not chance.

Paired metrics join variants of the same scenario (sid):
  upd_bind  = same_k1 AND noupd right       (first-value-wins: 0, latest-value-wins: 0, chance ~0.25)
  key_pair  = keyorig AND keycorr right     (a pure key matcher: 0)
  all5      = same_k1, noupd, twoslot, keyorig, keycorr all right (only real updating + binding passes)
"""
import math
from collections import defaultdict

PAIRS = {
    "upd_bind": ("same_k1", "noupd"),
    "key_pair": ("keyorig", "keycorr"),
    "all5": ("same_k1", "noupd", "twoslot", "keyorig", "keycorr"),
}


def right(scores):
    g = scores["gold"]
    return all(g > v for k, v in scores.items() if k != "gold")


def margin(scores):
    """gold minus the best other candidate (nats)."""
    g = scores["gold"]
    return g - max(v for k, v in scores.items() if k != "gold")


def acc(xs):
    n = len(xs)
    if n == 0:
        return None
    p = sum(xs) / n
    return {"acc": round(p, 3), "n": n, "se": round(math.sqrt(max(p * (1 - p), 1e-9) / n), 3)}


def picks(scores):
    """Label of the strict argmax, or None on a tie at the top."""
    best = max(scores.values())
    top = [k for k, v in scores.items() if v == best]
    return top[0] if len(top) == 1 else None


def summarize_new(recs):
    """recs: dicts with var, fam, d, sid, render, scores. Returns {key: acc-dict or float}."""
    cells = defaultdict(list)
    marg = defaultdict(list)
    choice = defaultdict(lambda: defaultdict(int))
    by_sid = defaultdict(dict)
    for r in recs:
        ok = right(r["scores"])
        for fam in (r["fam"], "all"):
            cells[(r["render"], r["var"], fam, r["d"])].append(ok)
            if r["d"] in (4, 10):
                cells[(r["render"], r["var"], fam, "d4-10")].append(ok)
            marg[(r["render"], r["var"], fam, r["d"])].append(margin(r["scores"]))
            p = picks(r["scores"])
            choice[(r["render"], r["var"], fam, r["d"])][p or "tie"] += 1
        by_sid[(r["render"], r["sid"])][r["var"]] = ok
    for (render, sid), v in by_sid.items():
        fam, d, _ = sid.split("|")
        d = int(d)
        for pname, vars_ in PAIRS.items():
            if all(x in v for x in vars_):
                ok = all(v[x] for x in vars_)
                for f in (fam, "all"):
                    cells[(render, "PAIR_" + pname, f, d)].append(ok)
                    if d in (4, 10):
                        cells[(render, "PAIR_" + pname, f, "d4-10")].append(ok)
    out = {}
    for k, xs in cells.items():
        out["|".join(map(str, k))] = acc([bool(x) for x in xs])
    for k, xs in marg.items():
        out["margin|" + "|".join(map(str, k))] = round(sum(xs) / len(xs), 3)
    for k, c in choice.items():
        out["picks|" + "|".join(map(str, k))] = dict(c)
    return out


def summarize_old(recs):
    """The old capacity_probe summary (copied logic), for the old items on newly probed models."""
    cells = defaultdict(list)
    for r in recs:
        s, t, d = r["scores"], r["task"], r["d"]
        g = s["gold"]
        others = {k: v for k, v in s.items() if k != "gold"}
        cells[(t, d, "all")].append(g > max(others.values()))
        for k, v in others.items():
            cells[(t, d, "vs_" + k)].append(g > v)
        if t == "K_closedbook":
            cells[(t, r["cond"], "all")].append(g > max(others.values()))
        cells[(t, d, "margin")].append(g - max(others.values()))
    for a_, b_ in zip(recs, recs[1:]):
        ok = lambda r: r["scores"]["gold"] > max(v for k, v in r["scores"].items() if k != "gold")
        if a_["task"] == "R1_owner" and a_["cond"] == "mine" and b_["task"] == "R1_owner" and b_["cond"] == "sister":
            cells[("R1_both", a_["d"], "all")].append(ok(a_) and ok(b_))
        if a_["task"] == "P_myname" and b_["task"] == "P_yourname":
            cells[("P_both", a_["d"], "all")].append(ok(a_) and ok(b_))
    for i in range(len(recs) - 3):
        w = recs[i:i + 4]
        if [x["task"] for x in w] == ["R2_twohop", "R2_onehop", "R2_twohop", "R2_onehop"]:
            ok = lambda r: r["scores"]["gold"] > r["scores"]["foil"]
            cells[("R2two_both", w[0]["d"], "all")].append(ok(w[0]) and ok(w[2]))
            cells[("R2one_both", w[0]["d"], "all")].append(ok(w[1]) and ok(w[3]))
    out = {}
    for (t, d, m), xs in cells.items():
        key = f"{t}|{d}|{m}"
        out[key] = round(sum(xs) / len(xs), 3) if m == "margin" else acc([bool(x) for x in xs])
    return out
