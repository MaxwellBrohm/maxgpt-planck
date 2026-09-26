"""parse AL items from their TEXT (user turns), independent of the stmts metadata."""
import re, sys
sys.dont_write_bytecode = True
import mygrade as G
NAME = re.compile(r"\b(Mr|Mrs|Ms|Dr)\. ([A-Z][a-z]+)")
def wb(term, text):
    return re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", text, re.I) is not None
def parse(it):
    objs = it["objects"]
    out = []
    for ti, (u, a) in enumerate(it["turns"]):
        vals = [v for v in it["values"] if G.mentions(u, v)]
        if not vals:
            continue
        assert len(vals) == 1, (it["idx"], ti, u, vals)
        names = [f"{m.group(1)}. {m.group(2)}" for m in NAME.finditer(u)]
        om = [o for o, (ph, hd) in enumerate(objs) if wb(ph, u) or wb(hd, u)]
        assert len(om) <= 1, (it["idx"], u)
        out.append(dict(turn=ti, value=vals[0], names=names, obj=om[0] if om else None, text=u))
    return out
def check_vs_meta(it, P):
    S = it["stmts"]
    assert len(S) == len(P), (it["idx"], len(S), len(P))
    for s, p in zip(S, P):
        assert s["turn"] == p["turn"] and s["value"] == p["value"], it["idx"]
        if s["ref"] == "alias":
            assert p["obj"] is None and p["names"] == [s["alias"]], (it["idx"], p)
        else:
            assert p["obj"] == s["obj"], (it["idx"], p, s)
    return True
