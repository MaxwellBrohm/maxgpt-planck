"""E006 position statistics of a training stream (notes.txt DEFINITIONS). Pure Python: no model, no tokenizer.
For one training example (train_e004 / train_e005 / train_e006p format: stmts in dialogue order, obj None for an
incidental statement, n_obj, asked):
  intro   1-based order in which the asked object is first mentioned among the example's objects
  GL      1 if the dialogue's last value statement is about the asked object (an incidental last counts as 0)
Aggregates over a list of examples (2-object examples only, n_obj == 2):
  GL2     mean GL where the asked object is introduced second
  GL1     mean GL where it is introduced first
  Delta   GL2 - GL1
  GLX2    mean GL over every 2-object example (the critique's "2-object gold-last", item 6)
  share2  share of 2-object examples whose asked object is introduced second
stats(exs) returns these for the whole list, per block and per block:case; gates(...) applies the P gates."""
from collections import defaultdict

KEYS = ("GL1", "GL2", "Delta", "GLX2", "share2", "n2", "n_a1", "n_a2")


def intro_order(ex):
    seen = []
    for s in ex["stmts"]:
        if s["obj"] is not None and s["obj"] not in seen:
            seen.append(s["obj"])
    return seen.index(ex["asked"]) + 1


def gold_last(ex):
    return int(ex["stmts"][-1]["obj"] == ex["asked"])


def _agg(rows):
    """rows: [(intro, gl)] of 2-object examples -> dict of KEYS (None where undefined)."""
    a1 = [g for i, g in rows if i == 1]
    a2 = [g for i, g in rows if i == 2]
    m = lambda xs: sum(xs) / len(xs) if xs else None
    gl1, gl2 = m(a1), m(a2)
    return {"GL1": gl1, "GL2": gl2, "Delta": None if gl1 is None or gl2 is None else gl2 - gl1,
            "GLX2": m([g for _, g in rows]), "share2": len(a2) / len(rows) if rows else None,
            "n2": len(rows), "n_a1": len(a1), "n_a2": len(a2)}


def stats(exs):
    """-> {"all": {...}, "<block>": {...}, "<block>:<case>": {...}} over the 2-object examples of exs."""
    rows = defaultdict(list)
    for ex in exs:
        if ex["n_obj"] != 2:
            continue
        r = (intro_order(ex), gold_last(ex))
        b = ex.get("block", "e004")
        rows["all"].append(r)
        rows[b].append(r)
        if ex.get("case"):
            rows[f"{b}:{ex['case']}"].append(r)
    return {k: _agg(v) for k, v in sorted(rows.items())}


def pooled(per_seed):
    """per_seed: [stats(...)] -> pooled numbers for "all" (weighted by counts, i.e. over examples)."""
    def w(key, n):
        tot = sum(s["all"][n] for s in per_seed)
        return sum(s["all"][key] * s["all"][n] for s in per_seed if s["all"][n]) / tot if tot else None
    g1, g2, gx = w("GL1", "n_a1"), w("GL2", "n_a2"), w("GLX2", "n2")
    return {"GL1": g1, "GL2": g2, "Delta": None if g1 is None or g2 is None else g2 - g1, "GLX2": gx,
            "n2": sum(s["all"]["n2"] for s in per_seed)}


def gates(st, ref):
    """P's position gates for one seed (notes STREAM ACCEPTANCE, P). st: stats(kept P examples of one seed);
    ref: E004's pooled kept values {"GL2": R4, "Delta": DELTA4, "GLX2": X4}. -> list of failures (empty = pass)."""
    fails = []
    chk = [("whole-stream GL2", st["all"]["GL2"], ref["GL2"]), ("ALIAS GL2", st.get("alias", {}).get("GL2"), ref["GL2"]),
           ("IND GL2", st.get("ind", {}).get("GL2"), ref["GL2"]), ("whole-stream Delta", st["all"]["Delta"], ref["Delta"]),
           ("whole-stream GLX2", st["all"]["GLX2"], ref["GLX2"])]
    for name, v, lim in chk:
        if v is None or v > lim:
            fails.append(f"{name} {v if v is None else round(v, 4)} > E004 pooled {lim:.4f}")
    return fails


def fmt(st, keys=("all", "e004", "alias", "ind")):
    out = []
    for k in keys:
        if k in st:
            x = st[k]
            f = lambda v: "-" if v is None else f"{v:.3f}"
            out.append(f"{k}: GL2 {f(x['GL2'])} GL1 {f(x['GL1'])} Delta {f(x['Delta'])} GLX2 {f(x['GLX2'])} "
                       f"share2 {f(x['share2'])} (n2 {x['n2']}, a2 {x['n_a2']})")
    return "; ".join(out)
