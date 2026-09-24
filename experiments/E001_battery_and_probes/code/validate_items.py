"""Self-test for the E001 likelihood items and their scorer (no model needed).

1. Structural validators on every new item: no candidate value in the question or answer
   prefix (no question contains its own answer word), every candidate in context, one
   statement per value, mention order, key placement per variant, the position control's
   leading fillers, and the distance d.
2. Oracle checks: five fake "models" score every item from its text alone.
     first  : prefers the value mentioned first      (the primacy heuristic)
     latest : prefers the value mentioned last       (the recency heuristic)
     key    : prefers the value whose exchange shares most key phrases with the question
     ideal  : knows the gold
     empty  : returns the same score for everything (the likelihood analogue of an empty answer)
   Each must land exactly where the design says (0 or 1 per variant and per paired metric).
   This is what makes the controls able to tell the heuristics apart.
3. The old battery's items (items.py) and khard items are checked for answer words in their
   own question; violations are reported, not fixed (research/ is read-only).
Run: python validate_items.py   (exits non-zero on any failure)
"""
import re, sys
from collections import defaultdict

import items as I
import items_new as N
import metrics as M


def has_word(text, w):
    return re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", text, re.I) is not None


def exchanges_with(item, val):
    return [i for i, (u, a) in enumerate(item["turns"]) if has_word(u + " " + a, val)]


def has_key(fam, text):
    return any(k in text.lower() for k in N.FAMILIES[fam]["key_words"])


def has_other(fam, text):
    return any(k in text.lower() for k in N.FAMILIES[fam]["other_words"])


def validate(items):
    errs = []
    for it in items:
        fam, var = it["fam"], it["var"]
        vals = {lab: c.strip() for lab, c in it["cands"].items()}
        tag = f"{it['task']} {it['sid']}"
        if len(set(vals.values())) != len(vals):
            errs.append(f"{tag}: duplicate candidate values {vals}")
        for lab, v in vals.items():
            if has_word(it["question"], v) or has_word(it["prefix"], v):
                errs.append(f"{tag}: candidate {lab}={v} appears in the question or prefix")
            ex = exchanges_with(it, v)
            if not ex:
                errs.append(f"{tag}: candidate {lab}={v} not in context")
            elif len(ex) != 1:
                errs.append(f"{tag}: candidate {lab}={v} appears in {len(ex)} exchanges (fillers must not carry values)")
        # mention order
        first = {lab: min(exchanges_with(it, v) or [10 ** 6]) for lab, v in vals.items()}
        order = sorted(first, key=first.get)
        if order != N.ROLE_ORDER[var]:
            errs.append(f"{tag}: mention order {order} != {N.ROLE_ORDER[var]}")
        # key placement
        ex_text = {lab: " ".join(it["turns"][first[lab]]) if first[lab] < 10 ** 6 else "" for lab in vals}
        K = {lab: has_key(fam, t) for lab, t in ex_text.items()}
        if var.startswith("same") or var == "pos":
            if not all(K.values()):
                errs.append(f"{tag}: {var} needs the key in every statement {K}")
        elif var in ("keyorig", "neutral"):
            if not (K["orig"] and not K["gold"]):
                errs.append(f"{tag}: {var} needs the key only in the original {K}")
        elif var == "keycorr":
            if not (K["gold"] and not K["orig"]):
                errs.append(f"{tag}: keycorr needs the key only in the correction {K}")
        elif var == "noupd":
            if not (K["gold"] and not K["later"] and has_other(fam, ex_text["later"])):
                errs.append(f"{tag}: noupd needs key on gold, another object on the later value {K}")
        elif var == "twoslot":
            if not (K["gold"] and K["orig"] and not K["other"] and has_other(fam, ex_text["other"])):
                errs.append(f"{tag}: twoslot needs key on orig and gold, another object last {K}")
        if var == "neutral" and (has_key(fam, it["prefix"]) or has_key(fam, it["question"])):
            errs.append(f"{tag}: neutral question/prefix shares a key phrase")
        if var == "pos":
            if first["orig"] != 3 or any(first[l] == 0 for l in vals):
                errs.append(f"{tag}: pos must have 3 leading fillers (orig at exchange {first['orig']})")
        last_stmt = max(first.values())
        if len(it["turns"]) - 1 - last_stmt != it["d"]:
            errs.append(f"{tag}: {len(it['turns']) - 1 - last_stmt} exchanges after the last statement, d={it['d']}")
    # fillers never mention any value of any family
    for q, a in N.pool(N.EXCLUDE):
        for fam, F in N.FAMILIES.items():
            for v in F["values"]:
                if has_word(q + " " + a, v):
                    errs.append(f"filler mentions {v}: {q}")
    return errs


# ------------------------------------------------------------------ oracles
def oracle(item, kind):
    vals = {lab: c.strip() for lab, c in item["cands"].items()}
    ctx = I.transcript(item["turns"], "", "")
    wb = lambda v: r"(?<![A-Za-z])" + re.escape(v) + r"(?![A-Za-z])"
    if kind == "first":
        return {lab: -min(m.start() for m in re.finditer(wb(v), ctx)) for lab, v in vals.items()}
    if kind == "latest":
        return {lab: max(m.start() for m in re.finditer(wb(v), ctx)) for lab, v in vals.items()}
    if kind == "key":
        out = {}
        for lab, v in vals.items():
            best = 0
            for i in exchanges_with(item, v):
                t = " ".join(item["turns"][i]).lower()
                best = max(best, sum(t.count(k) for k in N.FAMILIES[item["fam"]]["key_words"]))
            out[lab] = best
        return out
    if kind == "ideal":
        return {lab: (1.0 if lab == "gold" else 0.0) for lab in vals}
    if kind == "empty":
        return {lab: 0.0 for lab in vals}
    raise ValueError(kind)


ALL1 = {v: 1 for v in N.VARIANTS}
ALL0 = {v: 0 for v in N.VARIANTS}
EXPECT = {
    "first": dict(ALL0, noupd=1),
    "latest": dict(ALL1, noupd=0, twoslot=0),
    "key": dict(ALL0, keycorr=1, noupd=1),
    "ideal": ALL1,
    "empty": ALL0,
}
EXPECT_PAIRS = {
    "first": {"upd_bind": 0, "key_pair": 0, "all5": 0},
    "latest": {"upd_bind": 0, "key_pair": 1, "all5": 0},
    "key": {"upd_bind": 0, "key_pair": 0, "all5": 0},
    "ideal": {"upd_bind": 1, "key_pair": 1, "all5": 1},
    "empty": {"upd_bind": 0, "key_pair": 0, "all5": 0},
}


def oracle_check(items):
    errs = []
    for kind in EXPECT:
        recs = [dict(var=it["var"], fam=it["fam"], d=it["d"], sid=it["sid"], render="plain",
                     scores={k: float(v) for k, v in oracle(it, kind).items()}) for it in items]
        s = M.summarize_new(recs)
        for var, want in EXPECT[kind].items():
            got = s[f"plain|{var}|all|d4-10"]["acc"]
            got0 = s[f"plain|{var}|all|0"]["acc"]
            if got != want or got0 != want:
                errs.append(f"oracle {kind}: {var} acc d0={got0} d4-10={got}, expected {want}")
        for p, want in EXPECT_PAIRS[kind].items():
            got = s[f"plain|PAIR_{p}|all|d4-10"]["acc"]
            if got != want:
                errs.append(f"oracle {kind}: pair {p} = {got}, expected {want}")
        if kind == "ideal" and any(M.margin(r["scores"]) <= 0 for r in recs):
            errs.append("ideal oracle has a non-positive margin")
        if kind == "empty" and any(M.margin(r["scores"]) != 0 for r in recs):
            errs.append("empty oracle has a non-zero margin")
    return errs


def old_item_answer_leaks():
    """Rule check on the OLD battery: no question may contain its own answer word."""
    leaks = []
    for it in I.build():
        q = [l for l in it["prompt"].split("\n") if l.startswith("User: ")][-1][6:]
        g = it["cands"]["gold"].strip()
        if has_word(q, g):
            leaks.append((it["task"], q, g))
    import khard_items as KH
    for it in KH.build():
        q = [l for l in it["prompt"].split("\n") if l.startswith("User: ")][-1][6:]
        g = it["cands"]["gold"].strip()
        if has_word(q, g):
            leaks.append((it["task"], q, g))
    return sorted(set(leaks))


def run(verbose=True):
    items = N.build()
    errs = validate(items) + oracle_check(items)
    if verbose:
        print(f"{len(items)} new items; {len(errs)} errors")
        for e in errs[:20]:
            print("  ", e)
    return errs


if __name__ == "__main__":
    errs = run()
    leaks = old_item_answer_leaks()
    print(f"old battery + khard items whose question contains the answer word: {len(leaks)}")
    for l in leaks:
        print("  ", l)
    if errs:
        sys.exit(1)
    print("validate_items OK")
