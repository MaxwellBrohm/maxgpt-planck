"""E004 training data generator (notes.txt (a)). Every example is an update dialogue; there are no owner/binding,
two-hop or perspective families. Deterministic per seed: stream(seed) draws from Random(1000 * seed + 44).

Example dict: kind, vtype, objects [(phrase, head)], asked (object index), turns [(user, assistant)],
stmts (one annotation per value statement: turn, obj (index or None for an incidental), value, role
orig/corr/rev/incid, ref full/head/pron/ell/None, marker), question, q_form, answer (" Sentence.\n": the
training target, loss on these tokens only), answer_pre (answer words before the value, for oracle O4), gold,
k (corrections of the asked object), k_other, n_obj, pre, gaps, d, tpl (every pool entry used, for coverage).
prompt(ex) renders the plain "User:/Assistant:" transcript that ends with "Assistant:" (no prefix)."""
import random

from text_e004 import value_re
from pools_train import (POOLS, ACKS, MARKERS, NONCORR_MARKERS, FILLERS_TRAIN, P_NO_MARKER_CORR,
                         P_MARKER_B_ORIG, P_MARKER_OTHER, cap, with_marker)

KINDS = {"upd": 0.25, "twoslot": 0.20, "twoslot_b": 0.05, "noupd": 0.15, "noupd_b": 0.05, "noupd_incid": 0.075,
         "upd_incid": 0.05, "mid": 0.075, "revert": 0.05, "cross": 0.05}
VTYPES = ["weekday", "colour", "month", "city"]
K_RANGE, D_RANGE, PRE_RANGE, GAP_RANGE = (1, 3), (0, 10), (0, 3), (0, 2)

# Reference-form weights. Pronoun and ellipsis are only possible directly after the same object's exchange;
# the eligible weights are raised so the overall rates land near REF_TARGET (30/20/25/25; about 9% of
# corrections are not eligible), measured in test_train_gen.py. REF_FAR is used when the previous statement is
# about another object.
REF_ELIG = {"full": 0.2692, "head": 0.1796, "pron": 0.2756, "ell": 0.2756}
REF_FAR = {"full": 0.30 / 0.50, "head": 0.20 / 0.50}
POOL_OF = {("orig", "full"): "orig", ("corr", "full"): "corr", ("corr", "head"): "corr", ("corr", "pron"): "pron",
           ("corr", "ell"): "ell", ("rev", "full"): "rev", ("rev", "head"): "rev", ("rev", "pron"): "rev_pron",
           ("rev", "ell"): "rev_ell", ("incid", None): "incid"}


def _pick(rng, table):
    r, acc = rng.random(), 0.0
    for key, p in table.items():
        acc += p
        if r < acc:
            return key
    return key


def pick_objects(rng, objects, n):
    """n objects with distinct head nouns, no head noun inside the other object's phrase."""
    while True:
        objs = rng.sample(objects, n)
        ok = len({h for _, h in objs}) == n and all(
            objs[i][1] not in objs[j][0].split() for i in range(n) for j in range(n) if i != j)
        if ok:
            return objs


def plan(rng, kind):
    """(events, n_obj, asked): events are (object index or None, role) in dialogue order."""
    k = lambda: rng.randint(*K_RANGE)
    if kind == "upd":
        return [(0, "orig")] + [(0, "corr")] * k(), 1, 0
    if kind in ("twoslot", "twoslot_b"):
        ev = [(0, "orig")] + [(0, "corr")] * k() + [(1, "orig")] + [(1, "corr")] * rng.choice((0, 1, 1, 2))
        return ev, 2, (0 if kind == "twoslot" else 1)
    if kind in ("noupd", "noupd_b"):
        ev = [(0, "orig"), (1, "orig")] + [(1, "corr")] * rng.randint(0, 3)
        return ev, 2, (0 if kind == "noupd" else 1)
    if kind == "noupd_incid":
        return [(0, "orig"), (None, "incid")], 1, 0
    if kind == "upd_incid":
        return [(0, "orig")] + [(0, "corr")] * k() + [(None, "incid")], 1, 0
    if kind == "mid":
        return [(0, "orig"), (1, "orig")] + [(0, "corr")] * k(), 2, 1
    if kind == "revert":
        return [(0, "orig")] + [(0, "corr")] * rng.randint(1, 2) + [(0, "rev")], 1, 0
    if kind == "cross":
        asked = rng.randint(0, 1)
        corrs = [(asked, "corr")] * k() + [(1 - asked, "corr")] * rng.randint(1, 2)
        rng.shuffle(corrs)
        return [(0, "orig"), (1, "orig")] + corrs, 2, asked
    raise ValueError(kind)


def assign_values(rng, events, pool):
    """distinct values for every statement, except a change back, which takes its object's original value."""
    fresh = rng.sample(pool, sum(1 for _, r in events if r != "rev"))
    out, orig_of, it = [], {}, iter(fresh)
    for o, role in events:
        v = orig_of[o] if role == "rev" else next(it)
        if role == "orig":
            orig_of[o] = v
        out.append(v)
    return out


def _marker(rng, role, is_first_obj):
    if role in ("corr", "rev"):
        return None if rng.random() < P_NO_MARKER_CORR else rng.choice(MARKERS)
    p = P_MARKER_B_ORIG if (role == "orig" and not is_first_obj) else P_MARKER_OTHER
    return rng.choice(NONCORR_MARKERS) if rng.random() < p else None


def gen(rng, kind=None, vtype=None):
    kind = kind or _pick(rng, KINDS)
    vtype = vtype or rng.choice(VTYPES)
    P = POOLS[vtype]
    events, n_obj, asked = plan(rng, kind)
    objs = pick_objects(rng, P["objects"], n_obj)
    vals = assign_values(rng, events, P["values"])
    turns, stmts, tpl, used, gaps = [], [], [], [], []

    def fill(n):
        idx = rng.sample([i for i in range(len(FILLERS_TRAIN)) if i not in used], n)
        used.extend(idx)
        for i in idx:
            turns.append(FILLERS_TRAIN[i])
            tpl.append(("*", "filler", i))

    pre = rng.randint(*PRE_RANGE)
    fill(pre)
    first_obj = events[0][0]
    for i, ((o, role), v) in enumerate(zip(events, vals)):
        if role in ("corr", "rev"):
            ref = _pick(rng, REF_ELIG if events[i - 1][0] == o else REF_FAR)
        else:
            ref = "full" if role == "orig" else None
        if i > 0:
            g = 0 if ref in ("pron", "ell") else rng.randint(*GAP_RANGE)
            gaps.append(g)
            fill(g)
        marker = _marker(rng, role, o == first_obj)
        pool = POOL_OF[(role, ref)]
        j = rng.randrange(len(P[pool]))
        term = objs[o][0] if ref == "full" else (objs[o][1] if ref == "head" else None)
        user = with_marker(cap(P[pool][j].format(o=term, v=v)), marker)
        acks = ACKS[role]
        ja = rng.randrange(len(acks))
        turns.append((user, cap(acks[ja].format(v=v))))
        stmts.append(dict(turn=len(turns) - 1, obj=o, value=v, role=role, ref=ref, marker=marker))
        tpl += [(vtype, pool, j), ("*", "ack_" + ("corr" if role == "rev" else role), ja)]
        if marker is not None:
            tpl.append(("*", "marker", MARKERS.index(marker)))
    d = rng.randint(*D_RANGE)
    fill(d)
    about = [s for s in stmts if s["obj"] == asked]
    gold = about[-1]["value"]
    k = sum(1 for s in about if s["role"] != "orig")
    k_other = sum(1 for s in stmts if s["obj"] not in (asked, None) and s["role"] != "orig")
    q_form = rng.choice(("full", "head"))
    term = objs[asked][0] if q_form == "full" else objs[asked][1]
    jq = rng.randrange(len(P["ask"]))
    apool = [("ans", j) for j in range(len(P["ans"]))]
    if k > 0:
        apool += [("ans_upd", j) for j in range(len(P["ans_upd"]))]
    ap, ja = rng.choice(apool)
    sent = cap(P[ap][ja].format(o=term, v=gold))
    tpl += [(vtype, "ask", jq), (vtype, ap, ja)]
    return dict(kind=kind, vtype=vtype, objects=objs, asked=asked, turns=turns, stmts=stmts,
                question=cap(P["ask"][jq].format(o=term)), q_form=q_form, answer=" " + sent + "\n",
                answer_pre=sent[:value_re(gold).search(sent).start()].strip(), gold=gold, k=k,
                k_other=k_other, n_obj=n_obj, pre=pre, gaps=gaps, d=d, tpl=tpl)


def prompt(ex):
    lines = []
    for u, a in ex["turns"]:
        lines += [f"User: {u}", f"Assistant: {a}"]
    lines += [f"User: {ex['question']}", "Assistant:"]
    return "\n".join(lines)


def stream(seed):
    """the per-seed training stream (before the 768-token rejection, which the trainer applies)."""
    rng = random.Random(1000 * seed + 44)
    while True:
        yield gen(rng)


def take(seed, n):
    it = stream(seed)
    return [next(it) for _ in range(n)]
