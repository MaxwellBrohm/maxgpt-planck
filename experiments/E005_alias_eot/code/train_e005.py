"""E005 training data generator (notes.txt THE THREE CHANGES). Wraps train_e004.py, which is not edited.
stream(seed): R_mix = Random(1000 * seed + 46) draws, per example, a block u then a render r:
  E004 block 70%  the next example of train_e004.stream(seed), unchanged, consumed in order
  ALIAS block 20% change (a), gen_alias(R_new), R_new = Random(1000 * seed + 45)
  IND block  10%  change (b), gen_ind(R_new)
  render     "chat" with probability 0.50 (change (c), rendered by the trainer), else "plain".
Every example keeps E004's fields (train_e004.py docstring) plus block, render; ALIAS and IND examples add case,
placement, both_aliased, aliases {object index: alias}, alias_obj (object whose alias correction occurs).
A statement's ref may be "alias" (an alias correction: the alias names the object, nothing else does)."""
import random

import train_e004 as T
from pools_train import POOLS, ACKS, FILLERS_TRAIN, MARKER_FMT, cap, with_marker
from pools_alias_train import (SURNAMES, HONORIFICS_TRAIN, ROLES, P_HONORIFIC, TITLES, JOINS, ALIAS_CORR,
                               CASES, PLACEMENTS, P_BOTH_ALIASED, N_B_CORR)
from text_e004 import value_re

BLOCKS = {"e004": 0.70, "alias": 0.20, "ind": 0.10}
P_CHAT = 0.50
P_ASK_X = 0.75
IND_LAST = {"ell": 2 / 3, "pron": 1 / 3}          # the asked-side indirect correction: ellipsis 2/3, pronoun 1/3
P_Y_INDIRECT = 2 / 3


def with_marker_e5(s, marker):
    """E004's marker render, except that a statement starting with a title keeps its capital."""
    if marker is not None and s.split(" ", 1)[0] in TITLES:
        return MARKER_FMT[marker].format(s=s)
    return with_marker(s, marker)


def ev(o, role="corr", ref=None, gap=None):
    """a planned statement: o = object label, ref/gap None = E004's rules decide at render time."""
    return dict(o=o, role=role, ref=ref, gap=gap)


def chain(o, k):
    return [ev(o, "orig", "full")] + [ev(o) for _ in range(k)]


def interleave(rng, a, b):
    """random merge of two lists, each keeping its order."""
    out, a, b = [], list(a), list(b)
    while a or b:
        src = a if (a and (not b or rng.random() < len(a) / (len(a) + len(b)))) else b
        out.append(src.pop(0))
    return out


def plan_alias(rng):
    """-> (case, placement, both_aliased, events, aliased label). Labels: A = asked, B = the other."""
    case = T._pick(rng, CASES)
    place = rng.choice(PLACEMENTS)
    both = rng.random() < P_BOTH_ALIASED
    alias_gap = 0 if place == "adjacent" else (rng.randint(1, 2) if place == "filler" else rng.randint(0, 1))
    if case in ("latest", "earlier"):
        k = rng.randint(1, 3) if case == "latest" else rng.randint(2, 3)
        p = k if case == "latest" else rng.randint(1, k - 1)       # position of the alias among A's corrections
        head = chain("A", p - 1)
        al = ev("A", "corr", "alias", alias_gap)
        tail = [ev("A") for _ in range(k - p)]
        b = chain("B", rng.choice(N_B_CORR))
        if place == "other_obj":
            b[0]["gap"] = rng.randint(0, 1)
            events = head + [b[0], al] + interleave(rng, tail, b[1:])
        elif rng.random() < 0.5:
            events = b + head + [al] + tail
        else:
            slot = rng.randint(0, len(tail))
            events = head + [al] + tail[:slot] + b + tail[slot:]
        return case, place, both, events, "A"
    ka, kb = rng.randint(0, 2), rng.randint(1, 2)                   # OTHER: B aliased, its LAST correction
    a = chain("A", ka)
    b = chain("B", kb - 1)
    al = ev("B", "corr", "alias", alias_gap)
    b_first = kb == 2 and rng.random() < 0.5                        # B's original before A's original
    if place == "other_obj":
        a[-1]["gap"] = rng.randint(0, 1)
        if b_first:
            events = [b[0]] + a[:-1] + b[1:] + [a[-1], al]
        else:
            events = a[:-1] + b + [a[-1], al]
    elif b_first:
        events = [b[0]] + a + b[1:] + [al]
    else:
        events = a + b + [al]
    return case, place, both, events, "B"


def plan_ind(rng):
    """-> (asked label, events). X's last correction indirect, then Y (its last indirect 2/3)."""
    kx, ky = rng.randint(1, 3), rng.randint(1, 2)
    x = chain("X", kx - 1) + [ev("X", "corr", T._pick(rng, IND_LAST), 0)]
    y = chain("Y", ky - 1)
    y[0]["gap"] = rng.randint(0, 2)
    if rng.random() < P_Y_INDIRECT:
        y.append(ev("Y", "corr", T._pick(rng, IND_LAST), 0))
    else:
        y.append(ev("Y", "corr", T._pick(rng, T.REF_FAR), None))
    asked = "X" if rng.random() < P_ASK_X else "Y"
    return asked, x + y


def make_alias(rng, vtype):
    s = rng.randrange(len(SURNAMES))
    t = rng.choice(HONORIFICS_TRAIN) if rng.random() < P_HONORIFIC else rng.choice(ROLES)
    return s, t


def render(rng, kind, vtype, events, asked_label, alias_labels=(), corr_label=None):
    """E004's rendering (templates, markers, acks, fillers, question, answer) of a planned event list."""
    P = POOLS[vtype]
    order = []
    for e in events:
        if e["o"] not in order:
            order.append(e["o"])
    idx = {lab: i for i, lab in enumerate(order)}
    objs = T.pick_objects(rng, P["objects"], len(order))
    vals = rng.sample(P["values"], len(events))
    turns, stmts, tpl, used, gaps, aliases = [], [], [], [], [], {}
    taken = set()
    for lab in alias_labels:
        while True:
            s, t = make_alias(rng, vtype)
            if s not in taken:
                break
        taken.add(s)
        j = rng.randrange(len(JOINS[vtype]))
        aliases[idx[lab]] = (f"{t} {SURNAMES[s]}", j)
        tpl += [("*", "surname", s), ("*", "title", TITLES.index(t)), (vtype, "join", j)]

    def fill(n):
        ids = rng.sample([i for i in range(len(FILLERS_TRAIN)) if i not in used], n)
        used.extend(ids)
        for i in ids:
            turns.append(FILLERS_TRAIN[i])
            tpl.append(("*", "filler", i))

    pre = rng.randint(*T.PRE_RANGE)
    fill(pre)
    first = idx[events[0]["o"]]
    for i, (e, v) in enumerate(zip(events, vals)):
        o, role, ref = idx[e["o"]], e["role"], e["ref"]
        if ref is None:
            ref = T._pick(rng, T.REF_ELIG if events[i - 1]["o"] == e["o"] else T.REF_FAR)
        if i > 0:
            g = e["gap"] if e["gap"] is not None else (0 if ref in ("pron", "ell") else rng.randint(*T.GAP_RANGE))
            gaps.append(g)
            fill(g)
        marker = T._marker(rng, role, o == first)
        if ref == "alias":
            pool, j = "alias_corr", rng.randrange(len(ALIAS_CORR[vtype]))
            user = ALIAS_CORR[vtype][j].format(a=aliases[o][0], v=v)
        else:
            pool = T.POOL_OF[(role, ref)]
            j = rng.randrange(len(P[pool]))
            term = objs[o][0] if ref == "full" else (objs[o][1] if ref == "head" else None)
            if role == "orig" and o in aliases:
                term += " " + JOINS[vtype][aliases[o][1]].format(a=aliases[o][0])
            user = P[pool][j].format(o=term, v=v)
        user = with_marker_e5(cap(user), marker)
        ja = rng.randrange(len(ACKS[role]))
        turns.append((user, cap(ACKS[role][ja].format(v=v))))
        stmts.append(dict(turn=len(turns) - 1, obj=o, value=v, role=role, ref=ref, marker=marker))
        tpl += [(vtype, pool, j), ("*", "ack_" + role, ja)]
        if marker is not None:
            tpl.append(("*", "marker", T.MARKERS.index(marker)))
    d = rng.randint(*T.D_RANGE)
    fill(d)
    asked = idx[asked_label]
    about = [s for s in stmts if s["obj"] == asked]
    gold = about[-1]["value"]
    k = sum(1 for s in about if s["role"] != "orig")
    k_other = sum(1 for s in stmts if s["obj"] != asked and s["role"] != "orig")
    q_form = rng.choice(("full", "head"))
    term = objs[asked][0] if q_form == "full" else objs[asked][1]
    jq = rng.randrange(len(P["ask"]))
    apool = [("ans", j) for j in range(len(P["ans"]))] + ([("ans_upd", j) for j in range(len(P["ans_upd"]))]
                                                           if k > 0 else [])
    ap, ja = rng.choice(apool)
    sent = cap(P[ap][ja].format(o=term, v=gold))
    tpl += [(vtype, "ask", jq), (vtype, ap, ja)]
    return dict(kind=kind, vtype=vtype, objects=objs, asked=asked, turns=turns, stmts=stmts,
                question=cap(P["ask"][jq].format(o=term)), q_form=q_form, answer=" " + sent + "\n",
                answer_pre=sent[:value_re(gold).search(sent).start()].strip(), gold=gold, k=k, k_other=k_other,
                n_obj=len(order), pre=pre, gaps=gaps, d=d, tpl=tpl,
                aliases={o: a for o, (a, _) in aliases.items()},
                alias_obj=idx[corr_label] if corr_label else None)


def gen_alias(rng):
    case, place, both, events, lab = plan_alias(rng)
    vtype = rng.choice(T.VTYPES)
    labs = ("A", "B") if both else (lab,)
    ex = render(rng, "alias_" + case, vtype, events, "A", labs, lab)
    return dict(ex, block="alias", case=case, placement=place, both_aliased=both)


def gen_ind(rng):
    asked, events = plan_ind(rng)
    vtype = rng.choice(T.VTYPES)
    ex = render(rng, "ind_" + asked.lower(), vtype, events, asked)
    return dict(ex, block="ind", case="ask_" + asked.lower(), placement=None, both_aliased=False)


def stream(seed):
    """the per-seed E005 training stream (before the trainer's 768-token rejection)."""
    r_mix, r_new = random.Random(1000 * seed + 46), random.Random(1000 * seed + 45)
    e4 = T.stream(seed)
    while True:
        u = r_mix.random()
        if u < BLOCKS["e004"]:
            ex = dict(next(e4), block="e004")
        elif u < BLOCKS["e004"] + BLOCKS["alias"]:
            ex = gen_alias(r_new)
        else:
            ex = gen_ind(r_new)
        ex["render"] = "chat" if r_mix.random() < P_CHAT else "plain"
        yield ex


def take(seed, n):
    it = stream(seed)
    return [next(it) for _ in range(n)]


prompt = T.prompt
