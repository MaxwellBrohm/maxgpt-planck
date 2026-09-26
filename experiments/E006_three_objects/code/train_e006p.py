"""E006 arm P training stream (notes.txt ARMS, P): E005's stream (train_e005.py, imported, not edited) with ONE
change, in the planners of the ALIAS and IND blocks only. Every new decision is drawn from a third generator
R_pos = Random(1000 * seed + 47); R_mix (block and render per drawn example, Random(1000 * seed + 46)), the E004
block (train_e004.stream(seed), unchanged, in order) and every PLAN draw of R_new = Random(1000 * seed + 45) are
E005's. After the first changed example the render consumes R_new differently, so later ALIAS/IND examples are fresh
draws from the same distributions, not C's examples.
  (a) ALIAS, cases LATEST and EARLIER, placements adjacent and filler: where E005 puts B's whole chain before A's
      original (its 1/2 branch) and B has k_B >= 1, with probability Q_SPLIT the chain is split: B's original stays
      before A's original, B's corrections move after A's latest statement, in order ("p_change": "split").
  (b) IND: with probability F_YFIRST per example Y's original moves to the front; X's chain follows unchanged, then
      Y's corrections in order, the first one after the 0-2 fillers E005 drew before Y's original; if that first
      moved correction was planned indirect it becomes full or head (E004's far weights, drawn from R_pos)
      ("p_change": "yfirst"). The ask (X 3/4, Y 1/4) is unchanged.
A changed example carries "p_change"; an unchanged one has no extra key, so Q_SPLIT = F_YFIRST = 0 reproduces
train_e005.stream exactly (test_train_e006p.py checks it on 20,000 draws per seed)."""
import random

import train_e004 as T
from train_e005 import ev, chain, interleave, render, IND_LAST, P_Y_INDIRECT, P_ASK_X, BLOCKS, P_CHAT
from pools_alias_train import CASES, PLACEMENTS, P_BOTH_ALIASED, N_B_CORR

Q_SPLIT = 1.0
F_YFIRST = 0.30
R_POS = 47          # R_pos = Random(1000 * seed + R_POS)


def plan_alias_p(rng, rp, q=Q_SPLIT):
    """train_e005.plan_alias line for line (same R_new draws in the same order), plus change (a) from rp.
    -> (case, placement, both_aliased, events, aliased label, changed)."""
    case = T._pick(rng, CASES)
    place = rng.choice(PLACEMENTS)
    both = rng.random() < P_BOTH_ALIASED
    alias_gap = 0 if place == "adjacent" else (rng.randint(1, 2) if place == "filler" else rng.randint(0, 1))
    if case in ("latest", "earlier"):
        k = rng.randint(1, 3) if case == "latest" else rng.randint(2, 3)
        p = k if case == "latest" else rng.randint(1, k - 1)
        head = chain("A", p - 1)
        al = ev("A", "corr", "alias", alias_gap)
        tail = [ev("A") for _ in range(k - p)]
        b = chain("B", rng.choice(N_B_CORR))
        if place == "other_obj":
            b[0]["gap"] = rng.randint(0, 1)
            return case, place, both, head + [b[0], al] + interleave(rng, tail, b[1:]), "A", False
        if rng.random() < 0.5:
            if len(b) > 1 and rp.random() < q:                    # change (a): split B's chain
                return case, place, both, [b[0]] + head + [al] + tail + b[1:], "A", True
            return case, place, both, b + head + [al] + tail, "A", False
        slot = rng.randint(0, len(tail))
        return case, place, both, head + [al] + tail[:slot] + b + tail[slot:], "A", False
    ka, kb = rng.randint(0, 2), rng.randint(1, 2)
    a = chain("A", ka)
    b = chain("B", kb - 1)
    al = ev("B", "corr", "alias", alias_gap)
    b_first = kb == 2 and rng.random() < 0.5
    if place == "other_obj":
        a[-1]["gap"] = rng.randint(0, 1)
        events = ([b[0]] + a[:-1] + b[1:] + [a[-1], al]) if b_first else (a[:-1] + b + [a[-1], al])
    elif b_first:
        events = [b[0]] + a + b[1:] + [al]
    else:
        events = a + b + [al]
    return case, place, both, events, "B", False


def y_first(rp, x, y):
    """change (b) on E005's planned chains x (X's) and y (Y's): Y's original first, then x, then Y's corrections."""
    post = [dict(e) for e in y[1:]]
    post[0]["gap"] = y[0]["gap"]
    if post[0]["ref"] in ("pron", "ell"):
        post[0]["ref"] = T._pick(rp, T.REF_FAR)
    return [dict(y[0], gap=None)] + x + post


def plan_ind_p(rng, rp, f=F_YFIRST):
    """train_e005.plan_ind line for line, plus change (b) from rp. -> (asked label, events, changed)."""
    kx, ky = rng.randint(1, 3), rng.randint(1, 2)
    x = chain("X", kx - 1) + [ev("X", "corr", T._pick(rng, IND_LAST), 0)]
    y = chain("Y", ky - 1)
    y[0]["gap"] = rng.randint(0, 2)
    if rng.random() < P_Y_INDIRECT:
        y.append(ev("Y", "corr", T._pick(rng, IND_LAST), 0))
    else:
        y.append(ev("Y", "corr", T._pick(rng, T.REF_FAR), None))
    asked = "X" if rng.random() < P_ASK_X else "Y"
    if rp.random() < f:
        return asked, y_first(rp, x, y), True
    return asked, x + y, False


def gen_alias_p(rng, rp, q=Q_SPLIT):
    case, place, both, events, lab, changed = plan_alias_p(rng, rp, q)
    vtype = rng.choice(T.VTYPES)
    labs = ("A", "B") if both else (lab,)
    ex = dict(render(rng, "alias_" + case, vtype, events, "A", labs, lab), block="alias", case=case,
              placement=place, both_aliased=both)
    if changed:
        ex["p_change"] = "split"
    return ex


def gen_ind_p(rng, rp, f=F_YFIRST):
    asked, events, changed = plan_ind_p(rng, rp, f)
    vtype = rng.choice(T.VTYPES)
    ex = dict(render(rng, "ind_" + asked.lower(), vtype, events, asked), block="ind", case="ask_" + asked.lower(),
              placement=None, both_aliased=False)
    if changed:
        ex["p_change"] = "yfirst"
    return ex


def stream_p(seed, q=Q_SPLIT, f=F_YFIRST):
    """the per-seed P training stream (before the trainer's 768-token rejection)."""
    r_mix, r_new = random.Random(1000 * seed + 46), random.Random(1000 * seed + 45)
    r_pos = random.Random(1000 * seed + R_POS)
    e4 = T.stream(seed)
    while True:
        u = r_mix.random()
        if u < BLOCKS["e004"]:
            ex = dict(next(e4), block="e004")
        elif u < BLOCKS["e004"] + BLOCKS["alias"]:
            ex = gen_alias_p(r_new, r_pos, q)
        else:
            ex = gen_ind_p(r_new, r_pos, f)
        ex["render"] = "chat" if r_mix.random() < P_CHAT else "plain"
        yield ex


def take_p(seed, n, q=Q_SPLIT, f=F_YFIRST):
    it = stream_p(seed, q, f)
    return [next(it) for _ in range(n)]


KNOBS = {"Q_SPLIT": Q_SPLIT, "F_YFIRST": F_YFIRST, "R_pos": f"Random(1000 * seed + {R_POS})"}
