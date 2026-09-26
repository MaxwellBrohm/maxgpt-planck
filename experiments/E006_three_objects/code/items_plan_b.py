"""E004 eval item plans, part 2: H5 (obj3), the diagnostic ID family and the two controls in held-out form
(notes (b)). Controls: 8 cells (one per axis H1-H7, plus IR = in range: differs from training only in wording and
objects), n/8 items each, factors balanced within each cell.

C_twoslot: A corrected (k 1-3; H1 cell 2-3 with the frame on A's first correction; H2 cell 1-3 with the frame on
  A's original and indirect/alias corrections only; H3 cell 4-5), then B stated after A's latest and corrected
  0/1/2 times (1/4, 1/2, 1/4); B's last correction is a pronoun or ellipsis in 2/3 of the items where B is
  corrected. gold = A's latest. H5 cell: a third object C (corrected 0-1) before A or after B.
C_noupd: A stated once, never corrected; B stated and corrected 1-3 times (H3 cell 4-5). Step-3 deviation: A comes
  first in half the items and in the MIDDLE of B's statements in the other half (with A always first, "first
  mention" scored 1.00). B's last correction is indirect in 2/3 of items. Echo cells: the question and prefix
  frame-echo one of B's named corrections (H1 cell) or B's original (H2 cell). H5 cell: C stated and corrected
  0-2 times, as a block at the start or the end."""
from items_plan import (FORMS, bal, ev, corr_marker, a_chain, b_block, fit_pool, spec, factors)
from heldout_e004 import EVAL_NONCORR_MARKERS
from pools_eval import ALL_EVAL_POOLS, TRAIN_VTYPES, H6_CELLS

CELLS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "IR"]


def plan_h5(rng, n):
    F = factors(rng, n, pool=TRAIN_VTYPES, asked=[0, 1, 2], ak0=[True, False, False, False],
                ind=[True, True, False], after_ind=[True, False], q=["full", "head"])
    return [_h5_one(rng, f) for f in F]


def _h5_one(rng, f):
    nval = len(ALL_EVAL_POOLS[f["pool"]]["values"])
    asked = f["asked"]
    for _ in range(20000):
        ks = [0 if (o == asked and f["ak0"]) else rng.randint(1 if o == asked else 0, 3) for o in range(3)]
        if 3 + sum(ks) > nval:
            continue
        seqs = {o: [(o, "orig")] + [(o, "corr")] * ks[o] for o in range(3)}
        order = []
        while any(seqs.values()):
            o = rng.choice([o for o in seqs if seqs[o]])
            order.append(seqs[o].pop(0))
        events = []
        for i, (o, role) in enumerate(order):
            if role == "orig":
                mk = rng.choice(EVAL_NONCORR_MARKERS) if (i > 0 and rng.random() < 1 / 3) else None
                events.append(ev(o, "orig", "full", marker=mk))
            else:
                adj = order[i - 1][0] == o
                events.append(ev(o, "corr", rng.choice(FORMS if adj else ["full", "head"]),
                                 marker=corr_marker(rng)))
        last = max(i for i, e in enumerate(events) if e["obj"] == asked)
        after = events[last + 1:]
        if not after:
            continue
        if ks[asked] and (events[last]["ref"] in ("pron", "ell")) != f["ind"]:
            continue
        if any(e["ref"] in ("pron", "ell") for e in after) != f["after_ind"]:
            continue
        return spec("H5", f["pool"], events, 3, asked=asked, q_form=f["q"], asked_k=ks[asked],
                    after_ind=f["after_ind"])
    raise RuntimeError("H5 plan failed")


def plan_id(rng, n):
    F = factors(rng, n, pool=TRAIN_VTYPES, kind=["upd", "twoslot", "noupd"], q=["full", "head"],
                b_marker=[True, False, False])
    out = []
    for f in F:
        if f["kind"] == "upd":
            events, n_obj = a_chain(rng, rng.randint(1, 3), rng.choice(FORMS)), 1
        elif f["kind"] == "twoslot":
            events = a_chain(rng, rng.randint(1, 3), rng.choice(FORMS)) + \
                b_block(rng, f["b_marker"], rng.choice([0, 1, 1, 2]))
            n_obj = 2
        else:
            events, n_obj = [ev(0, "orig", "full")] + b_block(rng, f["b_marker"], rng.randint(0, 3)), 2
        out.append(spec("ID", f["pool"], events, n_obj, q_form=f["q"], kind=f["kind"]))
    return out


def _cell_factors(rng, cell, m, extra):
    pools = H6_CELLS if cell == "H6" else TRAIN_VTYPES
    return factors(rng, m, pool=pools, q=["full", "head"], b_marker=[True, False, False], **extra)


def plan_c_twoslot(rng, n):
    out = []
    for cell in CELLS:
        F = _cell_factors(rng, cell, n // len(CELLS), dict(
            n_b=[0, 1, 1, 2], b_ind=[True, True, False], ref=FORMS, ind=["pron", "ell", "alias"], k=[1, 2, 3],
            k23=[2, 3], k45=[4, 5], lure=[True, False], c_first=[True, False]))
        for f in F:
            alias = None
            if cell == "H1":
                a = a_chain(rng, f["k23"], f["ind"], lure_idx=(0, 1) if f["lure"] else (1,))
            elif cell == "H2":
                a = a_chain(rng, f["k"], f["ind"], earlier=["pron", "ell"], lure_idx=(0,))
            else:
                a = a_chain(rng, f["k45"] if cell == "H3" else f["k"], f["ref"])
            if cell in ("H1", "H2") and f["ind"] == "alias":
                alias, a[-1]["gap"] = 0, rng.randint(0, 2)
            n_b, k_c = f["n_b"], rng.randint(0, 1)
            if cell == "H5":      # trim to the value pool (weekday has 7): C's correction first, then B's
                nval = len(ALL_EVAL_POOLS[f["pool"]]["values"])
                k_c = max(0, min(k_c, nval - len(a) - 2 - n_b))
                n_b = max(0, min(n_b, nval - len(a) - 2 - k_c))
            b = b_block(rng, f["b_marker"], n_b, last_indirect=f["b_ind"] if n_b else None)
            events, n_obj = a + b, 2
            if cell == "H5":
                c = b_block(rng, False, k_c, obj=2)
                events, n_obj = (c + events if f["c_first"] else events + c), 3
            fit_pool(rng, events, f["pool"])
            out.append(spec("C_twoslot", f["pool"], events, n_obj, q_form=f["q"], cell=cell,
                            d=20 if cell == "H4" else 10, fillers="h7" if cell == "H7" else "default",
                            frame=rng.randrange(4) if cell in ("H1", "H2") else None, alias_obj=alias,
                            n_b=n_b))
    return out


def _noupd_events(rng, f, cell, nval):
    k_b = f["k45"] if cell == "H3" else f["k"]
    if (cell == "H1" or not f["a_first"]) and f["b_ind"] and k_b == 1:
        k_b = 2
    k_c = rng.randint(0, 2) if cell == "H5" else 0
    while 2 + k_b + (1 + k_c if cell == "H5" else 0) > nval:
        k_c -= 1
    n_before = 0 if f["a_first"] else rng.randint(0, k_b - 1 - (1 if f["b_ind"] and k_b >= 2 else 0))
    b = b_block(rng, f["b_marker"], k_b, last_indirect=f["b_ind"])
    a = [ev(0, "orig", "full")]
    if f["a_first"]:
        events = a + b
    else:
        events = b[:1 + n_before] + a + b[1 + n_before:]
        events[len(b[:1 + n_before]) + 1]["ref"] = rng.choice(["full", "head"])   # first B after A is named
    if cell == "H1":
        named = [e for e in events if e["obj"] == 1 and e["role"] == "corr" and e["ref"] in ("full", "head")]
        if not named:     # a_first: B's first correction (never its last, since k_b >= 2 when b_ind)
            first_corr = [e for e in events if e["obj"] == 1 and e["role"] == "corr"][0]
            first_corr["ref"] = "full"
            named = [first_corr]
        rng.choice(named)["lure"] = True
    elif cell == "H2":
        events[[i for i, e in enumerate(events) if e["obj"] == 1][0]]["lure"] = True
    if cell == "H5":
        c = b_block(rng, False, k_c, obj=2)
        events = c + events if f["c_first"] else events + c
    return events


def plan_c_noupd(rng, n):
    out = []
    for cell in CELLS:
        F = _cell_factors(rng, cell, n // len(CELLS), dict(
            a_first=[True, False], k=[1, 2, 3], k45=[4, 5], b_ind=[True, True, False], c_first=[True, False]))
        for f in F:
            events = _noupd_events(rng, f, cell, len(ALL_EVAL_POOLS[f["pool"]]["values"]))
            out.append(spec("C_noupd", f["pool"], events, 3 if cell == "H5" else 2, q_form=f["q"], cell=cell,
                            d=20 if cell == "H4" else 10, fillers="h7" if cell == "H7" else "default",
                            frame=rng.randrange(4) if cell in ("H1", "H2") else None, a_first=f["a_first"]))
    return out
