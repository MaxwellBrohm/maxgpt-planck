"""E004 eval item plans, part 1: shared helpers and the U-family plans (notes (b)). A plan is a spec for
items_render.render(). Structural factors are BALANCED within a family draw (bal(): exact shares, shuffled), so a
cheap oracle's score on a factor-determined structure is fixed by design rather than by sampling luck.

Eval defaults (every family unless it changes one): value type per item cycles the four training types (16
each at n = 64), eval objects and pools, d = 10, 0-3 fillers before, 0-2 between, k = 1-3; A = object 0 is asked.
Object B (same value type) is stated before A's original in half the items and after A's latest correction in
the other half; step-3 deviation: after A's latest, B is always followed directly by its own pronoun or ellipsis
correction, so "the latest statement that names no other object" (oracle O7) is wrong in that half (with B
stated once, O7 scored 1.00 on every U family). B's first statement carries a marker in 1/3 of items. Latest
correction of A: full 1/4, head 1/4, pron 1/4, ell 1/4 (H1/H2: pron/ell/alias 1/3 each). Every correction
carries an eval correction marker with probability 1/2. Step-3 deviation for H1/H2 alias items with B not after
A: B sits between A's previous statement and the alias correction (not before A's original), so that tracking
the topic by adjacency does not resolve the alias (the diagnostic topic tracker scored 1.00 on alias items).
Value type, B position and latest reference form are balanced jointly (joint())."""
import itertools

from heldout_e004 import EVAL_CORR_MARKERS, EVAL_NONCORR_MARKERS
from pools_eval import ALL_EVAL_POOLS, TRAIN_VTYPES, H6_CELLS

FORMS = ["full", "head", "pron", "ell"]


def bal(rng, n, levels):
    out = [levels[i % len(levels)] for i in range(n)]
    rng.shuffle(out)
    return out


def ev(obj, role, ref, marker=None, lure=False, gap=None, reuse=None):
    return dict(obj=obj, role=role, ref=ref, marker=marker, lure=lure, gap=gap, reuse=reuse)


def corr_marker(rng):
    return rng.choice(EVAL_CORR_MARKERS) if rng.random() < 0.5 else None


def a_chain(rng, k, latest_ref, earlier=FORMS, lure_idx=(), alias=False):
    """A's original plus k corrections; lure_idx: event indices rendered from the frame (0 = the original);
    lure corrections are named (full/head). earlier: forms allowed for non-latest, non-lure corrections."""
    out = [ev(0, "orig", "full", lure=0 in lure_idx)]
    for c in range(1, k + 1):
        if c == k:
            ref = latest_ref
        elif c in lure_idx:
            ref = rng.choice(["full", "head"])
        else:
            ref = rng.choice(earlier)
        out.append(ev(0, "corr", ref, marker=corr_marker(rng), lure=c in lure_idx))
    return out


def b_block(rng, b_marker, n_corr, obj=1, last_indirect=None):
    """object obj's original (named, maybe marked) plus n_corr corrections, all adjacent to it; last_indirect
    forces the last correction's form to pron/ell (True) or full/head (False)."""
    out = [ev(obj, "orig", "full", marker=rng.choice(EVAL_NONCORR_MARKERS) if b_marker else None)]
    for c in range(1, n_corr + 1):
        if c == n_corr and last_indirect is not None:
            ref = rng.choice(["pron", "ell"] if last_indirect else ["full", "head"])
        else:
            ref = rng.choice(FORMS)
        out.append(ev(obj, "corr", ref, marker=corr_marker(rng)))
    return out


def fit_pool(rng, events, pool_key):
    """if the dialogue needs more distinct values than the pool has (weekday, 7), later non-latest corrections
    of object 0 repeat an earlier value of object 0 (never adjacent, never the gold, never a lure)."""
    need = len(events) - len(ALL_EVAL_POOLS[pool_key]["values"])
    a_idx = [i for i, e in enumerate(events) if e["obj"] == 0]
    latest = a_idx[-1]
    for _ in range(max(0, need)):
        opts = [(j, s) for pos, j in enumerate(a_idx) for s in a_idx[:max(0, pos - 1)]
                if j != latest and events[j]["reuse"] is None and not events[j]["lure"] and not events[s]["lure"]
                and events[s]["reuse"] is None and events[j]["role"] == "corr"]
        if not opts:
            raise RuntimeError("pool too small")
        j, s = rng.choice(opts)
        events[j]["reuse"] = s
    return events


def spec(family, pool_key, events, n_obj, asked=0, d=10, q_form="full", frame=None, fillers="default",
         alias_obj=None, cell=None, **meta):
    return dict(family=family, cell=cell, pool_key=pool_key, events=events, n_obj=n_obj, asked=asked, d=d,
                pre=None, q_form=q_form, frame=frame, fillers=fillers, alias_obj=alias_obj, meta=meta)


def u_default(rng, family, pool_key, k, b_after, latest_ref, b_marker, q_form, d=10, fillers="default",
              cell=None):
    """the default U structure: A corrected k times; B before A's original or after A's latest (then B's own
    adjacent pronoun/ellipsis correction follows)."""
    a = a_chain(rng, k, latest_ref)
    if b_after:
        events = a + b_block(rng, b_marker, 1, last_indirect=True)
    else:
        events = b_block(rng, b_marker, 0) + a
    fit_pool(rng, events, pool_key)
    return spec(family, pool_key, events, 2, d=d, q_form=q_form, fillers=fillers, cell=cell,
                b_after=b_after, latest_ref=latest_ref)


def echo_u(rng, family, pool_key, k, b_after, latest_ref, b_marker, q_form, orig_lure, cell=None, b_corr=1):
    """H1 (orig_lure None/False/True with k >= 2: the frame is on A's first correction, and on the original
    too when orig_lure) and H2 (orig_lure = "only": the frame is on the original and every correction is a
    pronoun, ellipsis or alias, so no correction shares a content word with the question or prefix)."""
    alias = latest_ref == "alias"
    if orig_lure == "only":
        a = a_chain(rng, k, latest_ref, earlier=["pron", "ell"], lure_idx=(0,))
    else:
        a = a_chain(rng, k, latest_ref, lure_idx=(0, 1) if orig_lure else (1,))
    if alias:
        a[-1]["gap"] = rng.randint(0, 2)
    if b_after:
        events = a + b_block(rng, b_marker, b_corr, last_indirect=True if b_corr else None)
    elif alias:       # B sits between A's previous statement and the alias correction, so the alias is needed
        events = a[:-1] + b_block(rng, b_marker, 0) + a[-1:]
    else:
        events = b_block(rng, b_marker, 0) + a
    fit_pool(rng, events, pool_key)
    return spec(family, pool_key, events, 2, q_form=q_form, frame=rng.randrange(4), cell=cell,
                alias_obj=0 if alias else None, b_after=b_after, latest_ref=latest_ref, orig_lure=orig_lure)


def factors(rng, n, **levels):
    cols = {name: bal(rng, n, lv) for name, lv in levels.items()}
    return [{name: cols[name][i] for name in cols} for i in range(n)]


def joint(rng, n, **levels):
    """factors balanced JOINTLY: every combination equally often (up to n's divisibility), shuffled."""
    names, combos = list(levels), list(itertools.product(*levels.values()))
    rng.shuffle(combos)
    out = [dict(zip(names, combos[i % len(combos)])) for i in range(n)]
    rng.shuffle(out)
    return out


def plan_u(rng, family, n):
    """H1, H2, H3, H4, H6, H7 (H5 lives in items_plan_b.py)."""
    F = factors(rng, n, b_marker=[True, False, False], q=["full", "head"], lure=[True, False], k45=[4, 5],
                k=[1, 2, 3], ind=["pron", "ell", "alias"], pool=H6_CELLS if family == "H6" else TRAIN_VTYPES)
    # B position exactly half/half with the value type (O2, O7 and X6 depend on it alone)
    if family == "H1":
        J = joint(rng, n, pool=TRAIN_VTYPES, b_after=[True, False], k23=[2, 3])
    elif family == "H2":
        J = joint(rng, n, pool=TRAIN_VTYPES, b_after=[True, False])
    else:
        J = joint(rng, n, b_after=[True, False], ref=FORMS)     # exact at n = 16, 32 and 64
    out = []
    for f in [dict(a, **b) for a, b in zip(F, J)]:
        if family == "H1":
            s = echo_u(rng, family, f["pool"], f["k23"], f["b_after"], f["ind"], f["b_marker"], f["q"], f["lure"])
        elif family == "H2":
            s = echo_u(rng, family, f["pool"], f["k"], f["b_after"], f["ind"], f["b_marker"], f["q"], "only")
        else:
            k = f["k45"] if family == "H3" else f["k"]
            s = u_default(rng, family, f["pool"], k, f["b_after"], f["ref"], f["b_marker"], f["q"],
                          d=20 if family == "H4" else 10, fillers="h7" if family == "H7" else "default")
        out.append(s)
    return out
