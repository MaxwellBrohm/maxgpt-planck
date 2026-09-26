"""E004 eval item renderer: turns one planned item (items_plan.py) into text plus the builder's annotation.

spec: family, cell, pool_key, n_obj, asked, events, d, pre, q_form (full/head), frame (index into the pool's
frames or None), fillers ("default"/"h7"), alias_obj (object index whose original defines an alias, or None),
meta (plan factors, reported only). Each event: obj, role (orig/corr), ref (full/head/pron/ell/alias), marker
(eval marker or None), lure (True: rendered from the frame, so the question and prefix echo it), reuse (index of
an earlier event whose value it repeats, or None), gap (fillers before it, or None for 0-2 at random; pron and
ell always get 0, so they sit directly after the previous statement).

Item: the spec fields plus objects [(phrase, head)], alias, turns [(user, assistant)], stmts (turn, obj, value,
role, ref, marker, lure, tpl), question, prefix (forced answer prefix; the value follows it), gold (the value of
the latest statement about the asked object: the IDEAL oracle's answer), candidates (every value of the item's
value type mentioned, in order of first mention), values (the value pool), vtype, k, d, n_obj."""
import random

from text_e004 import words
from pools_eval import (ALL_EVAL_POOLS, ELL_COMMON_EVAL, ACKS_E, ALIASES, cap, with_marker, filler_pool)


def object_words(objects, alias=None):
    ws = {w for ph, h in objects for w in words(ph) + [h]}
    if alias:
        ws |= set(words(alias))
    return ws


def pick_objects(rng, objects, n):
    """n objects with distinct head nouns and no word shared between their phrases."""
    for _ in range(10000):
        objs = rng.sample(objects, n)
        ws = [set(words(ph)) | {h} for ph, h in objs]
        if all(not (ws[i] & ws[j]) for i in range(n) for j in range(i + 1, n)):
            return objs
    raise RuntimeError("no object set")


class _Templates:
    """sample templates without replacement within one item (reuse only when a pool is exhausted)."""

    def __init__(self, rng):
        self.rng, self.used = rng, {}

    def take(self, key, pool):
        used = self.used.setdefault(key, set())
        free = [j for j in range(len(pool)) if j not in used] or list(range(len(pool)))
        j = self.rng.choice(free)
        used.add(j)
        return j, pool[j]


def _filler_ok(f, banned):
    return not (set(words(f[0] + " " + f[1])) & (banned | {b + "s" for b in banned}))


def render(rng, spec):
    P = ALL_EVAL_POOLS[spec["pool_key"]]
    objs = pick_objects(rng, P["objects"], spec["n_obj"])
    alias = rng.choice(ALIASES) if spec.get("alias_obj") is not None else None
    banned = object_words(objs, alias)
    fillers = [f for f in filler_pool(spec["fillers"]) if _filler_ok(f, banned)]
    rng.shuffle(fillers)
    events = spec["events"]
    n_fresh = sum(1 for e in events if e.get("reuse") is None)
    fresh = iter(rng.sample(P["values"], n_fresh))
    vals = []
    for e in events:
        vals.append(vals[e["reuse"]] if e.get("reuse") is not None else next(fresh))
    T = _Templates(rng)
    frame = P["frames"][spec["frame"]] if spec.get("frame") is not None else None
    turns, stmts = [], []

    def fill(n):
        if n > len(fillers):
            raise RuntimeError("filler pool exhausted")
        for _ in range(n):
            turns.append(fillers.pop())

    fill(spec["pre"])
    for i, (e, v) in enumerate(zip(events, vals)):
        ref = e["ref"]
        if i > 0:
            fill(0 if ref in ("pron", "ell") else (e["gap"] if e.get("gap") is not None else rng.randint(0, 2)))
        phrase, head = objs[e["obj"]]
        if e["role"] == "orig":
            term = phrase + (f" {P['alias_join']} {alias}" if spec.get("alias_obj") == e["obj"] else "")
            key, pool = ("frame_orig", [frame["orig"]]) if e.get("lure") else ("orig", P["orig"])
        elif ref in ("full", "head"):
            term = phrase if ref == "full" else head
            key, pool = ("frame_corr", [frame["corr"]]) if e.get("lure") else ("corr", P["corr"])
        elif ref == "pron":
            term, key, pool = None, "pron", P["pron"]
        elif ref == "ell":
            term, key, pool = None, "ell", P["ell"] + ELL_COMMON_EVAL
        elif ref == "alias":
            term, key, pool = None, "alias_corr", P["alias_corr"]
        else:
            raise ValueError(ref)
        j, tpl = T.take(key, pool)
        user = with_marker(cap(tpl.format(o=term, v=v, a=alias)), e.get("marker"))
        ja, atpl = T.take("ack_" + e["role"], ACKS_E[e["role"]])
        turns.append((user, cap(atpl.format(v=v))))
        stmts.append(dict(turn=len(turns) - 1, obj=e["obj"], value=v, role=e["role"], ref=ref,
                          marker=e.get("marker"), lure=bool(e.get("lure")), tpl=(key, j)))
    fill(spec["d"])
    asked = spec["asked"]
    term = objs[asked][0] if spec["q_form"] == "full" else objs[asked][1]
    if frame is not None:
        question, prefix = frame["ask"].format(o=term), frame["pre"].format(o=term)
    else:
        question = T.take("ask", P["ask"])[1].format(o=term)
        prefix = T.take("pre", P["pre"])[1].format(o=term)
    about = [s for s in stmts if s["obj"] == asked]
    cands = []
    for s in stmts:
        if s["value"] not in cands:
            cands.append(s["value"])
    item = dict(spec, objects=objs, alias=alias, turns=turns, stmts=stmts, question=cap(question),
                prefix=cap(prefix), gold=about[-1]["value"], candidates=cands, values=P["values"],
                vtype=P["vtype"], k=sum(1 for s in about if s["role"] == "corr"), n_obj=spec["n_obj"])
    return item


def prompt(item, with_prefix=False):
    """plain "User:/Assistant:" render ending "Assistant:" (free generation) or "Assistant: <prefix>" (LIK)."""
    lines = []
    for u, a in item["turns"]:
        lines += [f"User: {u}", f"Assistant: {a}"]
    lines += [f"User: {item['question']}", "Assistant:" + (" " + item["prefix"] if with_prefix else "")]
    return "\n".join(lines)


def seeded(seed, family):
    return random.Random(f"E004:{seed}:{family}")
