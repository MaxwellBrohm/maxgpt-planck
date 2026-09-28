"""SPEC 4: skill item structure, rendering to id-carrying chat records, and the world rule (generator side).

An item is a list of exchanges. A statement exchange is a user turn of 1-3 statements and the ack " ok"; a question
exchange is "F M L a ?" (form 0) or "a F M L ?" (form 1) and the answer " v" or " none" (P: " y v"). Statement
forms: full "F M L a v", reord "a F M L v", ell "a v" (the entity of the statement just before it, in the same or
the previous user turn), alias "X a v" after a definition "F M L = X". The latest statement of a key wins.
oracles.py re-derives the same rule from the tokens alone; tests require the two to agree on every item.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import kcommon as K


@dataclass
class St:
    kind: str            # full | reord | ell | alias | def
    ent: int
    attr: int = -1       # attribute token id
    val: int = -1        # value token id
    alias: int = -1      # alias token id (alias, def)


@dataclass
class Qn:
    ent: int
    attr: int
    form: int            # 0: F M L a ?   1: a F M L ?


@dataclass
class Item:
    fam: str
    names: list          # per entity, a 3-tuple of token ids
    ex: list = field(default_factory=list)     # exchanges: ("S", [St, ...]) or ("Q", Qn)
    marker: int = -1     # P: the system marker y
    meta: dict = field(default_factory=dict)

    def questions(self) -> list[int]:
        return [i for i, e in enumerate(self.ex) if e[0] == "Q"]


def stmt_ids(s: St, names) -> list[int]:
    n = list(names[s.ent])
    if s.kind == "full":
        return [*n, s.attr, s.val]
    if s.kind == "reord":
        return [s.attr, *n, s.val]
    if s.kind == "ell":
        return [s.attr, s.val]
    if s.kind == "alias":
        return [s.alias, s.attr, s.val]
    if s.kind == "def":
        return [*n, K.EQ, s.alias]
    raise ValueError(s.kind)


def q_ids(q: Qn, names) -> list[int]:
    n = list(names[q.ent])
    return [*n, q.attr, K.Q] if q.form == 0 else [q.attr, *n, K.Q]


def world_rule(it: Item) -> list[dict]:
    """Replay the exchanges; per question -> {gold, key_last (exchange of the key's latest statement or -1)}.
    Asserts the generator's own contract: an ellipsis follows a statement of its entity with no question
    between, an alias is used only after its definition."""
    state, last_at, alias_of = {}, {}, {}
    prev_ent, out = None, []
    for xi, (kind, body) in enumerate(it.ex):
        if kind == "Q":
            key = (body.ent, body.attr)
            out.append({"gold": state.get(key, K.NONE), "key_last": last_at.get(key, -1), "at": xi})
            prev_ent = None
            continue
        for s in body:
            if s.kind == "def":
                alias_of[s.alias] = s.ent
                prev_ent = s.ent
                continue
            if s.kind == "ell":
                assert prev_ent == s.ent, "ellipsis without its entity's statement just before it"
            if s.kind == "alias":
                assert alias_of.get(s.alias) == s.ent, "alias used before or without its definition"
            state[(s.ent, s.attr)] = s.val
            last_at[(s.ent, s.attr)] = xi
            prev_ent = s.ent
    return out


def render(it: Item, rid: str, full: bool = True) -> dict:
    """-> chat record {"id", "fam", "turns"[, "qs", "meta"]}. qs: per question the index (in turns) of its answer
    turn, the gold id, the asked key and distance d (exchanges between the key's latest statement and it)."""
    golds = world_rule(it)
    turns, qs, gi = [], [], 0
    if it.marker >= 0:
        turns.append({"role": "system", "ids": [it.marker], "loss": False})
    for xi, (kind, body) in enumerate(it.ex):
        if kind == "S":
            ids = [t for s in body for t in stmt_ids(s, it.names)]
            turns += [{"role": "user", "ids": ids}, {"role": "assistant", "ids": [K.OK]}]
            continue
        g = golds[gi]
        gi += 1
        ans = [g["gold"]] if it.marker < 0 else [it.marker, g["gold"]]
        turns += [{"role": "user", "ids": q_ids(body, it.names)}, {"role": "assistant", "ids": ans}]
        kl = g["key_last"]
        qs.append({"turn": len(turns) - 1, "gold": g["gold"], "ask": [*it.names[body.ent], body.attr],
                   "form": body.form, "d": (xi - kl - 1) if kl >= 0 else -1})
    rec = {"id": rid, "fam": it.fam, "turns": turns}
    if full:
        rec["qs"] = qs
        rec["meta"] = it.meta
    return rec


def n_tokens(rec: dict) -> int:
    return sum(len(t["ids"]) + 2 for t in rec["turns"])


def group(units: list[list[St]], r, sizes=(1, 2, 3)) -> list[list[St]]:
    """Flatten ordered units (an antecedent and its ellipsis form one unit) into user turns of 1-3 statements.
    A unit may cross into the next turn (allowed: 'the same or the previous user turn')."""
    flat = [s for u in units for s in u]
    turns, i = [], 0
    while i < len(flat):
        k = int(r.choice(sizes))
        turns.append(flat[i:i + k])
        i += k
    return turns


def intro_rank(stmts: list[St], ent: int) -> int:
    seen = []
    for s in stmts:
        if s.kind in ("full", "reord", "def") and s.ent not in seen:
            seen.append(s.ent)
    return seen.index(ent) if ent in seen else -1


def follows(stmts: list[St], ent: int, attr: int) -> bool:
    """Another entity's statement of `attr` after the asked key's latest statement (SPEC 4 balance rule)."""
    last = max((i for i, s in enumerate(stmts) if s.ent == ent and s.attr == attr and s.kind != "def"), default=-1)
    return any(s.attr == attr and s.ent != ent and s.kind != "def" for s in stmts[last + 1:])
