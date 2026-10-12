"""Fitted cue model (acceptance gate, every family; K3 round 1, 2026-09-27): the measured shortcut ceiling.

A per-candidate choice model over non-resolution features, trained on training items (all families pooled at their
stream shares, so " none" competes with every family's values as in the real mix), 2-fold cross-fitted on the
training items (even items fit the odd ones and back) and fitted on all of them for the eval sets. Its accuracy per
family (max over the learners, LEARNERS) must sit at or under the family's cheap-rule bar BAR (C5 tolerance).
Candidates: every value stated before the final question (values are distinct within an item) and " none"; SPEC 9's
3 out-of-item values are never gold and every learner rules them out, so they are left out.
"Non-resolution": no feature compares a statement's name tokens with the question's (full or partial: that is R, U
and B's skill, and B's partial matches have their own O3_S bars) and no feature says which entity an ellipsis or an
alias refers to (F's skill). Allowed: positions, recency, turn layout, statement kinds and neighbours, the asked
attribute, counts, entity order and repeats read between statements (never against the question), the question's
frame, whether the asked name or attribute occurs at all (O9's and O10's cues), and earlier questions and answers
in the item, including an earlier question identical to the final one (a copy cue). check.json lists every column.
K3 round 2: NAMES columns (REVIEW 3a B-CENTER; token structure between the item's names, blind to the question: a
name's centrality, its tokens' positional frequency, its majority tokens, its distinct attributes) and a separate F
model (f_item_features, REVIEW 3a GATE-F-BLIND): its candidates are the asked attribute's statements, and it may
read the asked name, but only as the N or O form of the attribute's named statements (O4's and SURF's cue), the
reference statements' place against the asked name's first mention and definition (F-INTRO's and F-DEFX's cues)
and the asked name's intro rank; no per-entity column and no asked-name count near a statement (an ellipsis refers
to the statement just before it, so either rebuilds the resolution: REVIEW 3a read F 0.75-1.00 with them).
The ceiling covers these columns only. Chained full-name identity (an earlier question with the final name, its
answer's statement, that statement's name: REVIEW 3a LINK) is not a column: it joins the final question's name to a
statement, the skill itself (SPEC 5).
K3 round 4 (REVIEW 5a S-1; SPEC 5's ruling on name identity): question-to-question name identity (the final name
among earlier questions' names) and statement-to-statement grouping (entity sizes) make no such join, so both are
cheap; _q_cols adds the first (the second was in since round 1). No column compares a candidate's name with any
question's name.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import kcommon as K
import oracles as O

KINDS = ("full", "reord", "ell", "alias")
NEIGH = ("start", "Q", "def", *KINDS)


def _b(x: int, edges) -> int:
    return next((i for i, e in enumerate(edges) if x <= e), len(edges))


def _oh(name: str, x: int, n: int) -> list:
    return [(f"{name}{i}", float(i == x)) for i in range(n)]


def _events(rec: dict, lex) -> tuple[list[str], list[int]]:
    """Every user-turn event in order ("Q", "def", or a statement kind) and each statement's event index (the
    tokenization of oracles.parse, definitions and questions kept)."""
    ev, at = [], []
    for t in rec["turns"]:
        ids = t["ids"]
        if t["role"] != "user":
            continue
        if ids and ids[-1] == K.Q:
            ev.append("Q")
            continue
        i = 0
        while i < len(ids):
            a = ids[i]
            if a in lex.pos[0]:
                kind, w = ("def", 5) if ids[i + 3] == K.EQ else ("full", 5)
            elif a in lex.attr and i + 1 < len(ids) and ids[i + 1] in lex.pos[0]:
                kind, w = "reord", 5
            else:
                kind, w = ("ell", 2) if a in lex.attr else ("alias", 3)
            if kind != "def":
                at.append(len(ev))
            ev.append(kind)
            i += w
    return ev, at


def _names_cols(s: dict, S: list[dict], intro: list, am: bool, avals: list) -> list:
    """NAMES (K3 round 2), per statement, blind to the question: zeros on reference statements."""
    if s["name"] is None or len(intro) < 2:
        return [(c, 0.0) for c in NAMES_COLS[:-1]] + [("vrank", _vrank(s, am, avals))]
    cent = {u: sum(sum(x == y for x, y in zip(u, w)) for w in intro if w != u) for u in intro}
    cs = sorted(set(cent.values()), reverse=True)
    cnt = [Counter(x["name"][i] for x in S if x["name"] is not None) for i in range(3)]
    tf = {u: sum(cnt[i][u[i]] for i in range(3)) for u in intro}
    ts = sorted(set(tf.values()), reverse=True)
    maj = [Counter(u[i] for u in intro).most_common(1)[0][0] for i in range(3)]
    u = s["name"]
    return ([("cent_max", float(cent[u] == cs[0])),
             ("cent_umax", float(cent[u] == cs[0] and sum(c == cs[0] for c in cent.values()) == 1)),
             ("cent", cent[u] / (2.0 * (len(intro) - 1)))] + _oh("cent_rank", min(cs.index(cent[u]), 3), 4)
            + [("tf", tf[u] / max(1.0, 3.0 * sum(x["name"] is not None for x in S)))]
            + _oh("tf_rank", min(ts.index(tf[u]), 2), 3) + _oh("maj", sum(u[i] == maj[i] for i in range(3)), 4)
            + [("n_attrs", len({x["attr"] for x in S if x["name"] == u}) / 6.0), ("vrank", _vrank(s, am, avals))])


def _vrank(s, am, avals) -> float:
    return avals.index(s["val"]) / max(1, len(avals) - 1) if am else 0.0


NAMES_COLS = ["cent_max", "cent_umax", "cent", *[f"cent_rank{i}" for i in range(4)], "tf",
              *[f"tf_rank{i}" for i in range(3)], *[f"maj{i}" for i in range(4)], "n_attrs", "vrank"]


def _q_cols(q, prev, S, intro, a) -> list:
    """K3 round 4 (REVIEW 5a S-1, ruled cheap, SPEC 5): the final name among earlier questions' names (question to
    question: times asked 0/1/2+, the latest earlier question, distinct attributes asked of it 0/1/2+), the distinct
    names earlier questions named (0-3+), and the item's names stating nothing of the asked attribute in a named
    statement that no earlier question named (0-3+; A's " none" side). Item level: no candidate's name is compared
    with any question's, so no column joins the final question's name to a statement (the skill, LINK)."""
    qn = [p["name"] for p in prev]
    free = [u for u in intro if not any(s["name"] == u and s["attr"] == a for s in S)]
    return (_oh("qn_asked", min(qn.count(q["name"]), 2), 3) + [("qn_last", float(bool(qn) and qn[-1] == q["name"]))]
            + _oh("qn_attrs", min(len({p["attr"] for p in prev if p["name"] == q["name"]}), 2), 3)
            + _oh("q_names", min(len(set(qn)), 3), 4)
            + _oh("q_free_unasked", min(sum(u not in qn for u in free), 3), 4))


def item_features(rec: dict, lex) -> dict:
    """-> {"fam", "grp" (B cube groups), "gold" (index of the gold candidate), "X" [candidates, columns], "names"}."""
    st, qs, intro = O.parse(rec, lex)
    q, prev = qs[-1], qs[:-1]
    S, a = st[:q["n"]], q["attr"]
    n, ev, at = len(S), *_events(rec, lex)
    A = [i for i, s in enumerate(S) if s["attr"] == a]
    same_q = [p for p in prev if p["name"] == q["name"] and p["attr"] == a]
    overlap = max((sum(x == y for x, y in zip(u, v)) for u in intro for v in intro if u != v), default=0)
    keys = [(s["name"], s["attr"]) for s in S if s["name"] is not None]
    reps = sorted((keys.count(k) for k in set(keys)), reverse=True) or [0]
    item = (_oh("form", q["form"], 2) + _oh("acnt", min(len(A), 5), 6) + _oh("k", min(len(intro), 5), 6)
            + _oh("nst", _b(n, (4, 8, 12, 16)), 5) + _oh("nref", min(sum(s["kind"] in O.REF_KINDS for s in S), 4), 5)
            + _oh("npq", min(len(prev), 3), 4) + _oh("npqa", min(sum(p["attr"] == a for p in prev), 2), 3)
            + _oh("ovl", min(overlap, 2), 3) + _oh("rep_max", min(reps[0], 4), 5)
            + _oh("rep_keys", min(sum(c > 1 for c in reps), 2), 3)
            + _oh("a_named", min(sum(S[i]["name"] is not None for i in A), 3), 4)
            + _oh("a_ents", min(len({S[i]["name"] for i in A if S[i]["name"] is not None}), 3), 4)
            + [("name_absent", float(q["name"] not in intro)), ("attr_absent", float(not A)),
               ("has_sys", float(rec["turns"][0]["role"] == "system")), ("same_q_before", float(bool(same_q)))]
            + _q_cols(q, prev, S, intro, a))
    rows, gold = [], -1
    last_turn = S[-1]["turn"] if S else -1
    avals = sorted(S[i]["val"] for i in A)
    for i, s in enumerate(S):
        e = at[i]
        in_turn = [j for j, x in enumerate(S) if x["turn"] == s["turn"]]
        named = s["name"] is not None
        own = [j for j, x in enumerate(S) if named and x["name"] == s["name"] and x["attr"] == s["attr"]]
        ans = [p for p in prev if p["gold"] == s["val"]]
        cur = any(p in same_q and not any(x["attr"] == a for x in S[p["n"]:]) for p in ans)
        am = s["attr"] == a
        row = ([("is_none", 0.0)] + [(f"kind_{k}", float(s["kind"] == k)) for k in KINDS]
               + [("frame_match", float((s["kind"], q["form"]) in (("full", 0), ("reord", 1)))),
                  ("is_last", float(i == n - 1)), ("is_pen", float(i == n - 2)), ("is_first", float(i == 0)),
                  ("rel", i / max(1, n - 1)), ("last_in_turn", float(i == in_turn[-1])),
                  ("in_last_turn", float(s["turn"] == last_turn))]
               + _oh("rev", _b(n - 1 - i, (0, 1, 2, 3, 4, 6, 9)), 8)
               + _oh("dist", _b(q["ex"] - s["ex"] - 1, (0, 1, 2, 3, 5, 8, 12)), 8)
               + _oh("tpos", in_turn.index(i), 3) + _oh("tlen", len(in_turn) - 1, 3)
               + [(f"prev_{k}", float((ev[e - 1] if e else "start") == k)) for k in NEIGH]
               + [(f"next_{k}", float(ev[e + 1] == k)) for k in NEIGH[1:]]
               + [("a_match", float(am))] + [(c, v * am) for c, v in _oh("a_rev", min(len(A) - 1 - A.index(i), 3)
                                                                         if am else 0, 4)]
               + [(c, v * am) for c, v in _oh("a_fwd", min(A.index(i), 3) if am else 0, 4)]
               + [("ref_unknown", float(not named)), ("own_sup", float(named and own[-1] != i))]
               + [(c, v * named) for c, v in _oh("own_cnt", min(len(own), 3) - 1 if named else 0, 3)]
               + [(c, v * named) for c, v in _oh("ent_rank", min(intro.index(s["name"]), 3) if named else 0, 4)]
               + [("ent_newest", float(named and intro.index(s["name"]) == len(intro) - 1))]
               + [(c, v * named) for c, v in _oh("ent_named", min(sum(x["name"] == s["name"] for x in S), 4) - 1
                                                  if named else 0, 4)]
               + [(c, v * named) for c, v in _oh("shares", min(sum(any(x == y for x, y in zip(s["name"], u))
                                                                   for u in intro if u != s["name"]), 2)
                                                  if named else 0, 3)]
               + [("ans_same_q", float(any(p in same_q for p in ans))), ("ans_same_q_cur", float(cur)),
                  ("ans_attr_other_q", float(any(p["attr"] == a and p not in same_q for p in ans))),
                  ("ans_other_attr", float(any(p["attr"] != a for p in ans))), ("ans_any", float(bool(ans)))]
               + _names_cols(s, S, intro, am, avals))
        rows.append(row + item)
        gold = i if s["val"] == q["gold"] else gold
    assert rows, rec.get("id")                     # every SPEC 4 item states something before its question
    none = [(c, 0.0) for c, _ in rows[0][:-len(item)]]
    none[0] = ("is_none", 1.0)
    rows.append(none + item)
    gold = len(rows) - 1 if q["gold"] == K.NONE else gold
    return {"fam": rec["fam"], "grp": rec.get("meta", {}).get("grp", -1), "gold": gold, "id": rec.get("id"),
            "cheap_absent": q["name"] not in intro or not A,
            "names": [c for c, _ in rows[0]], "X": np.array([[v for _, v in r] for r in rows], dtype=np.float32)}


def _window(s, i, nm, D, gaps, nearest) -> list:
    """The F model's alias-window columns (K3 round 3): see f_item_features."""
    al = s["kind"] == "alias" and i in gaps
    g = gaps.get(i, 99)
    win = [(d[0], d[2]) for d in D if 1 <= s["ex"] - d[0] <= 10]
    before = [d for d in D if d[0] < s["ex"] or (d[0] == s["ex"] and d[1] <= i)]
    return ([("y_xgap", min(max(g, 0), 15) / 15.0 * al)]
            + [(c, v * al) for c, v in _oh("y_xg", min(max(g, 0), 11), 12)]
            + [("y_kwin", float(al and any(u == nm for _, u in win))),
               ("y_owin", float(al and any(u != nm for _, u in win))),
               ("y_owin_n", sum(u != nm for _, u in win) / 3.0 * al),
               ("y_near_is_k", float(al and bool(before) and before[-1][2] == nm)),
               ("y_gaprank", float(al and i == nearest))])


F_ITEM = ("q_rank", "k", "nst", "npq", "a_N", "a_O", "form", "y_")     # the F model's item columns (prefixes)


def f_item_features(rec: dict, lex) -> dict:
    """The F model (K3 round 2, REVIEW 3a GATE-F-BLIND; F items only): candidates are the asked attribute's statements
    before the final question (F's gold is always one of them). Per candidate: kind; N or O (a named statement of the
    attribute with the asked name or another: O4's and SURF's cue); item-last, penultimate, last in its turn, in the
    last user turn; recency and exchange-distance buckets; rank among the attribute's statements from either end;
    turn position and size; the kinds of the events just before and after it (never their names); frame match; for
    reference statements: before the asked name's first explicit mention (F-INTRO's cue), an alias before the asked
    name's definition (F-DEFX's), the gap after that definition. Per item: the asked name's intro rank, names,
    statements, earlier questions, N and O statements of the attribute, the question's form. Nothing else reads the
    asked name, and no column describes a statement's entity, so no column joins a reference to its antecedent.
    K3 round 3 (REVIEW 4a S-2): the alias-window columns y_* (REVIEW 4a feats4.f_extra): per alias, its exchange gap
    after the asked name's definition (raw and one-hot 0-11), inside that definition's 1-10 window, inside another
    definition's window (flag, count), the nearest preceding definition is the asked name's, the nearer of the
    references after the asked definition; per item the definitions, the asked one and its exchange, exchanges."""
    st, qs, intro = O.parse(rec, lex)
    q = qs[-1]
    S, a, nm = st[:q["n"]], q["attr"], q["name"]
    n, (ev, at) = len(S), _events(rec, lex)
    A = [i for i, s in enumerate(S) if s["attr"] == a]
    ndef = [d[0] - 0.5 for d in q["defs"] if d[1] == nm]
    first = min([i for i, s in enumerate(S) if s["name"] == nm] + ndef + [1e9])
    D = [(ex, nb, u) for (nb, u, _), (ex, _) in zip(q["defs"], q["dex"])]      # (exchange, statements before, name)
    dk = [d[0] for d in D if d[2] == nm]
    gaps = {i: S[i]["ex"] - dk[0] for i in A if S[i]["kind"] in O.REF_KINDS and dk}
    nearest = min((g, i) for i, g in gaps.items() if g >= 1)[1] if any(g >= 1 for g in gaps.values()) else -1
    item = (_oh("q_rank", min(intro.index(nm), 3) if nm in intro else 4, 5) + _oh("k", min(len(intro), 5) - 1, 5)
            + _oh("nst", _b(n, (4, 8, 12, 16)), 5) + _oh("npq", min(len(qs) - 1, 3), 4)
            + _oh("a_N", min(sum(S[i]["name"] == nm for i in A), 2), 3)
            + _oh("a_O", min(sum(S[i]["name"] not in (None, nm) for i in A), 2), 3) + _oh("form", q["form"], 2)
            + [("y_ndefs", min(len(D), 4) / 4.0), ("y_kdef", float(bool(dk))),
               ("y_kdef_ex", min(dk[0], 20) / 20.0 if dk else 0.0), ("y_nex", min(q["ex"], 30) / 30.0)])
    rows, gold = [], -1
    for j, i in enumerate(A):
        s, e = S[i], at[i]
        turn = [x for x, y in enumerate(S) if y["turn"] == s["turn"]]
        ref = s["kind"] in O.REF_KINDS
        gap = min([i - p for p in ndef if p < i] or [99])
        row = ([(f"kind_{k}", float(s["kind"] == k)) for k in KINDS]
               + [("N", float(s["name"] == nm)), ("O", float(s["name"] not in (None, nm)))]
               + [("is_last", float(i == n - 1)), ("is_pen", float(i == n - 2)), ("last_in_turn", float(i == turn[-1])),
                  ("in_last_turn", float(s["turn"] == S[-1]["turn"])),
                  ("frame_match", float((s["kind"], q["form"]) in (("full", 0), ("reord", 1))))]
               + _oh("rev", _b(n - 1 - i, (0, 1, 2, 3, 4, 6, 9)), 8)
               + _oh("dist", _b(q["ex"] - s["ex"] - 1, (0, 1, 2, 3, 5, 8, 12)), 8)
               + _oh("a_rev", min(len(A) - 1 - j, 3), 4) + _oh("a_fwd", min(j, 3), 4)
               + _oh("tpos", turn.index(i), 3) + _oh("tlen", len(turn) - 1, 3)
               + [(f"prev_{k}", float((ev[e - 1] if e else "start") == k)) for k in NEIGH]
               + [(f"next_{k}", float(ev[e + 1] == k)) for k in NEIGH[1:]]
               + [("pre_intro", float(ref and i < first)),
                  ("pre_def", float(s["kind"] == "alias" and bool(ndef) and i < ndef[0]))]
               + [(c, v * ref) for c, v in _oh("gap", _b(gap, (1, 2, 4, 8, 98)), 6)]
               + _window(s, i, nm, D, gaps, nearest))
        rows.append(row + item)
        gold = j if s["val"] == q["gold"] else gold
    assert rows and gold >= 0, rec.get("id")
    return {"fam": rec["fam"], "grp": -1, "gold": gold, "id": rec.get("id"), "cheap_absent": False,
            "names": [c for c, _ in rows[0]], "X": np.array([[v for _, v in r] for r in rows], dtype=np.float32)}
