"""SPEC 5: the token-level world rule (IDEAL) and the shortcut oracles, read from rendered ids alone.

parse() walks a chat record's turns with the pool classes of pools.Lex: a statement starts with a position-1 name
token (full "F M L a v" or definition "F M L = X"), an attribute (reord "a F M L v" or ellipsis "a v") or an alias
("X a v"); a user turn ending in " ?" is a question. Nothing here reads the generator's structures.
Every oracle returns a token id or -1 (no answer, always wrong), or a tuple (a tie, oracles.credit shares it). Only
IDEAL, O9, O10, O9|10 and the reported A rules (A1H, A-MANYFREE, A-QNEW, A-QREPEAT, A-CNTJOIN) can say " none".
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import kcommon as K

SUBSETS = [s for n in (1, 2) for s in itertools.combinations(range(3), n)]      # the 6 proper subsets
O3 = {"O3_" + "".join(str(i + 1) for i in s): s for s in SUBSETS}
REFS = ["O11", "O11|2", "O12", "O12p", "O13e", "O13a", "O14e", "O14a"]     # reference shortcuts (review R-1)
BCUE = ["B-CENTER", "B-TOKFREQ", "B-MAJOR"]     # K3 round 2: name structure, blind to the question's name
BPART = ["B-ADAPT", "B-FRAME", "B-SURPRISE"]    # K3 round 3: partial keys with order-free tie-breaks (REVIEW 4a S-1);
                                                # round 5: B-SURPRISE's tie-break reads rank and size (REVIEW 6a S-2)
FINTRO = ["F-INTRO", "F-DEFX"]                  # K3 round 2: introduction order, no reference resolved
REPORTED = ["A1H", "A-MANYFREE", "A-QNEW", "A-QREPEAT", "F-ANTE", "F-TURNK",       # reported, no bar (K3 rounds
            "A-CNTJOIN"]                                                            # 2, 3, 7 and 8)
TA = ["TA-SHARE", "TA-REP", "TA-CNT", "TA-SHREP"]   # K3 round 6 (REVIEW 7a S-1): T's class, blind to the attribute
NAMES = ["IDEAL", "O1", "O2", *O3, "O4", "O5_1", "O5_2", "O5_3", "O6", "O7", "O8", "T", *TA, "O9", "O10", "O9|10",
         *REFS, *BCUE, *BPART, *FINTRO, "R-POST", "U-TIGHTSPAN", "F-ANTESIZE", *REPORTED]
ALL = "RUBFAP"
TH = {"O2": (ALL, .50), "O6": (ALL, .50), "O7": (ALL, .50), "O8": (ALL, .50), "O1": ("U", .50),
      "T": ("RUFP", .65), **{o: ("RUFP", .65) for o in TA},    # K3 round 6 (REVIEW 7a S-1, S-2): T's bar on P too
      "O9": ("A", .50), "O10": ("A", .50), "O9|10": ("A", .50), "O4": ("F", .35),
      "O5_1": (ALL, .65), "O5_2": (ALL, .65), "O5_3": (ALL, .65), **{o: ("B", .65) for o in O3},
      **{o: ("F", .50) for o in REFS[:4]}, "O13e": ("F", .75), "O13a": ("F", .75), "O14e": ("F", .65),
      "O14a": ("F", .65), "SURF": ("F", .65), "SURFe": ("F", .85), "SURFa": ("F", .85),
      **{o: ("B", .65) for o in BCUE + BPART}, **{o: ("F", .65) for o in FINTRO}, "R-POST": ("RP", .65),
      "U-TIGHTSPAN": ("RUP", .65),     # K3 round 7 (REVIEW 8a G-1): a ceiling row too (check_k)
      "F-ANTESIZE": ("F", .65)}        # K3 round 8 (REVIEW 9a G-1): SURF's bar, a ceiling row (check_k)
# A1H (A-ONEHOLDER, REVIEW 3a): " none" when O9|10's cues fire or one name states the asked attribute among 3 or
# more names, else O4. Reported only: on A alone it beats 0.50, but it says " none" on R, U and P items too and
# loses in the mix (K3 round 2; SPEC 9 splits A's LIK by that context). A-MANYFREE, A-QNEW, A-QREPEAT (REVIEW 4a,
# A1H's class, K3 round 3): " none" when O9|10's cues fire or two or more names state no statement of the attribute /
# earlier questions asked only other names / an earlier question asked the final name; else O4. F-ANTE (REVIEW 4a
# S-3, F-ANTEATTR): O4's quarter, else drop an ellipsis whose antecedent's attribute the asked name states in another
# named statement, then the latest reference left; it reads the ellipsis's antecedent and a question-to-statement
# name match, the two pieces of the resolution, so it is reported, not barred.
# F-ANTESIZE (K3 round 8, REVIEW 9a G-1, its f_ante9.py): F-ANTE's count shadow, barred with SURF's bar and so a
# ceiling row (check_k): it reads the statement before each ellipsis for its attribute's frequency only and the asked
# name's statements for a count, no name against a reference. A-CNTJOIN (K3 round 8, REVIEW 9a O-1, its A9-CNTJOIN):
# " none" when O9|10's cues fire or no key of the asked name (T's join) has the asked attribute's statement count in
# the item, else O4. It compares the question's attribute with the asked name's keys through a count, a lossy form
# of A's binding, so SPEC 5 rules it skill: reported, and SPEC 9 splits A's LIK by the cell where it fires.
# SURF, SURFe, SURFa: purity.surf (a fit needs the whole checked set); SURFr (SURF on the items whose gold is a
# reference statement) is reported only: SURF <= 0.65 already holds it to 0.533, the named quarter being O4's.
VIEWS = {"SURF": ("pat", None), "SURFe": ("pat_e", "ell"), "SURFa": ("pat_a", "alias")}
REF_KINDS = ("ell", "alias")
FORM = {"ell": "E", "alias": "A"}
GROUP_TH = .125                                 # each O3_S and B-* rule on B cube groups (K3 round 3; twin pairs 0.30
                                                # before): twice a coin's group credit (1/16) on a partial key


class ParseError(ValueError):
    pass


def parse(rec: dict, lex) -> tuple[list[dict], list[dict], list[tuple]]:
    """-> (statements, questions, intro: names in order of first explicit mention, definitions included).
    A statement: kind, name (explicit triple or None), alias, attr, val, ent (world
    rule entity), tent (topic tracker entity: explicit name or ellipsis adjacency, never alias), toks, turn, ex.
    A question: name, attr, form, toks, gold (the value the answer turn carries), n (statements before it), defs
    (the definitions before it: (statements before the definition, name, alias)), prev (earlier questions' names),
    dex (the definitions before it: (exchange, name))."""
    st, qs, alias_of, intro, defs, dex = [], [], {}, [], [], []
    prev_ent = prev_tent = None
    ex = -1
    turns = rec["turns"]
    for ti, t in enumerate(turns):
        ids = t["ids"]
        if t["role"] != "user":
            continue
        ex += 1
        if ids and ids[-1] == K.Q:
            if len(ids) != 5:
                raise ParseError(f"question {ids}")
            name, attr, form = (tuple(ids[:3]), ids[3], 0) if ids[0] in lex.pos[0] else (tuple(ids[1:4]), ids[0], 1)
            ans = turns[ti + 1]["ids"]
            qs.append({"name": name, "attr": attr, "form": form, "toks": ids[:-1], "gold": ans[-1],
                       "n": len(st), "turn": ti + 1, "ex": ex, "defs": list(defs), "prev": [p["name"] for p in qs],
                       "prevq": [(p["name"], p["attr"], p["gold"]) for p in qs],
                       "dex": list(dex)})
            prev_ent = prev_tent = None
            continue
        i = 0
        while i < len(ids):
            a = ids[i]
            s = {"turn": ti, "ex": ex, "alias": -1, "name": None}
            if a in lex.pos[0]:
                s["name"] = tuple(ids[i:i + 3])
                if s["name"] not in intro:
                    intro.append(s["name"])
                if ids[i + 3] == K.EQ:
                    alias_of[ids[i + 4]] = s["name"]
                    defs.append((len(st), s["name"], ids[i + 4]))
                    dex.append((ex, s["name"]))
                    prev_ent = prev_tent = s["name"]
                    i += 5
                    continue
                s.update(kind="full", attr=ids[i + 3], val=ids[i + 4], toks=ids[i:i + 5])
                i += 5
            elif a in lex.attr and i + 1 < len(ids) and ids[i + 1] in lex.pos[0]:
                s.update(kind="reord", name=tuple(ids[i + 1:i + 4]), attr=a, val=ids[i + 4], toks=ids[i:i + 5])
                if s["name"] not in intro:
                    intro.append(s["name"])
                i += 5
            elif a in lex.attr:
                s.update(kind="ell", attr=a, val=ids[i + 1], toks=ids[i:i + 2])
                i += 2
            elif a in lex.alias:
                s.update(kind="alias", alias=a, attr=ids[i + 1], val=ids[i + 2], toks=ids[i:i + 3])
                i += 3
            else:
                raise ParseError(f"token {a} at {i} of turn {ti}")
            if s["attr"] not in lex.attr or s["val"] not in lex.val:
                raise ParseError(f"statement {s}")
            s["ent"] = s["name"] or (prev_ent if s["kind"] == "ell" else alias_of.get(s["alias"]))
            s["tent"] = s["name"] or (prev_tent if s["kind"] == "ell" else None)
            prev_ent, prev_tent = s["ent"], s["tent"]
            st.append(s)
    return st, qs, intro


def _last(sts, pred):
    for s in reversed(sts):
        if pred(s):
            return s["val"]
    return -1


def answers(st: list[dict], q: dict, prompt_ids: list[int], lex) -> dict:
    """Every oracle's answer to question q given the statements before it."""
    S, a, nm = st[:q["n"]], q["attr"], q["name"]
    out = {"IDEAL": _last(S, lambda s: s["ent"] == nm and s["attr"] == a)}
    out["IDEAL"] = K.NONE if out["IDEAL"] < 0 else out["IDEAL"]
    out["O1"] = next((s["val"] for s in S if s["attr"] == a), -1)
    out["O2"] = _last(S, lambda s: s["attr"] == a)
    for o, sub in O3.items():
        out[o] = _last(S, lambda s: s["attr"] == a and s["name"] is not None and all(s["name"][i] == nm[i]
                                                                                      for i in sub))
    out["O4"] = _last(S, lambda s: s["attr"] == a and s["name"] == nm)
    for n in (1, 2, 3):
        run = q["toks"][-n:]
        out[f"O5_{n}"] = _last(S, lambda s: any(s["toks"][j:j + n] == run for j in range(len(s["toks"]) - n + 1)))
    last_turn = S[-1]["turn"] if S else -1
    out["O6"] = _last(S, lambda s: s["turn"] == last_turn)
    vals = [t for t in prompt_ids if t in lex.val]
    cnt = Counter(vals)
    out["O7"] = max(reversed(vals), key=lambda v: cnt[v]) if vals else -1
    out["O8"] = S[-2]["val"] if len(S) >= 2 else -1
    out["T"] = _last(S, lambda s: s["tent"] == nm)
    out.update(_topic_attr(S, q))
    no_attr = not any(s["attr"] == a for s in S)
    no_name = not any(s["name"] == nm for s in S) and nm not in _defs(prompt_ids, lex)
    out["O9"] = K.NONE if no_attr else out["O4"]
    out["O10"] = K.NONE if no_name else out["O4"]
    out["O9|10"] = K.NONE if (no_attr or no_name) else out["O4"]
    ref = lambda s: s["kind"] in REF_KINDS                                                        # noqa: E731
    out["O11"] = _last(S, lambda s: s["attr"] == a and ref(s))
    out["O11|2"] = out["O11"] if out["O11"] >= 0 else out["O2"]
    out["O12"] = _last(S, lambda s: s["attr"] == a and (ref(s) or s["name"] == nm))
    out["O12p"] = _last(S, lambda s: s["attr"] == a and (ref(s) or any(x == y for x, y in zip(s["name"], nm))))
    for o, res, blind in (("O13e", "ell", "alias"), ("O13a", "alias", "ell")):
        out[o] = _last(S, lambda s: s["attr"] == a and (s["name"] == nm or s["kind"] == blind or
                                                        (s["kind"] == res and s["ent"] == nm)))
        out["O14" + o[-1]] = _last(S, lambda s: s["attr"] == a and s["kind"] != blind and s["ent"] == nm)
    out.update(_name_structure(S, q))
    out.update(_partial_keys(S, q))
    out["R-POST"] = _r_post(S, q)
    out.update(_intro_order(S, q, out["O4"]))
    holders = {s["name"] for s in S if s["attr"] == a and s["name"] is not None}
    names = {s["name"] for s in S if s["name"] is not None} | {d[1] for d in q.get("defs", [])}
    absent = no_attr or no_name
    out["A1H"] = K.NONE if (absent or (len(holders) == 1 and len(names) >= 3)) else out["O4"]
    prev = q.get("prev", [])
    for o, cue in (("A-MANYFREE", len(names - holders) >= 2), ("A-QNEW", bool(prev) and nm not in prev),
                   ("A-QREPEAT", nm in prev)):
        out[o] = K.NONE if (absent or cue) else out["O4"]
    out["F-ANTE"] = _ante_attr(S, q, out["O4"])
    out["F-ANTESIZE"] = _ante_size(S, q, out["O4"])
    out["A-CNTJOIN"] = K.NONE if (absent or _cnt_join(S, q)) else out["O4"]
    out["U-TIGHTSPAN"] = _u_tightspan(S, q)
    out["F-TURNK"] = _f_turnk(S, q)
    return out


def _topic_attr(S: list[dict], q: dict) -> dict:
    """TA-SHARE, TA-REP, TA-CNT, TA-SHREP (K3 round 6; REVIEW 7a S-1, its rules_ta.py): T's class, blind to the
    question's attribute. The asked entity's statements are T's (by name or ellipsis adjacency); of its keys TA-SHARE
    takes the one whose attribute the most holders state (an alias statement counts as a holder of its own), TA-REP
    the one the entity states most (then holders), TA-CNT the one with the most statements in the item, TA-SHREP
    holders then the entity's count; ties to the key stated latest; the answer is that key's latest value. Round 5's
    R and P gave the asked attribute 2 or more holders and U restated the asked key 2-4 times, unlike the entity's
    other keys: 0.69-0.76 against T's 0.65."""
    E = [s for s in S if s["tent"] == q["name"]]
    if not E:
        return {o: -1 for o in TA}
    hold, lat, X = {}, {}, []
    for i, s in enumerate(S):
        hold.setdefault(s["attr"], set()).add(s["tent"] if s["tent"] else ("ref", i))
    for i, s in enumerate(E):
        lat[s["attr"]] = i
        X += [s["attr"]] if s["attr"] not in X else []
    own, tot = Counter(s["attr"] for s in E), Counter(s["attr"] for s in S)
    key = {"TA-SHARE": lambda x: (len(hold[x]),), "TA-REP": lambda x: (own[x], len(hold[x])),
           "TA-CNT": lambda x: (tot[x],), "TA-SHREP": lambda x: (len(hold[x]), own[x])}
    return {o: E[lat[max(X, key=lambda x: (*f(x), lat[x]))]]["val"] for o, f in key.items()}


def _partial_keys(S: list[dict], q: dict) -> dict:
    """B-ADAPT, B-FRAME (REVIEW 4a S-1): the proper subset of the question's name positions with the fewest matching
    holders of the asked attribute (the first such subset in SUBSETS order), then among its holders the smallest
    value token id (B-FRAME: first the holders whose latest statement uses the question's frame). Their tie-breaks
    survive a reversal of the statements, so a reversed twin could not catch them; a cube group does."""
    hold = {}
    for s in S:
        if s["attr"] == q["attr"] and s["name"] is not None:
            hold[s["name"]] = s
    best = min(([u for u in hold if all(u[i] == q["name"][i] for i in sub)] for sub in SUBSETS),
               key=lambda m: len(m) if m else 99)
    if not best:
        return {"B-ADAPT": -1, "B-FRAME": -1, "B-SURPRISE": -1}
    want = "full" if q["form"] == 0 else "reord"
    fr = [u for u in best if hold[u]["kind"] == want] or best
    intro = []
    for s in S:
        if s["name"] is not None and s["name"] not in intro:
            intro.append(s["name"])

    def surprise(u):            # K3 round 5 (REVIEW 6a S-2): how unexpected u's intro rank is for its statement count
        x = intro.index(u) / max(1, len(intro) - 1)
        return x if sum(s["name"] == u for s in S) >= 2 else 1 - x
    top = max(surprise(u) for u in best)
    return {"B-ADAPT": min(hold[u]["val"] for u in best), "B-FRAME": min(hold[u]["val"] for u in fr),
            "B-SURPRISE": min(hold[u]["val"] for u in best if surprise(u) == top)}


def _r_post(S: list[dict], q: dict) -> int:
    """R-POST (K3 round 5, REVIEW 6a S-1): the round-5 R design's posterior pick, blind to the question's name. Holders
    (names with a named statement of the asked attribute) in the order they state it: one holder, its value; two, the
    first's (the key is followed in 5/8 of R's foil items); three or more, the last's (3/8 against at most 5/16 for
    each followed one). That was round 5's design ceiling, about 0.64 (P 0.50). K3 round 6: fams.fam_R draws the
    asked key uniformly over the item's keys, so every holder is equally likely and this reads E[1/m], about 0.43
    (purity.exch_stats checks the uniform place on every set); kept with its bar."""
    order = []
    for s in S:
        if s["attr"] == q["attr"] and s["name"] is not None and s["name"] not in order:
            order.append(s["name"])
    if not order:
        return -1
    pick = order[0] if len(order) <= 2 else order[-1]
    return _last(S, lambda s: s["attr"] == q["attr"] and s["name"] == pick)


def _latest_named(S, u, a) -> int:
    return _last(S, lambda s: s["name"] == u and s["attr"] == a)


def _u_tightspan(S: list[dict], q: dict):
    """U-TIGHTSPAN (K3 round 7, REVIEW 8a G-1, its rules_rup8.py), blind to the question's name but for question-to-
    question identity: an earlier question identical to the final one (U's lure) is copied (its answer's statement's
    name, that name's latest value of the attribute: the copy SPEC 5 carries on purpose); else, of the names with a
    named statement of the asked attribute, the one whose statements of it span the fewest user turns, ties sharing
    credit (a tuple). Where no lure was drawn in a key's span its holder is likelier the key (skill._questions: a
    lure at a turn end inside it with 1/2), so this reads U's lure-absence tilt: about 0.60 against the gate's 0.555
    on gen.py's stream in K3 round 6 (U-BAYES, the exact posterior, reads the same). A ceiling row (check_k)."""
    a = q["attr"]
    same = [g for u, x, g in q.get("prevq", []) if u == q["name"] and x == a]
    src = [s["name"] for s in S if same and s["val"] == same[-1] and s["name"] is not None]
    if src:
        return _latest_named(S, src[0], a)
    span = {}
    for s in S:
        if s["attr"] == a and s["name"] is not None:
            lo, hi = span.get(s["name"], (s["turn"], s["turn"]))
            span[s["name"]] = (min(lo, s["turn"]), max(hi, s["turn"]))
    if not span:
        return -1
    best = min(hi - lo for lo, hi in span.values())
    top = tuple(_latest_named(S, u, a) for u, (lo, hi) in span.items() if hi - lo == best)
    return top[0] if len(top) == 1 else top


def _f_turnk(S: list[dict], q: dict):
    """F-TURNK (K3 round 7, REVIEW 8a N-1, its f_foilturn8.py; reported, no bar): O4's quarter (a named statement of the
    asked key after the first reference statement of the attribute); else, of the two reference statements, the one
    sharing its user turn with a named statement of the asked name is the key's (both or neither: a tie, a tuple).
    It reads names in an ellipsis's turn (an ellipsis's antecedent is the statement just before it, often in the
    same turn), so it rebuilds part of the ellipsis link, the class SPEC 5 keeps out of F's ceiling: 0.669-0.684
    (ellipsis items 0.61-0.63, alias items 0.48-0.53) on K3 round 6's sets. Reported beside F's ceiling."""
    a, nm = q["attr"], q["name"]
    refs = [i for i, s in enumerate(S) if s["attr"] == a and s["kind"] in REF_KINDS]
    named = [i for i, s in enumerate(S) if s["attr"] == a and s["name"] == nm]
    if len(refs) != 2:
        return _last(S, lambda s: s["attr"] == a)
    if named and named[-1] > refs[0]:
        return S[named[-1]]["val"]
    has = [any(S[y]["turn"] == S[x]["turn"] and S[y]["name"] == nm for y in range(len(S)) if y != x) for x in refs]
    if has[0] == has[1]:
        return (S[refs[0]]["val"], S[refs[1]]["val"])
    return S[refs[0] if has[0] else refs[1]]["val"]


def _ante_attr(S: list[dict], q: dict, o4: int) -> int:
    nm, a = q["name"], q["attr"]
    refs = [i for i, s in enumerate(S) if s["attr"] == a and s["kind"] in REF_KINDS]
    named = [i for i, s in enumerate(S) if s["attr"] == a and s["name"] == nm]
    if not refs or (named and named[-1] > refs[0]):
        return o4
    keep = [i for i in refs if not (S[i]["kind"] == "ell" and i > 0 and any(
        s["name"] == nm and s["attr"] == S[i - 1]["attr"] for j, s in enumerate(S) if j != i - 1))]
    return S[(keep or refs)[-1]]["val"]


def _ante_size(S: list[dict], q: dict, o4: int):
    """F-ANTESIZE (K3 round 8, REVIEW 9a G-1): O4's quarter; else, with two ellipses as the asked attribute's reference
    statements and c(x) the item's statements of the attribute of the statement just before ellipsis x (read for its
    frequency, never its name), the ellipsis with the larger c when the asked name states one attribute besides the
    asked one in named statements, the smaller when it states two or more; ties, alias items and an asked name with no
    other attribute a tie of the two (a tuple). With the antecedents' attributes drawn apart it read F 0.636 (the key
    never restates its own antecedent's attribute, the foil's is one the key states in 0.20-0.40 of items); since
    BD-48 both antecedents state one attribute, so every ellipsis item is a tie and it reads the design's 0.625."""
    nm, a = q["name"], q["attr"]
    refs = [i for i, s in enumerate(S) if s["attr"] == a and s["kind"] in REF_KINDS]
    named = [i for i, s in enumerate(S) if s["attr"] == a and s["name"] == nm]
    if not refs or (named and named[-1] > refs[0]):
        return o4
    if len(refs) != 2:
        return S[refs[-1]]["val"]
    tie = (S[refs[0]]["val"], S[refs[1]]["val"])
    if any(S[i]["kind"] != "ell" or i == 0 for i in refs):
        return tie
    cnt = Counter(s["attr"] for s in S)
    ci, cj = (cnt[S[i - 1]["attr"]] for i in refs)
    nma = len({s["attr"] for s in S if s["name"] == nm and s["attr"] != a})
    if ci == cj or nma == 0:
        return tie
    return S[refs[0] if (ci > cj) == (nma == 1) else refs[1]]["val"]


def _cnt_join(S: list[dict], q: dict) -> bool:
    """A-CNTJOIN's cell (K3 round 8, REVIEW 9a O-1): the asked name has keys (T's join: named statements and ellipsis
    adjacency) and none of them has the asked attribute's statement count in the item. It never holds where the asked
    key is a named statement or an ellipsis, that key's count being the asked attribute's (R, U, P; on F it holds
    where the key is an alias, which T does not join)."""
    keys = {s["attr"] for s in S if s["tent"] == q["name"]}
    cnt = Counter(s["attr"] for s in S)
    return bool(keys) and cnt[q["attr"]] not in {cnt[x] for x in keys}


def _name_structure(S: list[dict], q: dict) -> dict:
    """B-CENTER, B-TOKFREQ, B-MAJOR (REVIEW 3a), blind to the question's name: among the names stating the asked
    attribute, the one sharing the most tokens position-wise with the item's other names, the one whose tokens occur
    most often per position over the named statements, the one holding the most per-position majority tokens of the
    item's names; the latest statement of the attribute among the best (ties: the latest)."""
    N = []
    for u in [s["name"] for s in S if s["name"] is not None] + [d[1] for d in q.get("defs", [])]:
        if u not in N:
            N.append(u)
    cand = [s for s in S if s["attr"] == q["attr"] and s["name"] is not None]
    cnt = [Counter(s["name"][i] for s in S if s["name"] is not None) for i in range(3)]
    maj = [Counter(u[i] for u in N).most_common(1)[0][0] for i in range(3)] if N else None
    score = {"B-CENTER": lambda u: sum(sum(x == y for x, y in zip(u, w)) for w in N if w != u),
             "B-TOKFREQ": lambda u: sum(cnt[i][u[i]] for i in range(3)),
             "B-MAJOR": lambda u: sum(u[i] == maj[i] for i in range(3))}
    out = {}
    for o, f in score.items():
        best = max((f(s["name"]) for s in cand), default=None)
        out[o] = _last(cand, lambda s: f(s["name"]) == best)
    return out


def _intro_order(S: list[dict], q: dict, o4: int) -> dict:
    """F-INTRO, F-DEFX (REVIEW 3a), no reference resolved: a named statement of the asked key after the first
    reference statement of the attribute (O4's quarter); else, of the attribute's reference statements, drop those
    before the asked name's first explicit mention (a named statement or its definition: they cannot be the asked
    entity's); F-DEFX also drops an alias before the asked name's definition. Of those left, the earliest (for two
    aliases after the definition, the one nearer after it): when nothing is dropped the key's reference is the
    earlier one more often, the dropped cases having been the key-later ones. No reference statement: O4."""
    nm, a = q["name"], q["attr"]
    refs = [i for i, s in enumerate(S) if s["attr"] == a and s["kind"] in REF_KINDS]
    named = [i for i, s in enumerate(S) if s["attr"] == a and s["name"] == nm]
    if not refs or (named and named[-1] > refs[0]):
        return {"F-INTRO": o4, "F-DEFX": o4}
    ndef = [n - 0.5 for n, u, _ in q.get("defs", []) if u == nm]
    first = min([i for i, s in enumerate(S) if s["name"] == nm] + ndef + [math.inf])
    keep = [i for i in refs if i > first] or refs
    kd = [i for i in keep if not (S[i]["kind"] == "alias" and ndef and i < ndef[0])] or keep
    return {"F-INTRO": S[keep[0]]["val"], "F-DEFX": S[kd[0]]["val"]}


def form_pattern(st: list[dict], q: dict, res: str | None = None) -> tuple[tuple, int]:
    """SURF's view of question q: the forms of the asked attribute's statements before it (N the asked name, O another
    name, E ellipsis, A alias) and the index of the statement carrying the gold (-1: none). res ("ell" or "alias"):
    that kind resolved (SURFe, SURFa), its statements of the asked entity in lower case (e, a)."""
    A = [s for s in st[:q["n"]] if s["attr"] == q["attr"]]
    pat = tuple(FORM[s["kind"]].lower() if s["kind"] == res and s["ent"] == q["name"] else
                FORM.get(s["kind"], "N" if s["name"] == q["name"] else "O") for s in A)
    return pat, next((i for i in range(len(A) - 1, -1, -1) if A[i]["val"] == q["gold"]), -1)


def _defs(ids, lex) -> set:
    return {tuple(ids[i - 3:i]) for i in range(3, len(ids) - 1) if ids[i] == K.EQ and ids[i + 1] in lex.alias}


def prompt_of(rec: dict, turn: int) -> list[int]:
    """Content ids of every turn before the answer turn `turn` (the scorer adds role and end tokens)."""
    return [x for t in rec["turns"][:turn] for x in t["ids"]]


def candidates(rec: dict, gold: int, lex, r) -> list[int]:
    """SPEC 9 C: gold, every V value in the item, 3 V values not in the item, " none" [C4]."""
    inside = sorted({x for t in rec["turns"] for x in t["ids"] if x in lex.val} | ({gold} - {K.NONE}))
    pool = sorted(v for v in lex.v_only if v not in set(inside))
    extra = [int(x) for x in r.choice(pool, 3, replace=False)]
    return [gold] + [v for v in inside if v != gold] + extra + ([K.NONE] if gold != K.NONE else [])


def credit(ans, gold) -> float:
    """An oracle's credit: 1 or 0, or a tie's share when it answers a tuple (U-TIGHTSPAN, F-TURNK; K3 round 7)."""
    return ans.count(gold) / len(ans) if isinstance(ans, tuple) else float(ans == gold)


def rate_ok(rate: float, t: float, n: int) -> bool:
    return rate <= t + 3 * math.sqrt(t * (1 - t) / max(1, n))
