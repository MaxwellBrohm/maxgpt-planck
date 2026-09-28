"""SPEC 5: the token-level world rule (IDEAL) and the shortcut oracles, read from rendered ids alone.

parse() walks a chat record's turns with the pool classes of pools.Lex: a statement starts with a position-1 name
token (full "F M L a v" or definition "F M L = X"), an attribute (reord "a F M L v" or ellipsis "a v") or an alias
("X a v"); a user turn ending in " ?" is a question. Nothing here reads the generator's structures.
Every oracle returns a token id or -1 (no answer, always wrong). Only IDEAL, O9, O10 and O9|10 can say " none".
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
BPART = ["B-ADAPT", "B-FRAME"]                  # K3 round 3: partial keys with order-free tie-breaks (REVIEW 4a S-1)
FINTRO = ["F-INTRO", "F-DEFX"]                  # K3 round 2: introduction order, no reference resolved
REPORTED = ["A1H", "A-MANYFREE", "A-QNEW", "A-QREPEAT", "F-ANTE"]      # reported, no bar (K3 rounds 2 and 3)
NAMES = ["IDEAL", "O1", "O2", *O3, "O4", "O5_1", "O5_2", "O5_3", "O6", "O7", "O8", "T", "O9", "O10", "O9|10", *REFS,
         *BCUE, *BPART, *FINTRO, *REPORTED]
ALL = "RUBFAP"
TH = {"O2": (ALL, .50), "O6": (ALL, .50), "O7": (ALL, .50), "O8": (ALL, .50), "O1": ("U", .50),
      "T": ("RUF", .65), "O9": ("A", .50), "O10": ("A", .50), "O9|10": ("A", .50), "O4": ("F", .35),
      "O5_1": (ALL, .65), "O5_2": (ALL, .65), "O5_3": (ALL, .65), **{o: ("B", .65) for o in O3},
      **{o: ("F", .50) for o in REFS[:4]}, "O13e": ("F", .75), "O13a": ("F", .75), "O14e": ("F", .65),
      "O14a": ("F", .65), "SURF": ("F", .65), "SURFe": ("F", .85), "SURFa": ("F", .85),
      **{o: ("B", .65) for o in BCUE + BPART}, **{o: ("F", .65) for o in FINTRO}}
# A1H (A-ONEHOLDER, REVIEW 3a): " none" when O9|10's cues fire or one name states the asked attribute among 3 or
# more names, else O4. Reported only: on A alone it beats 0.50, but it says " none" on R, U and P items too and
# loses in the mix (K3 round 2; SPEC 9 splits A's LIK by that context). A-MANYFREE, A-QNEW, A-QREPEAT (REVIEW 4a,
# A1H's class, K3 round 3): " none" when O9|10's cues fire or two or more names state no statement of the attribute /
# earlier questions asked only other names / an earlier question asked the final name; else O4. F-ANTE (REVIEW 4a
# S-3, F-ANTEATTR): O4's quarter, else drop an ellipsis whose antecedent's attribute the asked name states in another
# named statement, then the latest reference left; it reads the ellipsis's antecedent and a question-to-statement
# name match, the two pieces of the resolution, so it is reported, not barred.
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
    return out


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
        return {"B-ADAPT": -1, "B-FRAME": -1}
    want = "full" if q["form"] == 0 else "reord"
    fr = [u for u in best if hold[u]["kind"] == want] or best
    return {"B-ADAPT": min(hold[u]["val"] for u in best), "B-FRAME": min(hold[u]["val"] for u in fr)}


def _ante_attr(S: list[dict], q: dict, o4: int) -> int:
    nm, a = q["name"], q["attr"]
    refs = [i for i, s in enumerate(S) if s["attr"] == a and s["kind"] in REF_KINDS]
    named = [i for i, s in enumerate(S) if s["attr"] == a and s["name"] == nm]
    if not refs or (named and named[-1] > refs[0]):
        return o4
    keep = [i for i in refs if not (S[i]["kind"] == "ell" and i > 0 and any(
        s["name"] == nm and s["attr"] == S[i - 1]["attr"] for j, s in enumerate(S) if j != i - 1))]
    return S[(keep or refs)[-1]]["val"]


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


def rate_ok(rate: float, t: float, n: int) -> bool:
    return rate <= t + 3 * math.sqrt(t * (1 - t) / max(1, n))
