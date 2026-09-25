"""E005 step 2 checks, part 2: the declared structures of the ALIAS block (cases, placements, both-aliased) and the
IND block, the alias reference rules, and the per-example check table used by test_train_e005.py and
mutation_e005.py. Each check returns a list of problems. The design numbers are copied from notes.txt (a)/(b)."""
import re

import checks_train as C
import checks_train_b as CB
from checks_e005 import (_generic, check_heldout_e5, ALIAS_KINDS, IND_KINDS, PRONOUNS, TRAIN_TITLES)
from text_e004 import words

PLACES = ("adjacent", "filler", "other_obj")


def _named(text, phrase, head):
    t = text.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(x.lower()) + r"s?(?![a-z])", t) for x in (phrase, head))


def check_alias_refs(ex):
    """alias corrections name only their alias; an alias is defined once, in its object's original, right after
    the full phrase; nothing else (fillers, acks, question, answer) carries an alias."""
    p, al = [], ex.get("aliases", {})
    for s in ex["stmts"]:
        user = ex["turns"][s["turn"]][0]
        if s["ref"] == "alias":
            a = al.get(s["obj"])
            if a is None or user.count(a) != 1:
                p.append(f"alias correction without its object's alias: {user}")
            if any(_named(user, ph, h) for ph, h in ex["objects"]):
                p.append(f"alias correction names an object: {user}")
            if set(words(user)) & PRONOUNS:
                p.append(f"alias correction holds a pronoun: {user}")
            if any(b in user for o, b in al.items() if o != s["obj"]):
                p.append(f"alias correction names the other alias: {user}")
        elif s["role"] == "orig" and s["obj"] in al:
            ph = ex["objects"][s["obj"]][0]
            if not re.search(re.escape(ph) + r" (\w+ ){1,4}?" + re.escape(al[s["obj"]]), user):
                p.append(f"alias not defined right after the full phrase: {user}")
        for o, a in al.items():
            want = 1 if (s["obj"] == o and (s["role"] == "orig" or s["ref"] == "alias")) else 0
            if user.count(a) != want:
                p.append(f"alias {a} {'missing from' if want else 'in'} statement: {user}")
    st = {s["turn"] for s in ex["stmts"]}
    rest = [x for i, t in enumerate(ex["turns"]) for x in (t if i not in st else t[1:])]
    rest += [ex["question"], ex["answer"]]
    if any(t in x for x in rest for t in TRAIN_TITLES):
        p.append("a title outside the user statements")
    return p


def _prev_same(ex, i):
    return max((j for j in range(i) if ex["stmts"][j]["obj"] == ex["stmts"][i]["obj"]), default=None)


def check_alias_struct(ex):
    p = _generic(ex)
    case = ALIAS_KINDS.get(ex["kind"])
    if case is None or ex.get("case") != case or ex.get("placement") not in PLACES:
        return p + [f"undeclared alias kind/case/placement {ex['kind']} {ex.get('case')} {ex.get('placement')}"]
    S, A = ex["stmts"], ex["asked"]
    B = 1 - A
    ia = [i for i, s in enumerate(S) if s["ref"] == "alias"]
    if len(ia) != 1:
        return p + [f"{len(ia)} alias corrections (exactly 1 expected)"]
    i = ia[0]
    ao = S[i]["obj"]
    if ao != ex.get("alias_obj") or ao not in ex.get("aliases", {}):
        p.append("alias correction's object is not the declared aliased object")
    n_al = len(ex.get("aliases", {}))
    if n_al != (2 if ex.get("both_aliased") else 1):
        p.append(f"both_aliased {ex.get('both_aliased')} but {n_al} aliases")
    if n_al == 2 and len({a.split(" ", 1)[1] for a in ex["aliases"].values()}) != 2:
        p.append("two aliases share a surname")
    kA = sum(1 for s in S if s["obj"] == A and s["role"] == "corr")
    kB = sum(1 for s in S if s["obj"] == B and s["role"] == "corr")
    lastA = max(j for j, s in enumerate(S) if s["obj"] == A)
    after = sum(1 for s in S[i + 1:] if s["obj"] == A)
    if case == "latest" and not (ao == A and lastA == i and 1 <= kA <= 3 and kB <= 2):
        p.append(f"LATEST: alias not A's last, or k out of range (kA {kA}, kB {kB})")
    if case == "earlier" and not (ao == A and 1 <= after <= 2 and 2 <= kA <= 3 and kB <= 2):
        p.append(f"EARLIER: {after} A corrections after the alias (1-2), kA {kA} (2-3), kB {kB}")
    if case == "other":
        lastB = max(j for j, s in enumerate(S) if s["obj"] == B)
        if not (ao == B and lastB == i and i > lastA and kA <= 2 and 1 <= kB <= 2):
            p.append(f"OTHER: alias not B's last after A's latest, or k out of range (kA {kA}, kB {kB})")
    j = _prev_same(ex, i)
    if j is None:
        return p + ["alias correction before its object's original"]
    t0, t1 = S[j]["turn"], S[i]["turn"]
    between = [s for s in S if t0 < s["turn"] < t1]
    place = ex["placement"]
    if place == "adjacent" and t1 != t0 + 1:
        p.append("adjacent placement: alias correction is not the next turn")
    if place == "filler" and (between or not 1 <= t1 - t0 - 1 <= 2):
        p.append(f"filler placement: {t1 - t0 - 1} turns, {len(between)} statements between")
    if place == "other_obj":
        want = [S[lastA]] if case == "other" else [s for s in S if s["obj"] == B and s["role"] == "orig"]
        if between != want or t1 - t0 - 1 > 3:
            p.append("other-object placement: not exactly the other object's statement between (0-1 fillers)")
    if place in ("adjacent", "filler"):
        tb = [s["turn"] for s in S if s["obj"] == B]
        if case == "other":
            if not S[lastA]["turn"] < t0:
                p.append("OTHER adjacent/filler: B's previous statement is not after A's latest")
        elif not (max(tb) < next(s["turn"] for s in S if s["obj"] == A) or min(tb) > t1):
            p.append("adjacent/filler with A aliased: B not all before A's original or all after the alias")
    return p


def check_ind_struct(ex):
    p = _generic(ex)
    if ex["kind"] not in IND_KINDS or ex["asked"] != IND_KINDS[ex["kind"]] or ex.get("aliases"):
        return p + [f"undeclared IND kind/asked/aliases {ex['kind']} {ex['asked']}"]
    S = ex["stmts"]
    objs = [s["obj"] for s in S]
    x_last = max(i for i, o in enumerate(objs) if o == 0)
    if any(o == 1 for o in objs[:x_last + 1]):
        p.append("Y stated before X's last correction")
    kx, ky = objs.count(0) - 1, objs.count(1) - 1
    if not (1 <= kx <= 3 and 1 <= ky <= 2):
        p.append(f"IND k out of range: X {kx}, Y {ky}")
    if S[x_last]["ref"] not in ("ell", "pron"):
        p.append(f"X's last correction is {S[x_last]['ref']}, not ellipsis/pronoun")
    if any(s["ref"] == "alias" for s in S):
        p.append("alias correction in IND")
    return p


def check_structure_e5(ex):
    b = ex.get("block")
    if b == "e004":
        return C.check_structure(ex)
    if b == "alias":
        return check_alias_struct(ex)
    if b == "ind":
        return check_ind_struct(ex)
    return [f"undeclared block {b}"]


def check_render(ex):
    return [] if ex.get("render") in ("plain", "chat") else [f"render {ex.get('render')}"]


# per-example checks for every block; E004's own references/values/answer/echo checks apply to all blocks
PER_EX = [("structure", check_structure_e5), ("references", C.check_references), ("alias refs", check_alias_refs),
          ("values", C.check_values), ("answer", CB.check_answer), ("echo", CB.check_echo),
          ("heldout", check_heldout_e5), ("render", check_render)]


def all_problems(ex):
    out = {name: fn(ex) for name, fn in PER_EX}
    return {k: v for k, v in out.items() if v}
