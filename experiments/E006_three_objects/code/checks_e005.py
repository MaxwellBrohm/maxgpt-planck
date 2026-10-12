"""E005 step 2 training-data checks for the ALIAS and IND blocks (changes (a), (b)) and the E005 purity lines that
replace E004's "0 alias/honorific in training". Each check returns a list of problem strings (empty = pass).
No model, no tokenizer. E004-block examples go through E004's own checks (checks_train*.py) unchanged."""
import functools
import re

import heldout_e004 as H
import checks_train as C
import checks_train_b as CB
from text_e004 import words, normalize, echo_runs, grams, values_in
from pools_train import POOLS
from train_e004 import prompt
from pools_alias_train import SURNAMES as TRAIN_SURNAMES

# own copies of the notes (a) vocabulary and design, so a generator or pool edit cannot move both at once
TRAIN_TITLES = ("Mr.", "Mrs.", "Ms.", "Dr.", "Pastor", "Professor", "Chef", "Captain")
ROLE_TITLES = ("Pastor", "Professor", "Chef", "Captain")
EVAL_JOIN_WORDS = ("with", "from", "run by")
PRONOUNS = {"it", "it's", "its", "they", "they're", "them", "one"}
ALIAS_KINDS = {"alias_latest": "latest", "alias_earlier": "earlier", "alias_other": "other"}
IND_KINDS = {"ind_x": 0, "ind_y": 1}


@functools.lru_cache(maxsize=1)
def eval_vocab():
    """E004 eval-side alias vocabulary: surnames, honorific+surname pairs, joins, alias templates, all draw text."""
    import items_e004 as I
    import pools_eval as PE
    pairs, texts = set(PE.ALIASES), []
    for name in I.DRAWS:
        for items in I.draw(name).values():
            for it in items:
                if it.get("alias"):
                    pairs.add(it["alias"])
                texts.append(I.prompt(it, with_prefix=True) + " " + it["gold"])
    surnames = set(H.ALIAS_SURNAMES) | {p.split(" ", 1)[1] for p in pairs}
    tpls = [t for P in PE.POOLS_E.values() for t in P.get("alias_corr", [])]
    joins = sorted({P["alias_join"] for P in PE.POOLS_E.values() if "alias_join" in P})
    obj_words = {w for P in PE.ALL_EVAL_POOLS.values() for ph, h in P["objects"] for w in words(ph) + [h]}
    return dict(pairs=pairs, surnames=surnames, tpls=tpls, joins=joins, texts=texts, obj_words=obj_words)


def _tgrams(t):
    """word 3-grams of a template, {a} and {o} -> <o>, {v} -> <v> (placeholders, as the notes define)."""
    return grams(normalize(t.replace("{a}", "{o}")), 3)


def check_alias_pools(PA):
    """pool-level: sizes, training-only vocabulary, joins and templates disjoint from the eval's."""
    p, E = [], eval_vocab()
    if len(PA.SURNAMES) != 24 or len(set(PA.SURNAMES)) != 24:
        p.append("surname pool is not 24 distinct surnames")
    if tuple(PA.TITLES) != TRAIN_TITLES or set(PA.HONORIFICS_TRAIN) != set(H.HONORIFICS):
        p.append(f"titles differ from notes (a): {PA.TITLES}")
    p += [f"eval surname in training pool: {s}" for s in PA.SURNAMES if s in E["surnames"]]
    all_vals = [v for P in POOLS.values() for v in P["values"]] + [v for vs in H.HELDOUT_VALUE_POOLS.values()
                                                                   for v in vs]
    train_obj = {w for P in POOLS.values() for ph, h in P["objects"] for w in words(ph) + [h]}
    for s in list(PA.SURNAMES) + list(ROLE_TITLES):
        if values_in(s, all_vals) or s.lower() in train_obj | E["obj_words"]:
            p.append(f"alias word is a value or object word: {s}")
        if any(re.search(r"(?<![A-Za-z])" + re.escape(s) + r"(?![A-Za-z])", t) for t in E["texts"]):
            p.append(f"training alias word occurs in an eval/dev/probe item: {s}")
    eval_g = set().union(*(_tgrams(t) for t in E["tpls"] + [j + " {a}" for j in E["joins"]]))
    for vt in POOLS:
        if len(PA.JOINS[vt]) != 3 or len(PA.ALIAS_CORR[vt]) != 6:
            p.append(f"{vt}: join/template pool size")
        for j in PA.JOINS[vt]:
            if set(words(j)) & {"with", "from", "run", "by"} or j.count("{a}") != 1:
                p.append(f"{vt}: join uses an eval join word or lacks {{a}}: {j}")
        for t in PA.ALIAS_CORR[vt] + PA.JOINS[vt]:
            ws = set(words(t.replace("{a}", "").replace("{v}", "")))
            if ws & PRONOUNS or ws & (train_obj | E["obj_words"]) or "{o}" in t or re.search(r"\d", t):
                p.append(f"{vt}: alias template holds a pronoun, object word or digit: {t}")
            if values_in(t, all_vals):
                p.append(f"{vt}: alias template holds a literal value: {t}")
            if _tgrams(t) & eval_g:
                p.append(f"{vt}: 3-gram shared with an eval alias template/join: {t} {sorted(_tgrams(t) & eval_g)}")
        for t in PA.ALIAS_CORR[vt]:
            if not t.startswith("{a} ") or t.count("{v}") != 1:
                p.append(f"{vt}: alias correction does not start with the alias or lacks one {{v}}: {t}")
            for qa in [x for r in ("ask", "ans", "ans_upd") for x in POOLS[vt][r]]:
                r = echo_runs(normalize(qa), normalize(t.replace("{a}", "")))
                if r:
                    p.append(f"{vt}: alias template echoes a question/answer template {qa!r} ~ {t!r}: {r}")
    return p


@functools.lru_cache(maxsize=1)
def _res():
    """one compiled alternation per list; a text matches it iff it matches one entry's own pattern (the
    checks_train_b._has_phrase patterns: objects with an optional plural s, markers without)."""
    alt = lambda xs: "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))
    return dict(surname=re.compile(r"(?<![A-Za-z])(" + alt(eval_vocab()["surnames"]) + r")(?![A-Za-z])"),
                object=re.compile(r"(?<![a-z])(" + alt(p.lower() for p in H.all_heldout_objects()) + r")s?(?![a-z])"),
                marker=re.compile(r"(?<![a-z])(" + alt(m.lower() for m in CB.EVAL_MARKER_WORDS) + r")(?![a-z])"))


def _text(ex):
    return prompt(ex) + ex["answer"]


def check_heldout_e5(ex):
    """E004's held-out check with the one pre-registered change: a DECLARED training alias (alias block only) may
    appear. Also: no eval surname, no eval honorific+surname pair, no eval join before a title, no other name."""
    text, E, p = _text(ex), eval_vocab(), []
    aliases = list(ex.get("aliases", {}).values())
    if aliases and ex.get("block") != "alias":
        p.append("aliases outside the alias block")
    for a in aliases:
        t, _, s = a.partition(" ")
        if t not in TRAIN_TITLES or s not in TRAIN_SURNAMES:
            p.append(f"alias not from the training pools: {a}")
    p += [f"eval surname {m}" for m in _res()["surname"].findall(text)]
    p += [f"eval alias pair {a}" for a in E["pairs"] if a in text]
    tit = "|".join(re.escape(t) for t in TRAIN_TITLES + tuple(H.HONORIFICS))
    for j in EVAL_JOIN_WORDS:
        if re.search(r"(?<![A-Za-z])" + j + r" (" + tit + r")(?![A-Za-z])", text, re.I):
            p.append(f"eval join used as a join: {j} + title")
    rest = text
    for a in sorted(aliases, key=len, reverse=True):
        rest = rest.replace(a, "the planner")
    p += [f"honorific or role outside a declared alias: {t}" for t in TRAIN_TITLES
          if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", rest)]
    p += [f"held-out object {m}" for m in _res()["object"].findall(rest.lower())]
    p += [f"eval marker {m}" for m in _res()["marker"].findall(rest.lower())]
    p += [f"name-like word {w}" for w in CB._name_like(rest)]
    if re.search(r"\bnames?\b", rest.lower()):
        p.append("mentions a name (binding family?)")
    return p


def _generic(ex):
    """E004's structure rules that do not depend on the kind (checks_train.check_structure, kind-free part)."""
    p = []
    if ex["vtype"] not in POOLS:
        return [f"undeclared value type {ex['vtype']}"]
    P = POOLS[ex["vtype"]]
    if ex["n_obj"] != 2 or len(ex["objects"]) != 2 or {s["obj"] for s in ex["stmts"]} != {0, 1}:
        p.append("not exactly 2 objects, both used")
    if any(o not in P["objects"] for o in ex["objects"]):
        p.append("object not in the training list of its type")
    (p0, h0), (p1, h1) = ex["objects"][:2] if len(ex["objects"]) >= 2 else (("", "a"), ("", "b"))
    if h0 == h1 or h0 in p1.split() or h1 in p0.split():
        p.append("head nouns clash")
    if not (0 <= ex["d"] <= 10 and 0 <= ex["pre"] <= 3 and all(0 <= g <= 2 for g in ex["gaps"])):
        p.append(f"d/pre/gap out of range {ex['d']} {ex['pre']} {ex['gaps']}")
    for o in (0, 1):
        if sum(1 for s in ex["stmts"] if s["obj"] == o and s["role"] == "corr") > 3:
            p.append(f"object {o} has more than 3 corrections")
        if [s["role"] for s in ex["stmts"] if s["obj"] == o][:1] != ["orig"]:
            p.append(f"object {o} does not start with its original")
    if any(s["role"] not in ("orig", "corr") for s in ex["stmts"]):
        p.append("role other than orig/corr")
    about = [s for s in ex["stmts"] if s["obj"] == ex["asked"]]
    if not about or about[-1]["value"] != ex["gold"]:
        p.append("gold is not the latest statement about the asked object")
    if ex["k"] != sum(1 for s in about if s["role"] != "orig"):
        p.append("k mismatch")
    if len(ex["turns"]) - 1 - ex["stmts"][-1]["turn"] != ex["d"] or ex["stmts"][0]["turn"] != ex["pre"]:
        p.append("d or pre does not match the turns")
    vals = [s["value"] for s in ex["stmts"]]
    if len(set(vals)) != len(vals):
        p.append("values not distinct")
    return p
