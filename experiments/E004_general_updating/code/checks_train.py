"""E004 training-data checks used by test_train_gen.py (and later by validate_e004.py). Each check returns a
list of problem strings (empty = pass). No model, no tokenizer."""
import re

import heldout_e004 as H
from text_e004 import (normalize, echo_runs, values_in, count_value, words, has_cue, NEGATION_CUES, HEDGES)
from pools_train import POOLS, MARKER_FMT, FILLERS_TRAIN, pool_sizes, ELL_COMMON, REV_ELL_COMMON
from train_e004 import KINDS, prompt

# the checks keep their own copy of the marker spec, so a generator edit cannot move both at once
NOTES_MARKERS = ("actually", "wait", "change of plans", "sorry, I meant", "update:", "oh,", "never mind,")
NONCORR_OK = ("actually", "wait", "oh,", "update:")        # markers that do not read as a retraction

ALL_TRAIN_VALUES = {vt: P["values"] for vt, P in POOLS.items()}
HEAD_NOUNS = sorted({h for P in POOLS.values() for _, h in P["objects"]})
STMT_ROLES = ("orig", "corr", "pron", "ell", "rev", "rev_pron", "rev_ell", "incid")
QA_ROLES = ("ask", "ans", "ans_upd")
# structure string per kind: A/B = original of object 0/1, a/b = correction, r = change back of object 0,
# i = incidental; asked object index; allowed k of the asked object
STRUCT = {"upd": (r"^Aa{1,3}$", 0), "twoslot": (r"^Aa{1,3}Bb{0,2}$", 0), "twoslot_b": (r"^Aa{1,3}Bb{0,2}$", 1),
          "noupd": (r"^ABb{0,3}$", 0), "noupd_b": (r"^ABb{0,3}$", 1), "noupd_incid": (r"^Ai$", 0),
          "upd_incid": (r"^Aa{1,3}i$", 0), "mid": (r"^ABa{1,3}$", 1), "revert": (r"^Aa{1,2}r$", 0),
          "cross": (r"^AB[ab]{2,5}$", None)}


def struct_string(ex):
    out = []
    for s in ex["stmts"]:
        if s["role"] == "incid":
            out.append("i")
        elif s["role"] == "rev":
            out.append("r" if s["obj"] == 0 else "R")
        else:
            c = "a" if s["obj"] == 0 else "b"
            out.append(c.upper() if s["role"] == "orig" else c)
    return "".join(out)


def check_structure(ex):
    p = []
    if ex["kind"] not in KINDS:
        p.append(f"undeclared kind {ex['kind']}")
        return p
    if ex["vtype"] not in POOLS:
        p.append(f"undeclared value type {ex['vtype']}")
        return p
    P = POOLS[ex["vtype"]]
    pat, asked = STRUCT[ex["kind"]]
    st = struct_string(ex)
    if not re.match(pat, st):
        p.append(f"structure {st} not {pat}")
    if asked is not None and ex["asked"] != asked:
        p.append("asked object wrong for kind")
    if ex["kind"] == "cross":
        ka, kb = st.count("a"), st.count("b")
        k_ask, k_oth = (ka, kb) if ex["asked"] == 0 else (kb, ka)
        if not (1 <= k_ask <= 3 and 1 <= k_oth <= 2):
            p.append(f"cross counts {st}")
    if ex["n_obj"] not in (1, 2) or len(ex["objects"]) != ex["n_obj"]:
        p.append("object count")
    if any(o not in P["objects"] for o in ex["objects"]):
        p.append("object not in the training list of its type")
    heads = [h for _, h in ex["objects"]]
    if len(set(heads)) != len(heads) or any(heads[i] in ex["objects"][j][0].split()
                                            for i in range(len(heads)) for j in range(len(heads)) if i != j):
        p.append("head nouns clash")
    n_obj_used = len({s["obj"] for s in ex["stmts"] if s["obj"] is not None})
    if n_obj_used != ex["n_obj"]:
        p.append("objects declared but unused")
    if not (0 <= ex["d"] <= 10 and 0 <= ex["pre"] <= 3 and all(0 <= g <= 2 for g in ex["gaps"])):
        p.append(f"d/pre/gap out of range {ex['d']} {ex['pre']} {ex['gaps']}")
    for o in range(ex["n_obj"]):
        kc = sum(1 for s in ex["stmts"] if s["obj"] == o and s["role"] in ("corr", "rev"))
        if kc > 3:
            p.append(f"object {o} has {kc} corrections")
    # the ideal rule from the annotation: latest statement about the asked object
    about = [s for s in ex["stmts"] if s["obj"] == ex["asked"]]
    if not about or about[-1]["value"] != ex["gold"]:
        p.append("gold is not the latest statement about the asked object")
    if ex["kind"] == "revert" and ex["gold"] != about[0]["value"]:
        p.append("revert gold is not the original value")
    if ex["k"] != sum(1 for s in about if s["role"] != "orig"):
        p.append("k mismatch")
    # d = filler exchanges after the last statement
    if len(ex["turns"]) - 1 - ex["stmts"][-1]["turn"] != ex["d"]:
        p.append("d does not match the turns after the last statement")
    if ex["stmts"][0]["turn"] != ex["pre"]:
        p.append("pre does not match")
    return p


def check_references(ex):
    """pronoun/ellipsis only directly after an exchange about the same object; markers only from the right set."""
    p = []
    for i, s in enumerate(ex["stmts"]):
        user = ex["turns"][s["turn"]][0]
        if s["ref"] in ("pron", "ell"):
            prev = ex["stmts"][i - 1] if i else None
            if prev is None or prev["obj"] != s["obj"] or prev["turn"] != s["turn"] - 1:
                p.append(f"indirect reference not adjacent (stmt {i})")
        if s["role"] == "orig" and s["ref"] != "full":
            p.append("original without the full phrase")
        m = s["marker"]
        if m is not None:
            if m not in NOTES_MARKERS:
                p.append(f"marker outside the notes list: {m}")
            elif s["role"] in ("orig", "incid") and m not in NONCORR_OK:
                p.append(f"retracting marker on a non-correction: {m}")
            if not user.startswith(MARKER_FMT[m].split("{s}")[0]):
                p.append("marker annotation does not match the text")
        elif any(user.startswith(f.split("{s}")[0]) for f in MARKER_FMT.values()):
            p.append(f"unannotated marker: {user}")
        if s["obj"] is not None:
            phrase, head = ex["objects"][s["obj"]]
            named = re.search(r"\b" + re.escape(head) + r"\b", user, re.I) is not None
            if s["ref"] in ("full", "head") and not named:
                p.append(f"object not named in a {s['ref']} statement: {user}")
            if s["ref"] == "full" and phrase.lower() not in user.lower():
                p.append(f"full phrase missing: {user}")
            if s["ref"] in ("pron", "ell") and named:
                p.append(f"indirect statement names the object: {user}")
        for o, (phrase, head) in enumerate(ex["objects"]):
            if o != s["obj"] and re.search(r"\b" + re.escape(head) + r"\b", user, re.I):
                p.append(f"statement names another object: {user}")
    return p


def check_values(ex):
    """one value type; each statement carries exactly its value; fillers and question carry none; the answer
    names the gold once and no other value of the pool."""
    p = []
    pool = POOLS[ex["vtype"]]["values"]
    others = [v for vt, vs in ALL_TRAIN_VALUES.items() if vt != ex["vtype"] for v in vs]
    others += [v for vs in H.HELDOUT_VALUE_POOLS.values() for v in vs]
    text = prompt(ex) + ex["answer"]
    if values_in(text, others) or re.search(r"\d", text):
        p.append(f"value of another or held-out type: {values_in(text, others)}")
    stmt_turns = {s["turn"]: s for s in ex["stmts"]}
    for t, (u, a) in enumerate(ex["turns"]):
        found = values_in(u + " " + a, pool)
        if t in stmt_turns:
            v = stmt_turns[t]["value"]
            if found != [v] or count_value(u, v) != 1:
                p.append(f"statement turn {t} values {found}")
        elif found:
            p.append(f"filler turn {t} mentions {found}")
        elif (u, a) not in FILLERS_TRAIN:
            p.append(f"turn {t} is neither a statement nor a training filler")
    if values_in(ex["question"], pool):
        p.append("question contains a value")
    fills = [x for t, x in enumerate(ex["turns"]) if t not in stmt_turns]
    if len(set(fills)) != len(fills):
        p.append("filler repeated within a dialogue")
    if values_in(ex["answer"], pool) != [ex["gold"]] or count_value(ex["answer"], ex["gold"]) != 1:
        p.append(f"answer values {values_in(ex['answer'], pool)} gold {ex['gold']}")
    return p


SHARED_N = {"ell": len(ELL_COMMON), "rev_ell": len(REV_ELL_COMMON)}   # leading entries shared by all types


def coverage_missing(exs):
    """pool entries (vtype or '*', role, index) that no example in exs used; an entry of a pool shared by every
    value type (the common ellipsis corrections) counts as used if any value type used it."""
    used = {t for ex in exs for t in ex["tpl"]}
    shared_used = {(role, j) for _, role, j in used}
    miss = set()
    for (vt, role), n in pool_sizes().items():
        for j in range(n):
            if j < SHARED_N.get(role, 0):
                if (role, j) not in shared_used:
                    miss.add(("shared", role, j))
            elif (vt, role, j) not in used:
                miss.add((vt, role, j))
    return sorted(miss)
