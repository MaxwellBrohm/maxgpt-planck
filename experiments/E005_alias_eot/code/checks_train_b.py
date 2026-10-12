"""E004 training-data checks, part 2: answers as sentences (grader clauses 1-2, 5-7), frame echoes, held-out
vocabulary and names per example; fillers and templates at pool level. Each check returns a list of problems."""
import os
import re
import sys

import heldout_e004 as H
from text_e004 import (normalize, echo_runs, values_in, words, has_cue, NEGATION_CUES, HEDGES)
from pools_train import POOLS, ACKS, FILLERS_TRAIN, CAP_VALUES
from checks_train import STMT_ROLES, QA_ROLES, HEAD_NOUNS, ALL_TRAIN_VALUES
from train_e004 import prompt

ALL_VALUES = [v for vs in ALL_TRAIN_VALUES.values() for v in vs] + \
             [v for vs in H.HELDOUT_VALUE_POOLS.values() for v in vs]
MARKER_WORDS = ["actually", "wait", "change of plans", "sorry", "update", "oh", "never mind"]
EVAL_MARKER_WORDS = ["hold on", "correction", "one more change", "oops", "on second thought"]
N_NOTES = {"weekday": (7, 14), "colour": (10, 14), "month": (12, 12), "city": (16, 12)}   # values, objects


def _has_phrase(text, ph, plural=True):
    return re.search(r"(?<![a-z])" + re.escape(ph.lower()) + ("s?" if plural else "") + r"(?![a-z])",
                     text.lower()) is not None


def _name_like(text):
    """capitalized words that do not start a sentence and are not I/I'm/... or a capitalized training value."""
    out = []
    for line in text.split("\n"):
        line = re.sub(r"^(User|Assistant):\s*", "", line)
        for sent in re.split(r"(?<=[.?!:])\s+", line):
            for w in sent.split()[1:]:
                w = w.strip(",.?!:;")
                if w[:1].isupper() and w not in CAP_VALUES and not (w == "I" or w.startswith("I'")):
                    out.append(w)
    return out


def check_answer(ex):
    p = []
    a = ex["answer"]
    if not (a.startswith(" ") and a.endswith("\n") and a.count("\n") == 1):
        p.append(f"answer not ' sentence\\n': {a!r}")
    s = a.strip()
    ws = words(s)
    if not s[:1].isupper() or not s.endswith("."):
        p.append(f"not a sentence: {s!r}")
    if len(ws) < 3:
        p.append(f"bare answer: {s!r}")
    tri = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if len(tri) != len(set(tri)) or any(ws[i] == ws[i + 1] == ws[i + 2] for i in range(len(ws) - 2)):
        p.append(f"repetition: {s}")
    if has_cue(s, NEGATION_CUES):
        p.append(f"negation cue: {s}")
    if "?" in s or has_cue(s, HEDGES):
        p.append(f"question or hedge: {s}")
    obj_words = {w for ph, h in ex["objects"] for w in words(ph) + words(h)}
    if any(w == "my" and obj_words & set(ws[i + 1:i + 4]) for i, w in enumerate(ws)):
        p.append(f"user's voice: {s}")
    if s.startswith(("User:", "Assistant:")):
        p.append("answer starts with a role tag")
    if not s.startswith(ex["answer_pre"]):
        p.append("answer_pre is not the answer's start")
    return p


def check_echo(ex):
    """0 frame echoes: question and answer vs every statement (user turn) and its acknowledgement."""
    terms = [t for ph, h in ex["objects"] for t in (ph, h)]
    pool = POOLS[ex["vtype"]]["values"]
    q = normalize(ex["question"], terms, pool)
    a = normalize(ex["answer"], terms, pool)
    p = []
    for s in ex["stmts"]:
        for side, txt in zip(("statement", "ack"), ex["turns"][s["turn"]]):
            nt = normalize(txt, terms, pool)
            for name, x in (("question", q), ("answer", a)):
                r = echo_runs(x, nt)
                if r:
                    p.append(f"echo {name} vs {side}: {r}")
    return p


def check_heldout(ex):
    text = prompt(ex) + ex["answer"]
    p = [f"held-out object {ph}" for ph in H.all_heldout_objects() if _has_phrase(text, ph)]
    p += [f"eval marker {m}" for m in EVAL_MARKER_WORDS if _has_phrase(text, m, plural=False)]
    p += [f"honorific {h}" for h in H.HONORIFICS if h in text]
    p += [f"name-like word {w}" for w in _name_like(text)]
    if re.search(r"\bnames?\b", text.lower()):
        p.append("mentions a name (binding family?)")
    return p


def check_fillers():
    p = []
    if len(FILLERS_TRAIN) < 40 or len(set(FILLERS_TRAIN)) != len(FILLERS_TRAIN):
        p.append(f"filler pool size {len(FILLERS_TRAIN)} or duplicates")
    phrases = [ph for P in POOLS.values() for ph, _ in P["objects"]]
    for u, a in FILLERS_TRAIN:
        t = u + " " + a
        if values_in(t, ALL_VALUES) or re.search(r"\d", t):
            p.append(f"filler has a value or digit: {u}")
        p += [f"filler names object word {h}: {u}" for h in HEAD_NOUNS + phrases if _has_phrase(t, h)]
        p += [f"filler has marker {m}: {u}" for m in MARKER_WORDS + EVAL_MARKER_WORDS
              if _has_phrase(t, m, plural=False)]
        p += [f"filler has held-out object {ph}" for ph in H.all_heldout_objects() if _has_phrase(t, ph)]
        p += [f"filler name-like word {w}" for w in _name_like(u) + _name_like(a)]
        if not u.endswith("?") or not a.endswith("."):
            p.append(f"filler is not a Q&A: {u}")
    return p


def e001_filler_overlap():
    """word 5-grams shared between the training fillers and E001's filler pools (the E004 eval fillers
    include E001's pool). Imports E001 read-only, writing no bytecode."""
    sys.dont_write_bytecode = True
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "E001_battery_and_probes", "code")
    sys.path.insert(0, os.path.abspath(d))
    import items as I1
    import items_new as I1n
    sys.path.pop(0)
    g = lambda t, n=5: {tuple(words(t)[i:i + n]) for i in range(len(words(t)) - n + 1)}
    ours = set().union(*(g(u + " " + a) for u, a in FILLERS_TRAIN))
    theirs = set().union(*(g(u + " " + a) for u, a in I1.DISTRACTORS + I1n.EXTRA_DISTRACTORS))
    return sorted(ours & theirs)


def check_pools():
    p = []
    for vt, P in POOLS.items():
        nv, no = N_NOTES[vt]
        if len(P["values"]) != nv or len(P["objects"]) != no:
            p.append(f"{vt}: values/objects differ from notes.txt")
        n_corr = len(P["corr"]) + len(P["pron"]) + len(P["ell"])
        if len(P["orig"]) < 8 or n_corr < 10 or len(P["ask"]) < 8 or len(P["ans"]) < 8:
            p.append(f"{vt}: pool below the notes minimum")
        for role in STMT_ROLES + QA_ROLES:
            for t in P[role]:
                need_o = role in ("orig", "corr", "rev", "ask")
                need_v = role != "ask"
                if (("{o}" in t) != need_o and role not in ("ans", "ans_upd")) or (("{v}" in t) != need_v):
                    p.append(f"{vt}/{role} placeholder: {t}")
                if any(t.lower().startswith(m) for m in MARKER_WORDS):
                    p.append(f"{vt}/{role} starts with a marker: {t}")
                if values_in(t, ALL_VALUES) or re.search(r"\d", t):
                    p.append(f"{vt}/{role} has a literal value: {t}")
        stmts = [t for r in STMT_ROLES for t in P[r]] + [t for pool in ACKS.values() for t in pool]
        for qa in [t for r in QA_ROLES for t in P[r]]:
            for st in stmts:
                r = echo_runs(normalize(qa), normalize(st))
                if r:
                    p.append(f"{vt}: pool echo {qa!r} ~ {st!r}: {r}")
        for t in P["ans"] + P["ans_upd"]:
            if has_cue(t, NEGATION_CUES) or has_cue(t, HEDGES):
                p.append(f"{vt}: answer template with negation or hedge: {t}")
    for pool in ACKS.values():
        for t in pool:
            if any(t.lower().startswith(m) for m in MARKER_WORDS):
                p.append(f"ack starts with a marker: {t}")
    return p
