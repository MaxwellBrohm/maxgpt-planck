"""E004 eval checks at pool and draw level (notes (b)). Each returns a list of problems (empty = pass).
  check_pools   every eval pool: placeholders, no literal value/digit/marker, object sets clean (eval objects of
                the training types share no word with any training object or E001 object; every eval object is
                listed in heldout_e004), 0 frame echoes between any default ask/prefix and any statement or
                acknowledgement template; each frame's ask and prefix echo its own orig and corr and no default
                template; no pronoun/ellipsis/alias template shares a content word with any frame ask/prefix.
  check_fillers every eval filler: no value, digit, marker word, name-like word; H7 fillers not in the default
                pool; 0 word 5-grams between eval fillers and training fillers.
  check_draws   eval/dev/probe share no prompt; family sizes; exact balance of the B position in U families."""
import re

import heldout_e004 as H
from text_e004 import normalize, echo_runs, values_in, words, content_words
from pools_eval import (ALL_EVAL_POOLS, POOLS_E, ELL_COMMON_EVAL, ACKS_E, ALL_VALUES, H7_FILLERS, eval_fillers)
from pools_train import POOLS as TRAIN_POOLS, FILLERS_TRAIN
from checks_eval import MARKER_WORDS

STMT_ROLES = ("orig", "corr", "pron", "ell", "alias_corr")


def _templates(P, roles):
    out = []
    for r in roles:
        out += [(r, t) for t in P.get(r, [])]
    if "ell" in roles:
        out += [("ell", t) for t in ELL_COMMON_EVAL]
    return out


def check_pools():
    p = []
    train_words = {w for P in TRAIN_POOLS.values() for ph, h in P["objects"] for w in words(ph) + [h]}
    e001 = {w for ph in H.E001_OBJECTS for w in words(ph)}
    listed = set(H.all_heldout_objects())
    for key, P in ALL_EVAL_POOLS.items():
        heads = [h for _, h in P["objects"]]
        if len(set(heads)) != len(heads) or len(P["objects"]) < 12:
            p.append(f"{key}: duplicate heads or fewer than 12 objects")
        for ph, h in P["objects"]:
            if ph not in listed:
                p.append(f"{key}: object {ph} not listed in heldout_e004")
            if (set(words(ph)) | {h}) & (train_words | e001):
                p.append(f"{key}: object {ph} shares a word with a training or E001 object")
        stmts = _templates(P, STMT_ROLES)
        for role, t in stmts + [(r, t) for r in ("ask", "pre") for t in P[r]]:
            bad = (("{v}" in t) == (role in ("ask", "pre"))
                   or (role in ("orig", "corr", "ask") and "{o}" not in t)
                   or (role in ("pron", "ell", "alias_corr") and "{o}" in t)
                   or (("{a}" in t) != (role == "alias_corr")))
            if bad:
                p.append(f"{key}/{role} placeholder: {t}")
            if values_in(t, ALL_VALUES) or re.search(r"\d", t):
                p.append(f"{key}/{role} literal value: {t}")
            if any(t.lower().startswith(m) for m in MARKER_WORDS):
                p.append(f"{key}/{role} starts with a marker: {t}")
        acks = [("ack", t) for pool in ACKS_E.values() for t in pool]
        for qa in [t for r in ("ask", "pre") for t in P[r]]:
            for role, st in stmts + acks + [("frame", f[x]) for f in P.get("frames", []) for x in ("orig", "corr")]:
                r = echo_runs(normalize(qa), normalize(st))
                if r:
                    p.append(f"{key}: default {qa!r} echoes {role} {st!r}: {r}")
        for i, f in enumerate(P.get("frames", [])):
            for x in ("orig", "corr"):
                if not (echo_runs(normalize(f["ask"]), normalize(f[x])) and
                        echo_runs(normalize(f["pre"]), normalize(f[x]))):
                    p.append(f"{key}: frame {i} ask/pre does not echo its {x}")
            for role, st in stmts + acks:
                if echo_runs(normalize(f["ask"]), normalize(st)) or echo_runs(normalize(f["pre"]), normalize(st)):
                    p.append(f"{key}: frame {i} echoes default {role} {st!r}")
            fq = set(content_words(f["ask"])) | set(content_words(f["pre"]))
            for role, st in _templates(P, ("pron", "ell", "alias_corr")):
                if set(content_words(st)) & fq:
                    p.append(f"{key}: {role} {st!r} shares content words with frame {i}")
    for key in POOLS_E:
        for x in ("frames", "alias_corr", "alias_join"):
            if not POOLS_E[key].get(x):
                p.append(f"{key}: missing {x}")
    return p


def _name_like(text):
    out = []
    for sent in re.split(r"(?<=[.?!:/])\s+", text):
        out += [w.strip(",.?!:;") for w in sent.split()[1:] if w[:1].isupper() and w.strip(",.?!:;") != "I"
                and not w.startswith("I'")]
    return out


def ngrams(text, n):
    ws = words(text)
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def check_fillers():
    p = []
    default, h7 = eval_fillers(), list(H7_FILLERS)
    if set(default) & set(h7) or len(set(default)) != len(default) or len(set(h7)) != len(h7):
        p.append("filler pools overlap or repeat")
    if len(default) < 45 or len(h7) < 26:
        p.append(f"filler pools too small: {len(default)} default, {len(h7)} h7")
    train5 = set().union(*(ngrams(u + " " + a, 5) for u, a in FILLERS_TRAIN))
    for u, a in default + h7:
        t = u + " " + a
        if values_in(t, ALL_VALUES) or re.search(r"\d", t):
            p.append(f"filler has a value or digit: {u}")
        if any(re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", t.lower()) for m in MARKER_WORDS):
            p.append(f"filler has a marker word: {u}")
        if _name_like(u) + _name_like(a):
            p.append(f"filler has a name-like word: {u} {_name_like(u) + _name_like(a)}")
        if ngrams(t, 5) & train5:
            p.append(f"filler shares a 5-gram with training fillers: {u}")
    return p


def check_draws(draws):
    """draws: {name: {family: items}}"""
    from items_render import prompt
    p = []
    seen = {}
    for name, D in draws.items():
        for fam, items in D.items():
            for it in items:
                seen.setdefault(prompt(it, True), set()).add(name)
            want = {"eval": 64, "dev": 32, "probe": 16}[name]
            if len(items) != want:
                p.append(f"{name}/{fam}: {len(items)} items")
            if fam in ("H1", "H2", "H3", "H4", "H6", "H7") and name == "eval":
                share = sum(it["meta"]["b_after"] for it in items) / len(items)
                if share != 0.5:
                    p.append(f"{name}/{fam}: B-after share {share}")
    dup = [k for k, v in seen.items() if len(v) > 1]
    if dup:
        p.append(f"{len(dup)} prompts shared between draws")
    return p
