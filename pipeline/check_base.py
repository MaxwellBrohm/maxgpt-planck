"""Shared context and helpers for the checker modules (check_text, check_events, check_behav).
A check is a function f(ctx) -> [(code, turn index or None, detail)]. ctx.text holds the parsed turn texts; a turn
the parse could not find is absent, and every check skips absent turns (FORMAT_LINES already fired)."""
import re
from functools import lru_cache

import golds
import lexicons as L
import render_prompt as R

WORD_RE = re.compile(r"[a-z0-9']+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")


class Ctx:
    def __init__(self, skel, turn_texts, built=None, wordlist=None):
        self.skel = skel
        self.by_i = {t["i"]: t for t in skel["turns"]}
        self.text = dict(turn_texts)
        self.built = built
        self.wordlist = wordlist
        self.lower_user = skel["user"]["style"] == "lowercase"
        self.card = R.card_name(skel)
        self.has_sys = bool(skel["assistant"].get("system_text"))
        self.uname = R.user_name(skel)
        self.events = {e["id"]: e for e in skel["events"]}
        self.first = first_scheduled(skel)

    def turns(self, role=None, mode=None):
        """(spec, text) for every parsed turn, filtered by role and mode."""
        for t in self.skel["turns"]:
            if t["i"] in self.text and (role is None or t["role"] == role) and (mode is None or t["mode"] == mode):
                yield t, self.text[t["i"]]

    def has(self, i, v):
        """value or required span v occurs in turn i (word-bounded, optional plural; capitalized values are
        case-sensitive except in the turns of a lowercase-style user)."""
        s = self.text.get(i)
        if s is None:
            return False
        rx = golds.value_re(v)
        if self.lower_user and self.by_i[i]["role"] == "user":
            rx = value_re_i(v)
        return bool(rx.search(s))

    def answer_turn(self, e):
        return e["turns"].get("answer")


@lru_cache(maxsize=None)
def value_re_i(v):
    return re.compile(golds.value_re(v).pattern, re.I)


def words(s):
    return WORD_RE.findall(s.lower())


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def stem(w):
    """crude suffix stripper for the OFFTOPIC overlap: chews/chewing -> chew, worried/worry -> worry."""
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf, rep, min_len in (("ies", "y", 5), ("ied", "y", 5), ("ing", "", 6), ("ed", "", 5), ("es", "", 5),
                              ("s", "", 4)):
        if len(w) >= min_len and w.endswith(suf) and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def content(s):
    return {stem(w) for w in words(s) if w not in L.STOPWORDS and len(w) > 2}


def sentences(s):
    return [x for x in SENT_SPLIT.split(s.strip()) if x.strip()]


def event_values(skel):
    """every value a conversation schedules: slot values, op values, list items, lookup stale values."""
    vals = {s["value"] for s in skel["slots"].values()}
    for e in skel["events"]:
        p = e["params"]
        for op in p.get("ops", []):
            vals |= set(op.get("items") or [])
            if op.get("value"):
                vals.add(op["value"])
        vals |= set(p.get("stale") or [])
    return vals


def first_scheduled(skel):
    """{value: first turn index whose exact text or must_include states it}; the system text counts as turn -1.
    Values never scheduled (a card name without a system text) are absent."""
    out = {}
    sys_text = skel["assistant"].get("system_text") or ""
    for v in event_values(skel):
        rx = golds.value_re(v)
        if rx.search(sys_text):
            out[v] = -1
            continue
        for t in skel["turns"]:
            if (t.get("text") and rx.search(t["text"])) or any(rx.search(x) for x in t["must_include"]):
                out[v] = t["i"]
                break
    return out


def same_type(skel, vtype):
    return set(golds.same_type_values(skel, vtype)) if vtype else set()


def slot_type(skel, value):
    for s in skel["slots"].values():
        if s["value"] == value:
            return s["type"]
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            if op.get("value") == value and op.get("slot"):
                return skel["slots"][op["slot"]]["type"]
    return None


def mentioned(ctx, i, values):
    return sorted(v for v in values if v and ctx.has(i, v))
