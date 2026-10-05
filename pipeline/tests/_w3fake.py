"""Test-only fake teacher for the W3 hold loop (no model): answers each call in its contract, from the planned call
it is handed, with FAKE text Claude wrote for tests. Its rows carry the decode record a real bankserve row has, so
client.Client.verify passes. bad_kinds: kinds it answers with junk (parse problems), to test the BAD_KIND brake."""
import re

import banks as B
from bankpass import client as CL, judge3 as J3, prompts4 as P4, w3render as WR
from teachers import decode as D

WORDS = ("amber", "brisk", "cedar", "dune", "ember", "fable", "grove", "harbor", "inlet", "juniper", "kettle",
         "lantern", "meadow", "nutmeg", "orchard", "pebble", "quill", "ripple", "saddle", "thimble")


def _vals(f):
    out = []
    for k, v in (f or {}).items():
        if k == "items":
            out.append(B.join_items(v))
        elif isinstance(v, str) and k not in ("article", "p"):
            out.append(v)
    return out


def line_text(c, i):
    f = (c.get("fills") or [None] * c["n"])[i] or {}
    role = c["bank"].split(".")[2] if c["class"] == "K" else None
    word = WORDS[(i + c["seed"]) % len(WORDS)]
    if role in ("query", "bait"):
        return f"Can you remind me about my {f.get('o') or 'thing'} and the {word} again?"
    if role == "corr":
        return f"my {f['o']} is {f['v']} near the {word}." if f.get("o") else f"it is {f['v']} near the {word}."
    if role:
        return f"My {f['o']} is {f['v']} by the {word}." if f.get("o") else f"I have {f['v']} by the {word}."
    vals = _vals(f)
    if c["bank"].startswith("marker."):
        return f"Oh {word},"
    return (f"Hello, about {' and '.join(vals)} near the {word} today." if vals
            else f"Hello there, a {word} kind of day.")


def answer(c):
    k, n = c["kind"], c["n"]
    if k == "line":
        rows = [line_text(c, i) + (" | rare" if c.get("verbalized") else "") for i in range(n)]
    elif k == "pool":
        rows = [f"{(c.get('seed_words') or WORDS)[(i + c['seed']) % 3]} {WORDS[i % len(WORDS)]}" for i in range(n)]
    elif k == "topic":
        rows = [f"looking after a {WORDS[(c['seed'] >> (2 * i)) % 20]} {WORDS[(c['seed'] >> (2 * i + 9)) % 20]} "
                f"{c['seed_words'][i % 3]} | {'common' if i % 3 == 0 else 'rare'}" for i in range(n)]
    elif k == "label":
        rows = [f"{c.get('o') or 'my'} {WORDS[i]}" for i in range(n)]
    elif k in ("notewas", "listname"):
        rows = [f"{WORDS[i]} note" for i in range(n)]
    elif k == "vote":
        rows = ["1"]
    elif k == "relfeat":
        rows = [f"{v}: either, one" for v in c["rows"]]
    elif k == "attr":
        rows = ["opening day: day of the week", "home town: city", "paint shade: colour"][:n]
    elif k == "pred":
        rows = [f"is set for {v}" for _, _, v in c["rows"]]
    elif k == "pos":
        rows = [f"{w}: noun" if i % 3 else f"{w}: verb" for i, w in enumerate(c["rows"])]
    elif k == "verbs":
        rows = [f"{w}: {w}ed, {w}ed" for w in c["rows"]]
    elif k == "ly":
        rows = [f"{a}, {b}: same" for a, b in c["rows"]]
    elif k == "wordset":
        rows = ["nouns: " + ", ".join(WORDS[:12]), "verbs: walk, fold, carry", "adjectives: brisk, warm, quiet"]
    elif k == "intent":
        rows = [f"ask whether the {WORDS[i]} part takes long" for i in range(n)]
    elif k == "topen":
        rows = [f"Hi, I would like to talk about my {WORDS[i]} plans today" for i in range(n)]
    elif k == "para":
        h, t = P4.base_block()
        rows = ["Please " + h.lower(), "Also, " + t.lower()]
    elif k == "check":
        rows = [f"{i}: yes" for i in range(1, len(P4.CHECKLIST) + 1)] + ["added: no"]
    elif k == "group":
        rows = ["home and chores"]
    elif k == "judge":
        rows = []
        for q, _, e, kind in c["qs"]:
            a = ", ".join(e) if isinstance(e, list) else e
            rows.append(f"{q}: {a}")
    else:
        raise KeyError(k)
    return "\n".join(rows + ["END"])


class Fake:
    """drop-in for client.Client in hold.run_calls."""

    def __init__(self, teacher, bad_kinds=(), gone_after=None):
        self.teacher, self.bad_kinds, self.gone_after, self.n = teacher, set(bad_kinds), gone_after, 0

    def generate(self, prompt, spec, seed, preset=None, call=None):
        self.n += 1
        if self.gone_after is not None and self.n > self.gone_after:
            raise CL.ServerGone("fake server stopped")
        text = "junk" if call["kind"] in self.bad_kinds else answer(call)
        dec = {"regex_sha256": D.sha(spec["regex"]), "ban": {"rule": "dash-v1"}, "phrases": {"rule": D.PHRASE_RULE},
               "preset": preset or "custom", "sampling": {"temperature": 0.0} if not preset else {}}
        return {"text": text, "finish": "stop", "n_out": len(text.split()), "n_prompt": len(prompt.split()),
                "decode": dec}


def matches(spec, text):
    """the fake's text against the call's regex (Python re; the PC preflight checks xgrammar)."""
    return re.fullmatch(spec["regex"], text) is not None
