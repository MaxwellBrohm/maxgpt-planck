"""Round 3 checks (2026-10-03, SPEC 15; the dry pilot 3 v1 quality review, notes.txt): told frames (PERSPECTIVE),
copied guidance frames (PROMPT_ECHO), false memory (FALSE_MEMORY), live weather (LIVE_DATA), the assistant's own case
(ASSIST_CASE) and a request glued to the line before it (RUN_ON). Same contract as the other checker modules:
f(ctx) -> [(code, turn index or None, detail)]. Patterns are in lexicons.py; thresholds are module constants so the
checker mutants can move them."""
import re

import lexicons as L
from check_base import words
from check_text import script_guidance

# a guided user line copies its guidance's frame when every required frame word comes back in order (a frame word may
# be followed by FRAME_GAP other words, a masked slot by FRAME_HOLE): "What usually helps with planning a picnic?"
# for "ask what usually helps with {t}" (dp3: 616 of 672 form 0 lines), "which step should I try next" for "ask which
# step to try next with {t}" (363 of 681), "look up the X and report its Y" (64 of 210)
FRAME_GAP = 2
FRAME_HOLE = 8
FRAME_MIN_REQ = 3          # required frame words
FRAME_MIN_CONTENT = 2      # of them outside lexicons.STOPWORDS ("how my garden is going" has none)
FRAME_OPTIONAL = {"to", "a", "an", "the", "with", "about", "of", "for", "on", "in", "its", "my", "your", "and", "at",
                  "it", "is"}
FRAME_LEAD = {"ask", "say", "tell", "mention", "share", "the", "assistant", "to", "have", "react", "bring", "give",
              "pass"}
# list guidance is the natural question itself ("what comes first on my shopping list?"), never a template
NATURAL_Q = ("ask list ", "list ", "state the list")
HOLE = "zzhole"


def frame_masks(skel):
    """the slot-like spans of a skeleton, longest first: values, topic texts, required spans, required words, event
    entities, attributes, list names and reference expressions, slot nouns and person relations."""
    m = [s["value"] for s in skel["slots"].values()] + list(skel["topic_text"].values())
    m += [x for t in skel["turns"] for x in t["must_include"] + t["must_exclude"]]
    m += [skel["required_words"][k] for k in ("noun", "verb", "adj")]
    for e in skel["events"]:
        m += [str(e["params"][k]) for k in ("entity", "attribute", "list_name", "ref_expr", "noun") if e["params"].get(k)]
    m += [s["noun"] for s in skel["slots"].values() if s.get("noun")] + [p["relation"] for p in skel["persons"].values()]
    return sorted({x for x in m if x}, key=len, reverse=True)


def guide_frame(skel, g):
    """the guidance's words with slot spans as holes and the leading instruction words dropped (user guidance only:
    rule notes are on assistant lines)."""
    for x in frame_masks(skel):
        g = re.sub(r"(?<![A-Za-z])" + re.escape(x) + r"(?![A-Za-z])", f" {HOLE} ", g, flags=re.I)
    toks = [HOLE if w.startswith(HOLE) else w for w in words(g)]
    while toks and toks[0] in FRAME_LEAD:
        toks.pop(0)
    return toks


def frame_re(toks):
    """a regex for the frame's required words in order, or None for a frame too small to be a template."""
    req = [w for w in toks if w != HOLE and w not in FRAME_OPTIONAL]
    if len(req) < FRAME_MIN_REQ or len([w for w in req if w not in L.STOPWORDS]) < FRAME_MIN_CONTENT:
        return None
    parts, hole = [], False
    for w in toks:
        if w == HOLE:
            hole = True
            continue
        if w in FRAME_OPTIONAL:
            continue
        if parts:
            parts.append(r"\W+(?:[\w']+\W+){0,%d}" % (FRAME_HOLE if hole else FRAME_GAP))
        parts.append(re.escape(w))
        hole = False
    return re.compile(r"(?<![\w'])" + "".join(parts) + r"(?![\w'])", re.I)


def chk_frame_copy(ctx):
    """PROMPT_ECHO: a guided user line that carries its own guidance's frame (list guidance excepted)."""
    if not ctx.built:
        return []
    out = []
    for t, s in ctx.turns(role="user", mode="guided"):
        if (t["intent"] or "").startswith(NATURAL_Q):
            continue
        rx = frame_re(guide_frame(ctx.skel, script_guidance(ctx.built, t)))
        m = rx.search(s) if rx else None
        if m:
            out.append(("PROMPT_ECHO", t["i"], "guidance frame: " + m.group(0)))
    return out


def _names(ctx):
    return sorted(v for v in ctx.values | {ctx.card} if v)


def _said_before(ctx, vals, upto):
    """some value of vals occurs in a turn before upto."""
    return any(ctx.has(j, v) for j in ctx.text if j < upto for v in vals)


def _first_source(ctx, v, i):
    """the role of the first turn before i that names v, or None."""
    return next((ctx.by_i[j]["role"] for j in sorted(ctx.text) if j < i and ctx.has(j, v)), None)


def chk_told_frame(ctx):
    """PERSPECTIVE: "you were told", "as told", "as I said", "I told you" in an assistant turn, unless the assistant
    (or a lookup result) was the first to name a value this turn names; repeating the user's value in an
    acknowledgement does not make the assistant its source."""
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        m = L.TOLD_FRAME_RE.search(s)
        if m and not any(_first_source(ctx, v, t["i"]) in ("assistant", "tool") for v in _names(ctx)
                         if ctx.has(t["i"], v)):
            out.append(("PERSPECTIVE", t["i"], "told frame: " + m.group(0)))
    return out


def chk_false_memory(ctx):
    """FALSE_MEMORY: forgetting outside an abstain answer; the assistant's own error before any assistant_err fix
    (the user corrected themselves); "earlier" when every value the turn names was first said in the line before.
    The assistant_err recap is a planned misstatement and is not checked; a correction's acknowledgement is not
    checked for "earlier" (dp3 re-check: "June works better than the month you mentioned before" names the old value
    without saying it)."""
    sk = ctx.skel
    abstain = {e["turns"].get("answer") for e in sk["events"] if e["kind"] == "S7"
               and e["params"].get("variant") == "abstain"}
    errs = [e for e in sk["events"] if e["kind"] == "S3" and e["params"].get("variant") == "assistant_err"]
    fixes = [e["turns"]["fix"] for e in errs if "fix" in e["turns"]]
    recaps = {e["turns"].get("recap_reply") for e in errs}
    acks = {e["turns"][op["turn"]] + 1 for e in sk["events"] for op in e["params"].get("ops", [])
            if op.get("op") in ("set", "fix") and op["turn"] in e["turns"]}
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        i = t["i"]
        if t.get("lookup_call") or i in recaps:
            continue
        m = L.FORGOT_RE.search(s)
        if m and i not in abstain:
            out.append(("FALSE_MEMORY", i, "forgot: " + m.group(0)))
        m = L.SELF_ERR_RE.search(s)
        if m and not any(f < i for f in fixes):
            out.append(("FALSE_MEMORY", i, "own error: " + m.group(0)))
        m = L.EARLIER_RE.search(s) if i not in acks else None
        vals = [v for v in _names(ctx) if ctx.has(i, v)] if m else []
        if vals and not _said_before(ctx, vals, i - 1):
            out.append(("FALSE_MEMORY", i, "earlier: " + m.group(0)))
    return out


def chk_live_data(ctx):
    """LIVE_DATA: current weather in an assistant turn, with no hedge in the match or the three words before it."""
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        for m in L.LIVE_RE.finditer(s):
            if not L.LIVE_HEDGE_RE.search(" ".join(s[:m.start()].split()[-3:]) + " " + m.group(0)):
                out.append(("LIVE_DATA", t["i"], m.group(0)))
                break
    return out


LONE_I_RE = re.compile(r"(?<![A-Za-z'])i(?![A-Za-z])")


def chk_assist_case(ctx):
    """ASSIST_CASE: an assistant turn whose first letter is lowercase, or with a lone lowercase "i" (D1: the
    assistant writes sentence case whatever the user does; dp3: Qwen copied lowercase users into 32 accepted turns)."""
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        if t.get("lookup_call"):
            continue
        m = re.search(r"[A-Za-z]", s)
        if m and m.group(0).islower():
            out.append(("ASSIST_CASE", t["i"], "starts " + s[m.start():m.start() + 12]))
        elif LONE_I_RE.search(s):
            out.append(("ASSIST_CASE", t["i"], "lowercase i"))
    return out


def chk_run_on(ctx):
    """RUN_ON: "please share / tell / let / give / send" straight after a word with no stop and no joining word."""
    out = []
    for t, s in ctx.turns(role="assistant", mode="guided"):
        for m in L.RUN_ON_RE.finditer(s):
            prev = s[:m.start()].rstrip()
            last = words(prev)[-1:] if prev else []
            if prev and prev[-1].isalpha() and last and last[0] not in L.RUN_ON_JOIN:
                out.append(("RUN_ON", t["i"], prev.split()[-1] + " " + m.group(0)))
                break
    return out


CHECKS = [chk_told_frame, chk_frame_copy, chk_false_memory, chk_live_data, chk_assist_case, chk_run_on]
