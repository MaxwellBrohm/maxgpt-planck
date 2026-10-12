"""RC-12 v4_voice with report frames (STEP 10, 2026-09-26; prereg draft s6 v4). Kept apart from grade_text.py
(at 250 lines when STEP 10 began). graders.g_val calls voice_error on every VAL probe; world-held probes (K) keep
grade_text.captures unchanged, and so do G-ROLEX r4 and the role-leak scan (grade_role.py); G-ROLEX r6 uses
given_away.

Per mention of the gold, the nearest person marker before it in its sentence is the last marker that ENDS within
the 8 words before it (a frame may start earlier: "You told me that the one in question was green"). Frames are
matched before the single markers (grade_text.FIRST_BIND_RX, you / your):
  you_rep  the user's words reported back: "you ('d / had / have) (already) told me", "you were telling me", "you
           asked (me) about / if / for", "you said / mentioned / shared / described (it) (to / with me)",
           "according to you"
  i_rep    the speaker's own earlier words: "I ('d / had) (already) said / mentioned / told / asked / did", "I told /
           gave you", "I told you to call me"
  neutral  any other "you ... me" (1 to 3 words between: "you asked me to", "you gave me", "you wanted me to",
           "you filled me in") and "let me": the "me" in them binds nothing to the speaker
A word inside a frame is never a pronoun ("You know I said green" holds an i_rep). With no marker before the
mention, the first frame after it in the sentence counts ("It's green, like I told you"). A mention inside a quote
that a you_rep opens (You said, "My bike is green.") counts as a you_rep, no reversal applies to it, and
graders.g_val runs grade_text.my_object on the reply without such quotes (unquote).
Modes (mode_of: the probe's holder):
  user   a fact the user gave about themselves, a named other or a table. Fails on a first-person binding
         ("My bike is green", "I'm Marcus") or an i_rep (the model speaks as the user: "I said green", "I'd told
         you it was green"). No marker at all: as captures (a first-person binding in the sentence, no "you").
  given  a fact the user gave the assistant ("What did I call you?"; OOD-H: a name, persona or reply language).
         Fails on an i_rep ("I said Cosmo", "I told you to call me Cosmo") or when the value is given to the user
         right before it (GIVEN_REV: "Your name's Cosmo", "I'll call you Cosmo", "You go by Cosmo", "I said you'd
         be Cosmo"). "My name is Cosmo", "You told me Cosmo", "You chose the name Cosmo for me" pass.
  own    the assistant's own earlier answer (OWN picks: probes with a gold_fn). Fails on a you_rep ("You said
         Quizzards", "You'd said", "According to you") or when the pick is given to someone else right before it
         (OWN_REV: "You'd gone with", "Your pick was", "Your team picked") or right after it (OWN_AFTER:
         "Quizzards, which you chose", "Quizzards was your pick"). "I said Quizzards", "You asked me to pick one,
         and Quizzards was my choice", "Out of the three you gave me, I picked Quizzards" pass.
  world  knowledge facts (holder "world"): grade_text.captures, unchanged."""
import re

import grade_text as T

GAP = r"(?:(?!(?:i|me|we|he|she|they)\b)\w+\s+)"
YOU_REP = (r"according\s+to\s+you|you(?:'ve|'d|\s+have|\s+had)?\s+" + GAP + r"{0,2}?(?:(?:told|telling|reminded)\s+me"
           r"|asked\s+(?:me\s+)?(?:about|if|whether|what|which|for)|(?:said|saying|mentioned|wrote|shared|described)"
           r"(?:\s+(?:\w+\s+)?(?:to|with)\s+me)?)")
I_REP = (r"i(?:'ve|'d|\s+have|\s+had)?\s+" + GAP + r"{0,2}?(?:(?:told|asked|reminded|gave|showed)\s+you"
         r"(?:\s+to\s+(?:\w+\s+)?me)?|said|mentioned|told|asked|did)")
NEUTRAL = r"you(?:'d|'ve|'re|'ll)?\s+" + GAP + r"{1,3}?me|let\s+me"
MARK = re.compile(r"\b(?:(?P<you_rep>" + YOU_REP + r")|(?P<i_rep>" + I_REP + r")|(?P<neutral>" + NEUTRAL + r")|"
                  + T.FIRST_BIND_RX + r"|you|your|you're|yours|you've)\b", re.I)
QUOTE_OPEN = re.compile(r"\b(?:" + YOU_REP + r")\W*$", re.I)
QUOTE_SPAN = re.compile(r"(\b(?:" + YOU_REP + r")\W*)\"[^\"]*\"", re.I)
BAD = {"user": ("first", "i_rep"), "given": ("i_rep",), "own": ("you_rep",)}
TAIL = r"[\s:\"']*$"
OWN_REV = T.rx(r"\b(?:you(?:'ve|'d|\s+have|\s+had)?|your\s+\w+)\s+" + GAP + r"?(?:picked|chose|chosen|(?:went|gone)"
               r"\s+(?:with|for)|opted\s+for|selected|suggested|named|decided\s+on|settled\s+on|wanted|voted\s+for|liked)"
               r"(?:\s+(?:the|a|an|it|name|one))*" + TAIL + r"|\byour\s+(?:\w+\s+)?(?:pick|choice)\s+(?:was|is)" + TAIL)
OWN_AFTER = T.rx(r"^\W*(?:(?:which|that)\s+you\s+" + GAP + r"?(?:picked|chose|went\s+with|wanted|liked|suggested)"
                 r"|(?:was|is)\s+your\s+(?:\w+\s+)?(?:pick|choice))\b")
GIVEN_REV = T.rx(r"\b(?:(?:call|called|calling|name|named)\s+you|your\s+(?:name|nickname)(?:'s|\s+is)|you\s+go\s+by"
                 r"|you(?:'d|'ll|\s+would|\s+will)\s+be|you(?:'re|\s+are)(?:\s+(?:called|named))?)" + TAIL)


def mode_of(probe):
    h = probe.get("holder")
    if h == "assistant":
        return "own" if probe.get("gold_fn") else "given"
    return "world" if h in ("world", None) else "user"


def kind(m):
    return m.lastgroup or ("first" if T.FIRST_BIND.fullmatch(m.group(0)) else "second")


def nearest(pre, n):
    """kind of the last person marker in pre that ends within its last n words: you_rep, i_rep, neutral, first,
    second, or None (no marker)."""
    words = [w.start() for w in re.finditer(r"\S+", pre)]
    lo = words[-n] if len(words) >= n else 0
    marks = [m for m in MARK.finditer(pre) if m.end() > lo]
    return kind(marks[-1]) if marks else None


def frame_after(post, n):
    """the frame kind of the first person marker in the first n words after the mention (None if not a frame)."""
    m = MARK.search(" ".join(post.split()[:n]))
    return m.lastgroup if m else None


def quoted(pre):
    """the mention sits inside a quote that a you_rep opens: You said, "My bike is green." """
    return pre.count('"') % 2 == 1 and QUOTE_OPEN.search(pre[:pre.rfind('"')]) is not None


def unquote(text):
    """text with every quote that a you_rep opens emptied (graders.g_val: grade_text.my_object reads it)."""
    return QUOTE_SPAN.sub(r'\1""', text)


def reversed_to(pre, post, mode):
    """the value given away right before (or, own, right after) the mention: GIVEN_REV, OWN_REV, OWN_AFTER."""
    if "assistrev" in T.OFF:
        return False
    if mode == "own":
        return bool(OWN_REV.search(pre) or OWN_AFTER.search(post))
    return mode == "given" and GIVEN_REV.search(pre) is not None


def given_away(text, v):
    """G-ROLEX r6_own (grade_role.py): the name the user gave the assistant given back to the user (GIVEN_REV)."""
    return any(reversed_to(text[T.sentence_span(text, m.start())[0]:m.start()], "", "given") for m in T.value_hits(text, v))


def voice_error(v, text, mode):
    """True when a mention of v stands in the wrong voice for the mode (module docstring). text is norm()-ed."""
    if mode == "world":
        return T.captures(v, text)
    if mode == "user" and "capture" in T.OFF:
        return False
    n = T.P["capture_window"]
    for m in T.value_hits(text, v):
        s0, s1 = T.sentence_span(text, m.start())
        pre, post = text[s0:m.start()], text[m.end():s1]
        q = quoted(pre)
        k = "you_rep" if q else nearest(pre, n) or frame_after(post, n)
        if k in BAD[mode] or (not q and reversed_to(pre, post, mode)):
            return True
        if k is None and mode == "user" and T.FIRST_BIND.search(text[s0:s1]) and not T.SECOND_P.search(text[s0:s1]):
            return True
    return False
