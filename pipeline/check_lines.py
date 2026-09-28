"""Line checks added or moved on 2026-09-28 (round 2 of the teacher pilot fixes, SPEC 14; v1 quality review and dry
pilot 2): required words (moved here from check_text, forms now depend on the part of speech), END inside a turn,
the lowercase user style, copied guidance clauses (PROMPT_ECHO) and the assistant-only service phrases (AI_ISM).
Same contract as the other checker modules: f(ctx) -> [(code, turn index or None, detail)]."""
import re

import lexicons as L
import render_prompt as R

# -ly adverbs whose meaning left the adjective (v1 review: fair -> fairly, short -> shortly, clear -> "Clearly, ..."):
# not a form of the required adjective. smooth -> smoothly and gentle -> gently stay forms (the 09-27 audit).
LY_SHIFTED = {"hardly", "lately", "nearly", "shortly", "fairly", "barely", "highly", "largely", "mostly", "really",
              "clearly", "surely", "simply", "presently", "scarcely", "justly", "directly", "evenly", "readily"}
# nouns whose plural really is -ves (shelf, half, scarf, leaf, loaf, knife, wife): not chef, roof, belief or cafe
VES_RE = re.compile(r"(?:[lr]f|[aeo]af|ife)$")


def word_forms_re(w, part=None):
    """the forms of a required word that count as using it: inflections for every part of speech, the -ly adverb
    only for an adjective (never a meaning-shifted one), the -ves plural only for a noun that takes it."""
    stems = {w, w + "s", w + "es", w + "d", w + "ed", w + "ing", w + "er", w + "est"}
    if w.endswith("e"):
        stems |= {w[:-1] + "ing", w[:-1] + "ed", w[:-1] + "er", w[:-1] + "est"}
    if w.endswith("y"):
        stems |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ier", w[:-1] + "iest"}
    if part == "adj":
        ly = w[:-1] + "ily" if w.endswith("y") else (w[:-1] + "y" if w.endswith("le") else w + "ly")
        if ly not in LY_SHIFTED:
            stems.add(ly)
    if part == "noun" and VES_RE.search(w):
        stems.add(w[:-1 if w.endswith("f") else -2] + "ves")
    if re.search(r"[^aeiou][aeiou][bdgmnprt]$", w):
        stems |= {w + w[-1] + "ing", w + w[-1] + "ed", w + w[-1] + "er", w + w[-1] + "est"}
    return re.compile(r"(?<![a-z])(?:" + "|".join(sorted(map(re.escape, stems), key=len, reverse=True)) + r")(?![a-z])",
                      re.I)


def placed_words(skel):
    """[(part, word)] of the required words the skeleton placed (turn_hint holds one turn per placed word)."""
    rw = skel["required_words"]
    return [(p, rw[p]) for p in R.REQ_WORD_ORDER[:len(rw["turn_hint"])]]


def chk_req_word(ctx):
    alltext = " ".join(s for _, s in ctx.turns())
    return [("REQ_WORD", None, w) for p, w in placed_words(ctx.skel) if not word_forms_re(w, p).search(alltext)]


def chk_end_in_turn(ctx):
    """END_IN_TURN: the END marker inside a turn, a turn that ends on "End." or "The end.", or script talk."""
    out = []
    for t, s in ctx.turns():
        m = L.END_TOKEN_RE.search(s) or L.END_TAIL_RE.search(s) or L.SCRIPT_TALK_RE.search(s)
        if m:
            out.append(("END_IN_TURN", t["i"], m.group(0).strip()))
    return out


def chk_user_style(ctx):
    """USER_STYLE: a lowercase-style user's own (guided) line with a capital letter. Their copied lines are stored
    lowercased (parse.exact_text), so one capitalized teacher line breaks the style inside the chat (v1 review:
    capitalized lines in 49 of 52 accepted lowercase Qwen card chats, after a sentence-case forced literal)."""
    if not ctx.lower_user:
        return []
    out = []
    for t, s in ctx.turns(role="user", mode="guided"):
        m = re.search(r"[A-Za-z']*[A-Z][A-Za-z']*", s)
        if m:
            out.append(("USER_STYLE", t["i"], m.group(0)))
    return out


def chk_guide_meta(ctx):
    """PROMPT_ECHO: a guided line that carries an instruction clause of the guidance (lexicons.GUIDE_META_RE), or a
    guided query line (the recall question of an event) with a withholding clause (lexicons.WITHHOLD_RE)."""
    queries = {e["turns"].get("query") for e in ctx.skel["events"]}
    out = []
    for t, s in ctx.turns(mode="guided"):
        m = L.GUIDE_META_RE.search(s) or (t["i"] in queries and t["role"] == "user" and L.WITHHOLD_RE.search(s))
        if m:
            out.append(("PROMPT_ECHO", t["i"], "guidance clause: " + m.group(0)))
    return out


def chk_assist_ism(ctx):
    """AI_ISM on assistant turns only: the service-greeting family (lexicons.ASSIST_ISM_RE)."""
    return [("AI_ISM", t["i"], L.ASSIST_ISM_RE.search(s).group(0)) for t, s in ctx.turns(role="assistant", mode="guided")
            if not t.get("lookup_call") and L.ASSIST_ISM_RE.search(s)]


CHECKS = [chk_req_word, chk_end_in_turn, chk_user_style, chk_guide_meta, chk_assist_ism]
