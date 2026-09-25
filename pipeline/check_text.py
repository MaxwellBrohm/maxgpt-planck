"""Text-level checks (SPEC section 6): hygiene, length, exact lines, spans, required words, degeneracy, voice,
language, safety, persona and prompt echo, and the held-out gate on the rendered text. Thresholds are module
constants so the checker mutants in mutation_checker.py can move them."""
import re
from functools import lru_cache

import gate
import heldout
import lexicons as L
import parse
from check_base import words, ngrams, content

REPEAT_MIN = 3        # a 4-gram this many times in one turn -> REPEAT_4GRAM
CONSEC_MAX = 0.25     # immediate-repeat coverage at or above this -> CONSEC_REP (turns of 8+ words)
SELF_COPY_MAX = 0.5   # share of the previous assistant turn's 4-grams -> SELF_COPY
ECHO_MAX = 0.6        # share of the previous user turn's 4-grams -> ECHO_USER
PERSONA_N = 8
PROMPT_N = 6
OOL_MAX = 0.02
SOCIAL_MAX_W = 20


def _lookup_turn(t):
    return t["role"] == "tool" or t.get("lookup_call")


def chk_empty(ctx):
    return [("EMPTY_TURN", t["i"], "") for t, s in ctx.turns() if not s.strip()]


def chk_format_text(ctx):
    out = []
    for t, s in ctx.turns():
        if L.THOUGHT_TAG_RE.search(s):
            out.append(("THOUGHT_TAG", t["i"], L.THOUGHT_TAG_RE.search(s).group(0)))
        if _lookup_turn(t):
            continue
        if L.ROLE_LABEL_RE.search(s):
            out.append(("ROLE_LABEL", t["i"], L.ROLE_LABEL_RE.search(s).group(0).strip()))
        if L.MARKDOWN_RE.search(s):
            out.append(("MARKDOWN", t["i"], L.MARKDOWN_RE.search(s).group(0)))
        if L.EMOJI_RE.search(s):
            out.append(("EMOJI", t["i"], L.EMOJI_RE.search(s).group(0).strip()))
        if L.STAGE_RE.search(s):
            out.append(("STAGE_DIR", t["i"], L.STAGE_RE.search(s).group(0)))
    for t, s in ctx.turns():
        if gate.DASH_RE.search(s):
            out.append(("DASH", t["i"], gate.DASH_RE.search(s).group(0)))
        if heldout.DIGIT_RE.search(s):
            out.append(("DIGIT", t["i"], heldout.DIGIT_RE.search(s).group(0)))
    return out


def social_answers(ctx):
    return {e["turns"]["answer"] for e in ctx.skel["events"] if e["kind"] == "S8"}


def chk_len(ctx):
    out, social = [], social_answers(ctx)
    for t, s in ctx.turns(mode="guided"):
        n = len(s.split())
        hi = min(t["max_w"], SOCIAL_MAX_W) if t["i"] in social else t["max_w"]
        if not (t["min_w"] <= n <= hi):
            out.append(("LEN_USER" if t["role"] == "user" else "LEN_ASSIST", t["i"], f"{n} not in {t['min_w']}-{hi}"))
    return out


def chk_exact(ctx):
    return [("EXACT_MISMATCH", t["i"], s[:40]) for t, s in ctx.turns(mode="exact")
            if parse.normalize(s) != parse.normalize(t["text"])]


def chk_spans(ctx):
    out = []
    for t, s in ctx.turns(mode="guided"):
        out += [("REQ_SPAN", t["i"], x) for x in t["must_include"] if not ctx.has(t["i"], x)]
        out += [("FORBID_SPAN", t["i"], x) for x in t["must_exclude"] if ctx.has(t["i"], x)]
    return out


def word_forms_re(w):
    stems = {w, w + "s", w + "es", w + "d", w + "ed", w + "ing", w + "er", w + "est"}
    if w.endswith("e"):
        stems |= {w[:-1] + "ing", w[:-1] + "ed", w[:-1] + "er", w[:-1] + "est"}
    if w.endswith("y"):
        stems |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ier", w[:-1] + "iest", w[:-1] + "ily"}
    if re.search(r"[^aeiou][aeiou][bdgmnprt]$", w):
        stems |= {w + w[-1] + "ing", w + w[-1] + "ed", w + w[-1] + "er", w + w[-1] + "est"}
    return re.compile(r"(?<![a-z])(?:" + "|".join(sorted(map(re.escape, stems), key=len, reverse=True)) + r")(?![a-z])",
                      re.I)


def chk_req_word(ctx):
    rw = ctx.skel["required_words"]
    alltext = " ".join(s for _, s in ctx.turns())
    return [("REQ_WORD", None, rw[k]) for k in ("noun", "verb", "adj") if not word_forms_re(rw[k]).search(alltext)]


def chk_vocab(ctx):
    """VOCAB_OOL (report only while the word list is FAKE): out-of-list rate over non-slot words."""
    if not ctx.wordlist:
        return []
    slot_words = {w for s in ctx.skel["slots"].values() for w in words(s["value"])}
    ws = [w for _, s in ctx.turns(mode="guided") for w in words(s) if w not in slot_words]
    ool = [w for w in ws if w not in ctx.wordlist]
    rate = len(ool) / max(1, len(ws))
    return [("VOCAB_OOL", None, f"{rate:.3f} {sorted(set(ool))[:5]}")] if rate > OOL_MAX else []


def consec_coverage(ws):
    marked = set()
    for n in range(1, 5):
        for j in range(len(ws) - 2 * n + 1):
            if ws[j:j + n] == ws[j + n:j + 2 * n]:
                marked |= set(range(j + n, j + 2 * n))
    return len(marked) / max(1, len(ws))


def chk_repeat(ctx):
    out = []
    for t, s in ctx.turns(mode="guided"):
        ws = words(s)
        g = ngrams(ws, 4)
        top = max((g.count(x) for x in set(g)), default=0)
        if top >= REPEAT_MIN:
            out.append(("REPEAT_4GRAM", t["i"], f"x{top}"))
        if len(ws) >= 8 and consec_coverage(ws) >= CONSEC_MAX:
            out.append(("CONSEC_REP", t["i"], f"{consec_coverage(ws):.2f}"))
    return out


def _share(src, dst):
    g = set(ngrams(words(src), 4))
    if len(g) < 2:
        return 0.0
    return len(g & set(ngrams(words(dst), 4))) / len(g)


def chk_copy(ctx):
    out, prev_a, prev_u = [], None, None
    for t in ctx.skel["turns"]:
        s = ctx.text.get(t["i"])
        if t["role"] == "user":
            prev_u = s
        if t["role"] != "assistant" or s is None or t.get("lookup_call"):
            continue
        if t["mode"] == "guided":
            if prev_a and _share(prev_a, s) >= SELF_COPY_MAX:
                out.append(("SELF_COPY", t["i"], f"{_share(prev_a, s):.2f}"))
            if prev_u and _share(prev_u, s) >= ECHO_MAX:
                out.append(("ECHO_USER", t["i"], f"{_share(prev_u, s):.2f}"))
        prev_a = s
    return out


def chk_language(ctx):
    out = []
    for t, s in ctx.turns():
        if any(c.isalpha() and not ("A" <= c <= "Z" or "a" <= c <= "z") for c in s):
            out.append(("NON_ENGLISH", t["i"], "non-latin or accented letter"))
        elif len(words(s)) >= 8 and not set(words(s)) & (L.FUNCTION_WORDS | L.STOPWORDS):
            out.append(("NON_ENGLISH", t["i"], "no English function word"))
        if L.SAFETY_RE.search(s):
            out.append(("SAFETY", t["i"], L.SAFETY_RE.search(s).group(0)))
        if not _lookup_turn(t) and L.AI_ISM_RE.search(s):
            out.append(("AI_ISM", t["i"], L.AI_ISM_RE.search(s).group(0)))
    return out


def chk_voice(ctx):
    out = []
    for t, s in ctx.turns(mode="guided"):
        rx, code = (L.USER_VOICE_RE, "USER_VOICE") if t["role"] == "user" else (L.ASSIST_VOICE_RE, "ASSIST_VOICE")
        if rx.search(s):
            out.append((code, t["i"], rx.search(s).group(0)))
    return out


def chk_persona(ctx):
    if not ctx.built:
        return []
    g = set(ngrams(words(ctx.built["persona"]), PERSONA_N))
    return [("PERSONA_LEAK", t["i"], "") for t, s in ctx.turns() if g & set(ngrams(words(s), PERSONA_N))]


def prompt_grams(ctx):
    """PROMPT_N-grams of the Claude-written prompt text (instruction, card, script guidance), with slot values,
    required spans, topic texts, the system line, required words and bank lines masked out."""
    b = ctx.built["blocks"]
    guide = [ln.split("]: ", 1)[-1] for ln in b["script"].split("\n")[1:] if "[copy exactly]" not in ln]
    text = "\n".join([b["head"], b["card"], b["tail"]] + guide)
    sk = ctx.skel
    masks = [s["value"] for s in sk["slots"].values()] + list(sk["topic_text"].values())
    masks += [x for t in sk["turns"] for x in t["must_include"] + t["must_exclude"]]
    masks += [sk["assistant"].get("system_text") or "", ctx.built["persona"]]
    masks += [sk["required_words"][k] for k in ("noun", "verb", "adj")]
    for m in sorted({m for m in masks if m}, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z])" + re.escape(m) + r"(?![A-Za-z])", " \n ", text, flags=re.I)
    grams = set()
    for piece in re.split(r"[\n.:;\"]", text):
        grams |= set(ngrams(words(piece), PROMPT_N))
    return grams


def chk_prompt_echo(ctx):
    if not ctx.built:
        return []
    g = prompt_grams(ctx)
    out = []
    for t, s in ctx.turns(mode="guided"):
        hit = g & set(ngrams(words(s), PROMPT_N))
        if hit:
            out.append(("PROMPT_ECHO", t["i"], " ".join(sorted(hit)[0])))
    return out


@lru_cache(maxsize=200000)
def _vocab_hits(s):
    return [h for h in heldout.vocab_hits(s) if h != "digit"]


@lru_cache(maxsize=200000)
def _echo_hits(s, terms, vals):
    return heldout.echo_hits(s, list(terms), list(vals))


def chk_heldout(ctx):
    out = []
    terms, vals = tuple(gate._obj_terms(ctx.skel)), tuple(gate._values(ctx.skel))
    for t, s in ctx.turns():
        v = _vocab_hits(s)
        if v:
            out.append(("HELDOUT_VOCAB", t["i"], v[0]))
        e = _echo_hits(s, terms, vals)
        if e:
            out.append(("HELDOUT_ECHO", t["i"], e[0]))
    out += [(c, None, d) for c, d in gate.struct_violations(ctx.skel)]
    return out


CHECKS = [chk_empty, chk_format_text, chk_len, chk_exact, chk_spans, chk_req_word, chk_vocab, chk_repeat, chk_copy,
          chk_language, chk_voice, chk_persona, chk_prompt_echo, chk_heldout]
