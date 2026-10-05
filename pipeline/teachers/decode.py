"""Decoding controls for the served teachers (D3, 2026-09-27): pure Python, no vLLM import, so the Mac tests, the
driver side and serve_http share one definition. DRY prep: nothing here is training data.

  label_regex(lines, sep)   structured output (vLLM StructuredOutputsParams(regex=...), xgrammar backend): the planned
                            label lines in order, one line each, then END (SPEC 5f). lines = [[label, literal or None]];
                            a literal (an exact or tool line, --structured labels_exact) is written as fixed text.
                            sep joins the lines: "\\n", or "\\n+" for Ministral, which puts a blank line between lines
                            (probe 2026-09-27, CPU: the "\\n" form accepts 0 of 69 of its own well-formed outputs, "\\n+"
                            61 of 69; Qwen and Gemma pass 61/62 and 58/58 either way; on the GPU Ministral under "\\n"
                            ran on with " END. END." in 5 of 30).
  dash_ban_ids(vocab)       the dash-token ban (applied as logit_bias -100): every token whose bytes hold U+2012 to
                            U+2015, U+2212, "--" or " - ", plus every token ending in " -" (it makes " - " before a
                            token that starts with a space). The probe's set: 140 / 185 / 149 ids for Ministral / Qwen
                            / Gemma, under the V2 runner's 1024-id logit_bias cap; it took Ministral from 26 to 28 of
                            30 outputs with a dash to 0 of 30 at 1.00 to 1.03x throughput. Byte-fallback fragments of a
                            dash are NOT banned (their first bytes are shared with curly quotes); the checker's DASH
                            still rejects whatever gets through.
  (2026-09-28, round 2) LABEL_RULE "labels-v2": every free line is noend(), so no line holds "END" and END is written
                            once, on its own line, at the end. Dry pilot 2: Ministral's last line ran on with
                            " END. END." until max_tokens (293 runs), and 10 accepted chats kept END inside a turn.
                            "labels-v1" was LINE_ANY on every line (dp2 FIX arms). NO length cap: every capped form
                            measured costs too much in xgrammar 0.2.8 (Qwen vocab, PC CPU): a flat "[^\\n]{1,300}"
                            last line 1.6 s of compile per request (LINE_ANY 0.14 s), the no-END unit repeated
                            {1,300} 0.61 s of mask per token, nested repeats did not finish compiling in 10 minutes.
  phrase_words(literals)    the AI-ism phrase ban (PHRASE_RULE): vLLM bad_words, one surface form per string (vLLM adds
                            the space-prefixed form when it has as many tokens); a phrase that occurs in a forced
                            literal of the chat is left out, so the grammar and the ban never forbid the same token.
Presets (D4) live in serve.TEACHERS next to the teacher they belong to."""
import hashlib
import re

STRUCTURED = ("off", "labels", "labels_exact")
SEPS = ("\\n", "\\n+")                       # regex text: one newline, or one or more
LABEL_OK = re.compile(r"^[UAT][1-9][0-9]*$")
LINE_ANY = "[^\\n]+"
META = set("\\.^$|?*+()[]{}")               # regex metacharacters; nothing else is escaped (xgrammar's parser)

LABEL_RULE = "labels-v2"
# A capital E is followed by neither E nor N-then-D/E, so "END" cannot occur; a line may end in "E" or "EN". Stricter
# than "no END" only for capital "EE" and "ENE" (all-caps words). The exact no-END automaton, with E-runs as a loop,
# cost xgrammar 57 to 59 ms of mask time per token on Qwen's vocab (PC CPU, 2026-09-28) against 0.002 ms for this
# form and for LINE_ANY, so it is not used.
_UNIT = "(?:[^E\\n]|E[^EN\\n]|EN[^DE\\n])"


def noend():
    """a non-empty line (no newline) that never contains "END": units of 1 to 3 chars, then an optional "E" or "EN"."""
    return f"(?:{_UNIT}+(?:EN?)?|EN?)"

# ---- AI-ism phrase ban (PHRASE_RULE): the checker's lexicons.AI_ISM_RE phrases in the forms the teachers write
# (dp2 records: "happy to help" 113 Gemma, "I am here to help" 52 Qwen, Ministral's curly "I’m here to help") plus
# the greeting family (v1 review: "Hello, how can I help you today?" 33 times in Gemma accepts). Capitalized and
# lowercase starts; "is there anything else" covers AI_ISM's "is there anything else i can". vLLM bans a phrase after
# a space only when " phrase" has as many tokens as "phrase"; Ministral's "happy to help" and "feel free to" split
# their bare first word, so their dp2 contexts ("I'd be happy to help" 61, "I'm happy to help", "and feel free to")
# are listed whole (teachers/test_decode_pc.py checks the dp2 sentences are covered on every teacher's tokenizer).
PHRASE_RULE = "aiism-v4"
# aiism-v3 (2026-10-04, round 4): the phrase ban is ON for every served teacher run (drive.py --serve; Max, 10-04:
# service filler stays rejected), and it adds the service family dry pilot 4 measured (assistant lines over all R3
# attempts, Qwen / Ministral / Gemma): "I can certainly ..." 4 / 0 / 237, "I can help" 6 / 23 / 153, "anything
# else" 10 / 54 / 166 (13 closer variants: one bare "anything else" covers them; user lines hold it 6 / 5 / 1 times
# in about 11,000 each, "Anything else I should think about?", which the ban also moves), "here to help" 44 / 37 /
# 10 (incl. "an assistant here to help"), "ready to help" 17 / 18 / 7, "glad I could help/assist" 13 / 3 / 1,
# "I'd love to help" 0 / 17 / 0, "how else can I" 1 / 2 / 1, "I can certainly do that" (Gemma 64). The aiism-v2
# whole-context forms stay for tokenizers that split a bare first word (Ministral's "happy", "feel"), and the
# "anything else" closers are listed in context too, since Ministral's tekken splits a bare "anything" (vLLM then keeps
# no space form of "anything else"). "here to help" (every space form kept) replaces the three "I'm / I am here to
# help" forms. Measured on the real tokenizers (PC CPU, test_decode_pc): see notes.txt round 4. Not coverable here:
# "glad" splits when bare on Qwen and Ministral, so "Glad I could assist" passes their ban (AI_ISM still rejects it).
# aiism-v4 (10-04, after the GPU hold that measured aiism-v3): adds "I can definitely", Gemma's main substitute under
# v3 (14 lines in 64 attempts, against 0 without the ban); 125 sequences on Gemma's tokenizer, under vLLM's 128.
PHRASES = ("as an AI", "As an AI", "language model", "Language model", "great question", "Great question",
           "feel free to", "Feel free to", "and feel free to", "please feel free to", "Please feel free to",
           "certainly!", "Certainly!", "I hope this helps", "as a virtual", "As a virtual", "I'm just an AI",
           "I’m just an AI", "I am just an AI", "artificial intelligence", "Artificial intelligence", "large language",
           "Large language",
           "here to help", "here to assist",
           "happy to help", "Happy to help", "be happy to help", "am happy to help", "I'm happy to help",
           "I’m happy to help", "happy to assist", "Happy to assist", "glad to help", "Glad to help", "glad I could help",
           "Glad I could help",
           "glad I could assist", "Glad I could assist", "ready to help", "Ready to help", "ready to assist",
           "love to help",
           "I can help", "I can assist", "I can certainly", "I can definitely",
           "how can I help", "How can I help", "how can I assist", "How can I assist", "how may I help",
           "How may I help", "how else can I", "How else can I", "anything else", "Anything else",
           "is there anything else", "Is there anything else", "you need anything else", "to know anything else",
           "to add anything else", "with anything else")
GREETING = ("how can i help", "anything else", "here to help")   # the family added beyond AI_ISM (v2: greetings)
BAD_WORDS_CAP = 128                          # vLLM 0.30 VLLM_MAX_NUM_BAD_WORDS: token sequences per request
BAD_TOKENS_CAP = 1024                        # VLLM_MAX_BAD_WORDS_TOTAL_TOKENS (V2 runner)


def _fold(s):
    return " ".join(s.replace("’", "'").replace("‘", "'").lower().split())


def covered(seqs, ids):
    """True when one banned token sequence occurs in ids: then vLLM's kernel would have masked its last token."""
    ids = list(ids)
    return any(ids[k:k + len(q)] == list(q) for q in seqs for k in range(len(ids) - len(q) + 1))


def phrase_words(literals=()):
    """the bad_words of one request: PHRASES minus every phrase whose folded form occurs in a forced literal."""
    lit = [_fold(x) for x in literals or () if x]
    return [p for p in PHRASES if not any(_fold(p) in x for x in lit)]

DASH_RULE = "dash-v1"
DASH_CHARS = "\u2012\u2013\u2014\u2015\u2212"
DASH_BYTES = tuple(c.encode("utf-8") for c in DASH_CHARS) + (b"--", b" - ")
LOGIT_BIAS_CAP = 1024                         # vLLM 0.30 V2 model runner: logit_bias ids per request (probe)
BAN_BIAS = -100.0


def esc(text):
    """a literal line as regex text: metacharacters backslashed, every other character as itself."""
    return "".join("\\" + c if c in META else c for c in text)


def label_regex(lines, sep="\\n", rule=LABEL_RULE):
    """[[label, literal or None], ...] -> the regex of the whole output: 'U1: <line>' + sep + ... + sep + 'END'.
    labels-v2: free lines are noend(); labels-v1: LINE_ANY (kept for comparison)."""
    if sep not in SEPS:
        raise ValueError(f"line separator {sep!r} is not one of {SEPS}")
    if rule not in ("labels-v1", "labels-v2"):
        raise ValueError(f"unknown label rule {rule!r}")
    if not lines:
        raise ValueError("no planned lines")
    parts, seen = [], set()
    for item in lines:
        lab, lit = item
        if not isinstance(lab, str) or not LABEL_OK.match(lab) or lab in seen:
            raise ValueError(f"bad or repeated label {lab!r}")
        seen.add(lab)
        if lit is None and rule == "labels-v1":
            parts.append(f"{lab}: {LINE_ANY}")
        elif lit is None:
            parts.append(f"{lab}: {noend()}")
        elif isinstance(lit, str) and lit.strip() and "\n" not in lit and "\r" not in lit:
            parts.append(f"{lab}: {esc(lit)}")
        else:
            raise ValueError(f"bad literal for {lab}: {lit!r}")
    return sep.join(parts + ["END"])


def constraint_ok(c):
    """a request's constraint: {"mode": "labels" | "labels_exact", "lines": [[label, literal or None], ...]}."""
    if not isinstance(c, dict) or c.get("mode") not in STRUCTURED[1:] or not isinstance(c.get("lines"), list):
        return False
    lines = c["lines"]
    if not all(isinstance(x, list) and len(x) == 2 for x in lines):
        return False
    if c["mode"] == "labels" and any(x[1] is not None for x in lines):
        return False
    return True


def dash_ban_ids(vocab):
    """vocab = [bytes] per token id -> (sorted banned ids, {set: count})."""
    contains, end_sh = [], []
    for i, b in enumerate(vocab):
        if not b:
            continue
        if any(d in b for d in DASH_BYTES):
            contains.append(i)
        elif b.endswith(b" -"):
            end_sh.append(i)
    return sorted(contains + end_sh), {"contains": len(contains), "end_space_hyphen": len(end_sh)}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha(ids):
    return sha(",".join(str(int(i)) for i in ids))


def logit_bias(ids):
    if len(ids) > LOGIT_BIAS_CAP:
        raise ValueError(f"{len(ids)} banned ids exceed vLLM's logit_bias cap of {LOGIT_BIAS_CAP} per request")
    return {int(i): BAN_BIAS for i in ids}
