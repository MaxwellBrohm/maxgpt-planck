"""RC-12 grader text helpers (SPEC s4). Pure Python, no model. Adapted from E001 battery.py (sentence-scoped
assertion analysis, person binding) and E004 gen_grade.py (strict clauses, user-voice object words).

Every rule that a mutant can switch off is named; OFF holds the names switched off (mutation_graders.py only).
P holds the numeric thresholds (also mutated). Nothing outside the mutation script ever touches OFF or P."""
import re

import grade_cues as GC
import pools_vals as V

OFF = set()
P = dict(neg_window=3, capture_window=8, echo=0.80, alt_chars=30)


def rx(p):
    return re.compile(p, re.I)


def norm(t):
    return (t or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def toks(t):
    return re.findall(r"[A-Za-z0-9']+", t or "")


def lwords(t):
    return [w.lower() for w in toks(t)]


SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")
NEG_CUE = rx(r"\b(?:not|no|never|neither|nor|without|cannot|none)\b|n't\b|\binstead of\b|\brather than\b")
NO_INTERJ = rx(r"\bno(?:\s+(?:problem|worries|doubt))?\s*[,.!;:]")
POST_NEG = rx(r"^\s+(?:(?:is|was|are|were|has|had|does|did)\s+(?:not|never|no longer)\b|(?:isn't|wasn't|aren't|weren't"
              r"|hasn't|doesn't|didn't)\b|(?:is|was)\s+wrong\b)")
HEDGE = rx(r"\b(?:maybe|perhaps|possibly|probably|might|may be|could be|not sure|not certain|i guess|guessing|i think"
           r"|i believe|i suppose|presumably|likely|" + GC.HEDGE_MORE + r")\b")      # GC: step 5 audit additions
HYPO = rx(r"\b(?:for example|for instance|e\.g|such as|hypothetically)\b")
COND = rx(r"\b(?:if|suppose|supposing|imagine|let's say|assuming|in case)\b")
VOC = rx(r"\b(?:hi|hello|hey|dear|thanks|thank you|welcome|nice to meet you|good to meet you|okay|ok)\W+(?:there\W+)?$")
LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+")
UNSURE = rx(r"\b(?:i'm not sure|i am not sure|not sure (?:what|which|who|when|where)|i'm not certain|i am not certain"
            r"|i don't know|i do not know|i (?:can't|cannot|don't|do not) (?:remember|recall)|remind me"
            r"|(?:could|can) you (?:please )?(?:tell|remind|share)|you (?:haven't|have not|didn't|did not|never) "
            r"(?:told|mentioned|said|shared|say|tell|mention|share)|i don't have (?:any |that |the |enough )?"
            r"(?:information|details|record|access|memory)|as an ai)\b")
ABS_CUE = rx(r"\byou (?:haven't|have not|didn't|did not|never) (?:yet )?(?:told|mentioned|said|shared|given|stated|say"
             r"|tell|mention|share|give)\b|\bi (?:don't|do not) know\b|\bi'm not sure\b|\bi am not sure\b"
             r"|\bi (?:don't|do not) have (?:that|this|any|the|enough)?\s*(?:information|detail|details|record)"
             r"|\b(?:isn't|is not|aren't|are not|not) (?:listed )?(?:on|in|part of|included in) (?:the|that|this|your)\b"
             r"|\b(?:there's|there is|there are) (?:no|nobody|no one|nothing)\b|\b(?:nobody|no one) is (?:listed|on)\b"
             r"|\b(?:doesn't|does not|don't|do not) (?:include|list|have|mention|show|say|cover)\b|\bnot (?:listed|included|mentioned|specified|given)\b|\bwasn't mentioned\b"
             r"|" + GC.ABS_MORE)
# person binding (E001 battery.py): first person that BINDS a fact to the speaker
NEUTRAL_I = (r"think|remember|believe|recall|guess|know|told|said|mentioned|understand|see|hope|noted|note|can|will"
             r"|would|could|should|might|may|must|do|did|don't|didn't|apologize|suggest|recommend")
FIRST_BIND_RX = r"(?:my|mine|i'm|i am|i was|i've|i have|me|i\s+(?!(?:" + NEUTRAL_I + r")\b)[a-z]+)"
FIRST_BIND = re.compile(r"\b" + FIRST_BIND_RX + r"\b", re.I)
PERSON = re.compile(r"\b(?:" + FIRST_BIND_RX + r"|you|your|you're|yours|you've)\b", re.I)
SECOND_P = rx(r"\b(?:you|your|you're|yours)\b")
FIRST_ANY = rx(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b")
# assistant-held facts (OWN picks, the name the user gave the assistant): the reply gives the fact to the user
ASSIST_REV = rx(r"\b(?:you|you've|you have)\s+(?:\w+\s+)?(?:picked|chose|chosen|went with|selected|suggested|listed)\b"
                r"|\b(?:i|i've|i have)\s+(?:\w+\s+)?(?:called|named|call|name)\s+you\b")
ASSIST_REV_END = rx(r"\b(?:your name is|you're|you are|you're called|you are called)\s*$")


def sentence_span(text, pos):
    start = 0
    for m in SENT_END.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.start()
    return start, len(text)


def is_question(sent):
    s = re.sub(r"[\s\"'*_)\]]+$", "", sent)
    s = re.sub(r"[^\x00-\x7F]+$", "", s).rstrip()
    return s.endswith("?")


def list_lines(text):
    return [ln for ln in (text or "").splitlines() if LIST_LINE.match(ln)]


def value_hits(text, v):
    """match objects for every mention of pool value v (whole word; capitalized values case-sensitive)."""
    return list(V.value_re(v).finditer(text or ""))


def mentioned(text, v):
    return bool(value_hits(text, v))


def negated_before(text, s0, pos):
    if "neg" in OFF:
        return False
    before = NO_INTERJ.sub(" ", text[s0:pos])
    window = " ".join(toks(before)[-P["neg_window"]:])
    return NEG_CUE.search(window) is not None


def negated_after(text, end):
    return "postneg" not in OFF and POST_NEG.search(text[end:end + 40]) is not None


def in_condition(before):
    if "cond" in OFF:
        return False
    last = None
    for c in COND.finditer(before):
        last = c
    return last is not None and not re.search(r",|\bthen\b", before[last.end():], re.I)


def alternative(text, m, s0, s1, altvals):
    """the value sits next to an 'or' whose other side offers a candidate-like word (capitalized, digit, or a
    value of the probe's pool), or next to a slash: "Friday or Saturday", "red or blue", "red/blue". An 'or' inside
    a negated list after the value ("Friday, not Monday or Tuesday") or before a "but" that precedes it ("Not
    Monday or Tuesday, but Friday") offers no alternative."""
    if "alt" in OFF:
        return False
    if text[max(0, m.start() - 1):m.start()] == "/" or text[m.end():m.end() + 1] == "/":
        return True
    low_alt = {a.lower() for a in altvals}
    for o in re.finditer(r"\bor\b", text[s0:s1], re.I):
        a, b = s0 + o.start(), s0 + o.end()
        if b < m.start() - P["alt_chars"] or a > m.end() + P["alt_chars"]:
            continue
        if a >= m.end() and NEG_CUE.search(text[m.end():a]):
            continue
        if b <= m.start() and re.search(r"\bbut\b", text[b:m.start()], re.I):
            continue
        left = toks(text[max(s0, a - 25):a])[-2:]
        right = toks(text[b:min(s1, b + 25)])[:2]
        gold = m.group(0).lower()
        for w in left + right:
            if w.lower() == gold or w == "I":
                continue
            if w[0].isupper() or w[0].isdigit() or w.lower() in low_alt:
                return True
    return False


def bare_list_value(text, m):
    """a bare value on a bullet/numbered line (<= 4 words, no "Label:") in a reply with 3+ list lines."""
    if "list" in OFF or len(list_lines(text)) < 3:
        return False
    ls = text.rfind("\n", 0, m.start()) + 1
    nl = text.find("\n", m.start())
    line = text[ls:nl if nl >= 0 else len(text)]
    if not LIST_LINE.match(line):
        return False
    body = re.sub(r"[*_`]", "", LIST_LINE.sub("", line, count=1))
    head = re.sub(r"[*_`]", "", text[ls:m.start()])
    return len(toks(body)) <= 4 and ":" not in head


def why_not_asserted(text, m, altvals=()):
    """None when this mention is ASSERTED; else the reason (question, neg, postneg, hedge, hypo, cond, alt, list,
    than, voc). text must already be norm()-ed."""
    s0, s1 = sentence_span(text, m.start())
    sent, before = text[s0:s1], text[s0:m.start()]
    if negated_before(text, s0, m.start()):
        return "neg"
    if negated_after(text, m.end()):
        return "postneg"
    if "question" not in OFF and is_question(sent):
        return "question"
    if "hedge" not in OFF and (HEDGE.search(sent) or GC.near_hedge(text, s0, s1, HEDGE)):
        return "hedge"
    if "hypo" not in OFF and HYPO.search(before):
        return "hypo"
    if in_condition(before):
        return "cond"
    if alternative(text, m, s0, s1, altvals):
        return "alt"
    if bare_list_value(text, m):
        return "list"
    if "than" not in OFF and re.search(r"\bthan\W+(?:(?:the|your|my|his|her|their)\W+)?$", before, re.I):
        return "than"
    if "voc" not in OFF and VOC.search(before):
        return "voc"
    return None


def asserted_hits(text, v, altvals=()):
    return [m for m in value_hits(text, v) if why_not_asserted(text, m, altvals) is None]


def captures(v, text):
    """perspective error (E001 battery.captures): a mention of v whose nearest preceding person marker in its
    sentence (within 8 words) binds it to the first person; with no marker that close, a sentence with
    first-person binding and no "you" anywhere."""
    if "capture" in OFF:
        return False
    for m in value_hits(text, v):
        s0, s1 = sentence_span(text, m.start())
        window = " ".join(text[s0:m.start()].split()[-P["capture_window"]:])
        marks = list(PERSON.finditer(window))
        if marks:
            if FIRST_BIND.fullmatch(marks[-1].group(0)):
                return True
        elif FIRST_BIND.search(text[s0:s1]) and not SECOND_P.search(text[s0:s1]):
            return True
    return False


def my_object(text, objwords):
    """E004 clause 7: "my" within 3 words before a word naming the asked object or its holder."""
    if "myobj" in OFF or not objwords:
        return False
    ws = lwords(text)
    return any(w == "my" and set(ws[i + 1:i + 4]) & objwords for i, w in enumerate(ws))


def assist_reversed(text, v):
    """an assistant-held value given to the user: "You picked Quizzards", "Your name is Kestrel"."""
    if "assistrev" in OFF:
        return False
    for m in value_hits(text, v):
        s0, _ = sentence_span(text, m.start())
        before = text[s0:m.start()]
        if ASSIST_REV.search(before) or ASSIST_REV_END.search(before):
            return True
    return False


def unsure(text):
    return "unsure" not in OFF and UNSURE.search(text) is not None


def echo(reply, question):
    """>= 80% of the reply's word 4-grams copied from the question (SPEC v5), or the reply contains the whole
    question (4+ words) word for word ("What instrument am I learning? Trombone."); replies under 4 words:
    identical words."""
    if "echo" in OFF:
        return False
    r, q = lwords(reply), lwords(question)
    if not r:
        return False
    if len(r) < 4:
        return r == q
    if "echo_contains" not in OFF and len(q) >= 4 and any(r[i:i + len(q)] == q for i in range(len(r) - len(q) + 1)):
        return True
    qg = {tuple(q[i:i + 4]) for i in range(len(q) - 3)}
    rg = [tuple(r[i:i + 4]) for i in range(len(r) - 3)]
    return sum(g in qg for g in rg) / len(rg) >= P["echo"]
