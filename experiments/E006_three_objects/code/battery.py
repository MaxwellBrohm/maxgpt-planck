"""Multi-turn coherence battery for small chat models (E001 hardened graders).

Copied from research/probe/battery.py; the conversations and fixtures are unchanged, the
graders are hardened against the false passes the audit (lanes/probe.verify.md 2a-2d) found:

  negation      "Your name is not Priya." / "It won't hurt at all." / "prevents burns"
  guess lists   "Is it Priya, Maria, or Anna?" / "Maybe 415, 416 or 417?" / bullet lists
  hedges        "a cat that you might call Biscuit" / "I'm not sure. ... Priya"
  hypotheticals "For example, if your appointment is on Tuesday, ..."
  "you" + first person  "You told me my name was Priya." (nearest person marker wins)
  incidental numbers    "The apartment number for building 417 is 12345." /
                        "Bake them at 350F for 12 minutes." / "4. Arrive early." for 4 pm
  both values   "Your appointment is on Monday. Tuesday is free." (the corrected value must
                be the one asserted, or appear in a change phrase, when the stale one is too)
  truncation    a check that FAILS on a reply that hit the token cap is ungradable (None),
                except format checks whose violation is already visible (one sentence, caps)

Run `python battery.py` for the self-test: every check must pass its good fixture, fail an
empty answer, fail its plausible wrong answer, fail every adversarial fixture generated for
its kind, and pass every legitimate variant; no gold answer may appear in its own question
or in any distractor. mutation_test_graders.py breaks each guard and checks this goes red.
"""
import re

# ---------------------------------------------------------------- distractors
# No names, numbers, cities or pets in these, so they cannot leak a gold answer.
D = [
    "What's a good way to keep houseplants healthy?",          # 0
    "Can you give me a tip for sleeping better?",              # 1
    "Why is the sky blue?",                                    # 2
    "What are some healthy breakfast ideas?",                  # 3
    "How do I make my coffee taste less bitter?",              # 4
    "What's the difference between weather and climate?",      # 5
    "Can you explain what a noun is?",                         # 6
    "How can I stay focused while studying?",                  # 7
    "What's a fun indoor activity for a rainy day?",           # 8
    "Why do leaves change color in the fall?",                 # 9
    "How do I make a paper airplane fly farther?",             # 10
    "What does a librarian do?",                               # 11
]

# ---------------------------------------------------------------- helpers
def norm(s):
    return (s or "").lower()


def rx(pattern):
    return re.compile(pattern, re.I)


def unquote(t):
    return (t or "").replace("’", "'").replace("‘", "'")


def mentions(pattern, text):
    return bool(rx(pattern).search(unquote(text)))


# ---- assertion analysis: is an occurrence of the gold actually ASSERTED as the answer?
NEGATION = rx(r"(?:\bnot|n't|\bnever|\bno longer|\bwithout|\bprevents?|\bpreventing|\bavoids?|\bavoiding|\binstead of|\brather than|\bisnt|\bwont)\W+(?:\w+\W+){0,2}$")
NO_BEFORE = rx(r"\bno\W+$")
STALE_CTX = rx(r"(?:\bfrom|\bchanged|\boriginally|\bpreviously|\bearlier|\bbefore|\bused to|\bwas|\bwere|\bold)\W+(?:\w+\W+){0,3}$")
STALE_POST = rx(r"^\W*(?:\w+\W+){0,1}(?:was|were|originally|previously|before|earlier|no longer)\b")
CHANGE_CTX = rx(r"(?:\bmoved|\bchanged|\bswitched|\brescheduled|\bupdated|\bnow|\bnew|\bactually|\binstead|\bcorrect(?:ed|ion)?)\W+(?:\w+\W+){0,3}$")
HEDGE = rx(r"\b(?:maybe|perhaps|possibly|probably|might|could be|not sure|not certain|i guess|guessing)\b")
HYPO = rx(r"\b(?:for example|for instance|e\.g|such as|hypothetically)\b")  # examples: to the end of the sentence
COND = rx(r"\b(?:if|suppose|imagine|let's say)\b")  # conditionals: up to the next comma or "then"
UNSURE = rx(r"\b(?:i'm not sure|i am not sure|not sure what|i'm not certain|i am not certain|i don't know|i do not know)\b")
LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_span(text, pos):
    start = 0
    for m in SENT_END.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.start()
    return start, len(text)


def is_question(sent):
    s = re.sub(r"[\s\"'*_)\]”]+$", "", sent)
    s = re.sub(r"[^\x00-\x7F]+$", "", s).rstrip()
    return s.endswith("?")


def list_lines(text):
    return [l for l in (text or "").splitlines() if LIST_LINE.match(l)]


def or_alternative(text, m, s0, s1):
    """The gold sits next to an 'or' whose other side offers a candidate-like alternative
    (a capitalized word or a number): "Priya, Maria, or Anna", "Rome or Milan", "3 pm, or 4 pm",
    "417 or 420". Plain enumerations ("burns, injuries, or even death") do not count."""
    for o in re.finditer(r"\bor\b", text[s0:s1], re.I):
        a, b = s0 + o.start(), s0 + o.end()
        if b < m.start() - 30 or a > m.end() + 30:
            continue
        left = re.findall(r"[\w']+", text[max(s0, a - 25):a])[-2:]
        right = re.findall(r"[\w']+", text[b:min(s1, b + 25)])[:2]
        gold = m.group(0).lower()
        for w in left + right:
            if w.lower() in gold or gold in w.lower() or w == "I":
                continue
            if w[0].isupper() or w[0].isdigit():
                return True
    return False


def occ_ok(text, m, single=True, strict=True):
    """Is this regex match an assertion (not negated, questioned, hedged, hypothetical or one
    of a list of guesses)? strict=False (knowledge / follow-up checks) keeps only the
    negation, question and 'or' rules, because hedges and examples are fine in explanations."""
    s0, s1 = sentence_span(text, m.start())
    sent = text[s0:s1]
    before = text[max(s0, m.start() - 40):m.start()]
    if NEGATION.search(before) or NO_BEFORE.search(before):
        return False
    if is_question(sent):
        return False
    if or_alternative(text, m, s0, s1) or text[max(0, m.start() - 1):m.start()] == "/" or text[m.end():m.end() + 1] == "/":
        return False
    if not strict:
        return True
    if HEDGE.search(sent) or HYPO.search(text[s0:m.start()]) or in_condition(text[s0:m.start()]):
        return False
    if single and bare_list_value(text, m):
        return False
    return True


def in_condition(before):
    """The match sits inside an if-clause: an 'if' earlier in the sentence with no comma or
    'then' between it and the match. "If it moved to 4 pm, you should arrive at 4 pm" asserts
    the second 4 pm (main clause) but not the first."""
    last = None
    for c in COND.finditer(before):
        last = c
    if last is None:
        return False
    return not re.search(r",|\bthen\b", before[last.end():], re.I)


def bare_list_value(text, m):
    """The match is a bare value on a bullet/numbered line (4 words or fewer, no "Label:"
    before it) in a reply with 3+ list lines: a list of guesses, not a statement."""
    if len(list_lines(text)) < 3:
        return False
    line_start = text.rfind("\n", 0, m.start()) + 1
    nl = text.find("\n", m.start())
    line = text[line_start:nl if nl >= 0 else len(text)]
    if not LIST_LINE.match(line):
        return False
    body = re.sub(r"[*_`]", "", LIST_LINE.sub("", line, count=1))
    head = re.sub(r"[*_`]", "", text[line_start:m.start()])
    return len(re.findall(r"[\w']+", body)) <= 4 and ":" not in head


def asserted_spans(pattern, text, single=True, strict=True):
    t = unquote(text)
    return [(m.start(), m.end()) for m in rx(pattern).finditer(t) if occ_ok(t, m, single, strict)]


def answered(pattern, text, single=True, strict=True):
    return bool(asserted_spans(pattern, text, single, strict))


def stale_asserted(pattern, text):
    t = unquote(text)
    out = []
    for m in rx(pattern).finditer(t):
        before = t[max(0, m.start() - 40):m.start()]
        if STALE_CTX.search(before) or STALE_POST.search(t[m.end():m.end() + 30]):
            continue
        if occ_ok(t, m, single=False, strict=True):
            out.append(m.start())
    return out


def asserted(pattern, text):
    """Kept from the original battery (used by its self-test): pattern occurs somewhere not
    preceded by a negation or stale ("changed from", "was") context."""
    t = unquote(text)
    for m in rx(pattern).finditer(t):
        before = t[max(0, m.start() - 40):m.start()]
        if not (NEGATION.search(before) or STALE_CTX.search(before)):
            return True
    return False


def words(t):
    return re.findall(r"[A-Za-z0-9']+", t or "")


def n_sentences(t):
    t = (t or "").strip()
    t = re.sub(r"\b(e\.g|i\.e|etc|dr|mr|mrs|ms|st|vs|approx|u\.s)\.", lambda m: m.group(0).replace(".", ""), t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", t) if len(re.findall(r"[A-Za-z]+", p)) >= 2]
    return len(parts)


def ends_with_question(t):
    t = (t or "").rstrip()
    t = re.sub(r"[\s\"'*_)\]’”]+$", "", t)
    t = re.sub(r"[^\x00-\x7F]+$", "", t).rstrip()  # trailing emoji
    return t.endswith("?")


def caps_ratio(t):
    letters = [c for c in (t or "") if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def parse_list(t):
    """Items of a numbered / bulleted list, cleaned to their head words."""
    items = []
    for line in (t or "").splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)", line)
        if m:
            s = m.group(1)
            s = re.sub(r"[*_`\"]", "", s)
            s = re.split(r"\s[-\u2013\u2014:(]|[:(,]", s)[0].strip()
            if s:
                items.append(s)
    if len(items) < 3:  # fallback: "1. Apple 2. Kiwi 3. Orange" on one line
        found = re.findall(r"(?:^|\s)\d+[.)]\s*([A-Za-z][A-Za-z' ]{1,30}?)(?=\s+\d+[.)]|[,.;\n]|$)", t or "")
        if len(found) >= 3:
            items = [f.strip() for f in found]
    return items


def stems(t):
    return [w.lower()[:5] for w in re.findall(r"[A-Za-z]+", t or "") if len(w) >= 3]


def item_key(item):
    """A matching key for a list item: the sequence of 5-char stems of its words."""
    k = tuple(stems(item))
    return k or None


def find_seq(seq, key):
    for i in range(len(seq) - len(key) + 1):
        if tuple(seq[i:i + len(key)]) == key:
            return i
    return -1


def last_int(t):
    nums = re.findall(r"(?<![\d.])\d+(?!\d|\.\d)", (t or "").replace(",", ""))
    return int(nums[-1]) if nums else None


def num_pat(n):
    return r"(?<![\d.,])" + str(n) + r"(?![\d])"


def competing_numbers(text, gold, allowed=()):
    """Asserted numbers with 2+ digits other than the gold and the numbers the user gave,
    ignoring list markers ("4. Arrive early"). A second number is a guess or an incidental
    answer ("building 417 is 12345", "350F for 12 minutes")."""
    t = unquote(text).replace(",", "")
    out = []
    for m in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])", t):
        n = m.group(0)
        if n == str(gold) or n in {str(a) for a in allowed} or len(n.replace(".", "")) < 2:
            continue
        line_start = t.rfind("\n", 0, m.start()) + 1
        if re.match(r"\s*$", t[line_start:m.start()]) and re.match(r"[.)]\s", t[m.end():m.end() + 2]):
            continue
        if not occ_ok(t, m, single=False, strict=False):
            continue
        out.append(n)
    return out


# Deflection: claims it cannot remember / has no access / cites privacy instead of answering.
NO_MEMORY = rx(r"(don't|do not|cannot|can't|unable to|not able to)\s+(\w+\s+){0,3}(access|remember|recall|know|store|retain|provide)|as an ai|as a helpful ai|i don't have (any |the )?(personal|memory|information|ability)|do not have (any |the )?(personal|memory|information|ability)|personal (information|data|details)|privacy|no (memory|access)|haven't (told|mentioned|shared)|you (haven't|didn't) (tell|mention|share|say)")
# E001 additions: phrasings the audit found the original regex missed
NO_MEMORY_2 = rx(r"(?:don't|do not) have (?:any |the |enough |specific |exact )?(?:specific |exact )?(?:information|details|data)"
                 r"|(?:i'm|i am) not able to answer|not sure what you (?:told|meant|said|mentioned)|didn't know your"
                 r"|as a helpful assistant|could you (?:please )?(?:share|tell me|remind me)")

# ---------------------------------------------------------------- check factories
# Every check is a dict: name, turn (index of the assistant reply it grades),
# fn(ans, replies) -> bool, good (fixture that must pass), bad (plausible wrong
# answer that must fail), ctx (fake earlier replies for dynamic checks).
# E001 adds: adv (adversarial fixtures that must fail) and ok (legit variants that must pass).

IDENTITY_DEFLECT = rx(r"\bi(?:'m| am) (?:just |only )?(?:an ai\b|a (?:large )?language model|an artificial intelligence|a virtual assistant|a chatbot|a text-based ai|qwen\b|gemma\b|smollm\b)"
                      r"|\bas an? (?:ai|(?:large )?language model|chatbot|text-based ai)\b"
                      r"|\b(?:don't|do not|can't|cannot) have (?:a|any) (?:physical|name|cat|dog|pet|favorite|personal)")


def deflects(ans):
    """Memory/privacy deflection or an identity disclaimer instead of an answer about the user."""
    a = unquote(ans)
    return bool(NO_MEMORY.search(a) or NO_MEMORY_2.search(a) or IDENTITY_DEFLECT.search(a))


FIRST_P = rx(r"\b(i|i'm|i've|my|me|mine)\b")
SECOND_P = rx(r"\b(you|your|you're|yours)\b")
# first person that BINDS a fact to the speaker: possessives, "I'm", and "I <verb>" except
# cognition, speech and modal verbs ("I remember", "I think", "I told you", "I would")
NEUTRAL_I = r"think|remember|believe|recall|guess|know|told|said|mentioned|understand|see|hope|noted|note|can|will|would|could|should|might|may|must|do|did|don't|didn't|apologize|suggest|recommend"
FIRST_BIND_RX = r"(?:my|mine|i'm|i am|i was|i've|i have|me|i\s+(?!(?:" + NEUTRAL_I + r")\b)[a-z]+)"
FIRST_BIND = re.compile(r"\b" + FIRST_BIND_RX + r"\b", re.I)
PERSON = re.compile(r"\b(?:" + FIRST_BIND_RX + r"|you|your|you're|yours|you've)\b", re.I)


def captures(pattern, ans):
    """Perspective error: an occurrence of the gold whose NEAREST preceding person marker in
    its sentence (within 8 words) is first person ("You told me my name was Priya", "I'm
    Marcus"); with no marker that close, the original sentence rule applies, restricted to
    identity/possessive first person ("Priya is my name" fails, "It's Priya, if I remember
    correctly" passes) with no "you" anywhere in the sentence."""
    t = unquote(ans)
    for m in rx(pattern).finditer(t):
        s0, s1 = sentence_span(t, m.start())
        before = t[s0:m.start()]
        window = " ".join(before.split()[-8:])
        marks = list(PERSON.finditer(window))
        if marks:
            if FIRST_BIND.fullmatch(marks[-1].group(0)):
                return True
        else:
            sent = t[s0:s1]
            if FIRST_BIND.search(sent) and not SECOND_P.search(sent):
                return True
    return False


VOCATIVE = r"\b(hi|hello|hey|okay|ok|dear|thanks|thank you|welcome|nice to meet you|good to meet you)\W+(?:there\W+)?"


def strip_vocative(pattern, ans):
    """Remove greeting uses of the gold ("Hello Marcus!", "Okay, Priya!") so an echo in a
    greeting is not counted as an answer."""
    return re.sub(VOCATIVE + r"(?:" + pattern + r")\b\W?", " ", ans or "", flags=re.I)


def adv_fixtures(kind, val, alt):
    """Adversarial wrong answers that each contain the gold value (they must all fail)."""
    if kind == "name":
        return [f"Your name is not {val}.", f"Is your name {val}, {alt}, or Anna?", f"Maybe {val}? Or {alt}.",
                f"You told me my name was {val}.", f"I'm {val}, and you asked about my name.",
                f"Here you go:\n- {alt}\n- {val}\n- Anna", f"I'm not sure about that. Your name is {val}.",
                f"For example, if your name were {val}, that would be lovely.", f"Did you say {val}?",
                f"Your name is {val} or {alt}.", f"For example, {val} is a lovely name."]
    if kind == "word":
        return [f"It is not {val}.", f"Is it {val} or {alt}?", f"It might be {val}, or {alt}.",
                f"For example, if it were {val}, that would be nice.", f"Here you go:\n- {alt}\n- {val}\n- Other",
                f"I'm not sure about that. It is {val}.", f"You know my answer: {val}.", f"Did you say {val}?",
                f"It is {val} or {alt}.", f"For example, {val} is a popular choice."]
    if kind == "number":
        return [f"It is not {val}.", f"Maybe {val - 1}, {val} or {val + 1}?", f"The number for building {val} is 12345.",
                f"It could be {val} or {val + 3}.", f"- {val}\n- {val + 1}\n- {val + 2}\n- {val + 3}",
                f"If it were {val}, that would be easy to remember.", f"Is it {val}?"]
    raise ValueError(kind)


def has(name, turn, pattern, good, bad, guard=True, capture=True, val=None, alt=None, kind=None,
        single=True, wrong=None, number=None, allowed=(), units=None, adv=(), ok=()):
    """Gold pattern ASSERTED (outside a greeting). With guard=True the reply must also not
    deflect or sound unsure and, if capture=True, must not state the user's fact as its own.
    number: gold integer; any other asserted 2+ digit number (except `allowed`) fails.
    units: a regex of words that make a gold number incidental ("12 minutes").
    wrong: a regex whose presence fails the reply (a known wrong binding)."""
    def fn(ans, replies):
        if guard:
            ans = strip_vocative(pattern, ans)
        spans = asserted_spans(pattern, ans, single=single, strict=guard)
        if units:
            t = unquote(ans)
            spans = [(a, b) for a, b in spans if not re.match(r"\s*(?:" + units + r")\b", t[b:], re.I)]
        if not spans:
            return False
        if wrong and mentions(wrong, ans):
            return False
        if number is not None and competing_numbers(ans, number, allowed):
            return False
        if guard and (deflects(ans) or UNSURE.search(unquote(ans))):
            return False
        if guard and capture and captures(pattern, ans):
            return False
        return True
    adv_all = list(adv) + (adv_fixtures(kind, val, alt) if kind else [])
    if kind and not capture:  # summaries may quote the user; drop the perspective fixtures
        adv_all = [a for a in adv_all if not re.search(r"\b(my|I'm)\b", a)]
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=pattern, guard=guard, adv=adv_all, ok=list(ok))


def has_not_stale(name, turn, new, old, good, bad, adv=(), ok=()):
    """Corrected value asserted, stale value never asserted (negated, "was", "changed from"
    and "originally" mentions are fine). If both are asserted, the reply passes only when it
    ends on the corrected value AND the corrected value appears in a change phrase
    ("moved to", "now", "actually", "instead")."""
    def fn(ans, replies):
        if deflects(ans) or UNSURE.search(unquote(ans)) or captures(new, ans):
            return False
        pos_new = [a for a, b in asserted_spans(new, ans, single=True, strict=True)]
        if not pos_new:
            return False
        pos_old = stale_asserted(old, ans)
        if not pos_old:
            return True
        t = unquote(ans)
        changed = any(CHANGE_CTX.search(t[max(0, p - 40):p]) for p in pos_new)
        return max(pos_new) > max(pos_old) and changed
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=new, guard=True, adv=list(adv), ok=list(ok))


def binding(name, turn, right, wrong_assert, good, bad, adv=(), ok=()):
    def fn(ans, replies):
        return (answered(right, ans) and not mentions(wrong_assert, ans) and not deflects(ans)
                and not UNSURE.search(unquote(ans)) and not captures(right, ans))
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=right, guard=True, adv=list(adv), ok=list(ok))


TRUNC_SENSITIVE = ("end_q",)  # checks decided by the END of the reply, which a cap cuts off


def grade(check, replies, hit_max=False):
    """Run a check. A FAIL on a reply that hit the token cap is ungradable (None) only for
    checks decided by how the reply ENDS (end with a question). Content checks keep their
    fail: 120-160 tokens without the answer is a failure a user would see.
    Returns (result, capped_fail_flag); the flag also marks capped content fails for reporting."""
    r = check["fn"](replies[check["turn"]], replies)
    if r is False and hit_max:
        if check["name"].startswith(TRUNC_SENSITIVE):
            return None, True
        return False, True
    return (None if r is None else bool(r)), False


def degenerate(ans):
    """A loop: some word 3-gram occurs 4+ times in the reply."""
    ws = [w.lower() for w in words(ans)]
    g = {}
    for i in range(len(ws) - 2):
        k = tuple(ws[i:i + 3]); g[k] = g.get(k, 0) + 1
    return max(g.values(), default=0) >= 4


def fresh(ans, replies, turn):
    """Format checks only count a real reply: not a loop, not a verbatim copy of an earlier reply."""
    a = norm(ans).strip()
    return not degenerate(ans) and all(a != norm(x).strip() for x in replies[:turn])


def one_sentence(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: n_sentences(a) == 1 and fresh(a, r, turn), good=good, bad=bad)


def end_q(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: ends_with_question(a) and fresh(a, r, turn), good=good, bad=bad)


def all_caps(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: caps_ratio(a) >= 0.9 and fresh(a, r, turn), good=good, bad=bad)


def has_fmt(name, turn, pattern, good, bad):
    """Format marker present in a fresh (non-looping, non-copied) reply."""
    return dict(name=name, turn=turn, fn=lambda a, r: mentions(pattern, a) and fresh(a, r, turn), good=good, bad=bad)


def shorter_than(name, turn, src_turn, topic, good, bad, ctx):
    def fn(ans, replies):
        prev = replies[src_turn]
        wa, wp = len(words(ans)), len(words(prev))
        return 3 <= wa < wp and mentions(topic, ans) and norm(ans).strip() != norm(prev).strip()
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx)


def list_ref(name, turn, src_turn, idx, good, bad, ctx):
    """Answer names item idx of the list the model produced at src_turn, and it is
    the first list item mentioned (so reciting the whole list does not pass).
    None (ungradable) if the model produced no 3-item list or the target item is
    not distinct from the others."""
    def fn(ans, replies):
        items = parse_list(replies[src_turn])
        if len(items) < 3:
            return None
        keys = [item_key(i) for i in items[:3]]
        k = keys[idx]
        if not k or sum(1 for kk in keys if kk == k) > 1:
            return None
        a = stems(ans)
        pos_k = find_seq(a, k)
        if pos_k < 0:
            return False
        others = [find_seq(a, kk) for kk in keys if kk and kk != k]
        return all(p < 0 or pos_k <= p for p in others)
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx, dynamic=True)


def add_ref(name, turn, src_turn, delta, good, bad, ctx):
    """Answer contains (model's own earlier number) + delta."""
    def fn(ans, replies):
        n = last_int(replies[src_turn])
        if n is None:
            return None
        return mentions(num_pat(n + delta), ans)
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx, dynamic=True)


def not_claim(name, turn, claims, good, bad):
    """Fails if any sentence speaks in the first person about the USER's job/identity
    (e.g. "As a chef in a busy restaurant, my day is dynamic") without addressing the user."""
    job = rx(r"\b(" + claims + r")\b")
    first = rx(r"\b(i|i'm|i've|my|me)\b")
    second = rx(r"\b(you|your|you're)\b")
    def fn(ans, replies):
        if len(words(ans)) < 2:
            return False
        for sent in re.split(r"(?<=[.!?])\s+|\n+", ans):
            if job.search(sent) and first.search(sent) and not second.search(sent):
                return False
        return True
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad)


# ---------------------------------------------------------------- the battery
TESTS = []


def T(**kw):
    kw.setdefault("system", None)
    kw.setdefault("distance", None)
    kw.setdefault("forced", [])
    kw.setdefault("control_for", None)
    kw.setdefault("max_new", {})
    TESTS.append(kw)


# (1) recall of a user-stated fact at increasing distance (distance = number of
# intervening user turns between the fact and the question).
# Extra fields (E001): grader kind, gold value, alternative value, numeric gold, verbatim
# false passes from the audit that must now fail.
RECALL = [
    ("R_d2_name", 2, "Hi! My name is Priya, and I'm trying to get more organized this year.", [0, 1],
     "Quick check: what did I tell you my name was?", r"\bpriya\b", "Your name is Priya.", "Your name is Maria.",
     ("Your name is", " Priya", " Maria"),
     dict(kind="name", val="Priya", alt="Maria",
          adv=["Your name is not Priya.", "Is your name Priya, Maria, or Anna?", "I told you my name was Priya.",
               "You told me my name was Priya.", "I'm Priya, but you could say hi."],
          ok=["You told me your name is Priya.", "I remember! Your name is Priya.", "It's Priya, if I remember correctly.",
              "Your name, I believe, is Priya."])),
    ("R_d2_number", 2, "I just moved into apartment 417 in a new building downtown.", [2, 3],
     "What's my apartment number again?", num_pat(417), "Your apartment number is 417.", "Your apartment number is 714.",
     ("Your apartment number is", " 417", " 714"),
     dict(kind="number", val=417, number=417,
          adv=["The apartment number for building 417 is 12345.", "Maybe 415, 416 or 417?",
               "For apartment 417, you can use the following apartment numbers:\n- 417\n- 418\n- 419\n- 420"],
          ok=["You live in apartment 417.", "Your apartment number is 417, in the new building downtown.",
              "Here's a quick note:\n- **Apartment Number**: 417\n- **Building**: new, downtown\n- **Move-in**: this month"])),
    ("R_d4_pet", 4, "I have a cat named Biscuit who loves sitting in cardboard boxes.", [4, 5, 6, 7],
     "What's my cat called?", r"\bbiscuit", "Your cat is called Biscuit.", "Your cat is called Whiskers.",
     ("Your cat is called", " Biscuit", " Whiskers"),
     dict(kind="word", val="Biscuit", alt="Whiskers",
          adv=["A cat that you might call \"Biscuit\" would love boxes.", "My cat is called Biscuit."],
          ok=["Your cat is called Biscuit, the one who loves cardboard boxes."])),
    ("R_d4_city", 4, "I live in Tucson and I work at a small bakery.", [8, 9, 10, 11],
     "Which city do I live in?", r"\btucson\b", "You live in Tucson.", "You live in Phoenix.",
     ("You live in", " Tucson", " Phoenix"),
     dict(kind="word", val="Tucson", alt="Phoenix",
          adv=["I live in Tucson, Arizona. Tucson is known for its desert landscapes.",
               "I don't have specific information about your location, but if you live in Tucson you might enjoy it.",
               "You don't live in Tucson; you live in Phoenix.",
               "I don't have specific information about your exact location, but I can tell you that Tucson, Arizona, is a beautiful city."],
          ok=["You live in Tucson, Arizona.", "Tucson! You mentioned it at the start.", "I think you live in Tucson."])),
    ("R_d6_name", 6, "Hello, I'm Marcus. I'm learning to play the guitar.", [0, 2, 4, 6, 8, 10],
     "What's my name?", r"\bmarcus\b", "Your name is Marcus.", "Your name is Michael.",
     ("Your name is", " Marcus", " Michael"),
     dict(kind="name", val="Marcus", alt="Michael", adv=["Hello Marcus! My name is Alex."],
          ok=["Your name is Marcus, and you're learning the guitar."])),
    ("R_d6_number", 6, "My grandmother turns 94 next week and I want to plan something nice for her.", [1, 3, 5, 7, 9, 11],
     "How old is my grandmother about to be?", num_pat(94) + r"|ninety[- ]four", "She is turning 94.", "She is turning 90.",
     ("Your grandmother is turning", " 94", " 90"),
     dict(kind="number", val=94, number=94,
          adv=["She is turning 94? No, she is turning 90.", "She will be 94 or 95 next week."],
          ok=["Your grandmother is turning 94 next week.", "She's about to be ninety-four!",
              "She is turning 94 next week. Here are some ideas to celebrate her birthday."])),
    ("R_d10_pet", 10, "By the way, my dog is named Waffles.", list(range(10)),
     "What's my dog's name?", r"\bwaffles?\b", "Your dog's name is Waffles.", "Your dog's name is Buddy.",
     ("Your dog's name is", " Waffles", " Buddy"),
     dict(kind="word", val="Waffles", alt="Buddy", adv=["By the way, my dog is named **Waffles**!"],
          ok=["Your dog is named Waffles."])),
]
for tid, dist, fact, dix, q, gold, good, bad, forced, ext in RECALL:
    turns = [fact] + [D[i] for i in dix] + [q]
    T(id=tid, cat="recall", distance=dist, turns=turns,
      checks=[has("recall", len(turns) - 1, gold, good, bad, **ext)], forced=[forced])
    T(id=tid + "__ctrl", cat="control", control_for=tid, distance=0, turns=[fact + " " + q],
      checks=[has("recall", 0, gold, good, bad, **ext)], forced=[forced])

multi_fact = "Quick intro: I'm Elena, I live in Oslo, and I have a parrot named Kiwi."
multi_q = "Can you summarize the three things I told you about myself at the very start?"
# "summarize what I told you" invites quoting the user in the first person, so the
# role-capture guard is off here (deflection guard stays on). E001: a known wrong binding
# ("you are a parrot") fails all three checks; the list-line guess rule is off (a summary
# is naturally a list).
PARROT = r"\b(?:you(?:'re| are)|i am|i'm) a parrot\b"
MULTI_GOOD = "You're Elena, you live in Oslo, and your parrot is Kiwi."
MULTI_BAD = "You're Anna, you live in Bergen, and your parrot is Polly."
MULTI_ADV = ["Elena: you are a parrot named Kiwi living in Oslo.", "Is it Elena, Anna, or Maria? Oslo or Bergen? Kiwi or Polly?"]
MULTI_OK = ["1. Your name is Elena\n2. You live in Oslo\n3. You have a parrot named Kiwi"]
multi_checks = lambda t: [
    has("name", t, r"\belena\b", MULTI_GOOD, MULTI_BAD, capture=False, single=False, wrong=PARROT, adv=MULTI_ADV, ok=MULTI_OK),
    has("city", t, r"\boslo\b", MULTI_GOOD, MULTI_BAD, capture=False, single=False, wrong=PARROT, adv=MULTI_ADV, ok=MULTI_OK),
    has("pet", t, r"\bkiwi\b", MULTI_GOOD, MULTI_BAD, capture=False, single=False, wrong=PARROT, adv=MULTI_ADV, ok=MULTI_OK),
]
T(id="R_d5_multi", cat="recall", distance=5, turns=[multi_fact] + [D[i] for i in [2, 4, 6, 8, 10]] + [multi_q],
  checks=multi_checks(6))
T(id="R_d5_multi__ctrl", cat="control", control_for="R_d5_multi", distance=0, turns=[multi_fact + " " + multi_q],
  checks=multi_checks(0))

# (2) coreference / elliptical follow-ups
ROME_ADV = ["The capital of Italy is not Rome.", "Is it Rome or Milan?", "Is the capital of Italy Rome?",
            "The capital of Italy is Rome or Milan."]
T(id="F_ellipsis_capital", cat="followup", turns=["What is the capital of France?", "And what about Italy?"],
  checks=[has("knows_paris", 0, r"\bparis\b", "The capital of France is Paris.", "The capital of France is Lyon.", guard=False,
              adv=["It is not Paris.", "Paris or Lyon?"]),
          has("ellipsis", 1, r"\brome\b|\broma\b", "The capital of Italy is Rome.", "The capital of Italy is Milan.", guard=False,
              adv=ROME_ADV)])
T(id="F_ellipsis_capital__ctrl", cat="control", control_for="F_ellipsis_capital", turns=["What is the capital of Italy?"],
  checks=[has("ellipsis", 0, r"\brome\b|\broma\b", "The capital of Italy is Rome.", "The capital of Italy is Milan.", guard=False,
              adv=ROME_ADV)])

T(id="F_translate", cat="followup", turns=["How do you say 'thank you' in Spanish?", "And in French?"],
  checks=[has("knows_gracias", 0, r"gracias", "It's 'gracias'.", "It's 'danke'.", guard=False),
          has("ellipsis", 1, r"\bmerci\b", "In French it's 'merci'.", "In French it's 'gracias'.", guard=False,
              adv=["In French it's not 'merci'.", "Is it 'merci' or 'danke'?"])])
T(id="F_translate__ctrl", cat="control", control_for="F_translate", turns=["How do you say 'thank you' in French?"],
  checks=[has("ellipsis", 0, r"\bmerci\b", "In French it's 'merci'.", "In French it's 'gracias'.", guard=False)])

COREF_WRONG = r"\b(she|ana|your sister)\s+(\w+\s+)?lives in denver"
COREF_ADV = ["She doesn't live in Miami.", "Does she live in Miami or Denver?", "Maybe Miami, maybe Denver."]
T(id="F_coref_she", cat="followup", turns=["My brother Tom lives in Denver, and my sister Ana lives in Miami.", "Where does she live?"],
  checks=[binding("coref", 1, r"\bmiami\b", COREF_WRONG, "She lives in Miami.", "She lives in Denver.", adv=COREF_ADV)])
T(id="F_coref_she__ctrl", cat="control", control_for="F_coref_she",
  turns=["My brother Tom lives in Denver, and my sister Ana lives in Miami. Where does she live?"],
  checks=[binding("coref", 0, r"\bmiami\b", COREF_WRONG, "She lives in Miami.", "She lives in Denver.", adv=COREF_ADV)])

WHY = r"\bburn|\bpain|\binjur|\bhurt|\bblister|damage (to )?(your )?skin"
WHY_ADV = ["Touching it is safe and prevents burns.", "It won't hurt at all.", "It will not burn you."]
WHY_OK = ["You might get a painful burn.", "Because it can burn your skin badly.",
          "Hot surfaces can cause serious burns, injuries, or even death."]
T(id="F_whynot", cat="followup", turns=["Is it a good idea to touch a hot stove?", "Why not?"],
  checks=[has("why", 1, WHY, "Because you could burn yourself.", "Because it keeps the kitchen tidy.", guard=False,
              adv=WHY_ADV, ok=WHY_OK)])
T(id="F_whynot__ctrl", cat="control", control_for="F_whynot", turns=["Why is it a bad idea to touch a hot stove?"],
  checks=[has("why", 0, WHY, "Because you could burn yourself.", "Because it keeps the kitchen tidy.", guard=False,
              adv=WHY_ADV, ok=WHY_OK)])

EX_TOPIC = r"exercis|health|heart|body|fit|strong|mood|energy|active|muscle|weight|stress|mind|sleep|feel"
T(id="F_shorter", cat="followup", turns=["Write two sentences about why exercise is good for you.", "Make it shorter."],
  checks=[shorter_than("shorter", 1, 0, EX_TOPIC, "Exercise keeps you healthy.",
                       "Exercise strengthens your heart and muscles, and it also improves your mood, sleep, and energy every single day of the week.",
                       ctx=["Exercise strengthens your heart and muscles. It also improves your mood and helps you sleep better."])])
T(id="F_shorter__ctrl", cat="control", control_for="F_shorter", turns=["Write one short sentence about why exercise is good for you."],
  checks=[dict(name="shorter", turn=0, fn=lambda a, r: 3 <= len(words(a)) <= 20 and mentions(EX_TOPIC, a),
               good="Exercise keeps your body and mind healthy.",
               bad="Exercise is good for you because it strengthens your heart, builds muscle, improves your mood, helps you sleep, and lowers stress over time.")])

# (3) instruction persistence
q1 = ["How does a rainbow form?", "What causes earthquakes?", "How do vaccines work?"]
T(id="I_one_sentence", cat="instruction", turns=["From now on, answer every question in exactly one sentence."] + q1,
  checks=[one_sentence(f"one_sentence_t{t}", t, "Vaccines train your immune system to recognize a germ.",
                       "Vaccines train your immune system. They contain weakened germs.") for t in (1, 2, 3)])
T(id="I_one_sentence__ctrl", cat="control", control_for="I_one_sentence", turns=["In exactly one sentence: how do vaccines work?"],
  checks=[one_sentence("one_sentence_t3", 0, "Vaccines train your immune system to recognize a germ.",
                       "Vaccines train your immune system. They contain weakened germs.")])

q2 = ["I like hiking on weekends.", "My favorite food is pasta.", "I'm reading a mystery novel right now."]
T(id="I_end_question", cat="instruction", turns=["For the rest of our chat, always end each of your replies with a question for me."] + q2,
  checks=[end_q(f"end_q_t{t}", t, "Mysteries are fun! Who is your favorite detective?",
                "Mysteries are fun. Enjoy the book!") for t in (1, 2, 3)])
T(id="I_end_question__ctrl", cat="control", control_for="I_end_question",
  turns=["I'm reading a mystery novel right now. Reply briefly and end your reply with a question for me."],
  checks=[end_q("end_q_t3", 0, "Mysteries are fun! Who is your favorite detective?", "Mysteries are fun. Enjoy the book!")])

q3 = ["What is your favorite season and why?", "Name two animals that live in the desert.", "What do bees make?"]
T(id="I_caps", cat="instruction", turns=["Please reply ONLY in capital letters for the rest of this conversation."] + q3,
  checks=[all_caps(f"caps_t{t}", t, "BEES MAKE HONEY AND WAX.", "Bees make honey and wax.") for t in (1, 2, 3)])
T(id="I_caps__ctrl", cat="control", control_for="I_caps", turns=["Reply only in capital letters: what do bees make?"],
  checks=[all_caps("caps_t3", 0, "BEES MAKE HONEY AND WAX.", "Bees make honey and wax.")])

PIRATE_SYS = "You are Captain Pip, a cheerful pirate. Always say 'Arr' somewhere in every reply."
ARR = r"\ba+r{2,}h*\b"
T(id="I_persona", cat="instruction", system=PIRATE_SYS,
  turns=["Hi there!", "What's a good snack for a long trip?", "Tell me about the weather at sea.", "What's your name?"],
  checks=[has_fmt(f"arr_t{t}", t, ARR, "Arr, matey! Hardtack be the best snack.", "Ahoy, matey! Hardtack is the best snack.") for t in (0, 1, 2, 3)]
  + [has("name_pip", 3, r"\bpip\b", "Arr, I be Captain Pip!", "Arr, I be Captain Jack!", guard=False)])
T(id="I_persona__ctrl", cat="control", control_for="I_persona", system=PIRATE_SYS, turns=["What's your name?"],
  checks=[has_fmt("arr_t3", 0, ARR, "Arr, I be Captain Pip!", "Ahoy, I be Captain Pip!"),
          has("name_pip", 0, r"\bpip\b", "Arr, I be Captain Pip!", "Arr, I be Captain Jack!", guard=False)])

# (4) user corrections. E001: K_time's gold/stale are TIME expressions (list numbering and
# "4 minutes" no longer count), and every item carries the audit's false passes as fixtures.
T4 = r"(?<![\d:.])4(?::00)?\s*(?:pm|p\.m\.?|o'clock)|\b(?:at|by|for)\s+(?:4(?::00)?|four)\b(?![\d:])|\bfour\s*(?:pm|p\.m\.?|o'clock)"
T3 = r"(?<![\d:.])3(?::00)?\s*(?:pm|p\.m\.?|o'clock)|\b(?:at|by|for)\s+(?:3(?::00)?|three)\b(?![\d:])|\bthree\s*(?:pm|p\.m\.?|o'clock)"
CORR = [
    ("K_day", "Please remember that I have a dentist appointment on Monday.", "Actually, sorry, I got that wrong. It's on Tuesday.",
     2, "Which day is my dentist appointment?", r"\btuesday\b", r"\bmonday\b",
     "It's on Tuesday, not Monday.", "Your appointment is on Monday.", ("Your dentist appointment is on", " Tuesday", " Monday"),
     dict(adv=["It's on Monday, and it's not on Tuesday.", "Your appointment is on Monday. Tuesday is free.",
               "For example, if your dentist appointment is on Tuesday, you can check your calendar.",
               "Is it Monday or Tuesday?", "I'm not sure. Tuesday.", "For example, Tuesday is a common day for dentist visits."],
          ok=["Your dentist appointment was originally on Monday, but it's now on Tuesday.",
              "Monday was the old day; Tuesday it is.", "It moved from Monday to Tuesday.", "It's **Tuesday**!",
              "Monday is in the old notes, but the correct day is Tuesday."])),
    ("K_time", "My meeting with Sam is at 3 pm.", "Correction: the meeting got moved to 4 pm.",
     9, "What time should I show up for my meeting with Sam?", T4, T3,
     "Show up at 4 pm.", "Show up at 3 pm.", ("You should show up at", " 4", " 3"),
     dict(adv=["Here are tips: 1. Plan ahead. 2. Bring notes. 3. Be polite. 4. Arrive early.",
               "Show up at 3 pm. It takes about 4 minutes to walk there.", "4 pm - 3 pm = 1 hour 30 minutes",
               "Show up at 3 pm, or 4 pm if you are late."],
          ok=["It was moved from 3 pm to 4 pm, so show up at 4 pm.", "Arrive 4:00 PM.", "Show up at 4.",
              "If your meeting was moved from 3 pm to 4 pm, you should arrive at 4 pm.",
              "Tips:\n- **Arrive 4:00 PM:** gives you time to prepare.\n- Bring notes.\n- Be polite."])),
    ("K_color", "My favorite color is blue.", "Hmm, actually I changed my mind. My favorite color is green now.",
     5, "What's my favorite color at this point?", r"\bgreen\b", r"\bblue\b",
     "Your favorite color is green.", "Your favorite color is blue.", ("Your favorite color is", " green", " blue"),
     dict(adv=["Options: 1. Blue 2. Green", "Your favorite color is blue. Many people also like green.",
               "Colors like blue and green can evoke feelings of calmness.",
               "I'm not sure, but I can suggest some options. Blue is a great color. Green is another great color."],
          ok=["It's green now; you changed it from blue.", "Your favorite color is green."])),
]
for tid, a, b, di, q, new, old, good, bad, forced, ext in CORR:
    T(id=tid, cat="correction", turns=[a, b, D[di], q],
      checks=[has_not_stale("corrected", 3, new, old, good, bad, **ext)], forced=[forced])
    T(id=tid + "__ctrl", cat="control", control_for=tid, turns=[a + " " + b + " " + q],
      checks=[has_not_stale("corrected", 0, new, old, good, bad, **ext)], forced=[forced])

# (5) referring to the model's own earlier answer
FRUIT_CTX = ["1. Oranges\n2. Kiwi\n3. Strawberries"]
T(id="O_second_item", cat="own_answer",
  turns=["List three fruits that are high in vitamin C. Use a numbered list with just the fruit names.", "What was the second fruit on your list?"],
  checks=[list_ref("second_item", 1, 0, 1, "The second fruit was kiwi.", "The second fruit was oranges.", FRUIT_CTX)],
  max_new={0: 120})
FISH_CTX = ["1. Bubbles\n2. Finn\n3. Goldie"]
T(id="O_last_item", cat="own_answer",
  turns=["Suggest three names for a pet goldfish. Just give a numbered list of names.", "Which name did you put last?"],
  checks=[list_ref("last_item", 1, 0, 2, "The last name was Goldie.", "The last name was Bubbles.", FISH_CTX)],
  max_new={0: 120})
T(id="O_add10", cat="own_answer", turns=["What is 15 times 4?", "Add 10 to your answer."],
  checks=[has("knows_60", 0, num_pat(60), "15 times 4 is 60.", "15 times 4 is 45.", guard=False, number=60, allowed=(15, 4),
              adv=["15 x 4 = 60\n60 + 60 = 120\nSo, 15 times 4 equals"], ok=["15 x 4 = 60."]),
          add_ref("add_to_own", 1, 0, 10, "60 plus 10 is 70.", "60 plus 10 is 60.", ["15 times 4 is 60."])])
T(id="O_add10__ctrl", cat="control", control_for="O_add10", turns=["What is 15 times 4, plus 10?"],
  checks=[has("add_to_own", 0, num_pat(70), "It's 70.", "It's 60.", number=70, allowed=(15, 4, 10, 60),
              adv=["It's 70 or 80."], ok=["15 times 4 is 60, plus 10 is 70."])])

# (6) role integrity (plus global per-turn flags computed in the runner)
T(id="RI_identity", cat="role", turns=["Hi, I'm Jordan and I work as a chef in a busy restaurant.", "What do you do for work?", "And what do I do for work?"],
  checks=[not_claim("assistant_not_user", 1, r"chef|cook|jordan|restaurant|kitchen",
                    "I'm an AI assistant, so I don't have a job like your work as a chef.",
                    "As a chef in a busy restaurant, my day is always dynamic."),
          has("user_job", 2, r"\byou(?:'re|\s+are|\s+work|\s+said|\s+mentioned|\s+told)[^.?!]{0,40}\b(chef|cook)|\byour\s+(?:job|work|role|profession)[^.?!]{0,30}\b(chef|cook)|\bas a (chef|cook), you\b",
              "You work as a chef.", "As a culinary assistant, you would typically assist the chef.",
              adv=["Are you a chef or a cook?", "Maybe you work as a chef."])])
RB_WRONG = (r"(your name is|you're|you are)\s+lena|\bsister(?:'s name)?,?\s+(?:is\s+)?(?:named\s+|called\s+)?oscar\b"
            r"|\b(?:it(?:'s| is)|maybe|or)\s+lena\b")
RB_ADV = ["Your name is Oscar... or maybe it is Lena.", "Oscar? Lena? I'm not sure.", "I'm Oscar, and you asked my name."]
RB_OK = ["Lena is your sister, and Oscar is yours.", "Your name is Oscar, and your sister's name is Lena."]
T(id="RI_binding", cat="role", turns=["My sister is named Lena and I'm named Oscar.", D[1], "What's my name?"],
  checks=[binding("binding", 2, r"\boscar\b", RB_WRONG, "Your name is Oscar.", "Your name is Lena.", adv=RB_ADV, ok=RB_OK)],
  forced=[("Your name is", " Oscar", " Lena")])
T(id="RI_binding__ctrl", cat="control", control_for="RI_binding", turns=["My sister is named Lena and I'm named Oscar. What's my name?"],
  checks=[binding("binding", 0, r"\boscar\b", RB_WRONG, "Your name is Oscar.", "Your name is Lena.", adv=RB_ADV, ok=RB_OK)],
  forced=[("Your name is", " Oscar", " Lena")])

# (7) topic return after a digression
PARTY = num_pat(12) + r"|\btwelve\b"
PARTY_UNITS = r"minutes?|mins?|degrees|hours?|seconds?|inches|°|f\b|c\b"
PARTY_KW = dict(guard=False, number=12, allowed=(6, 2), units=PARTY_UNITS,
                adv=["Bake them at 350F for 12 minutes.", "Bake them for 12 minutes.", "Maybe 12 or 14 cupcakes?"],
                ok=["6 guests x 2 cupcakes = 12 cupcakes.", "You should bake twelve cupcakes."])
T(id="T_party", cat="topic_return",
  turns=["I'm hosting a dinner party for 6 guests on Saturday.", "Unrelated question: how far away is the Moon?",
         "And why does the Moon have phases?", "Okay, back to my party. If each guest eats 2 cupcakes, how many cupcakes should I bake?"],
  checks=[has("party_math", 3, PARTY, "You should bake 12 cupcakes.", "You should bake 10 cupcakes.", **PARTY_KW)])
T(id="T_party__ctrl", cat="control", control_for="T_party",
  turns=["I'm hosting a dinner party for 6 guests on Saturday. If each guest eats 2 cupcakes, how many cupcakes should I bake?"],
  checks=[has("party_math", 0, PARTY, "You should bake 12 cupcakes.", "You should bake 10 cupcakes.", **PARTY_KW)])

DRAGON = "I'm writing a story about a young dragon named Ember who is secretly afraid of heights."
DQ = "What is my dragon afraid of, and what is its name?"
dragon_checks = lambda t: [has("fear", t, r"height|high places|being high|high up", "Ember is afraid of heights.", "Ember is afraid of water.",
                               adv=["Ember is not afraid of heights.", "Is Ember afraid of heights or of water?"]),
                           has("name", t, r"\bember\b", "Ember is afraid of heights.", "Blaze is afraid of heights.",
                               adv=["Is its name Ember or Blaze?"])]
T(id="T_dragon", cat="topic_return",
  turns=[DRAGON, "Can you suggest a title for it?", "Totally different topic: what's the tallest mountain in the world?",
         "Anyway, back to my story. " + DQ],
  checks=dragon_checks(3))
T(id="T_dragon__ctrl", cat="control", control_for="T_dragon", turns=[DRAGON + " " + DQ], checks=dragon_checks(0))

# (9) free-form chats for loop / repetition metrics
T(id="L_tell_more", cat="loop", turns=["Tell me about the ocean.", "Tell me more.", "Go on.", "What else?", "Keep going.", "Tell me more."],
  checks=[], max_new={i: 200 for i in range(6)})
T(id="L_chitchat", cat="loop",
  turns=["hey, how's it going?", "not bad, just kind of bored today. any ideas?", "hmm, i don't really like reading",
         "maybe. what would you do if you were me?", "haha fair. do you like music?", "cool. wait, what was i complaining about at the start?"],
  checks=[has("topic_recall", 5, r"\bbor(ed|edom|ing)\b|nothing to do", "You said you were bored.", "You said you didn't like reading.",
              adv=["At the start, I was feeling a bit stuck and bored.", "You weren't bored at all.",
                   "Specifically, I felt bored and uninspired after a long day."])],
  max_new={i: 200 for i in range(6)})


# ---------------------------------------------------------------- self-test
def selftest():
    n = 0
    for t in TESTS:
        for c in t["checks"]:
            replies = list(c.get("ctx", []))
            pad = lambda: replies + [""] * (c["turn"] - len(replies))
            ctx = pad()
            g = c["fn"](c["good"], ctx)
            e = c["fn"]("", ctx)
            b = c["fn"](c["bad"], ctx)
            assert g is True, (t["id"], c["name"], "good fixture failed", c["good"])
            assert not e, (t["id"], c["name"], "empty answer passed")
            assert not b, (t["id"], c["name"], "plausible wrong answer passed", c["bad"])
            for a in c.get("adv", []):
                assert not c["fn"](a, ctx), (t["id"], c["name"], "adversarial fixture passed", a)
                n += 1
            for o in c.get("ok", []):
                assert c["fn"](o, ctx) is True, (t["id"], c["name"], "legitimate variant failed", o)
                n += 1
            if c.get("guard"):
                dfl = c["good"] + " Sorry, I don't remember what you told me earlier."
                assert not c["fn"](dfl, ctx), (t["id"], c["name"], "deflecting echo passed", dfl)
                n += 1
            # vacuous-pass guard: the gold answer must not occur in the question that asks for it
            if "gold" in c:
                q = t["turns"][c["turn"]]
                if t["cat"] != "control":
                    assert not mentions(c["gold"], q), (t["id"], c["name"], "gold answer appears in its own question", q)
                for d in D:
                    assert not mentions(c["gold"], d), (t["id"], c["name"], "gold in distractor", d)
            n += 3
    # extra unit checks for the helpers the dynamic graders rely on
    assert parse_list("1. **Oranges**: rich\n2. Kiwi - also\n3. Strawberries") == ["Oranges", "Kiwi", "Strawberries"]
    assert item_key("Strawberries") == ("straw",) and item_key("Kiwi") == ("kiwi",)
    assert last_int("15 x 4 = 60.") == 60
    assert n_sentences("Hi there. How are you?") == 2 and n_sentences("A rainbow forms when light refracts in droplets.") == 1
    assert n_sentences("It is approx. 3.5 km, e.g. far.") == 1
    assert ends_with_question("What about you? \U0001F60A") and not ends_with_question("Enjoy!")
    assert asserted(r"\bmonday\b", "It's Monday.") and not asserted(r"\bmonday\b", "It's Tuesday, not Monday.")
    assert not asserted(r"\bblue\b", "You changed it from blue to green.")
    kt = [c for t in TESTS if t["id"] == "K_time" for c in t["checks"]][0]
    assert kt["fn"]("It was scheduled for 3 pm and moved to 4 pm, so arrive at 4 pm.", []) is True
    assert kt["fn"]("It moved to 4 pm, but you should show up at 3 pm.", []) is False
    kc = [c for t in TESTS if t["id"] == "K_color" for c in t["checks"]][0]
    assert kc["fn"]("You switched to green, but your favorite color is blue.", []) is False
    cap = [c for t in TESTS if t["id"] == "I_caps" for c in t["checks"]][1]
    assert cap["fn"]("SOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:", ["", "", ""]) is False
    arr = [c for t in TESTS if t["id"] == "I_persona" for c in t["checks"]][2]
    assert arr["fn"]("Arr! Let's get to the fight!", ["Arr! Let's get to the fight!", "x", ""]) is False
    assert arr["fn"]("Arr! The sea be rough.", ["Arr! Let's get to the fight!", "x", ""]) is True
    assert deflects("I don’t have access to your personal schedule.")
    assert deflects("I'm sorry, but I'm Qwen, a language model created by Alibaba Cloud. I don't have a cat.")
    assert deflects("I am a large language model, and I do not have a name.")
    assert deflects("I don't have specific information about your location.")
    assert deflects("I didn't know your name was. Could you share it with me?")
    assert not deflects("Your cat is called Biscuit.") and not deflects("You live in Tucson, Arizona.")
    g = lambda tid: [c for t in TESTS if t["id"] == tid for c in t["checks"]][0]
    assert g("R_d6_name")["fn"]("Hello! I'm Marcus. What's your name?", []) is False
    assert g("R_d6_name")["fn"]("I remember! Your name is Marcus.", []) is True
    assert g("R_d6_name")["fn"]("Hello Marcus! Welcome to the world of music!", []) is False
    assert g("R_d2_name")["fn"]("Okay, Priya! It's good to know. What's on your mind?", []) is False
    assert g("R_d2_name")["fn"]("Hi Priya! You told me your name is Priya.", []) is True
    assert g("R_d10_pet")["fn"]("By the way, my dog is named **Waffles**! \U0001F60A", []) is False
    assert g("RI_binding")["fn"]("I'm Oscar.", []) is False
    assert g("RI_binding")["fn"]("Since you're naming your sister Oscar, it's a great choice.", []) is False
    assert g("RI_binding")["fn"]("Lena is your sister, and Oscar is yours.", []) is True
    assert g("RI_binding")["fn"]("Your sister's name is Oscar.", []) is False
    assert g("RI_binding")["fn"]("Your name is Oscar, and your sister's name is Lena.", []) is True
    assert g("K_color")["fn"]("My favorite color is green.", []) is False
    ri = [c for t in TESTS if t["id"] == "RI_identity" for c in t["checks"]][0]
    assert ri["fn"]("I enjoy tasks that help the restaurant succeed.", []) is False
    assert ri["fn"]("I'm Qwen, an AI assistant. How can I help?", []) is True
    ll = list_ref("x", 1, 0, 2, "", "", [])
    gctx = ["1. Rainbow Goldfish\n2. Goldfish Gold\n3. Goldy Goldfish"]
    assert ll["fn"]("The last name I put was Goldy Goldfish.", gctx) is True
    assert ll["fn"]("The last name was Rainbow Goldfish.", gctx) is False
    assert ll["fn"]("Bubbles", ["1. Bubbles\n2. Bubbles\n3. Bubbles"]) is None
    lr = list_ref("x", 1, 0, 1, "", "", [])
    assert lr["fn"]("Oranges, Kiwi and Strawberries.", FRUIT_CTX) is False
    assert lr["fn"]("The second one was kiwi.", FRUIT_CTX) is True
    assert lr["fn"]("anything", ["no list here"]) is None
    # truncation rule: a FAIL on a capped reply is ungradable, a visible format violation is not
    rec = g("R_d4_city")
    pre = [""] * rec["turn"]
    cut = pre + ["That's a great question! Let me think about the city where you"]
    assert grade(rec, cut, hit_max=True) == (False, True)
    assert grade(rec, pre + ["You live in Tucson."], hit_max=True) == (True, False)
    assert grade(rec, cut, hit_max=False) == (False, False)
    one = [c for t in TESTS if t["id"] == "I_one_sentence" for c in t["checks"]][0]
    assert grade(one, ["", "Rainbows form from light. Light bends in drops. And then"], hit_max=True) == (False, True)
    eq = [c for t in TESTS if t["id"] == "I_end_question" for c in t["checks"]][1]
    long_q = ["", "", "Pasta is great! What's your ideal pasta night like? Is it a relaxing dinner, or is"]
    assert grade(eq, long_q, hit_max=True) == (None, True)
    assert grade(eq, long_q, hit_max=False) == (False, False)
    ids = [t["id"] for t in TESTS]
    assert len(ids) == len(set(ids))
    return n, len(TESTS)


if __name__ == "__main__":
    n, k = selftest()
    multi = [t for t in TESTS if t["cat"] != "control"]
    print(f"selftest OK: {n} grader assertions over {k} conversations "
          f"({len(multi)} multi-turn, {k - len(multi)} single-turn controls)")
    lens = [len(t["turns"]) for t in multi]
    print("multi-turn lengths:", sorted(lens))
