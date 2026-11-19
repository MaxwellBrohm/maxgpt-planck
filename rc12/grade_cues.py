"""RC-12 grader cue additions from the step 5 audit (audit_graders.py found them on real probes). Kept apart so
grade_text.py stays under the file limit; grade_text.HEDGE and grade_text.ABS_CUE splice these in.
  HEDGE_MORE  epistemic hedges the first list let through as confident assertions: "I'd guess X", "If I remember
              right, X", "X, as far as I know", "It should be X", "X, unless it changed", "I can't be certain,
              but X" (each passed all 392 value probes before the fix)
  ABS_MORE    ordinary abstentions the first list failed: "I don't think you've told me that", "That hasn't come
              up yet", "That list skips Thursday", "Nobody is down for Thursday there" (failed every ABS probe)
  near_hedge  a hedge that stands as its own short sentence next to the answer ("Maybe. It is X." / "X.
              Probably."): the sentence-scoped hedge rule could not see it
LOCK STEP A2 (2026-10-19; the OOD-H reviews' grader findings, oodh/DESIGN.txt s14 and s15; rc12/notes.txt STEP 16;
prereg s6). Each piece is pinned by a fixture (fixtures_a2.py) and a mutant (mutants_graders.py):
  ABS_NOT_GIVEN  "You never gave me that", "You didn't specify", "You never specified that", "You didn't provide
              that", "You never picked one", "You haven't chosen one yet", "You didn't name one", "You didn't rate
              it", "You haven't ordered anything yet", "You haven't decided": joins ABS_MORE (failed every ABS probe)
  ABS_DENIAL  the bare denial "No, you didn't." joins ABS_MORE
  asked_if    a condition word right after "you (first) asked (me)" is the user's reported question ("You asked if
              X"), not a condition, as "You asked whether X" never was one; "They asked if X", "I wonder if X" stay
              conditions ("I asked if X" also fails v4 as the model speaking as the user)
  reported_hedge  "should be" or "seem(s)" after a you-report frame (grade_voice.YOU_REP: "You said", "You told me",
              "You mentioned", "According to you" ...) and before the value, with no comma, semicolon or "but" between
              the frame and the value, is the user's wording reported back ("You said it should be X"), not the
              reply's own hedge. Every other hedge counts as before: "You said it might be X", "It should be X",
              "You said so, but it should be X", "You said X, and it seems right" still fail, and so does a frame
              after the hedge ("It should be X, you said."). Review r2 (2026-11-19): "so" is a break too ("You told
              me nothing so it should be X"), and the exemption is refused when the frame or the text from it to the
              value holds a negation cue (grade_text.NEG_CUE: "You never wrote it should be X", "You said you
              weren't sure it should be X") or the text after the frame a first-person word (grade_text.FIRST_ANY:
              "You said I should be X"), so such a reply fails as it did before A2
  reported_question  a sentence that is ONE quoted question opened by "you asked / said (me)" ('You asked, "Can I
              bring X?"') is not a question for a mention inside the quote; 'Is it "X?"' and 'They asked, "Is it
              X?"' stay questions
  neg_value_re  (2026-11-19, A2 (v)) a value that holds a negation ("is not vegan", "doesn't eat X", "not spicy")
              also matches with that negation in the other forms of its group (NEG_GROUPS: "is not", "isn't", "'s
              not"; "cannot", "can not", "can't" ...), and a "not" that opens the value also as "n't" on the word
              before it ("She isn't spicy" for "not spicy"). Only the auxiliary of the value's own negation: "wasn't
              X" is not "is not X". A value without a negation (every RC-12 value: one pool word or a holder) gets
              None, so its matching is exactly V.value_re's. Assertion rules apply to the match as to any mention"""
import re

HEDGE_MORE = (r"guess|can't be (?:sure|certain)|cannot be (?:sure|certain)|if i (?:remember|recall)|if i'm not mistaken"
              r"|if memory serves|sounds (?:about )?right|as far as i (?:know|remember|recall|can tell)"
              r"|from what i (?:remember|recall|know)|i (?:would |'d )?assume|seems?|apparently|should be"
              r"|i'd say|i would say|if that's (?:still )?(?:right|correct)|unless")
ABS_NOT_GIVEN = (r"\byou (?:haven't|have not|didn't|did not|never) (?:yet )?(?:gave|specif(?:y|ied)|provided?|named?"
                 r"|rated?|pick(?:ed)?|cho(?:se|sen|ose)|order(?:ed)?|decided?)\b")
ABS_DENIAL = r"\bno,? you (?:didn't|did not|haven't|have not)\b"
ABS_MORE = (r"\bi (?:don't|do not) (?:think|believe) you(?:'ve| have)? (?:ever )?(?:told|mentioned|said|shared)\b"
            r"|\b(?:hasn't|has not|didn't|did not|never) (?:come up|been mentioned)\b|\b(?:isn't|is not|not) covered\b"
            r"|\bskips\b|\b(?:nobody|no one)(?: is|'s)? (?:down|scheduled|assigned)\b"
            r"|" + ABS_NOT_GIVEN + r"|" + ABS_DENIAL)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
ASKED = re.compile(r"\byou\s+(?:first\s+)?asked(?:\s+me)?\s+$", re.I)
REPORT_HEDGE = re.compile(r"should be|seems?", re.I)
REPORT_BREAK = re.compile(r"[,;]|\bbut\b|\bso\b", re.I)
QUOTE_FRAME = re.compile(r"\byou\s+(?:asked|said)(?:\s+me)?\s*[,:]?\s*$", re.I)
QUOTED_Q = re.compile(r'(?P<pre>[^"]*)"(?P<q>[^"]*\?)"[\s.!]*')
_FRAME = []
NEG_GROUPS = (("is not", "isn't", "'s not"), ("are not", "aren't", "'re not"), ("am not", "'m not"),
              ("was not", "wasn't"), ("were not", "weren't"), ("do not", "don't"), ("does not", "doesn't"),
              ("did not", "didn't"), ("has not", "hasn't"), ("have not", "haven't", "'ve not"), ("had not", "hadn't"),
              ("will not", "won't", "'ll not"), ("would not", "wouldn't", "'d not"), ("cannot", "can not", "can't"),
              ("could not", "couldn't"), ("should not", "shouldn't"), ("must not", "mustn't"), ("need not", "needn't"))
GROUP_OF = {f: g for g in NEG_GROUPS for f in g}
_WORDS = sorted((f for f in GROUP_OF if f[0] != "'"), key=len, reverse=True)
_CLITICS = sorted((f for f in GROUP_OF if f[0] == "'"), key=len, reverse=True)
NEG_IN_VALUE = re.compile(r"(?<![A-Za-z0-9'])(?:" + "|".join(map(re.escape, _WORDS)) + r"|not)(?![A-Za-z0-9])"
                          r"|(?<=[A-Za-z])(?:" + "|".join(map(re.escape, _CLITICS)) + r")(?![A-Za-z0-9])", re.I)
_NEG = {}


def _alts(forms):
    """the forms as alternatives, matched in any case (review r1, 2026-11-19): a capitalized value is matched
    case-sensitively (V.value_re), and its own "Can't" or "Not" must still match as written"""
    return "(?i:" + "|".join(map(re.escape, forms)) + ")"


def neg_value_re(v, flags):
    """the matcher of value v with its negations in every form (module docstring); None when v holds none."""
    key = (v, flags)
    if key not in _NEG:
        s, parts, last = v.replace("’", "'"), [r"(?<![A-Za-z0-9])"], 0
        for m in NEG_IN_VALUE.finditer(s):
            group = GROUP_OF.get(m.group(0).lower())
            words = [f for f in group if f[0] != "'"] if group else ["not"]
            clitics = [f for f in group if f[0] == "'"] if group else []
            core = s[last:m.start()].rstrip()
            if m.start() == 0:                        # the value opens with the negation: a clitic needs a word
                parts[0] = ""                         # before it, a word form a boundary
                clitics += [] if group else ["n't"]
                parts.append(r"(?:(?<![A-Za-z0-9])(?:" + _alts(words) + ")"
                             + (r"|(?<=[A-Za-z])(?:" + _alts(clitics) + ")" if clitics else "") + ")")
            else:
                parts.append(re.escape(core) + "(?: (?:" + _alts(words) + ")"
                             + ("|(?:" + _alts(clitics) + ")" if clitics else "") + ")")
            last = m.end()
        _NEG[key] = re.compile("".join(parts) + re.escape(s[last:]) + r"(?![A-Za-z0-9])", flags) \
            if len(parts) > 1 else None
    return _NEG[key]


def near_hedge(text, s0, s1, hedge_rx, max_words=3):
    """True when the sentence right before [s0, s1) or right after it is short (<= max_words) and a hedge."""
    before = [x for x in SPLIT.split(text[:s0]) if x.strip()]
    after = [x for x in SPLIT.split(text[s1:]) if x.strip()]
    for sent in before[-1:] + after[:1]:
        if len(re.findall(r"[A-Za-z0-9']+", sent)) <= max_words and hedge_rx.search(sent):
            return True
    return False


def asked_if(before, c):
    """the condition-word match c (in before) follows "you (first) asked (me)": a reported question."""
    return ASKED.search(before[:c.start()]) is not None


def report_frame():
    """grade_voice.YOU_REP compiled once (imported late: grade_voice imports grade_text, which imports this)."""
    if not _FRAME:
        import grade_voice as GV
        _FRAME.append(re.compile(r"\b(?:" + GV.YOU_REP + r")\b", re.I))
    return _FRAME[0]


def reported_hedge(sent, h, pos):
    """hedge match h of sent is the user's wording reported back before the mention at pos (module docstring)."""
    if h.end() > pos or not REPORT_HEDGE.fullmatch(h.group(0)):
        return False
    import grade_text as T                    # late, as report_frame: grade_text imports this module
    frames = list(report_frame().finditer(sent[:h.start()]))
    if not frames:
        return False
    f = frames[-1]
    return (REPORT_BREAK.search(sent[f.end():pos]) is None and T.NEG_CUE.search(sent[f.start():pos]) is None
            and T.FIRST_ANY.search(sent[f.end():pos]) is None)


def own_hedge(sent, pos, hedge_rx, exempt=True):
    """True when sent holds a hedge that is the reply's own (every hedge but a reported_hedge when exempt)."""
    return any(not (exempt and reported_hedge(sent, h, pos)) for h in hedge_rx.finditer(sent))


def reported_question(sent, pos):
    """sent is one quoted question ending in "?" that "you asked / said" opens, and the mention at pos is inside it."""
    m = QUOTED_Q.fullmatch(sent)
    return bool(m) and m.start("q") <= pos < m.end("q") and QUOTE_FRAME.search(m.group("pre")) is not None
