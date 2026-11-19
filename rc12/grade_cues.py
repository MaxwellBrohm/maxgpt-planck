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
              None, so its matching is exactly V.value_re's. Assertion rules apply to the match as to any mention
LOCK STEP A2b (2026-11-19; rc12/notes.txt STEP 16A2B; prereg s6 (x); fixtures_a2b.py, mutants_a2b.py, audit_a2b.py).
LOOKUP's X probe only (graders.g_abs, on a LOOKUP record: its one ABS probe), never ABS_CUE, so none of these sees
it: another family, an OOD-H probe, oodh/gate.py's D12 check, mutation_graders' question check:
  LOOKUP_ABS  the not-found replies the Planck pipeline teaches (pipeline/SPEC.txt S9 "empty result reported as not
              found"; its accepted forms, pipeline/lexicons.NOT_FOUND_RE): "I couldn't find X", "I could not / can't /
              cannot / didn't / did not / don't / do not / was unable to / wasn't able to / was not able to (seem
              to) find, locate, see or spot X" (LOOKUP_FIND; the subject "I", "we", "the lookup / search" or "my
              lookup / search", then maybe one of just, still, really, honestly, simply; or no subject at a clause
              start, the text's start or after . ! ? ; : or a comma: "Couldn't find X.", "Sorry, couldn't find
              anything"), "Nothing came up / turned up / showed up", "nothing (was) found", "turned up nothing",
              "found nothing", "came up empty", "not found", "wasn't found", "(could) not / cannot be found", "No
              results / match / entry / row / record / information / luck" (a word start: "No matches", "No
              entries"), "missing from", "absent from"
  LOOKUP_ONLY "Only the <table> (up to 2 words) has / lists / includes / contains / shows" followed by the probe's
              key ("the" may come first), "it", "that", "a row", "an entry" or "a line" (fix round, 2026-11-19: "Only
              the old one has changed" says nothing about X), and "only on / in the <table>" ("X is only on the old
              rota"); each counted only when <table> is "other" or a table of the record other than the one asked
              (lookup_tables): "Only the new one has X" asked of the new table says X is there
  lookup_cue  a match counts unless negated in the 3 word tokens before it within its clause (from the last comma,
              semicolon or colon of its sentence: "I'm afraid not, I couldn't find X" is no negation of the cue;
              lookup_negated reads grade_text.negated_before there: "X isn't missing from", "Not only the old one has
              X"), "nothing" and "nobody" counting as negations too ("Nothing is missing from the new rota"), and none
              counts when the probe's question holds one, so an echo of the question never abstains through them
              (G-ABS has no echo clause, D12; test_gens refuses a LOOKUP question holding any of them)
  FIND_FRAME  LOOKUP_FIND and "I found nothing" are taken out of the reply before a4_voice on LOOKUP ("I couldn't
              find X" binds nothing): only the frame goes, so "I am on X myself" or "My X is free" beside it still
              fails"""
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
LOOKUP_FIND = (r"(?:\b(?:i|we|(?:the|my) (?:lookup|search))(?: (?:just|still|really|honestly|simply))? "
               r"|(?:^|(?<=[.!?;:,]))\s*)(?:couldn't|could not|can't|cannot|didn't|did not|don't|do not|was unable to"
               r"|wasn't able to|was not able to)"
               r"(?: seem to)? (?:find|locate|see|spot)\b")
LOOKUP_ABS = (LOOKUP_FIND + r"|\bnothing (?:came|turned|showed) up\b|\bnothing (?:was )?found\b"
              r"|\b(?:turned up|found) nothing\b|\bcame up empty\b|\b(?:not|cannot|\w+n't)(?: be)? found\b"
              r"|\bno (?:result|match|entr|row|record|information|luck)|\b(?:missing|absent) from\b")
LOOKUP_ONLY = r"\bonly the (\w+)(?: \w+){0,2} (?:has|lists|includes|contains|shows) "
ONLY_OBJ = r"(?:it|that|a row|an entry|a line|(?:the )?KEY)\b"
LOOKUP_ON = r"\bonly (?:on|in) the (\w+)\b"
LOOKUP_FRAME = LOOKUP_FIND + r"|\bi found nothing\b"
FIND_FRAME, _LOOKUP, _ON = (re.compile(x, re.I) for x in (LOOKUP_FRAME, LOOKUP_ABS, LOOKUP_ON))
CLAUSE = re.compile(r"[,;:]")
NEG_MORE = re.compile(r"\b(?:nothing|nobody)\b", re.I)
_ONLY = {}
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


def lookup_tables(rec, probe):
    """the record's tables other than the one the probe asks about (its holder "table:<name>"), lowercased."""
    asked = str(probe.get("holder") or "").split(":", 1)[-1].lower()
    objs = {f["object"].lower() for t in rec["turns"] for f in t.get("facts") or [] if f.get("role") == "table"}
    return objs - {asked}


def only_rx(key):
    """LOOKUP_ONLY with its object (ONLY_OBJ, the key in any case), compiled once per key; no key: never matches."""
    if key not in _ONLY:
        _ONLY[key] = re.compile(LOOKUP_ONLY + ONLY_OBJ.replace("KEY", re.escape(key) if key else "(?!)"), re.I)
    return _ONLY[key]


def lookup_negated(text, pos):
    """a negation (grade_text.negated_before, or "nothing" / "nobody") in the 3 word tokens before pos, read from the
    last comma, semicolon or colon of its sentence (module docstring)."""
    import grade_text as T                    # late, as report_frame: grade_text imports this module
    s0 = T.sentence_span(text, pos)[0]
    s0 = max([s0] + [c.end() for c in CLAUSE.finditer(text, s0, pos)])
    window = " ".join(T.toks(text[s0:pos])[-T.P["neg_window"]:])
    return T.negated_before(text, s0, pos) or ("neg" not in T.OFF and NEG_MORE.search(window) is not None)


def lookup_hit(text, others, key):
    """text holds a LOOKUP_ABS match, or a LOOKUP_ONLY (with its object) or LOOKUP_ON one naming "other" or a table
    in others, not negated."""
    ok = others | {"other"}
    hits = list(_LOOKUP.finditer(text)) + [m for m in only_rx(key).finditer(text) if m.group(1).lower() in ok] \
        + [m for m in _ON.finditer(text) if m.group(1).lower() in ok]
    return any(not lookup_negated(text, m.start()) for m in hits)


def lookup_any(text):
    """any LOOKUP_ABS, LOOKUP_ON or bare LOOKUP_ONLY match, negated or not, whatever the table: test_gens' content
    rule for LOOKUP questions (fix round, 2026-11-19)."""
    return any(re.search(x, text, re.I) for x in (LOOKUP_ABS, LOOKUP_ONLY, LOOKUP_ON))


def lookup_cue(text, rec, probe):
    """the A2b abstain cue of a LOOKUP X reply (module docstring); text already norm()-ed."""
    import grade_text as T
    others, key = lookup_tables(rec, probe), probe.get("key")
    return not lookup_hit(T.norm(probe.get("question")), others, key) and lookup_hit(text, others, key)
