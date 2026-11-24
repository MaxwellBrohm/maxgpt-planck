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
              "found nothing" (these four only when "for", "on", "in", "about", a mark or the end of the text comes
              next, END: "Nothing found for X", not "I found nothing wrong" or "Nothing was found wrong with it"),
              "came up empty", the passive found forms: "not found" (not with have, has, had, having or 've before it
              in its sentence, A2d (d): "I have not found any gap" and "The office has also not found a gap" are no
              cue), "not be found", "cannot be found", "wasn't / weren't / isn't / aren't found", "couldn't / can't
              / won't be found" ("I haven't found any gap" is no cue), "No results / match / entry / row / record /
              information / luck" (a word start: "No matches", "No entries"; A2d (g) what may follow), "missing
              from", "absent from" (A2d (j): not before the other table)
  LOOKUP_ONLY "only the <table> (up to 2 more words) has / lists / includes / contains / shows " and ONLY_ON "only on /
              in the": the broad forms, read only by lookup_any (test_gens' content rule for LOOKUP questions); the
              grader reads the shapes of A2d below
  lookup_cue  none counts when the probe's question holds one that counts (read as a reply is), so an echo of the
              question never abstains through them (G-ABS has no echo clause, D12; test_gens refuses a LOOKUP question
              holding any of them, lookup_any). The review rounds of 2026-11-22 and 23 (rc12/notes.txt STEP 16A2B-R2
              and R3) read negations through a 3-word window, asides, subject phrases and a perfect check; A2d replaced
              them
LOCK STEP A2d (2026-11-23; rc12/notes.txt STEP 16A2D and its fix round STEP 16A2D-R; prereg s6 (x)). Each round of
rules about how a negation voids a cue closed its findings and opened new wrong passes, and a grader that freezes at
the lock should fail closed, so the cues are read by a rule a reader can check by eye (lookup_hit, cue_counts,
voided):
  (a) the reply is read with * _ and backticks taken out (EMPHASIS), and for (c) also with each of them read as a
      space (fix round: "X is**not** missing", "X is not_really missing" are negated)
  (b) openings at the very start of a sentence, one or more, each followed by a mark, are ignored (OPENERS: I'm / I
      am afraid not, afraid not, sadly not, unfortunately not, (I) guess not, probably not, apparently not, (I) don't
      / do not think so, not really, not sure, not this time, no, sorry); "None" is no opening ("None, it seems, are
      missing" says nothing is). The mark is one of , ; : . en dash, em dash, ellipsis, or a hyphen that no letter
      follows directly (fix round: "No-one is missing" is no opening; a quote or bracket is no opening's mark)
  (c) a negation word anywhere in the cue's sentence before the cue voids it (LOOKUP_NEG: not, n't, no, never,
      nothing, nobody, none, nowhere, neither, nor, without, cannot, hardly, scarcely, barely, far from, anything but,
      instead of, rather than, and n't without its apostrophe after is, are, was, were, do, does, did, has, have, had,
      ca, could, wo; fix round: noone, zero, nary, nada, naught, nothin, n't without its apostrophe after ai, should,
      would, must, need, might, n't with an apostrophe look-alike (acute accent, modifier letter apostrophe, prime,
      fullwidth apostrophe, reversed comma) and 'nt); the cue's own words do not count. A sentence ends at . ! or ?
      followed by a space, a line break, a quote and a space, or the end, and at a line break, as grade_text.SENT_END
      has it, but a run of dots does not end one (BREAK). A cue whose sentence holds a question mark anywhere (QMARK:
      ?, the fullwidth one, the interrobang) does not count, and no cue counts in a reply that holds struck-through
      text (STRIKE: ~, <del, <strike, in any case)
  (d) have, has, had, having or 've anywhere before the passive "not (be) found" in its sentence voids it (HAVE)
  (e) ONLY and ON in their simplest shape only: "only the <table> (up to two more words) has / lists / includes <the
      key (maybe plural) / it / that>" (ONLY_SHAPE) and "<the key (maybe plural) / it> is / are / was / were / 's only
      on / in the <table> (up to two more words)" (LOOKUP_ON), each followed by . ! ; a line break or the end
      (SHAPE_END: not ? and not a comma, "and" or "but"), and <table> "other" or a table of the record other than the
      one asked (lookup_tables: "Only the new one has X" asked of the new table says X is there)
  (g) "no result / match / entr / row / record / information / luck" and "nothing came / turned / showed up" count only
      when for, on, in, about, of, with, a mark or the end comes next, maybe after "found" (QUANT, QUANT_END: "No
      results found for X"; not "No rows are missing", "Nothing came up missing"); and none of these, "nothing (was)
      found", "turned up nothing" or "found nothing" (NONE_CUE) counts when is, are, was, were, missing, absent,
      blank or wrong follows it in its sentence (PREDICATE, fix round: "No records on the new one are missing",
      "Nothing found in the new one looks wrong")
  (j) "missing from" / "absent from" with "other" or another table of the record as either of the next two words
      (any spaces, marks, quotes or brackets around them) is no cue (FROM_TABLE: "X is missing from the old one"
      asked of the new table; fix round: "... from the 'old' one", "... from (the old one)", "... from that old
      one", "... from old one")
  Right replies the rule fails are recorded in prereg s6 (x) (fixtures_a2b.X_CONSERVATIVE_A2D and its fix round's
  X_CONSERVATIVE_A2D_FR: "Nobody - X is missing", "X, not others, is missing", "Only the old one has X on it", "It'll
  be only on the old one", "No results for X; it is only on the old one" ...)
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
END = r"(?=\s*(?:for|on|in|about)\b|\s*[^\w\s']|\s*$)"
LOOKUP_ABS = (LOOKUP_FIND + r"|\bnothing (?:came|turned|showed) up\b|\bnothing (?:was )?found\b" + END
              + r"|\b(?:turned up|found) nothing\b" + END + r"|\bcame up empty\b"
              r"|\b(?:(?P<nf>not)(?: be)?|cannot be|(?:was|were|is|are)n't"
              r"|(?:could|ca|wo)n't be) found\b"
              r"|\bno (?:result|match|entr|row|record|information|luck)|\b(?:missing|absent) from\b")
LOOKUP_ONLY = r"\bonly the (\w+)(?: \w+){0,2} (?:has|lists|includes|contains|shows) "   # lookup_any's broad form
ONLY_ON = r"only (?:on|in) the"
LOOKUP_FRAME = LOOKUP_FIND + r"|\bi found nothing\b"
FIND_FRAME, _LOOKUP = (re.compile(x, re.I) for x in (LOOKUP_FRAME, LOOKUP_ABS))
SHAPE_END = r"(?=[ \t]*(?:[.!;\n]|$))"
ONLY_SHAPE = r"\bonly the (\w+)(?: \w+){0,2} (?:has|lists|includes) (?:KEY(?:s)?|it|that)" + SHAPE_END
LOOKUP_ON = r"\b(?:KEY(?:s)?|it)(?: (?:is|are|was|were)|'s) only (?:on|in) the (?P<t>\w+)(?: \w+){0,2}" + SHAPE_END
EMPHASIS = re.compile(r"[*_`]")
BREAK = re.compile(r"(?:(?<!\.)\.|[!?])(?=\s|[\"']\s|[\"']?$)|\n")
QMARK = re.compile(r"[?\uff1f\u203d]")
STRIKE = re.compile(r"~|</?(?:del|strike)\b", re.I)
OPENERS = re.compile(r"(?:\s*(?:(?:(?:i'm |i am )?afraid|sadly|unfortunately|(?:i )?guess|probably|apparently) not"
                     r"|(?:i )?(?:don't|do not) think so|not (?:really|sure|this time)|no|sorry)"
                     r"\s*(?:[,;:.\u2013\u2014\u2026]|-(?![a-z])))+", re.I)
LOOKUP_NEG = re.compile(r"\b(?:not|no|never|nothing|nobody|none|nowhere|neither|nor|without|cannot|hardly|scarcely"
                        r"|barely|far from|anything but|instead of|rather than)\b|n't\b"
                        r"|\b(?:is|are|was|were|do|does|did|has|have|had|ca|could|wo)nt\b"
                        r"|\b(?:noone|zero|nary|nada|naught|nothin)\b|\b(?:ai|should|would|must|need|might)nt\b"
                        r"|n[\u00b4\u02bc\u2032\uff07\u201b]t\b|'nt\b", re.I)
HAVE = re.compile(r"\b(?:have|has|had|having)\b|'ve\b", re.I)
QUANT = re.compile(r"no |nothing (?:came|turned|showed) up", re.I)
QUANT_END = re.compile(r"\w*(?: found)?(?=\s*(?:for|on|in|about|of|with)\b|\s*[^\w\s']|\s*$)", re.I)
NONE_CUE = re.compile(r"no |nothing |turned up nothing|found nothing", re.I)
PREDICATE = re.compile(r"\b(?:is|are|was|were|missing|absent|blank|wrong)\b", re.I)
FROM_TABLE = re.compile(r"[\W_]*(\w+)(?:[\W_]+(\w+))?")
_ONLY, _ONR = {}, {}
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
    """ONLY_SHAPE with the key (in any case), compiled once per key; no key: "it" and "that" only."""
    if key not in _ONLY:
        _ONLY[key] = re.compile(ONLY_SHAPE.replace("KEY", re.escape(key) if key else "(?!)"), re.I)
    return _ONLY[key]


def on_rx(key):
    """LOOKUP_ON with the key (in any case), compiled once per key; no key: "it" only."""
    if key not in _ONR:
        _ONR[key] = re.compile(LOOKUP_ON.replace("KEY", re.escape(key) if key else "(?!)"), re.I)
    return _ONR[key]


def sentence(t, s, e):
    """(start, end) of the sentence holding t[s:e]: from the last BREAK before s to the first at or after e."""
    s0 = max([b.end() for b in BREAK.finditer(t) if b.end() <= s], default=0)
    b = BREAK.search(t, e)
    return s0, (b.start() if b else len(t))


def cue_counts(t, m, ok):
    """the rules of one LOOKUP_ABS match m (module docstring, A2d): (g) a "no <word>" or "nothing came / turned / showed
    up" cue needs QUANT_END after it, and a NONE_CUE with a PREDICATE word after it in its sentence is no cue; (j)
    "missing / absent from" with a table in ok as either of the next two words (FROM_TABLE) is no cue; (d) the
    passive "not (be) found" with HAVE before it in its sentence is no cue."""
    if QUANT.match(m.group(0)) and not QUANT_END.match(t, m.end()):
        return False
    if NONE_CUE.match(m.group(0)) and PREDICATE.search(t[m.end():sentence(t, m.start(), m.end())[1]]):
        return False
    f = FROM_TABLE.match(t, m.end())
    words = {f.group(1).lower(), (f.group(2) or "").lower()} if f else set()
    if m.group(0).lower().startswith(("missing", "absent")) and words & ok:
        return False
    return m.group("nf") is None or HAVE.search(t, sentence(t, m.start(), m.end())[0], m.start()) is None


def voided(t, s, e, wide, at):
    """the cue t[s:e] does not count (module docstring, A2d): its sentence holds a QMARK, the reply holds STRIKE, or a
    LOOKUP_NEG word stands before the cue in its sentence once the openings at the sentence start (OPENERS) are taken
    out, in t or in wide (the reply with each EMPHASIS character read as a space; at maps t's offsets to wide's)."""
    import grade_text as T                    # late, as report_frame: grade_text imports this module
    s0, s1 = sentence(t, s, e)
    if QMARK.search(t, s0, s1 + 1) or STRIKE.search(t):
        return True
    before = t[s0:s]
    o = OPENERS.match(before)
    before = before[o.end():] if o else before
    before += "\n" + wide[at[s - len(before)]:at[s]]
    return "neg" not in T.OFF and LOOKUP_NEG.search(before) is not None


def lookup_hit(text, others, key):
    """text (read without EMPHASIS) holds a cue that counts: a LOOKUP_ABS match (cue_counts), an ONLY_SHAPE or
    LOOKUP_ON match naming "other" or a table in others, not voided."""
    t, ok = EMPHASIS.sub("", text), others | {"other"}
    wide, at = EMPHASIS.sub(" ", text), [i for i, c in enumerate(text) if not EMPHASIS.match(c)] + [len(text)]
    pos = [m.span() for m in _LOOKUP.finditer(t) if cue_counts(t, m, ok)] \
        + [m.span() for m in only_rx(key).finditer(t) if m.group(1).lower() in ok] \
        + [m.span() for m in on_rx(key).finditer(t) if m.group("t").lower() in ok]
    return any(not voided(t, s, e, wide, at) for s, e in pos)


def lookup_any(text):
    """any LOOKUP_ABS, LOOKUP_ON or bare LOOKUP_ONLY match, negated or not, whatever the table: test_gens' content
    rule for LOOKUP questions (fix round, 2026-11-19)."""
    return any(re.search(x, text, re.I) for x in (LOOKUP_ABS, LOOKUP_ONLY, r"\b" + ONLY_ON + r" \w+"))


def lookup_cue(text, rec, probe):
    """the A2b abstain cue of a LOOKUP X reply (module docstring); text already norm()-ed."""
    import grade_text as T
    others, key = lookup_tables(rec, probe), probe.get("key")
    return not lookup_hit(T.norm(probe.get("question")), others, key) and lookup_hit(text, others, key)
