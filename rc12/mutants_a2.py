"""RC-12 grader mutants for lock step A2 (2026-10-19; grade_cues.py's A2 block, fixtures_a2.py, rc12/notes.txt STEP 16).
SOURCE mutants as in mutants_graders.py (module, old text, new text; old occurs exactly once), which appends them to
its SOURCE; its OFF list holds the A2 switches (askedif, rephedge, repquest; negform since 2026-11-19). Kept apart so
mutants_graders.py stays under the file limit."""

SOURCE_A2 = [
    # lock step A2 (2026-10-19; grade_cues.py's A2 block; fixtures_a2.py): one mutant per clause of each fix, each
    # killed by the fixture named after it. asked_if: you, first, me
    ("grade_cues", r'r"\byou\s+(?:first\s+)?asked', r'r"\b\w+\s+(?:first\s+)?asked'),          # a2_they_asked_if
    ("grade_cues", r'(?:first\s+)?asked(?:\s+me)?\s+$"', r'asked(?:\s+me)?\s+$"'),               # a2_first_asked_ok
    ("grade_cues", r'asked(?:\s+me)?\s+$", re.I)', r'asked\s+$", re.I)'),                        # a2_asked_me_if_ok
    # reported_hedge: the two hedge words, only those words, only before the value, a frame needed, the LAST frame,
    # each break (comma, semicolon, "but")
    ("grade_cues", 'REPORT_HEDGE = re.compile(r"should be|seems?", re.I)', 'REPORT_HEDGE = re.compile(r"seems?", re.I)'),
    ("grade_cues", 'REPORT_HEDGE = re.compile(r"should be|seems?", re.I)', 'REPORT_HEDGE = re.compile(r"should be", re.I)'),
    ("grade_cues", "if h.end() > pos or not REPORT_HEDGE.fullmatch(h.group(0)):", "if h.end() > pos:"),     # a2_said_might
    ("grade_cues", "if h.end() > pos or not REPORT_HEDGE.fullmatch(h.group(0)):",
     "if not REPORT_HEDGE.fullmatch(h.group(0)):"),                                                     # a2_hedge_after
    # (these two and the r1 "break only before the value" mutant retargeted in review r2, 2026-11-19, to the
    # rewritten reported_hedge; same edits: a frame at the sentence start, the FIRST frame, the break to the end)
    ("grade_cues", "frames = list(report_frame().finditer(sent[:h.start()]))",
     "frames = [re.match(\"\", sent)] + list(report_frame().finditer(sent[:h.start()]))"),            # a2_should_bare
    ("grade_cues", "f = frames[-1]", "f = frames[0]"),                                               # a2_last_frame_ok
    ("grade_cues", 'REPORT_BREAK = re.compile(r"[,;]', 'REPORT_BREAK = re.compile(r"[;]'),                # a2_said_comma
    ("grade_cues", 'REPORT_BREAK = re.compile(r"[,;]', 'REPORT_BREAK = re.compile(r"[,]'),                # a2_said_semi
    ("grade_cues", r'|\bbut\b|\bso\b", re.I)', r'|\bso\b", re.I)'),                                   # a2_said_but
    # reported_question: you, asked, said, me, the comma and colon, the mention inside the quote
    ("grade_cues", r'QUOTE_FRAME = re.compile(r"\byou\s+', r'QUOTE_FRAME = re.compile(r"\b\w+\s+'),        # a2_quote_they
    ("grade_cues", r'(?:asked|said)(?:\s+me)?\s*[,:]', r'(?:said)(?:\s+me)?\s*[,:]'),
    ("grade_cues", r'(?:asked|said)(?:\s+me)?\s*[,:]', r'(?:asked)(?:\s+me)?\s*[,:]'),
    ("grade_cues", r'(?:asked|said)(?:\s+me)?\s*[,:]', r'(?:asked|said)\s*[,:]'),
    ("grade_cues", r'(?:\s+me)?\s*[,:]?\s*$"', r'(?:\s+me)?\s*:?\s*$"'),
    ("grade_cues", r'(?:\s+me)?\s*[,:]?\s*$"', r'(?:\s+me)?\s*,?\s*$"'),
    ("grade_cues", 'm.start("q") <= pos < m.end("q") and ', ""),                                          # a2_quote_outside
    # ABS_NOT_GIVEN and ABS_DENIAL (ABS_CUE is compiled from GC.ABS_MORE in grade_text, so the mutants edit there)
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("(?:gave|", "(?:"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|specif(?:y|ied)", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|provided?", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|named?", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|rated?", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|pick(?:ed)?", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|cho(?:se|sen|ose)", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|order(?:ed)?", ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|decided?)", ")"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("(?:yet )?(?:gave", "(?:gave"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace(r"\\byou (?:haven\'t", r"\\b\\w+ (?:haven\'t"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("(?:haven\'t|have not|didn\'t", "(?:have not|didn\'t"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("have not|didn\'t|did not|never) (?:yet", "have not|did not|never) (?:yet"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|did not|never) (?:yet", "|did not) (?:yet"))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|" + GC.ABS_DENIAL, ""))'),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace(r"\\bno,? you", r"\\bno,? \\w+"))'),
    # A2 (v), 2026-11-19 (grade_cues.neg_value_re; fixtures_a2.NEG, each mutant named after a fixture that kills it;
    # OFF "negform" switches the rule off): each form of a group, the opening "n't", the auxiliary kept, the separator
    ("grade_cues", "NEG_GROUPS = ((\"is not\", \"isn't\", \"'s not\"),", "NEG_GROUPS = ((\"is not\", \"'s not\"),"),
    ("grade_cues", "NEG_GROUPS = ((\"is not\", \"isn't\", \"'s not\"),", "NEG_GROUPS = ((\"is not\", \"isn't\"),"),
    ("grade_cues", "NEG_GROUPS = ((\"is not\", \"isn't\", \"'s not\"),", "NEG_GROUPS = ((\"isn't\", \"'s not\"),"),
    ("grade_cues", "(\"do not\", \"don't\")", "(\"do not\",)"),                                      # a2_neg_do_ok
    ("grade_cues", "(\"cannot\", \"can not\", \"can't\")", "(\"can not\", \"can't\")"),              # a2_neg_can_ok
    ("grade_cues", "(\"will not\", \"won't\", \"'ll not\")", "(\"won't\", \"'ll not\")"),            # a2_neg_will_ok
    ("grade_cues", "(\"have not\", \"haven't\", \"'ve not\")", "(\"have not\", \"haven't\")"),        # a2_neg_have_ok
    ("grade_cues", "clitics += [] if group else [\"n't\"]", "clitics += []"),                       # a2_neg_bare_ok
    ("grade_cues", "group = GROUP_OF.get(m.group(0).lower())", "group = None"),                   # a2_neg_other_aux
    ("grade_cues", "+ (\"|(?:\" + _alts(clitics) + \")\" if clitics else \"\") + \")\")", "+ \")\")"),
    ("grade_cues", "\"(?: (?:\" + _alts(words)", "\"(?:(?:\" + _alts(words)"),                     # a2_neg_mid_ok
    ("grade_cues", "+ (r\"|(?<=[A-Za-z])(?:\" + _alts(clitics) + \")\" if clitics else \"\") + \")\")", "+ \")\")"),
    ("grade_cues", "if m.start() == 0:", "if False:"),                                                # a2_neg_is_ok
    ("grade_cues", "parts[0] = \"\"", "pass"),                                                      # a2_neg_clitic_ok
    ("grade_cues", "|(?<=[A-Za-z])(?:\" + \"|\".join(map(re.escape, _CLITICS))",
     "|(?<=[A-Za-z])(?:\" + \"|\".join(map(re.escape, [\"'zz not\"]))"),                           # a2_neg_mid_ok
    ("grade_text", "GC.neg_value_re(v, V.value_re(v).flags)", "GC.neg_value_re(v, 0)"),             # a2_neg_mid_ok
    ("grade_text", "return m.group(0).lower() in {f.lower() for f in forms(v)}",
     "return m.group(0).lower() != v.lower()"),                                                        # a2_neg_is_ok
    # review r1 (2026-11-19; fixtures_a2, the r1 kinds): clauses that had no killing fixture. asked_if and
    # reported_question: the frame ends the text before the "if" / the quote; reported_hedge: a break counts only
    # between the frame and the value; ABS_NOT_GIVEN's "have not" and "did not"; ABS_DENIAL's optional comma
    ("grade_cues", r'asked(?:\s+me)?\s+$", re.I)', r'asked(?:\s+me)?\s+", re.I)'),                # a2_asked_to_check_if
    ("grade_cues", r'[,:]?\s*$", re.I)', r'[,:]?\s*", re.I)'),                                     # a2_quote_frame_far
    ("grade_cues", "REPORT_BREAK.search(sent[f.end():pos])", "REPORT_BREAK.search(sent[f.end():])"),
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace("|have not|didn\'t", "|didn\'t"))'),          # a2_have_not_ok
    ("grade_text", "GC.ABS_MORE)",
     'GC.ABS_MORE.replace("didn\'t|did not|never) (?:yet", "didn\'t|never) (?:yet"))'),               # a2_did_not_ok
    ("grade_text", "GC.ABS_MORE)", 'GC.ABS_MORE.replace(r"\\bno,? you", r"\\bno, you"))'),          # a2_no_comma_ok
    # (v): the negation forms in any case (a2_neg_cap_*), and the contraction of each group no fixture held before
    ("grade_cues", 'return "(?i:" + "|".join(map(re.escape, forms)) + ")"',
     'return "|".join(map(re.escape, forms))'),                                                       # a2_neg_cap_ok
    ("grade_cues", "(\"are not\", \"aren't\", \"'re not\")", "(\"are not\", \"'re not\")"),       # a2_neg_are_ok
    ("grade_cues", "(\"are not\", \"aren't\", \"'re not\")", "(\"are not\", \"aren't\")"),        # a2_neg_re_ok
    ("grade_cues", "(\"am not\", \"'m not\")", "(\"am not\",)"),                                   # a2_neg_am_ok
    ("grade_cues", "(\"was not\", \"wasn't\")", "(\"was not\",)"),                                 # a2_neg_was_ok
    ("grade_cues", "(\"were not\", \"weren't\")", "(\"were not\",)"),                              # a2_neg_were_ok
    ("grade_cues", "(\"does not\", \"doesn't\")", "(\"does not\",)"),                              # a2_neg_does_ok
    ("grade_cues", "(\"did not\", \"didn't\")", "(\"did not\",)"),                                 # a2_neg_did_ok
    ("grade_cues", "(\"has not\", \"hasn't\")", "(\"has not\",)"),                                 # a2_neg_has_ok
    ("grade_cues", "(\"had not\", \"hadn't\")", "(\"had not\",)"),                                 # a2_neg_had_ok
    ("grade_cues", "(\"would not\", \"wouldn't\", \"'d not\")", "(\"would not\", \"'d not\")"),   # a2_neg_would_ok
    ("grade_cues", "(\"would not\", \"wouldn't\", \"'d not\")", "(\"would not\", \"wouldn't\")"),  # a2_neg_d_ok
    ("grade_cues", "(\"will not\", \"won't\", \"'ll not\")", "(\"will not\", \"won't\")"),        # a2_neg_ll_ok
    ("grade_cues", "(\"could not\", \"couldn't\")", "(\"could not\",)"),                           # a2_neg_could_ok
    ("grade_cues", "(\"should not\", \"shouldn't\")", "(\"should not\",)"),                        # a2_neg_should_ok
    ("grade_cues", "(\"must not\", \"mustn't\")", "(\"must not\",)"),                              # a2_neg_must_ok
    ("grade_cues", "(\"need not\", \"needn't\")", "(\"need not\",)"),                              # a2_neg_need_ok
    # review r2 (2026-11-19; fixtures_a2, the r2 kinds). reported_hedge: the frame before the hedge (no mutant held
    # it), "so" as a break, the negation check (on, over the frame, over the text after it), the first-person check
    ("grade_cues", "report_frame().finditer(sent[:h.start()])",
     "report_frame().finditer(sent)"),                                                            # a2_should_said_after
    ("grade_cues", r'|\bso\b", re.I)', r'", re.I)'),                                                # a2_told_nothing_so
    ("grade_cues", "T.NEG_CUE.search(sent[f.start():pos]) is None", "True"),                    # a2_never_wrote_should
    ("grade_cues", "T.NEG_CUE.search(sent[f.start():pos])", "T.NEG_CUE.search(sent[f.end():pos])"),  # never_once
    ("grade_cues", "T.NEG_CUE.search(sent[f.start():pos])",
     "T.NEG_CUE.search(sent[f.start():f.end()])"),                                               # a2_said_unsure_should
    ("grade_cues", "T.FIRST_ANY.search(sent[f.end():pos]) is None", "True"),                       # a2_said_i_should
    # (v): the value's boundary before a mid-value negation and at its end, the curly apostrophe normalized
    ("grade_cues", 's, parts, last = v.replace("\u2019", "\'"), [r"(?<![A-Za-z0-9])"], 0',
     's, parts, last = v.replace("\u2019", "\'"), [""], 0'),                                         # a2_neg_mid_inside
    ("grade_cues", '+ re.escape(s[last:]) + r"(?![A-Za-z0-9])", flags)',
     '+ re.escape(s[last:]), flags)'),                                                              # a2_neg_tail_inside
    ("grade_cues", 's, parts, last = v.replace("\u2019", "\'"),', 's, parts, last = v,'),             # a2_neg_curly_ok
]
