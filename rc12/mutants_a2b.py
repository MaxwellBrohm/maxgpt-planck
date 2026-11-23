"""RC-12 grader mutants for lock step A2b (2026-11-19; grade_cues.py's A2b block, graders.g_abs, fixtures_a2b.py,
rc12/notes.txt STEP 16A2B). SOURCE mutants as in mutants_graders.py (module, old text, new text; old occurs exactly
once), which appends them to its SOURCE; its OFF list holds the A2b switch "lookabs" (the whole LOOKUP extension).
One mutant per clause: each alternative of LOOKUP_FIND, LOOKUP_ABS and LOOKUP_ONLY, the table rule, the negation
rule, the question rule, the scope and the a4 exemption, each named after a fixture that kills it; each survives the
fixtures without fixtures_a2b (rc12/notes.txt STEP 16A2B). The fix round (2026-11-19, notes STEP 16A2B-R) retargets
the mutants whose text it rewrote and adds one per new clause (the clause-start subject and each of its marks, the
subjects, adverbs and auxiliaries, the found forms, the ONLY verbs and objects, "only on / in", the clause-scoped
negation and its marks, "nothing" / "nobody", any clean hit, the question gate's norm, ONLY forms and negation, the
"I found nothing" frame); each survives the fixtures without X_OK_R1 and the fix round's guards (the A2b graders'
own fixtures) and is killed by the one named. The late review (2026-11-22, notes STEP 16A2B-R2) retargets the
mutants whose text it rewrote (the found forms, the ONLY objects, LOOKUP_ON, the hit list, the negation rule) and
adds one per new clause (its block below); each survives the fixtures without X_OK_R2, X_FAIL_R2 and the OODH copy
(the fix round's) and is killed by the one named, and a mutant named after an older kind survives the fixtures
without fixtures_a2b. Its second round (2026-11-22, notes STEP 16A2B-R3) does the same with X_OK_R3 and X_FAIL_R3
(a mutant named after one of their kinds survives the fixtures before that round), retargets the mutants of the text
it rewrote (the found forms' perfect, the ONLY objects, LOOKUP_ON's anchor, NEG_MORE, NEG_PHRASE) and pins the second
reviewer's survivors (LOOKUP_ON's leading boundary, TOKEN's apostrophe, END's word ends, NEG_MORE's colon, a row form
for any word, the NEG_MORE window). Not mutated: word boundaries (as ABS_MORE) but those the reviewers' survivors
showed, the 3-token window (grade_text.negated_before's, mutated there), re.escape of the key (dev keys are weekdays
and months), the aside's inner classes (within a sentence they differ from "no comma" or "no dash" only on ";" and
":"); equivalent, so not mutated: LOOKUP_ON's word count greedy, any clitic after its anchor (other clitics are no
English after a key, "it" or "that"), INTERJ blanked to one space, "no" among NEG_PHRASE's subjects (the window reads
it), more than two words after an aside (the window then holds only them), and the order of drop_aside and INTERJ
(the forms where it differs are NEG_PHRASE's). Not pinned: the words after an aside left out (it only brings earlier
words into the window; no right reply was found that it fails)."""

SOURCE_A2B = [
    # LOOKUP_FIND: the subject, each auxiliary, each verb
    ("grade_cues", r'r"(?:\b(?:i|we|', r'r"(?:\b(?:\w+|we|'),                                  # a2b_they_couldnt
    ("grade_cues", "(?:couldn't|could not|", "(?:could not|"),                                    # a2b_couldnt_find_ok
    ("grade_cues", "|could not|can't|", "|can't|"),                                      # a2b_could_not_anything_ok
    ("grade_cues", "|can't|cannot|", "|cannot|"),                                                    # a2b_cant_see_ok
    ("grade_cues", "|cannot|didn't|", "|didn't|"),                                              # a2b_cannot_locate_ok
    ("grade_cues", "|cannot|didn't|did not|", "|cannot|did not|"),                                  # a2b_didnt_spot_ok
    ("grade_cues", "|didn't|did not|don't|", "|didn't|don't|"),                                   # a2b_did_not_find_ok
    ("grade_cues", '|do not|was unable to"', '|do not"'),                                          # a2b_unable_find_ok
    ("grade_cues", "(?:find|locate|see|spot)", "(?:locate|see|spot)"),                            # a2b_couldnt_find_ok
    ("grade_cues", "|locate|see|", "|see|"),                                                    # a2b_cannot_locate_ok
    ("grade_cues", "|see|spot)", "|spot)"),                                                          # a2b_cant_see_ok
    ("grade_cues", r'|spot)\b"', r')\b"'),                                                         # a2b_didnt_spot_ok
    # fix round: the other subjects, the adverbs, no subject at a clause start, the other auxiliaries, "seem to"
    ("grade_cues", r'r"(?:\b(?:i|we|(?:the|my)', r'r"(?:\b(?:i|(?:the|my)'),                             # a2b_we_ok
    ("grade_cues", "|we|(?:the|my) (?:lookup|search))(?: (?:just", "|we)(?: (?:just"),              # a2b_the_lookup_ok
    ("grade_cues", "(?:the|my) (?:lookup", "(?:my) (?:lookup"),                                     # a2b_the_lookup_ok
    ("grade_cues", "(?:the|my) (?:lookup", "(?:the) (?:lookup"),                                     # a2b_my_search_ok
    ("grade_cues", "(?:lookup|search))", "(?:search))"),                                            # a2b_the_lookup_ok
    ("grade_cues", "(?:lookup|search))", "(?:lookup))"),                                             # a2b_my_search_ok
    ("grade_cues", "(?:just|still|", "(?:still|"),                                                         # a2b_just_ok
    ("grade_cues", "(?:just|still|really|", "(?:just|really|"),                                           # a2b_still_ok
    ("grade_cues", "|still|really|honestly|", "|still|honestly|"),                                       # a2b_really_ok
    ("grade_cues", "|really|honestly|simply)", "|really|simply)"),                                     # a2b_honestly_ok
    ("grade_cues", "|honestly|simply))?", "|honestly))?"),                                               # a2b_simply_ok
    ("grade_cues", '(?: (?:just|still|really|honestly|simply))? "', ' "'),                                # a2b_still_ok
    ("grade_cues", '(?: (?:just|still|really|honestly|simply))? "', '(?: \\w+)? "'),                # a2b_almost_couldnt
    ("grade_cues", r'r"|(?:^|(?<=[.!?;:,]))\s*)', 'r")'),                                      # a2b_subjless_comma_ok
    ("grade_cues", "(?:^|(?<=[.!?;:,]))", "(?:(?<=[.!?;:,]))"),                                # a2b_subjless_start_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[!?;:,]))"),                                        # a2b_subjless_period_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[.?;:,]))"),                                          # a2b_subjless_bang_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[.!;:,]))"),                                             # a2b_subjless_q_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[.!?:,]))"),                                          # a2b_subjless_semi_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[.!?;,]))"),                                         # a2b_subjless_colon_ok
    ("grade_cues", "(?<=[.!?;:,]))", "(?<=[.!?;:]))"),                                         # a2b_subjless_comma_ok
    ("grade_cues", r"(?<=[.!?;:,]))\s*)", "(?<=[.!?;:,])))"),                                  # a2b_subjless_comma_ok
    ("grade_cues", "|did not|don't|do not|", "|did not|do not|"),                                     # a2b_dont_see_ok
    ("grade_cues", "|don't|do not|was", "|don't|was"),                                              # a2b_do_not_see_ok
    ("grade_cues", "r\"|wasn't able to|", "r\"|"),                                                  # a2b_wasnt_able_ok
    ("grade_cues", '|was not able to)"', ')"'),                                                   # a2b_was_not_able_ok
    ("grade_cues", r'r"(?: seem to)? (?:find', r'r" (?:find'),                                             # a2b_seem_ok
    # LOOKUP_ABS: the find frame joined, each "nothing" form, "came up empty", the found forms, each "no" word,
    # missing, absent; the "no" words read as word starts (a trailing boundary would fail "No results")
    ("grade_cues", 'LOOKUP_ABS = (LOOKUP_FIND + r"|', 'LOOKUP_ABS = (r"'),                        # a2b_couldnt_find_ok
    ("grade_cues", "(?:came|turned|showed) up", "(?:turned|showed) up"),                          # a2b_nothing_came_ok
    ("grade_cues", "|turned|showed) up", "|showed) up"),                                        # a2b_nothing_turned_ok
    ("grade_cues", "|showed) up", ") up"),                                                      # a2b_nothing_showed_ok
    ("grade_cues", r'|\bnothing (?:was )?found\b" + END', '"'),                                 # a2b_nothing_found_ok
    ("grade_cues", "nothing (?:was )?found", "nothing found"),                               # a2b_nothing_was_found_ok
    ("grade_cues", "(?:turned up|found) nothing", "(?:found) nothing"),                      # a2b_turned_up_nothing_ok
    ("grade_cues", "|found) nothing", ") nothing"),                                              # a2b_found_nothing_ok
    ("grade_cues", r'r"|\bcame up empty\b"', 'r""'),                                           # a2b_came_up_empty_ok
    ("grade_cues", "(?P<nf>not)(?: be)?|", "(?P<nf>(?!))(?: be)?|"),                               # a2b_not_found_ok
    ("grade_cues", "|cannot be|", "|"),                                                         # a2b_cannot_be_found_ok
    ("grade_cues", "(?:was|were|is|are)n't", "(?:were|is|are)n't"),                              # a2b_wasnt_found_ok
    ("grade_cues", "|(?:could|ca|wo)n't be) found", ") found"),                                       # a2b_be_found_ok
    ("grade_cues", "(?:result|match|", "(?:match|"),                                               # a2b_no_results_ok
    ("grade_cues", "|match|entr|", "|entr|"),                                                        # a2b_no_match_ok
    ("grade_cues", "|entr|row|", "|row|"),                                                           # a2b_no_entry_ok
    ("grade_cues", "|row|record|", "|record|"),                                                        # a2b_no_row_ok
    ("grade_cues", "|record|information|", "|information|"),                                       # a2b_no_record_ok
    ("grade_cues", "|information|luck)", "|luck)"),                                            # a2b_no_information_ok
    ("grade_cues", "|information|luck)", "|information)"),                                           # a2b_no_luck_ok
    ("grade_cues", r"|luck)|\b(?:missing", r"|luck)\b|\b(?:missing"),                              # a2b_no_results_ok
    ("grade_cues", "(?:missing|absent) from", "(?:absent) from"),                                     # a2b_missing_ok
    ("grade_cues", "|absent) from", ") from"),                                                         # a2b_absent_ok
    # LOOKUP_ONLY: each verb, up to 2 words between the table and the verb (none, one, any number); its object
    ("grade_cues", "(?:has|lists|", "(?:lists|"),                                              # a2b_only_other_has_ok
    ("grade_cues", "|lists|includes|", "|includes|"),                                        # a2b_only_other_lists_ok
    ("grade_cues", "|lists|includes|contains|", "|lists|contains|"),                            # a2b_only_includes_ok
    ("grade_cues", "|includes|contains|shows)", "|includes|shows)"),                            # a2b_only_contains_ok
    ("grade_cues", '|contains|shows) "', '|contains) "'),                                          # a2b_only_shows_ok
    ("grade_cues", r"(?: \w+){0,2} (?:has", r"(?: \w+){0,1} (?:has"),                        # a2b_only_two_words_ok
    ("grade_cues", r"(?: \w+){0,2} (?:has", r"(?: \w+){0} (?:has"),                          # a2b_only_other_has_ok
    ("grade_cues", r"(?: \w+){0,2} (?:has", r"(?: \w+)* (?:has"),                                    # a2b_only_far
    # LOOKUP_ON: each preposition
    ("grade_cues", "only (?:on|in) the", "only (?:in) the"),                                         # a2b_only_on_ok
    ("grade_cues", "only (?:on|in) the", "only (?:on) the"),                                         # a2b_only_in_ok
    # every pattern in any case
    ("grade_cues", "(re.compile(x, re.I) for x in (LOOKUP_FRAME",
     "(re.compile(x) for x in (LOOKUP_FRAME"),                                                   # a2b_nothing_came_ok
    # lookup_tables: the asked table taken out (its name parsed from the holder); lookup_hit: each table rule (in any
    # case), "other", the negation rule, any clean hit; lookup_negated: its sentence start, its clause start and each
    # mark, "nothing" and "nobody"; lookup_cue: none counts when the question holds one (the question norm()-ed, its
    # ONLY forms read, a negated cue in it no cue), the probe's key read
    ("grade_cues", "    return objs - {asked}", "    return objs"),                                   # a2b_only_asked
    ("grade_cues", '.split(":", 1)[-1].lower()', ".lower()"),                                         # a2b_only_asked
    ("grade_cues", "in only_rx(key).finditer(text) if m.group(1).lower() in ok]",
     "in only_rx(key).finditer(text)]"),                                                              # a2b_only_asked
    ("grade_cues", "only_rx(key).finditer(text) if m.group(1).lower() in ok]",
     "only_rx(key).finditer(text) if m.group(1) in ok]"),                                            # a2b_only_cap_ok
    ("grade_cues", 'ok = others | {"other"}', "ok = others"),                                # a2b_only_other_lists_ok
    ("grade_cues", "(?:nothing|nobody)", "(?:nobody)"),                                 # a2b_r2_found_nothing_missing
    ("grade_cues", "(?:nothing|nobody)", "(?:nothing)"),                                      # a2b_r2_nobodys_missing
    ("grade_cues", 'return not lookup_hit(T.norm(probe.get("question")), others, key) and lookup_hit(text, others, '
     'key)', "return lookup_hit(text, others, key)"),                                          # a2b_echo_question
    ("grade_cues", 'lookup_hit(T.norm(probe.get("question")), others, key)',
     'lookup_hit(probe.get("question") or "", others, key)'),                                       # a2b_echo_curly
    ("grade_cues", 'lookup_hit(T.norm(probe.get("question")), others, key)',
     'lookup_hit(T.norm(probe.get("question")), set(), key)'),                                       # a2b_echo_only
    ("grade_cues", 'not lookup_hit(T.norm(probe.get("question")), others, key)',
     'not lookup_any(T.norm(probe.get("question")))'),                                                 # a2b_negq_ok
    ("grade_cues", 'lookup_tables(rec, probe), probe.get("key")',
     "lookup_tables(rec, probe), None"),                                                      # a2b_only_other_has_ok
    # late review (2026-11-22, notes STEP 16A2B-R2): the "found nothing" ends (END, at each of its two uses, each of
    # its parts), "nothing <word> found", the found forms (the "not" form after have / has / had / 've, its "be",
    # each passive auxiliary, the active forms), the ONLY objects ("it" / "that" at a clause end and its parts, a row
    # or an entry or a line followed by for / on the key, "an? <word>", any verb), LOOKUP_ON's anchor (the key, "it",
    # "that", "'s", at most three words, no mark, the negation read at "only", in any case, another preposition), the
    # negation rule (the openings blanked, each of them, the anchor, the mark after; the sentence start; "nothing" /
    # "nobody" set off by a mark, each mark kind; the window; NEG_PHRASE, each word, the aside, the words, each verb,
    # the clause end), LOOKUP_FIND's subjects and marks (no third person, no "and" / "but" / dash before a bare form,
    # no other auxiliary), any clean hit and the joined lists rewritten for the positions
    ("grade_cues", r'(?:was )?found\b" + END', r'(?:was )?found\b"'),                     # a2b_r2_nothing_found_wrong
    ("grade_cues", r'found) nothing\b" + END', r'found) nothing\b"'),                      # a2b_r2_found_nothing_wrong
    ("grade_cues", r"(?:for|on|in|about)\b|", r"(?:on|in|about)\b|"),                             # a2b_nothing_found_ok
    ("grade_cues", r"(?:for|on|in|about)\b|", r"(?:for|in|about)\b|"),                                # a2b_r2_end_on_ok
    ("grade_cues", r"(?:for|on|in|about)\b|", r"(?:for|on|about)\b|"),                                # a2b_r2_end_in_ok
    ("grade_cues", r"(?:for|on|in|about)\b|", r"(?:for|on|in)\b|"),                                # a2b_r2_end_about_ok
    ("grade_cues", r'''|\s*[^\w\s']|\s*$)"''', r'''|\s*$)"'''),                                     # a2b_r2_end_mark_ok
    ("grade_cues", r'''|\s*[^\w\s']|\s*$)"''', r'''|\s*[^\w\s'])"'''),                               # a2b_r2_end_end_ok
    ("grade_cues", "nothing (?:was )?found", r"nothing (?:\w+ )?found"),                     # a2b_r2_nothing_new_found
    ("grade_cues", r"(?:\b(?:have|has|had)|", r"(?:\b(?:has|had)|"),                           # a2b_r2_have_not_found
    ("grade_cues", "(?:have|has|had)|'ve)", "(?:have|had)|'ve)"),                               # a2b_r2_has_not_found
    ("grade_cues", "(?:have|has|had)|'ve)", "(?:have|has)|'ve)"),                               # a2b_r2_had_not_found
    ("grade_cues", "|had)|'ve)(?:", "|had))(?:"),                                                 # a2b_r2_ve_not_found
    ("grade_cues", "(?P<nf>not)(?: be)?|cannot", "(?P<nf>not)|cannot"),                # a2b_r2_could_not_be_found_ok
    ("grade_cues", "(?:was|were|is|are)n't", "(?:was|is|are)n't"),                            # a2b_r2_werent_found_ok
    ("grade_cues", "(?:was|were|is|are)n't", "(?:was|were|are)n't"),                            # a2b_r2_isnt_found_ok
    ("grade_cues", "(?:was|were|is|are)n't", "(?:was|were|is)n't"),                            # a2b_r2_arent_found_ok
    ("grade_cues", "(?:was|were|is|are)n't", "(?:was|were|is|are)(?:n't)?"),                        # a2b_r2_was_found
    ("grade_cues", "(?:was|were|is|are)n't", r"\w+n't"),                                         # a2b_r2_havent_found
    ("grade_cues", "(?:could|ca|wo)n't be", "(?:ca|wo)n't be"),                                       # a2b_be_found_ok
    ("grade_cues", "(?:could|ca|wo)n't be", "(?:could|wo)n't be"),                          # a2b_r2_cant_be_found_ok
    ("grade_cues", "(?:could|ca|wo)n't be", "(?:could|ca)n't be"),                          # a2b_r2_wont_be_found_ok
    ("grade_cues", 'ONLY_OBJ = (r"(?:', r'ONLY_OBJ = (r"(?:\w+|'),                                    # a2b_only_changed
    ("grade_cues", 'ONLY_OBJ = (r"(?:', r'ONLY_OBJ = (r"(?:an? \w+|'),                               # a2b_r2_only_a_gap
    ("grade_cues", '(?:has|lists|includes|contains|shows) "', r'\w+ "'),                       # a2b_r2_only_changed_it
    ("grade_cues", "(?:(?:it|that)(?=", "(?:(?:that)(?="),                                         # a2b_only_has_it_ok
    ("grade_cues", "(?:(?:it|that)(?=", "(?:(?:it)(?="),                                         # a2b_only_has_that_ok
    ("grade_cues", r"(?:it|that)(?=\s*(?:[^\w\s']|$))", r"(?:it|that)\b"),                 # a2b_r2_only_that_problem
    ("grade_cues", r"(?=\s*(?:[^\w\s']|$))", r"(?=\s*(?:$))"),                                    # a2b_only_has_it_ok
    ("grade_cues", r"(?=\s*(?:[^\w\s']|$))", r"(?=\s*(?:[^\w\s']))"),                          # a2b_r2_only_it_end_ok
    ("grade_cues", r"(?:[^\w\s']|$))", r"(?:[^\w\s]|$))"),                                       # a2b_r2_only_its_typo
    ("grade_cues", "(?:a row|an entry|a line) (?:for|on) KEY|", r"(?:a row|an entry|a line)(?: \w+)*|"),
    # a2b_r2_only_line_crossed
    ("grade_cues", "(?:a row|an entry|", "(?:an entry|"),                                           # a2b_only_a_row_ok
    ("grade_cues", "|an entry|a line)", "|a line)"),                                             # a2b_only_an_entry_ok
    ("grade_cues", "|an entry|a line)", "|an entry)"),                                             # a2b_only_a_line_ok
    ("grade_cues", " (?:for|on) KEY", " (?:on) KEY"),                                               # a2b_only_a_row_ok
    ("grade_cues", " (?:for|on) KEY", " (?:for) KEY"),                                       # a2b_r2_only_entry_on_ok
    ("grade_cues", r"(?:the KEY(?: \w+)?|KEY)", "(?:KEY)"),                                     # a2b_only_the_key_ok
    ("grade_cues", r'|(?:the KEY(?: \w+)?|KEY)"', r'|(?!)(?:the KEY(?: \w+)?|KEY)"'),        # a2b_only_other_has_ok
    ("grade_cues", 'ONLY_OBJ.replace("KEY", re.escape(key) if key else "(?!)"), re.I)',
     'ONLY_OBJ.replace("KEY", re.escape(key) if key else "(?!)"))'),                            # a2b_only_other_has_ok
    ("grade_cues", ' \\\n        + [m.start("o") for m in on_rx(key).finditer(text) if m.group("t").lower() in ok]',
     ""),                                                                                            # a2b_only_on_ok
    ("grade_cues", "only (?:on|in) the", r"only (?:on|in|\w+) the"),                                   # a2b_r2_on_about
    ("grade_cues", 'if m.group("t").lower() in ok]', "]"),                                           # a2b_only_on_asked
    ("grade_cues", 'm.group("t").lower() in ok', 'm.group("t") in ok'),                             # a2b_only_on_cap_ok
    ("grade_cues", "(?:KEY(?:s)?|it|that)", "(?:it|that)"),                                             # a2b_only_on_ok
    ("grade_cues", "(?:KEY(?:s)?|it|that)", "(?:KEY(?:s)?|that)"),                                    # a2b_r2_on_it_ok
    ("grade_cues", "(?:KEY(?:s)?|it|that)", "(?:KEY(?:s)?|it)"),                                    # a2b_r2_on_that_ok
    ("grade_cues", r"\b(?:'s|'ll|'d)?(?:\s+\w+)", r"\b(?:'ll|'d)?(?:\s+\w+)"),                    # a2b_r2_on_key_s_ok
    ("grade_cues", "{0,3}?", "{0,2}?"),                                                     # a2b_r2_on_three_words_ok
    ("grade_cues", "{0,3}?", "*?"),                                                                     # a2b_r2_on_far
    ("grade_cues", r"(?:\s+\w+){0,3}?", r"(?:\W+\w+){0,3}?"),                                         # a2b_r2_on_marks
    ("grade_cues", r'''LOOKUP_ON = r"\b(?:KEY(?:s)?|it|that)\b(?:'s|'ll|'d)?(?:\s+\w+){0,3}?\s+(?P<o>"''',
     r'''LOOKUP_ON = r"\b(?P<o>"'''),                                                           # a2b_r2_on_extra_rows
    ("grade_cues", '[m.start("o") for m in on_rx', "[m.start() for m in on_rx"),                    # a2b_r2_on_negated
    ("grade_cues", 'LOOKUP_ON.replace("KEY", re.escape(key) if key else "(?!)"), re.I)',
     'LOOKUP_ON.replace("KEY", re.escape(key) if key else "(?!)"))'),                                # a2b_r2_on_it_ok
    ("grade_cues", "return any(not lookup_negated(text, x) for x in pos)", "return bool(pos)"),       # a2b_not_missing
    ("grade_cues", "return any(not lookup_negated(", "return bool(pos) and all(not lookup_negated("),  # a2b_two_hits_ok
    ("grade_cues", "pos = [m.start() for m in _LOOKUP.finditer(text) if not perfect(text, m)]",
     "pos = [m.start() for m in _LOOKUP.finditer(text) if not perfect(text, m)][:1]"),              # a2b_two_hits_ok
    ("grade_cues", "    s0 = T.sentence_span(text, pos)[0]\n", "    s0 = 0\n"),              # a2b_r2_prev_sentence_ok
    ("grade_cues", 'INTERJ.sub(lambda m: " " * len(m.group(0)), drop_aside(raw))', "drop_aside(raw)"),
    # a2b_afraid_not_comma_ok
    ("grade_cues", "(?:^|(?<=,))", "(?:(?<=,))"),                                             # a2b_afraid_not_comma_ok
    ("grade_cues", "(?:^|(?<=,))", "(?:^)"),                                             # a2b_r3_interj_comma_colon_ok
    ("grade_cues", r'r"(?:^|(?<=,))\s*(?:', r'r"\b(?:'),                                    # a2b_r2_interj_midsentence
    ("grade_cues", "(?:i'm |i am )?afraid", "(?:i am )?afraid"),                              # a2b_afraid_not_comma_ok
    ("grade_cues", "(?:i'm |i am )?afraid", "(?:i'm )?afraid"),                                 # a2b_r2_i_am_afraid_ok
    ("grade_cues", "(?:i'm |i am )?afraid", "(?:i'm |i am )afraid"),                                 # a2b_r2_afraid_ok
    ("grade_cues", "|sadly|", "|"),                                                            # a2b_sadly_not_colon_ok
    ("grade_cues", "|unfortunately|", "|"),                                                   # a2b_r2_unfortunately_ok
    ("grade_cues", "|(?:i )?guess|", "|i guess|"),                                                    # a2b_r2_guess_ok
    ("grade_cues", "|(?:i )?guess|", "|guess|"),                                                    # a2b_r2_i_guess_ok
    ("grade_cues", '|probably"', '"'),                                                             # a2b_r2_probably_ok
    ("grade_cues", 'r"|apparently) not', 'r") not'),                                             # a2b_r2_apparently_ok
    ("grade_cues", "(?:i )?(?:don't|do not) think so", "(?:i )?(?:do not) think so"),        # a2b_dont_think_semi_ok
    ("grade_cues", "(?:i )?(?:don't|do not) think so", "(?:i )?(?:don't) think so"),         # a2b_r2_do_not_think_ok
    ("grade_cues", "(?:i )?(?:don't|do not) think so", "i (?:don't|do not) think so"),      # a2b_r2_dont_think_noi_ok
    ("grade_cues", "not (?:really|sure|this time)", "not (?:sure|this time)"),                   # a2b_r2_not_really_ok
    ("grade_cues", "not (?:really|sure|this time)", "not (?:really|this time)"),                   # a2b_r2_not_sure_ok
    ("grade_cues", "not (?:really|sure|this time)", "not (?:really|sure)"),                   # a2b_r2_not_this_time_ok
    ("grade_cues", r'''r"\s*[^\w\s']", re.I)''', r'''r"\s*,", re.I)'''),                       # a2b_dont_think_semi_ok
    ("grade_cues", r'''r"\s*[^\w\s']", re.I)''', r'''r"\s*[,;:]", re.I)'''),                    # a2b_r2_interj_dash_ok
    ("grade_cues", r'''r"\s*[^\w\s']", re.I)''', r'''r"", re.I)'''),                            # a2b_r2_interj_no_mark
    ("grade_cues", r"\b(?![\s\x22]*[^\w\s'\x22])", r"\b"),                                      # a2b_r2_nobody_comma_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*,)"),                               # a2b_r2_nobody_em_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*[\-\u2013\u2014])"),             # a2b_r2_nobody_comma_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*[,\-\u2014])"),                    # a2b_r2_nothing_en_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*[,\-\u2013])"),                     # a2b_r2_nobody_em_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*[,\u2013\u2014])"),             # a2b_r2_nobody_hyphen_ok
    ("grade_cues", "NEG_MORE.search(before, w0) is not None or ", ""),                   # a2b_r2_found_nothing_missing
    ("grade_cues", "NEG_MORE.search(before, w0)", "NEG_MORE.search(before)"),               # a2b_r2_nothing_window_ok
    ("grade_cues", "    if T.negated_before(before, 0, len(before)):\n        return True\n", ""),    # a2b_not_missing
    ("grade_cues", " or NEG_PHRASE.search(raw) is not None", ""),                                 # a2b_r2_nobody_aside
    ("grade_cues", "(?:nothing|nobody|no one|none|", "(?:nobody|no one|none|"),              # a2b_r2_nothing_aside_was
    ("grade_cues", "(?:nothing|nobody|no one|none|", "(?:nothing|no one|none|"),                  # a2b_r2_nobody_aside
    ("grade_cues", "(?:nothing|nobody|no one|none|", "(?:nothing|nobody|none|"),            # a2b_r3_no_one_words_aside
    ("grade_cues", "(?:nothing|nobody|no one|none|", "(?:nothing|nobody|no one|"),              # a2b_r2_none_aside_are
    ("grade_cues", r"(?:\s*,[^,.;:!?]*,|", "(?:"),                                                # a2b_r2_nobody_aside
    ("grade_cues", r'\b(?:\s+\w+){0,3}" + ASIDE', r'\b" + ASIDE'),                            # a2b_r2_nothing_at_all
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:are|was|were|be|been)"),                    # a2b_r2_nobody_aside
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:is|was|were|be|been)"),                   # a2b_r2_none_aside_are
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:is|are|were|be|been)"),                # a2b_r2_nothing_aside_was
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:is|are|was|be|been)"),                # a2b_r3_none_of_them_were
    ("grade_cues", r"be|been)(?:\s+found)?\s+$", r"be|been)(?:\s+found)?\s+"),                # a2b_r2_phrase_end_ok
    ("grade_cues", r'r"(?:\b(?:i|we|', r'r"(?:\b(?:i|we|you|he|she|it|'),                          # a2b_r2_she_couldnt
    ("grade_cues", r"(?:^|(?<=[.!?;:,]))\s*)", r"(?:^|(?<=[.!?;:,])|(?<=\band)|(?<=\bbut))\s*)"),   # a2b_r2_and_couldnt
    ("grade_cues", r"(?<=[.!?;:,]))\s*)", r"(?<=[.!?;:,\-]))\s*)"),                              # a2b_r2_dash_couldnt
    ("grade_cues", '|was not able to)"', "|was not able to|won't|wouldn't)\""),                   # a2b_r2_wouldnt_see
    # late review, second round (2026-11-22, notes STEP 16A2B-R3): the perfect check (applied, its "not" group, each
    # auxiliary above, at most one word between, none, two), ONLY's key object (its "the <key> <word>", plural, each
    # word that may end it, the clause end with "and" / "but", a row form with no key or any word for it), LOOKUP_ON's
    # plural key, "'ll", "'d" and leading boundary, the openings "none" and "no", the aside (each mark, the dash
    # kinds, taken out at all, the comma kept, the words after it: their count, any word, each verb), NEG_MORE's
    # quotes, apostrophe and colon, NEG_PHRASE (each new subject, its word count, its aside, "be", "been", "found",
    # read as written), and the second reviewer's survivors (END's word ends, the window)
    ("grade_cues", " if not perfect(text, m)]", "]"),                                            # a2b_r2_has_not_found
    ("grade_cues", r"|'ve)(?:\s+\w+)?\s+$", r"|'ve)\s+$"),                                  # a2b_r3_has_also_not_found
    ("grade_cues", r"|'ve)(?:\s+\w+)?\s+$", r"|'ve)(?:\s+\w+){0,2}\s+$"),                          # a2b_r3_have_here_ok
    ("grade_cues", 'return m.group("nf") is not None and PERFECT', "return PERFECT"),     # a2b_r3_has_gone_missing_ok
    ("grade_cues", r"(?:the KEY(?: \w+)?|KEY)", "(?:the KEY|KEY)"),                              # a2b_only_the_key_ok
    ("grade_cues", r"(?:the KEY(?: \w+)?|KEY)", r"(?:the KEY(?: \w+)*|KEY)"),                # a2b_r3_only_slot_crossed
    ("grade_cues", r'r"(?:s)?(?: (?:on|in)', r'r"(?: (?:on|in)'),                                # a2b_r3_only_plural_ok
    ("grade_cues", "(?: (?:on|in) (?:it|there)| listed)?)", ")"),                                 # a2b_r3_only_on_it_ok
    ("grade_cues", " (?:on|in) (?:it|there)", " (?:in) (?:it|there)"),                            # a2b_r3_only_on_it_ok
    ("grade_cues", " (?:on|in) (?:it|there)", " (?:on) (?:it|there)"),                         # a2b_r3_only_in_there_ok
    ("grade_cues", "(?:it|there)| listed", "(?:there)| listed"),                                  # a2b_r3_only_on_it_ok
    ("grade_cues", "(?:it|there)| listed", "(?:it)| listed"),                                  # a2b_r3_only_in_there_ok
    ("grade_cues", "| listed)?", ")?"),                                                          # a2b_r3_only_listed_ok
    ("grade_cues", r'|\s+(?:and|but)\b)"', ')"'),                                                   # a2b_r3_only_and_ok
    ("grade_cues", "(?:and|but)", "(?:but)"),                                                       # a2b_r3_only_and_ok
    ("grade_cues", "(?:and|but)", "(?:and)"),                                                       # a2b_r3_only_but_ok
    ("grade_cues", r"\s+(?:and|but)\b)", r"\s+\w+\b)"),                                         # a2b_r3_only_key_twice
    ("grade_cues", ' + CLAUSE_END + ")")', ' + ")")'),                                         # a2b_r3_only_key_crossed
    ("grade_cues", "(?:a row|an entry|a line) (?:for|on) KEY|", "(?:a row|an entry|a line)(?: (?:for|on) KEY)?|"),
    # a2b_r3_only_a_line
    ("grade_cues", " (?:for|on) KEY|", r" (?:for|on) \w+|"),                                       # a2b_r3_row_for_other
    ("grade_cues", "(?:KEY(?:s)?|it|that)", "(?:KEY|it|that)"),                                     # a2b_r3_on_plural_ok
    ("grade_cues", r"\b(?:'s|'ll|'d)?", r"\b(?:'s|'d)?"),                                              # a2b_r3_on_itll_ok
    ("grade_cues", r"\b(?:'s|'ll|'d)?", r"\b(?:'s|'ll)?"),                                            # a2b_r3_on_thatd_ok
    ("grade_cues", r'LOOKUP_ON = r"\b(?:KEY', 'LOOKUP_ON = r"(?:KEY'),                                # a2b_r3_split_only
    ("grade_cues", "|none|no)", "|no)"),                                                          # a2b_r3_none_comma_ok
    ("grade_cues", "|none|no)", "|none)"),                                                           # a2b_r3_no_dash_ok
    ("grade_cues", r'DASH = r"(?:\s+-{1,2}\s+|', 'DASH = r"(?:'),                                   # a2b_r3_not_dash_aside
    ("grade_cues", r'|\s*[\u2013\u2014]\s*)"', ')"'),                                             # a2b_r3_nothing_em_aside
    ("grade_cues", r'|" + DASH + r"[^.;:!?()\-\u2013\u2014]*?" + DASH + r"|', "|"),               # a2b_r3_not_dash_aside
    ("grade_cues", r'|\s*\([^()]*\))"', ')"'),                                                    # a2b_r3_not_paren_aside
    ("grade_cues", "    m = ASIDE_END.search(before)\n", "    m = None\n"),                      # a2b_r3_isnt_long_aside
    ("grade_cues", '("," if m.group(0).lstrip().startswith(",") else " ")', '" "'),                    # a2b_r3_no_sadly_ok
    ("grade_cues", r"\b){0,2}\s*)$", r"\b){0,1}\s*)$"),                                # a2b_r3_not_aside_been_found
    ("grade_cues", r"(?:\s*\b(?:be|been|is|are|was|were|found)\b)", r"(?:\s*\b[\w']+\b)"),        # a2b_r3_tail_word_ok
    ("grade_cues", "(?:be|been|is|", "(?:been|is|"),                                           # a2b_r3_will_not_aside_be
    ("grade_cues", "(?:be|been|is|", "(?:be|is|"),                                              # a2b_r3_never_aside_been
    ("grade_cues", "|been|is|are|", "|been|are|"),                                               # a2b_r3_not_others_is_ok
    ("grade_cues", "|is|are|was|were|found)", "|is|was|were|found)"),                           # a2b_r3_not_others_are_ok
    ("grade_cues", "|is|are|was|were|found)", "|is|are|were|found)"),                          # a2b_r3_no_day_aside_was
    ("grade_cues", "|were|found)", "|found)"),                                                  # a2b_r3_not_others_were_ok
    ("grade_cues", "|were|found)", "|were)"),                                                   # a2b_r3_not_aside_found
    ("grade_cues", 'INTERJ.sub(lambda m: " " * len(m.group(0)), drop_aside(raw))',
     'INTERJ.sub(lambda m: " " * len(m.group(0)), raw)'),                                       # a2b_r3_isnt_long_aside
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?!\s*[^\w\s'])"),                              # a2b_r3_nothing_quoted
    ("grade_cues", r"[^\w\s'\x22])", r"[^\w\s'])"),                                                # a2b_r3_nothing_quoted
    ("grade_cues", r"[^\w\s'\x22])", r"[^\w\s\x22])"),                                           # a2b_r2_nobodys_missing
    ("grade_cues", r"(?![\s\x22]*", r"(?!\s*"),                                                  # a2b_r3_nobody_quote_ok
    ("grade_cues", r"(?![\s\x22]*[^\w\s'\x22])", r"(?![\s\x22]*[,\-\u2013\u2014])"),               # a2b_r3_nobody_colon_ok
    ("grade_cues", "|none|not a single|not one|no single)", "|none|not one|no single)"),           # a2b_r3_not_a_single
    ("grade_cues", "|none|not a single|not one|no single)", "|none|not a single|no single)"),          # a2b_r3_not_one
    ("grade_cues", "|none|not a single|not one|no single)", "|none|not a single|not one)"),          # a2b_r3_no_single
    ("grade_cues", r'\b(?:\s+\w+){0,3}" + ASIDE', r'\b(?:\s+\w+){0,2}" + ASIDE'),             # a2b_r3_none_of_the
    ("grade_cues", r'\b(?:\s+\w+){0,3}" + ASIDE', r'\b(?:\s+\w+){0,4}" + ASIDE'),      # a2b_r3_nobody_noticed_ok
    ("grade_cues", '{0,3}" + ASIDE\n                        + r"?', '{0,3}"\n                        + r"'),
    # a2b_r2_nobody_aside
    ("grade_cues", '{0,3}" + ASIDE\n                        + r"?', r'{0,3}(?:\s*,[^,.;:!?]*,)?"' + '\n                        + r"'),
    # a2b_r3_nobody_words_dash
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:is|are|was|were|been)"),                       # a2b_r3_found_to_be
    ("grade_cues", "(?:is|are|was|were|be|been)", "(?:is|are|was|were|be)"),                     # a2b_r3_nobody_has_been
    ("grade_cues", r"be|been)(?:\s+found)?\s+$", r"be|been)\s+$"),                           # a2b_r3_nothing_comma_found
    ("grade_cues", "NEG_PHRASE.search(raw)", "NEG_PHRASE.search(before)"),                        # a2b_r2_none_aside_are
    ("grade_cues", r"(?:for|on|in|about)\b|", r"(?:for|on|in|about)|"),               # a2b_r3_found_nothing_inconsistent
    ("grade_cues", 'w0 = ([m.start() for m in TOKEN.finditer(before)][-T.P["neg_window"]:]',
     "w0 = ([m.start() for m in TOKEN.finditer(before)][-2:]"),                                     # a2b_r3_nothing_seems
    ("grade_cues", '''TOKEN = re.compile(r"[A-Za-z0-9']+")''', '''TOKEN = re.compile(r"[A-Za-z0-9]+")'''),
    # a2b_r3_nobodys_ever
    ("graders", 'rec.get("family") == "LOOKUP"', 'rec.get("family") in ("LOOKUP", "OODH")'),  # a2b_scope_relabelled_oodh
    # the a4 frame: "I found nothing" joined
    ("grade_cues", r' + r"|\bi found nothing\b"', ""),                                        # a2b_i_found_nothing_ok
    # graders.g_abs: the scope (LOOKUP records only), the cue joined, the a4 exemption (off; on every ABS probe; the
    # whole reply or the whole sentence instead of the frame's words; every LOOKUP_ABS match instead of the frame)
    ("graders", 'rec.get("family") == "LOOKUP"', "True"),                                          # a2b_scope_couldnt
    ("graders", " or (look and T.GC.lookup_cue(text, rec, probe))", ""),                          # a2b_nothing_came_ok
    ("graders", 'T.GC.FIND_FRAME.sub(" ", text) if look else text', "text"),                      # a2b_couldnt_find_ok
    ("graders", 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     'T.GC.FIND_FRAME.sub(" ", text)'),                                                              # a2b_scope_voice
    ("graders", 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     '("" if T.GC.FIND_FRAME.search(text) else text) if look else text'),                           # a2b_voice_on_key
    ("graders", 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     're.sub(r"[^.!?]*(?:" + T.GC.LOOKUP_FIND + r")[^.!?]*", " ", text, flags=re.I) if look '
     'else text'),                                                                               # a2b_voice_same_sent
    ("graders", 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     'T.GC._LOOKUP.sub(" ", text) if look else text'),                                        # a2b_i_found_nothing_ok
]
