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
reviewer's survivors. Lock step A2d (2026-11-23, notes STEP 16A2D) replaced the window, aside, subject-phrase and
perfect rules with the fail-closed rule (grade_cues' module docstring), so their mutants are gone; the mutants of the
lists it kept (LOOKUP_FIND, LOOKUP_ABS, END, the question gate, the scope, the a4 exemption, the openings) stay, and one
per A2d clause is added (emphasis, the sentence breaks and end, the "?" and "~~" rules, the openings at the start and
in a run, each LOOKUP_NEG word and auxiliary, HAVE, the quantifier ends, "missing / absent from the <other table>",
each part of ONLY_SHAPE and LOOKUP_ON, any clean hit). Each A2d mutant survives the fixtures without X_OK_A2D and
X_FAIL_A2D and is killed by the kind named beside it; one named after an older kind survives the fixtures without
fixtures_a2b and fixtures_a2c and is killed by that kind. A2d's fix round (2026-11-23, notes STEP 16A2D-R) retargets
the mutants of the text it rewrote (the openings' marks, the "?" and strike rules, the voided call) and adds one per
new clause (the emphasis read as a space, each opening mark and the hyphen's letter rule, each new LOOKUP_NEG word,
auxiliary and apostrophe look-alike, NONE_CUE and each PREDICATE word and its reach, each QMARK character and its
reach, each STRIKE mark, its case and its reach, each part of FROM_TABLE, a far HAVE, ON's verb); a mutant named after
one of its kinds (a2d_fr_*, a2d_conservative_fr_*) survives the fixtures without its three lists and is killed by
that kind, and the older quantifier and END mutants that the predicate rule hides from their old kinds are named
after the fix round's kinds that isolate them. LOOKUP_ONLY and ONLY_ON in their broad forms are now read
only by lookup_any (test_gens' content rule; its record mutants are in mutation_gens), so no grader mutant targets
them. Not mutated: word boundaries (as ABS_MORE) but those the reviewers' survivors showed, re.escape of the key (dev
keys are weekdays and months), the spaces allowed before a shape's end mark, the lower bound of ONLY's word count, and
"?" in SHAPE_END (equivalent: a cue whose sentence holds a "?" never counts)."""

SOURCE_A2B = [
    ('grade_cues', 'r"(?:\\b(?:i|we|', 'r"(?:\\b(?:\\w+|we|'),                                      # a2b_they_couldnt
    ('grade_cues', "(?:couldn't|could not|", '(?:could not|'),                                   # a2b_couldnt_find_ok
    ('grade_cues', "|could not|can't|", "|can't|"),                                        # a2b_could_not_anything_ok
    ('grade_cues', "|can't|cannot|", '|cannot|'),                                                    # a2b_cant_see_ok
    ('grade_cues', "|cannot|didn't|", "|didn't|"),                                              # a2b_cannot_locate_ok
    ('grade_cues', "|cannot|didn't|did not|", '|cannot|did not|'),                                 # a2b_didnt_spot_ok
    ('grade_cues', "|didn't|did not|don't|", "|didn't|don't|"),                                  # a2b_did_not_find_ok
    ('grade_cues', '|do not|was unable to"', '|do not"'),                                         # a2b_unable_find_ok
    ('grade_cues', '(?:find|locate|see|spot)', '(?:locate|see|spot)'),                           # a2b_couldnt_find_ok
    ('grade_cues', '|locate|see|', '|see|'),                                                    # a2b_cannot_locate_ok
    ('grade_cues', '|see|spot)', '|spot)'),                                                          # a2b_cant_see_ok
    ('grade_cues', '|spot)\\b"', ')\\b"'),                                                         # a2b_didnt_spot_ok
    ('grade_cues', 'r"(?:\\b(?:i|we|(?:the|my)', 'r"(?:\\b(?:i|(?:the|my)'),                               # a2b_we_ok
    ('grade_cues', '|we|(?:the|my) (?:lookup|search))(?: (?:just', '|we)(?: (?:just'),             # a2b_the_lookup_ok
    ('grade_cues', '(?:the|my) (?:lookup', '(?:my) (?:lookup'),                                    # a2b_the_lookup_ok
    ('grade_cues', '(?:the|my) (?:lookup', '(?:the) (?:lookup'),                                    # a2b_my_search_ok
    ('grade_cues', '(?:lookup|search))', '(?:search))'),                                           # a2b_the_lookup_ok
    ('grade_cues', '(?:lookup|search))', '(?:lookup))'),                                            # a2b_my_search_ok
    ('grade_cues', '(?:just|still|', '(?:still|'),                                                       # a2b_just_ok
    ('grade_cues', '(?:just|still|really|', '(?:just|really|'),                                         # a2b_still_ok
    ('grade_cues', '|still|really|honestly|', '|still|honestly|'),                                     # a2b_really_ok
    ('grade_cues', '|really|honestly|simply)', '|really|simply)'),                                   # a2b_honestly_ok
    ('grade_cues', '|honestly|simply))?', '|honestly))?'),                                             # a2b_simply_ok
    ('grade_cues', '(?: (?:just|still|really|honestly|simply))? "', ' "'),                              # a2b_still_ok
    ('grade_cues', '(?: (?:just|still|really|honestly|simply))? "', '(?: \\w+)? "'),              # a2b_almost_couldnt
    ('grade_cues', 'r"|(?:^|(?<=[.!?;:,]))\\s*)', 'r")'),                                      # a2b_subjless_comma_ok
    ('grade_cues', '(?:^|(?<=[.!?;:,]))', '(?:(?<=[.!?;:,]))'),                                # a2b_subjless_start_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[!?;:,]))'),                                        # a2b_subjless_period_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[.?;:,]))'),                                          # a2b_subjless_bang_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[.!;:,]))'),                                             # a2b_subjless_q_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[.!?:,]))'),                                          # a2b_subjless_semi_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[.!?;,]))'),                                         # a2b_subjless_colon_ok
    ('grade_cues', '(?<=[.!?;:,]))', '(?<=[.!?;:]))'),                                         # a2b_subjless_comma_ok
    ('grade_cues', '(?<=[.!?;:,]))\\s*)', '(?<=[.!?;:,])))'),                                  # a2b_subjless_comma_ok
    ('grade_cues', "|did not|don't|do not|", '|did not|do not|'),                                    # a2b_dont_see_ok
    ('grade_cues', "|don't|do not|was", "|don't|was"),                                             # a2b_do_not_see_ok
    ('grade_cues', 'r"|wasn\'t able to|', 'r"|'),                                                  # a2b_wasnt_able_ok
    ('grade_cues', '|was not able to)"', ')"'),                                                  # a2b_was_not_able_ok
    ('grade_cues', 'r"(?: seem to)? (?:find', 'r" (?:find'),                                             # a2b_seem_ok
    ('grade_cues', 'LOOKUP_ABS = (LOOKUP_FIND + r"|', 'LOOKUP_ABS = (r"'),                       # a2b_couldnt_find_ok
    ('grade_cues', '|\\bnothing (?:was )?found\\b" + END', '"'),                                # a2b_nothing_found_ok
    ('grade_cues', 'nothing (?:was )?found', 'nothing found'),                              # a2b_nothing_was_found_ok
    ('grade_cues', '(?:turned up|found) nothing', '(?:found) nothing'),                     # a2b_turned_up_nothing_ok
    ('grade_cues', '|found) nothing', ') nothing'),                                             # a2b_found_nothing_ok
    ('grade_cues', 'r"|\\bcame up empty\\b"', 'r""'),                                           # a2b_came_up_empty_ok
    ('grade_cues', '(?P<nf>not)(?: be)?|', '(?P<nf>(?!))(?: be)?|'),                                # a2b_not_found_ok
    ('grade_cues', '|cannot be|', '|'),                                                       # a2b_cannot_be_found_ok
    ('grade_cues', "(?:was|were|is|are)n't", "(?:were|is|are)n't"),                               # a2b_wasnt_found_ok
    ('grade_cues', "|(?:could|ca|wo)n't be) found", ') found'),                                      # a2b_be_found_ok
    ('grade_cues', '(?:result|match|', '(?:match|'),                                               # a2b_no_results_ok
    ('grade_cues', '|match|entr|', '|entr|'),                                                        # a2b_no_match_ok
    ('grade_cues', '|entr|row|', '|row|'),                                                           # a2b_no_entry_ok
    ('grade_cues', '|row|record|', '|record|'),                                                        # a2b_no_row_ok
    ('grade_cues', '|record|information|', '|information|'),                                        # a2b_no_record_ok
    ('grade_cues', '|information|luck)', '|luck)'),                                            # a2b_no_information_ok
    ('grade_cues', '|information|luck)', '|information)'),                                            # a2b_no_luck_ok
    ('grade_cues', '|luck)|\\b(?:missing', '|luck)\\b|\\b(?:missing'),                             # a2b_no_results_ok
    ('grade_cues', '(?:missing|absent) from', '(?:absent) from'),                                     # a2b_missing_ok
    ('grade_cues', '|absent) from', ') from'),                                                         # a2b_absent_ok
    ('grade_cues', '(re.compile(x, re.I) for x in (LOOKUP_FRAME', '(re.compile(x) for x in (LOOKUP_FRAME'),
    # a2b_nothing_came_ok
    ('grade_cues', '    return objs - {asked}', '    return objs'),                                   # a2b_only_asked
    ('grade_cues', '.split(":", 1)[-1].lower()', '.lower()'),                                         # a2b_only_asked
    ('grade_cues', 'return not lookup_hit(T.norm(probe.get("question")), others, key) and lookup_hit(text, others, key)',
     'return lookup_hit(text, others, key)'),
    # a2b_echo_question
    ('grade_cues', 'lookup_hit(T.norm(probe.get("question")), others, key)',
     'lookup_hit(probe.get("question") or "", others, key)'),
    # a2b_echo_curly
    ('grade_cues', 'lookup_hit(T.norm(probe.get("question")), others, key)',
     'lookup_hit(T.norm(probe.get("question")), set(), key)'),
    # a2b_echo_only
    ('grade_cues', 'not lookup_hit(T.norm(probe.get("question")), others, key)',
     'not lookup_any(T.norm(probe.get("question")))'),
    # a2b_negq_ok
    ('grade_cues', 'lookup_tables(rec, probe), probe.get("key")', 'lookup_tables(rec, probe), None'),
    # a2b_only_other_has_ok
    ('grade_cues', '(?:was )?found\\b" + END', '(?:was )?found\\b"'),               # a2d_fr_end_nothing_found_lacking
    ('grade_cues', 'found) nothing\\b" + END', 'found) nothing\\b"'),               # a2d_fr_end_found_nothing_lacking
    ('grade_cues', '(?:for|on|in|about)\\b|', '(?:on|in|about)\\b|'),                           # a2b_nothing_found_ok
    ('grade_cues', '(?:for|on|in|about)\\b|', '(?:for|in|about)\\b|'),                              # a2b_r2_end_on_ok
    ('grade_cues', '(?:for|on|in|about)\\b|', '(?:for|on|about)\\b|'),                              # a2b_r2_end_in_ok
    ('grade_cues', '(?:for|on|in|about)\\b|', '(?:for|on|in)\\b|'),                              # a2b_r2_end_about_ok
    ('grade_cues', 'nothing (?:was )?found', 'nothing (?:\\w+ )?found'),                # a2d_fr_end_nothing_new_found
    ('grade_cues', '(?P<nf>not)(?: be)?|cannot', '(?P<nf>not)|cannot'),                 # a2b_r2_could_not_be_found_ok
    ('grade_cues', "(?:was|were|is|are)n't", "(?:was|is|are)n't"),                            # a2b_r2_werent_found_ok
    ('grade_cues', "(?:was|were|is|are)n't", "(?:was|were|are)n't"),                            # a2b_r2_isnt_found_ok
    ('grade_cues', "(?:was|were|is|are)n't", "(?:was|were|is)n't"),                            # a2b_r2_arent_found_ok
    ('grade_cues', "(?:was|were|is|are)n't", "(?:was|were|is|are)(?:n't)?"),                        # a2b_r2_was_found
    ('grade_cues', "(?:was|were|is|are)n't", "\\w+n't"),                                         # a2b_r2_havent_found
    ('grade_cues', "(?:could|ca|wo)n't be", "(?:ca|wo)n't be"),                                      # a2b_be_found_ok
    ('grade_cues', "(?:could|ca|wo)n't be", "(?:could|wo)n't be"),                           # a2b_r2_cant_be_found_ok
    ('grade_cues', "(?:could|ca|wo)n't be", "(?:could|ca)n't be"),                           # a2b_r2_wont_be_found_ok
    ('grade_cues', 'if m.group("t").lower() in ok]', ']'),                                         # a2b_only_on_asked
    ('grade_cues', 'm.group("t").lower() in ok', 'm.group("t") in ok'),                           # a2b_only_on_cap_ok
    ('grade_cues', 'LOOKUP_ON.replace("KEY", re.escape(key) if key else "(?!)"), re.I)',
     'LOOKUP_ON.replace("KEY", re.escape(key) if key else "(?!)"))'),
    # a2b_r2_on_it_ok
    ('grade_cues', "(?:i'm |i am )?afraid", '(?:i am )?afraid'),                             # a2b_afraid_not_comma_ok
    ('grade_cues', "(?:i'm |i am )?afraid", "(?:i'm )?afraid"),                                # a2b_r2_i_am_afraid_ok
    ('grade_cues', "(?:i'm |i am )?afraid", "(?:i'm |i am )afraid"),                                # a2b_r2_afraid_ok
    ('grade_cues', '|sadly|', '|'),                                                           # a2b_sadly_not_colon_ok
    ('grade_cues', '|unfortunately|', '|'),                                                  # a2b_r2_unfortunately_ok
    ('grade_cues', '|(?:i )?guess|', '|i guess|'),                                                   # a2b_r2_guess_ok
    ('grade_cues', '|(?:i )?guess|', '|guess|'),                                                   # a2b_r2_i_guess_ok
    ('grade_cues', "(?:i )?(?:don't|do not) think so", '(?:i )?(?:do not) think so'),         # a2b_dont_think_semi_ok
    ('grade_cues', "(?:i )?(?:don't|do not) think so", "(?:i )?(?:don't) think so"),          # a2b_r2_do_not_think_ok
    ('grade_cues', "(?:i )?(?:don't|do not) think so", "i (?:don't|do not) think so"),      # a2b_r2_dont_think_noi_ok
    ('grade_cues', 'not (?:really|sure|this time)', 'not (?:sure|this time)'),                  # a2b_r2_not_really_ok
    ('grade_cues', 'not (?:really|sure|this time)', 'not (?:really|this time)'),                  # a2b_r2_not_sure_ok
    ('grade_cues', 'not (?:really|sure|this time)', 'not (?:really|sure)'),                  # a2b_r2_not_this_time_ok
    ('grade_cues', 'r"(?:\\b(?:i|we|', 'r"(?:\\b(?:i|we|you|he|she|it|'),                         # a2b_r2_she_couldnt
    ('grade_cues', '(?:^|(?<=[.!?;:,]))\\s*)', '(?:^|(?<=[.!?;:,])|(?<=\\band)|(?<=\\bbut))\\s*)'), # a2b_r2_and_couldnt
    ('grade_cues', '(?<=[.!?;:,]))\\s*)', '(?<=[.!?;:,\\-]))\\s*)'),                             # a2b_r2_dash_couldnt
    ('grade_cues', '|was not able to)"', '|was not able to|won\'t|wouldn\'t)"'),                  # a2b_r2_wouldnt_see
    ('grade_cues', 'LOOKUP_ON = r"\\b(?:KEY', 'LOOKUP_ON = r"(?:KEY'),                             # a2b_r3_split_only
    ('grade_cues', '(?:for|on|in|about)\\b|', '(?:for|on|in|about)|'),              # a2d_fr_end_found_nothing_in_word
    ('graders', 'rec.get("family") == "LOOKUP"', 'rec.get("family") in ("LOOKUP", "OODH")'), # a2b_scope_relabelled_oodh
    ('grade_cues', ' + r"|\\bi found nothing\\b"', ''),                                       # a2b_i_found_nothing_ok
    ('graders', 'rec.get("family") == "LOOKUP"', 'True'),                                          # a2b_scope_couldnt
    ('graders', ' or (look and T.GC.lookup_cue(text, rec, probe))', ''),                         # a2b_nothing_came_ok
    ('graders', 'T.GC.FIND_FRAME.sub(" ", text) if look else text', 'text'),                     # a2b_couldnt_find_ok
    ('graders', 'T.GC.FIND_FRAME.sub(" ", text) if look else text', 'T.GC.FIND_FRAME.sub(" ", text)'), # a2b_scope_voice
    ('graders', 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     '("" if T.GC.FIND_FRAME.search(text) else text) if look else text'),
    # a2b_voice_on_key
    ('graders', 'T.GC.FIND_FRAME.sub(" ", text) if look else text',
     're.sub(r"[^.!?]*(?:" + T.GC.LOOKUP_FIND + r")[^.!?]*", " ", text, flags=re.I) if look else text'),
    # a2b_voice_same_sent
    ('graders', 'T.GC.FIND_FRAME.sub(" ", text) if look else text', 'T.GC._LOOKUP.sub(" ", text) if look else text'),
    # a2b_i_found_nothing_ok
    ('grade_cues', '(?:came|turned|showed) up\\b', '(?:turned|showed) up\\b'),                   # a2b_nothing_came_ok
    ('grade_cues', '|turned|showed) up\\b', '|showed) up\\b'),                                 # a2b_nothing_turned_ok
    ('grade_cues', '|showed) up\\b', ') up\\b'),                                               # a2b_nothing_showed_ok
    ('grade_cues', 'about)\\b|\\s*[^\\w\\s\']|\\s*$)"', 'about)\\b|\\s*$)"'),                     # a2b_r2_end_mark_ok
    ('grade_cues', 'about)\\b|\\s*[^\\w\\s\']|\\s*$)"', 'about)\\b|\\s*[^\\w\\s\'])"'),            # a2b_r2_end_end_ok
    ('grade_cues', '|sadly|unfortunately|(?:i )?guess|probably|apparently) not',
     '|sadly|unfortunately|(?:i )?guess|apparently) not'),
    # a2b_r2_probably_ok
    ('grade_cues', '|(?:i )?guess|probably|apparently) not', '|(?:i )?guess|probably) not'),    # a2b_r2_apparently_ok
    ('grade_cues', 'EMPHASIS = re.compile(r"[*_`]")', 'EMPHASIS = re.compile(r"[_`]")'),              # a2d_md_star_ok
    ('grade_cues', 'EMPHASIS = re.compile(r"[*_`]")', 'EMPHASIS = re.compile(r"[*`]")'),             # a2d_md_under_ok
    ('grade_cues', 'EMPHASIS = re.compile(r"[*_`]")', 'EMPHASIS = re.compile(r"[*_]")'),              # a2d_md_tick_ok
    ('grade_cues', 't, ok = EMPHASIS.sub("", text), others', 't, ok = text, others'),                 # a2d_md_star_ok
    ('grade_cues', '(?:(?<!\\.)\\.|[!?])(?=', '(?:\\.|[!?])(?='),                                  # a2d_ellipsis_isnt
    ('grade_cues', '(?:(?<!\\.)\\.|[!?])(?=', '(?:[!?])(?='),                                # a2b_r2_prev_sentence_ok
    ('grade_cues', '|[!?])(?=\\s|', '|[?])(?=\\s|'),                                                     # a2d_bang_ok
    ('grade_cues', '|[!?])(?=\\s|', '|[!])(?=\\s|'),                                                  # a2d_q_break_ok
    ('grade_cues', '(?=\\s|[\\"\']\\s|[\\"\']?$)|\\n")', '(?=\\s|[\\"\']\\s|[\\"\']?$)")'),           # a2d_newline_ok
    ('grade_cues', '(?=\\s|[\\"\']\\s|', '(?=[\\"\']\\s|'),                                  # a2b_r2_prev_sentence_ok
    ('grade_cues', '(?=\\s|[\\"\']\\s|', '(?=\\s|'),                                              # a2d_quote_break_ok
    ('grade_cues', '    s0 = max([b.end() for b in BREAK.finditer(t) if b.end() <= s], default=0)\n', '    s0 = 0\n'),
    # a2b_r2_prev_sentence_ok
    ('grade_cues', '    return s0, (b.start() if b else len(t))\n', '    return s0, len(t)\n'),    # a2d_fr_q_later_ok
    ('grade_cues', '    if QMARK.search(t, s0, s1 + 1) or STRIKE.search(t):', '    if STRIKE.search(t):'),
    # a2d_q_missing
    ('grade_cues', 'QMARK = re.compile(r"[?\\uff1f', 'QMARK = re.compile(r"[\\uff1f'),              # a2d_q_interrobang
    ('grade_cues', ' or STRIKE.search(t):', ':'),                                                         # a2d_strike
    ('grade_cues', '    before = t[s0:s]\n', '    before = t[s0:e]\n'),                          # a2b_couldnt_find_ok
    ('grade_cues', '    o = OPENERS.match(before)\n', '    o = OPENERS.search(before)\n'), # a2b_r2_interj_midsentence
    ('grade_cues', '    before = before[o.end():] if o else before\n', ''),                          # a2b_no_comma_ok
    ('grade_cues', '(?![a-z])))+", re.I)', '(?![a-z])))", re.I)'),                       # a2b_r2_interj_after_comma_ok
    ('grade_cues', 'r"\\s*(?:[,;:.\\u2013\\u2014\\u2026]|-(?![a-z])))+", re.I)', 'r"\\s*)+", re.I)'),
    # a2b_r2_interj_no_mark
    ('grade_cues', '\\u2026]|-(?![a-z])))+', '\\u2026]))+'),
    # a2b_r2_interj_dash_ok
    ('grade_cues', '(?:[,;:.\\u2013', '(?:[,:.\\u2013'),                                        # a2b_dont_think_semi_ok
    ('grade_cues', 'OPENERS = re.compile(r"(?:\\s*(?:', 'OPENERS = re.compile(r"(?:(?:'), # a2b_r2_interj_after_comma_ok
    ('grade_cues', '|this time)|no|sorry)', '|this time)|sorry)'),                                   # a2b_no_comma_ok
    ('grade_cues', '|this time)|no|sorry)', '|this time)|no)'),                         # a2b_r2_interj_after_comma_ok
    ('grade_cues', '\\b(?:not|no|never|', '\\b(?:no|never|'),                                           # a2b_not_only
    ('grade_cues', '\\b(?:not|no|never|', '\\b(?:not|never|'),                                      # a2b_r3_no_single
    ('grade_cues', '|no|never|nothing|', '|no|nothing|'),                                           # a2b_never_absent
    ('grade_cues', '|never|nothing|nobody|', '|never|nobody|'),                                  # a2b_nothing_missing
    ('grade_cues', '|nothing|nobody|none|', '|nothing|none|'),                                    # a2b_nobody_missing
    ('grade_cues', '|nobody|none|nowhere|', '|nobody|nowhere|'),                                  # a2b_r3_none_of_the
    ('grade_cues', '|none|nowhere|neither|', '|none|neither|'),                                          # a2d_nowhere
    ('grade_cues', '|nowhere|neither|nor|', '|nowhere|nor|'),                                            # a2d_neither
    ('grade_cues', '|neither|nor|without|', '|neither|without|'),                                            # a2d_nor
    ('grade_cues', '|nor|without|cannot|', '|nor|cannot|'),                                              # a2d_without
    ('grade_cues', '|without|cannot|hardly|', '|without|hardly|'),                                        # a2d_cannot
    ('grade_cues', '|cannot|hardly|scarcely"', '|cannot|scarcely"'),                                      # a2d_hardly
    ('grade_cues', '|hardly|scarcely"', '|hardly"'),                                                    # a2d_scarcely
    ('grade_cues', 'r"|barely|far from|', 'r"|far from|'),                                                # a2d_barely
    ('grade_cues', '|barely|far from|anything but|', '|barely|anything but|'),                          # a2d_far_from
    ('grade_cues', '|far from|anything but|instead of|', '|far from|instead of|'),                  # a2d_anything_but
    ('grade_cues', '|anything but|instead of|rather than)', '|anything but|rather than)'),            # a2d_instead_of
    ('grade_cues', '|instead of|rather than)', '|instead of)'),                                      # a2d_rather_than
    ('grade_cues', 'rather than)\\b|n\'t\\b"', 'rather than)\\b"'),                                  # a2b_not_missing
    ('grade_cues', '\\b(?:is|are|was|were|do|', '\\b(?:are|was|were|do|'),                                  # a2d_isnt
    ('grade_cues', '\\b(?:is|are|was|were|do|', '\\b(?:is|was|were|do|'),                                  # a2d_arent
    ('grade_cues', '|are|was|were|do|does|', '|are|were|do|does|'),                                        # a2d_wasnt
    ('grade_cues', '|was|were|do|does|', '|was|do|does|'),                                                # a2d_werent
    ('grade_cues', '|were|do|does|did|', '|were|does|did|'),                                                # a2d_dont
    ('grade_cues', '|do|does|did|has|', '|do|did|has|'),                                                  # a2d_doesnt
    ('grade_cues', '|does|did|has|have|', '|does|has|have|'),                                              # a2d_didnt
    ('grade_cues', '|did|has|have|had|ca|', '|did|have|had|ca|'),                                          # a2d_hasnt
    ('grade_cues', '|has|have|had|ca|', '|has|had|ca|'),                                                  # a2d_havent
    ('grade_cues', '|have|had|ca|could|', '|have|ca|could|'),                                              # a2d_hadnt
    ('grade_cues', '|had|ca|could|wo)nt', '|had|could|wo)nt'),                                              # a2d_cant
    ('grade_cues', '|ca|could|wo)nt', '|ca|wo)nt'),                                                      # a2d_couldnt
    ('grade_cues', '|could|wo)nt', '|could)nt'),                                                            # a2d_wont
    ('grade_cues', '    return "neg" not in T.OFF and LOOKUP_NEG.search(before) is not None', '    return False'),
    # a2b_not_missing
    ('grade_cues', '(?:have|has|had|having)', '(?:has|had|having)'),                           # a2b_r2_have_not_found
    ('grade_cues', '(?:have|has|had|having)', '(?:have|had|having)'),                           # a2b_r2_has_not_found
    ('grade_cues', '(?:have|has|had|having)', '(?:have|has|having)'),                           # a2b_r2_had_not_found
    ('grade_cues', '(?:have|has|had|having)', '(?:have|has|had)'),                              # a2d_having_not_found
    ('grade_cues', '\\b|\'ve\\b", re.I)', '\\b", re.I)'),                                        # a2b_r2_ve_not_found
    ('grade_cues', 'HAVE.search(t, sentence(t, m.start(), m.end())[0], m.start())',
     're.search(r"(?:\\b(?:have|has|had|having)|\'ve)(?:\\s+\\w+)?\\s+$", t[:m.start()], re.I)'),
    # a2d_conservative_r3_have_here
    ('grade_cues', '    return m.group("nf") is None or HAVE.search(', '    return HAVE.search('),
    # a2b_r3_has_gone_missing_ok
    ('grade_cues', '    return m.group("nf") is None or HAVE.search(', '    return True or HAVE.search('),
    # a2b_r2_has_not_found
    ('grade_cues', 'QUANT = re.compile(r"no |nothing', 'QUANT = re.compile(r"nothing'),     # a2d_fr_quant_no_row_gets
    ('grade_cues', 'nothing (?:came|turned|showed) up", re.I)', 'nothing (?:turned|showed) up", re.I)'),
    # a2d_fr_quant_came_short
    ('grade_cues', 'nothing (?:came|turned|showed) up", re.I)', 'nothing (?:came|showed) up", re.I)'),
    # a2d_fr_quant_turned_short
    ('grade_cues', 'nothing (?:came|turned|showed) up", re.I)', 'nothing (?:came|turned) up", re.I)'),
    # a2d_fr_quant_showed_short
    ('grade_cues', '    if QUANT.match(m.group(0)) and not QUANT_END.match(t, m.end()):\n        return False\n', ''),
    # a2d_fr_quant_no_row_gets
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:on|in|about|of|with)'),                      # a2b_no_results_ok
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:for|in|about|of|with)'),                 # a2b_no_information_ok
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:for|on|about|of|with)'),                    # a2d_no_entry_in_ok
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:for|on|in|of|with)'),                     # a2d_no_info_about_ok
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:for|on|in|about|with)'),                      # a2b_no_record_ok
    ('grade_cues', '(?:for|on|in|about|of|with)', '(?:for|on|in|about|of)'),                          # a2b_no_luck_ok
    ('grade_cues', 'with)\\b|\\s*[^\\w\\s\']|\\s*$)", re.I)', 'with)\\b|\\s*$)", re.I)'),     # a2d_no_results_mark_ok
    ('grade_cues', '|\\s*[^\\w\\s\']|\\s*$)", re.I)', '|\\s*[^\\w\\s\'])", re.I)'),            # a2d_no_results_end_ok
    ('grade_cues', '\\w*(?: found)?(?=', '\\w*(?='),                                         # a2d_no_results_found_ok
    ('grade_cues', '    if m.group(0).lower().startswith(("missing", "absent")) and words & ok:',
     '    if False and words & ok:'),
    # a2d_missing_from_other
    ('grade_cues', 'startswith(("missing", "absent"))', 'startswith(("absent",))'),           # a2d_missing_from_other
    ('grade_cues', 'startswith(("missing", "absent"))', 'startswith(("missing",))'),           # a2d_absent_from_other
    ('grade_cues', 'and words & ok:', 'and words & (ok - {"other"}):'),                    # a2d_missing_from_other_word
    ('grade_cues', 'and words & ok:', 'and words & {"other"}:'),                                # a2d_missing_from_other
    ('grade_cues', '{f.group(1).lower(), (f.group(2) or "").lower()}', '{f.group(1), f.group(2) or ""}'),
    # a2d_missing_from_other_cap
    ('grade_cues', '{f.group(1).lower(), (f.group(2) or "").lower()}', '{f.group(1).lower()}'),
    # a2d_missing_from_other
    ('grade_cues', ' (?:has|lists|includes) (?:KEY', ' (?:lists|includes) (?:KEY'),            # a2b_only_other_has_ok
    ('grade_cues', ' (?:has|lists|includes) (?:KEY', ' (?:has|includes) (?:KEY'),            # a2b_only_other_lists_ok
    ('grade_cues', ' (?:has|lists|includes) (?:KEY', ' (?:has|lists) (?:KEY'),                  # a2b_only_includes_ok
    ('grade_cues', '(?: \\w+){0,2} (?:has|lists|includes) (?:KEY', '(?: \\w+){0,1} (?:has|lists|includes) (?:KEY'),
    # a2b_only_two_words_ok
    ('grade_cues', '(?: \\w+){0,2} (?:has|lists|includes) (?:KEY', '(?: \\w+)* (?:has|lists|includes) (?:KEY'),
    # a2b_only_far
    ('grade_cues', '(?:KEY(?:s)?|it|that)" + SHAPE_END', '(?:it|that)" + SHAPE_END'),          # a2b_only_other_has_ok
    ('grade_cues', '(?:KEY(?:s)?|it|that)" + SHAPE_END', '(?:KEY|it|that)" + SHAPE_END'),      # a2b_r3_only_plural_ok
    ('grade_cues', '(?:KEY(?:s)?|it|that)" + SHAPE_END', '(?:KEY(?:s)?|that)" + SHAPE_END'),      # a2b_only_has_it_ok
    ('grade_cues', '(?:KEY(?:s)?|it|that)" + SHAPE_END', '(?:KEY(?:s)?|it)" + SHAPE_END'),      # a2b_only_has_that_ok
    ('grade_cues', 'only_rx(key).finditer(t) if m.group(1).lower() in ok]', 'only_rx(key).finditer(t)]'),
    # a2b_only_asked
    ('grade_cues', 'only_rx(key).finditer(t) if m.group(1).lower() in ok]',
     'only_rx(key).finditer(t) if m.group(1) in ok]'),
    # a2b_only_cap_ok
    ('grade_cues', ' \\\n        + [m.span() for m in only_rx(key).finditer(t) if m.group(1).lower() in ok]', ''),
    # a2b_only_other_has_ok
    ('grade_cues', 'ONLY_SHAPE.replace("KEY", re.escape(key) if key else "(?!)"), re.I)',
     'ONLY_SHAPE.replace("KEY", re.escape(key) if key else "(?!)"))'),
    # a2b_only_other_has_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[!;\\n]|$))"'),                                     # a2b_only_other_has_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.;\\n]|$))"'),                                          # a2d_only_bang_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.!\\n]|$))"'),                                          # a2d_only_semi_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.!;]|$))"'),                                         # a2d_only_newline_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.!;\\n]))"'),                                      # a2b_r2_only_it_end_ok
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.!;\\n,]|$))"'),                                      # a2d_only_comma_but
    ('grade_cues', '(?:[.!;\\n]|$))"', '(?:[.!;\\n]|$)|\\s+(?:and|but)\\b)"'),          # a2d_conservative_r3_only_and
    ('grade_cues', 'r"\\b(?:KEY(?:s)?|it)(?: (?:is', 'r"\\b(?:it)(?: (?:is'),                         # a2b_only_on_ok
    ('grade_cues', 'r"\\b(?:KEY(?:s)?|it)(?: (?:is', 'r"\\b(?:KEY|it)(?: (?:is'),                # a2b_r3_on_plural_ok
    ('grade_cues', 'r"\\b(?:KEY(?:s)?|it)(?: (?:is', 'r"\\b(?:KEY(?:s)?)(?: (?:is'),                 # a2b_r2_on_it_ok
    ('grade_cues', 'r"\\b(?:KEY(?:s)?|it)(?: (?:is', 'r"\\b(?:KEY(?:s)?|it|that)(?: (?:is'),            # a2d_on_thats
    ('grade_cues', "(?: (?:is|are|was|were)|'s) only", "(?: (?:are|was|were)|'s) only"),              # a2b_only_on_ok
    ('grade_cues', "(?: (?:is|are|was|were)|'s) only", "(?: (?:is|was|were)|'s) only"),          # a2b_r3_on_plural_ok
    ('grade_cues', "(?: (?:is|are|was|were)|'s) only", "(?: (?:is|are|were)|'s) only"),                # a2d_on_was_ok
    ('grade_cues', "(?: (?:is|are|was|were)|'s) only", "(?: (?:is|are|was)|'s) only"),                # a2d_on_were_ok
    ('grade_cues', "|were)|'s) only", '|were)) only'),                                            # a2b_r2_on_key_s_ok
    ('grade_cues', "|were)|'s) only (?:on", "|were)|'s)(?: \\w+)* only (?:on"),                        # a2b_r2_on_far
    ('grade_cues', "|were)|'s) only (?:on", "|were)|'s) (?:only )?(?:on"),                         # a2d_on_needs_only
    ('grade_cues', 'only (?:on|in) the (?P<t>', 'only (?:in) the (?P<t>'),                            # a2b_only_on_ok
    ('grade_cues', 'only (?:on|in) the (?P<t>', 'only (?:on) the (?P<t>'),                            # a2b_only_in_ok
    ('grade_cues', '(?P<t>\\w+)(?: \\w+){0,2}" + SHAPE_END', '(?P<t>\\w+)(?: \\w+){0,1}" + SHAPE_END'),
    # a2d_on_two_words_ok
    ('grade_cues', '(?P<t>\\w+)(?: \\w+){0,2}" + SHAPE_END', '(?P<t>\\w+)(?: \\w+){0,3}" + SHAPE_END'),
    # a2d_on_tail_three
    ('grade_cues', '(?P<t>\\w+)(?: \\w+){0,2}" + SHAPE_END', '(?P<t>\\w+)(?: \\w+)*" + SHAPE_END'), # a2d_on_other_side
    ('grade_cues', ' \\\n        + [m.span() for m in on_rx(key).finditer(t) if m.group("t").lower() in ok]', ''),
    # a2b_only_on_ok
    ('grade_cues', '    return any(not voided(t, s, e, wide, at) for s, e in pos)',
     '    return bool(pos) and all(not voided(t, s, e, wide, at) for s, e in pos)'),
    # a2d_two_hits_ok
    ('grade_cues', '    return any(not voided(t, s, e, wide, at) for s, e in pos)', '    return bool(pos)'),
    # a2b_not_missing
    ('grade_cues', '_LOOKUP.finditer(t) if cue_counts(t, m, ok)]', '_LOOKUP.finditer(t) if cue_counts(t, m, ok)][:1]'),
    # a2d_two_hits_ok
    # A2d's fix round (2026-11-23, notes STEP 16A2D-R): one per new clause, each killed by the kind named beside it
    ('grade_cues', '    before += "\\n" + wide[at[s - len(before)]:at[s]]\n', ''),
    # a2d_fr_md_join_star
    ('grade_cues', '    before += "\\n" + wide[at[s - len(before)]:at[s]]\n',
     '    before = wide[at[s - len(before)]:at[s]]\n'),
    # a2d_fr_md_tick_isnt
    ('grade_cues', 'wide[at[s - len(before)]:at[s]]', 'wide[at[s - len(t[s0:s])]:at[s]]'),
    # a2b_no_comma_ok
    ('grade_cues', 'wide, at = EMPHASIS.sub(" ", text)', 'wide, at = EMPHASIS.sub("", text)'),
    # a2d_fr_md_join_star
    ('grade_cues', '(?:[,;:.\\u2013', '(?:[;:.\\u2013'),
    # a2b_no_comma_ok
    ('grade_cues', '(?:[,;:.\\u2013', '(?:[,;.\\u2013'),
    # a2b_sadly_not_colon_ok
    ('grade_cues', '(?:[,;:.\\u2013', '(?:[,;:\\u2013'),
    # a2d_fr_opener_dots_ok
    ('grade_cues', '[,;:.\\u2013\\u2014', '[,;:.\\u2014'),
    # a2d_fr_opener_en_ok
    ('grade_cues', '\\u2013\\u2014\\u2026]', '\\u2013\\u2026]'),
    # a2d_fr_opener_em_ok
    ('grade_cues', '\\u2014\\u2026]|-(?!', '\\u2014]|-(?!'),
    # a2d_fr_opener_ellipsis_ok
    ('grade_cues', '|-(?![a-z])))+', '|-))+'),
    # a2d_fr_noone_hyphen
    ('grade_cues', '\\s*(?:[,;:.\\u2013\\u2014\\u2026]|-(?![a-z])))+', '\\s*(?:[^\\w\\s\']|-(?![a-z])))+'),
    # a2d_fr_no_quote
    ('grade_cues', 'r"|\\b(?:noone|zero|', 'r"|\\b(?:zero|'),
    # a2d_fr_noone
    ('grade_cues', '(?:noone|zero|nary|', '(?:noone|nary|'),
    # a2d_fr_zero
    ('grade_cues', '|zero|nary|nada|', '|zero|nada|'),
    # a2d_fr_nary
    ('grade_cues', '|nary|nada|naught|', '|nary|naught|'),
    # a2d_fr_nada
    ('grade_cues', '|nada|naught|nothin)', '|nada|nothin)'),
    # a2d_fr_naught
    ('grade_cues', '|naught|nothin)', '|naught)'),
    # a2d_fr_nothin
    ('grade_cues', '\\b(?:ai|should|', '\\b(?:should|'),
    # a2d_fr_aint
    ('grade_cues', '(?:ai|should|would|', '(?:ai|would|'),
    # a2d_fr_shouldnt
    ('grade_cues', '|should|would|must|', '|should|must|'),
    # a2d_fr_wouldnt
    ('grade_cues', '|would|must|need|', '|would|need|'),
    # a2d_fr_mustnt
    ('grade_cues', '|must|need|might)nt', '|must|might)nt'),
    # a2d_fr_neednt
    ('grade_cues', '|need|might)nt', '|need)nt'),
    # a2d_fr_mightnt
    ('grade_cues', 'n[\\u00b4\\u02bc', 'n[\\u02bc'),
    # a2d_fr_apos_acute
    ('grade_cues', '\\u00b4\\u02bc\\u2032', '\\u00b4\\u2032'),
    # a2d_fr_apos_modifier
    ('grade_cues', '\\u02bc\\u2032\\uff07', '\\u02bc\\uff07'),
    # a2d_fr_apos_prime
    ('grade_cues', '\\u2032\\uff07\\u201b]', '\\u2032\\u201b]'),
    # a2d_fr_apos_fullwidth
    ('grade_cues', '\\uff07\\u201b]t\\b', '\\uff07]t\\b'),
    # a2d_fr_apos_reversed
    ('grade_cues', ']t\\b|\'nt\\b", re.I)', ']t\\b", re.I)'),
    # a2d_fr_apos_misplaced
    ('grade_cues', '    if NONE_CUE.match(m.group(0)) and PREDICATE.search(', '    if False and PREDICATE.search('),
    # a2d_fr_pred_is
    ('grade_cues', 'NONE_CUE = re.compile(r"no |nothing ', 'NONE_CUE = re.compile(r"nothing '),
    # a2d_fr_pred_is
    ('grade_cues', '(r"no |nothing |turned', '(r"no |turned'),
    # a2d_fr_pred_missing
    ('grade_cues', '|turned up nothing|found nothing", re.I)', '|found nothing", re.I)'),
    # a2d_fr_pred_turned_up
    ('grade_cues', '|turned up nothing|found nothing", re.I)', '|turned up nothing", re.I)'),
    # a2d_fr_pred_found_nothing
    ('grade_cues', 'r"\\b(?:is|are|was|were|missing|', 'r"\\b(?:are|was|were|missing|'),
    # a2d_fr_pred_is
    ('grade_cues', '(?:is|are|was|were|missing|', '(?:is|was|were|missing|'),
    # a2d_fr_pred_are
    ('grade_cues', '|are|was|were|missing|', '|are|were|missing|'),
    # a2d_fr_pred_was
    ('grade_cues', '|was|were|missing|absent|', '|was|missing|absent|'),
    # a2d_fr_pred_were
    ('grade_cues', '|were|missing|absent|', '|were|absent|'),
    # a2d_fr_pred_missing
    ('grade_cues', '|missing|absent|blank|', '|missing|blank|'),
    # a2d_fr_pred_absent
    ('grade_cues', '|absent|blank|wrong)', '|absent|wrong)'),
    # a2d_fr_pred_blank
    ('grade_cues', '|blank|wrong)\\b', '|blank)\\b'),
    # a2d_fr_pred_wrong
    ('grade_cues', 'PREDICATE.search(t[m.end():sentence',
     'PREDICATE.search(t[sentence(t, m.start(), m.end())[0]:sentence'),
    # a2b_nothing_was_found_ok
    ('grade_cues', 'PREDICATE.search(t[m.end():sentence(t, m.start(), m.end())[1]])', 'PREDICATE.search(t[m.end():])'),
    # a2d_fr_pred_next_sentence_ok
    ('grade_cues', '    if QMARK.search(t, s0, s1 + 1) or', '    if QMARK.search(t, s1, s1 + 1) or'),
    # a2d_q_interrobang
    ('grade_cues', 'QMARK = re.compile(r"[?\\uff1f\\u203d]")', 'QMARK = re.compile(r"[?\\u203d]")'),
    # a2d_fr_q_fullwidth
    ('grade_cues', 'QMARK = re.compile(r"[?\\uff1f\\u203d]")', 'QMARK = re.compile(r"[?\\uff1f]")'),
    # a2d_fr_q_interrobang
    ('grade_cues', 'STRIKE = re.compile(r"~|', 'STRIKE = re.compile(r"~~|'),
    # a2d_fr_strike_single
    ('grade_cues', '~|</?(?:del|strike)', '~|</?(?:strike)'),
    # a2d_fr_strike_del
    ('grade_cues', '~|</?(?:del|strike)', '~|</?(?:del)'),
    # a2d_fr_strike_tag
    ('grade_cues', 'STRIKE = re.compile(r"~|</?(?:del|strike)\\b", re.I)',
     'STRIKE = re.compile(r"~|</?(?:del|strike)\\b")'),
    # a2d_fr_strike_del_upper
    ('grade_cues', ' or STRIKE.search(t):', ' or STRIKE.search(t, s0, s1):'),
    # a2d_fr_strike_other_sentence
    ('grade_cues', ' or STRIKE.search(t):', ' or STRIKE.search(t, s0, s):'),
    # a2d_fr_strike_after
    ('grade_cues', ' or STRIKE.search(t):', ' or STRIKE.search(t, s, s1):'),
    # a2d_fr_strike_before
    ('grade_cues', 'FROM_TABLE = re.compile(r"[\\W_]*(', 'FROM_TABLE = re.compile(r"\\s+('),
    # a2d_fr_from_paren_the
    ('grade_cues', 'FROM_TABLE = re.compile(r"[\\W_]*(', 'FROM_TABLE = re.compile(r" ('),
    # a2d_fr_from_newline
    ('grade_cues', '(?:[\\W_]+(\\w+))?")', '(?:\\s+(\\w+))?")'),
    # a2d_fr_from_squote
    ('grade_cues', '(?:[\\W_]+(\\w+))?")', '(?: (\\w+))?")'),
    # a2d_fr_from_two_spaces
    ('grade_cues', '(?:[\\W_]+(\\w+))?")', '(?: +(\\w+))?")'),
    # a2d_fr_from_tab
    ('grade_cues', '{f.group(1).lower(), (f.group(2) or "").lower()}', '{(f.group(2) or "").lower()}'),
    # a2d_fr_from_no_article
    ('grade_cues', 'HAVE.search(t, sentence(t, m.start(), m.end())[0], m.start())',
     'HAVE.search(t, max(sentence(t, m.start(), m.end())[0], m.start() - 20), m.start())'),
    # a2d_fr_have_far
    ('grade_cues', "|were)|'s) only (?:on", "|were)|'s)? only (?:on"),
    # a2d_conservative_fr_on_no_verb
]
