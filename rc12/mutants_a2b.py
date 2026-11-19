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
own fixtures) and is killed by the one named. Not mutated: word boundaries (as ABS_MORE), the 3-token window
(grade_text.negated_before's, mutated there), re.escape of the key (dev keys are weekdays and months)."""

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
    ("grade_cues", r'|\bnothing (?:was )?found\b"', '"'),                                        # a2b_nothing_found_ok
    ("grade_cues", "nothing (?:was )?found", "nothing found"),                               # a2b_nothing_was_found_ok
    ("grade_cues", "(?:turned up|found) nothing", "(?:found) nothing"),                      # a2b_turned_up_nothing_ok
    ("grade_cues", "|found) nothing", ") nothing"),                                              # a2b_found_nothing_ok
    ("grade_cues", r'r"|\b(?:turned up|found) nothing\b|\bcame up empty\b|',
     r'r"|\b(?:turned up|found) nothing\b|'),                                                    # a2b_came_up_empty_ok
    ("grade_cues", r"\b(?:not|cannot|\w+n't)", r"\b(?:cannot|\w+n't)"),                             # a2b_not_found_ok
    ("grade_cues", r"\b(?:not|cannot|\w+n't)", r"\b(?:not|\w+n't)"),                         # a2b_cannot_be_found_ok
    ("grade_cues", r"\b(?:not|cannot|\w+n't)", r"\b(?:not|cannot)"),                              # a2b_wasnt_found_ok
    ("grade_cues", r"(?: be)? found\b", r" found\b"),                                                 # a2b_be_found_ok
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
    ("grade_cues", 'ONLY_OBJ = r"(?:it|', 'ONLY_OBJ = r"(?:\\w+|it|'),                             # a2b_only_changed
    ("grade_cues", 'ONLY_OBJ = r"(?:it|that|', 'ONLY_OBJ = r"(?:that|'),                        # a2b_only_has_it_ok
    ("grade_cues", "|that|a row|", "|a row|"),                                                # a2b_only_has_that_ok
    ("grade_cues", "|a row|an entry|", "|an entry|"),                                            # a2b_only_a_row_ok
    ("grade_cues", "|an entry|a line|", "|a line|"),                                          # a2b_only_an_entry_ok
    ("grade_cues", "|a line|(?:the )?KEY)", "|(?:the )?KEY)"),                                  # a2b_only_a_line_ok
    ("grade_cues", "|(?:the )?KEY)", "|KEY)"),                                                 # a2b_only_the_key_ok
    ("grade_cues", r'|(?:the )?KEY)\b"', r')\b"'),                                           # a2b_only_other_has_ok
    ("grade_cues", 'if key else "(?!)"), re.I)', 'if key else "(?!)"))'),                    # a2b_only_other_has_ok
    # LOOKUP_ON: joined, each preposition
    ("grade_cues", "\\\n        + [m for m in _ON.finditer(text) if m.group(1).lower() in ok]", ""),    # a2b_only_on_ok
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
    ("grade_cues", "in _ON.finditer(text) if m.group(1).lower() in ok]", "in _ON.finditer(text)]"),  # a2b_only_on_asked
    ("grade_cues", "only_rx(key).finditer(text) if m.group(1).lower() in ok]",
     "only_rx(key).finditer(text) if m.group(1) in ok]"),                                            # a2b_only_cap_ok
    ("grade_cues", "_ON.finditer(text) if m.group(1).lower() in ok]",
     "_ON.finditer(text) if m.group(1) in ok]"),                                                  # a2b_only_on_cap_ok
    ("grade_cues", 'ok = others | {"other"}', "ok = others"),                                # a2b_only_other_lists_ok
    ("grade_cues", "return any(not lookup_negated(text, m.start()) for m in hits)",
     "return bool(hits)"),                                                                           # a2b_not_missing
    ("grade_cues", "return any(not lookup_negated(",
     "return bool(hits) and all(not lookup_negated("),                                               # a2b_two_hits_ok
    ("grade_cues", "hits = list(_LOOKUP.finditer(text)) +",
     "hits = list(_LOOKUP.finditer(text))[:1] +"),                                                   # a2b_two_hits_ok
    ("grade_cues", "    s0 = T.sentence_span(text, pos)[0]\n", "    s0 = 0\n"),                    # a2b_afraid_not_ok
    ("grade_cues", "    s0 = max([s0] + [c.end() for c in CLAUSE.finditer(text, s0, pos)])\n",
     ""),                                                                                   # a2b_afraid_not_comma_ok
    ("grade_cues", 'CLAUSE = re.compile(r"[,;:]")', 'CLAUSE = re.compile(r"[;:]")'),        # a2b_afraid_not_comma_ok
    ("grade_cues", 'CLAUSE = re.compile(r"[,;:]")', 'CLAUSE = re.compile(r"[,:]")'),         # a2b_dont_think_semi_ok
    ("grade_cues", 'CLAUSE = re.compile(r"[,;:]")', 'CLAUSE = re.compile(r"[,;]")'),         # a2b_sadly_not_colon_ok
    ("grade_cues", "(?:nothing|nobody)", "(?:nobody)"),                                          # a2b_nothing_missing
    ("grade_cues", "(?:nothing|nobody)", "(?:nothing)"),                                          # a2b_nobody_missing
    ("grade_cues", ' or ("neg" not in T.OFF and NEG_MORE.search(window) is not None)', ""),     # a2b_nothing_missing
    ("grade_cues", "return T.negated_before(text, s0, pos) or (", "return ("),                       # a2b_not_missing
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
