"""Round 2 checker mutants (2026-09-28; fixtures_r2.py, fixtures_r2b.py): each reverts or overshoots one round 2
change. Same form as checker_mutants.SRC, (name, module, old, new); checker_mutants appends this list (split out to
keep that file under 250 lines). The agent-form limits of stemmer.topic_forms (verbs only, four letters or more) are
pinned and mutated by the unit tests (tests/test_round2_units.py and the unit-test mutant run), not here."""
import check_base
import check_behav
import check_events
import check_lines
import checker
import lexicons
import stemmer

SRC_R2 = [
    ("ly_any_part", check_lines, 'if part == "adj":\n        ly =', 'if True:\n        ly ='),
    ("ly_shifted_ok", check_lines, "if ly not in LY_SHIFTED:", "if True:"),
    ("ves_any_f", check_lines, 'if part == "noun" and VES_RE.search(w):', 'if w.endswith(("f", "fe")):'),
    ("end_token_off", check_lines, "m = L.END_TOKEN_RE.search(s) or ", "m = "),
    ("end_tail_off", check_lines, "or L.END_TAIL_RE.search(s) or", "or"),
    ("script_talk_off", check_lines, " or L.SCRIPT_TALK_RE.search(s)\n", "\n"),
    ("end_any_case", lexicons, 'END_TOKEN_RE = re.compile(r"(?<![A-Za-z])END(?![A-Za-z])")',
     'END_TOKEN_RE = re.compile(r"(?<![A-Za-z])END(?![A-Za-z])", re.I)'),
    ("user_style_off", check_lines, "if not ctx.lower_user:", "if True:"),
    ("user_style_all_users", check_lines, "if not ctx.lower_user:", "if False:"),
    ("user_style_exact_too", check_lines, 'ctx.turns(role="user", mode="guided"):\n        m = re.search',
     'ctx.turns(role="user"):\n        m = re.search'),
    ("guide_meta_off", check_lines, "m = L.GUIDE_META_RE.search(s) or (t", "m = None or (t"),
    ("withhold_off", check_lines, "and L.WITHHOLD_RE.search(s))", "and None)"),
    ("withhold_all_turns", check_lines, '(t["i"] in queries and t["role"] == "user" and', '(t["role"] == "user" and'),
    ("assist_ism_all_roles", check_lines, 'for t, s in ctx.turns(role="assistant", mode="guided")',
     'for t, s in ctx.turns(mode="guided")'),
    ("assist_how_help_dropped", lexicons, 'r"how (?:can|may) i ?(?:help|assist)", ', ""),
    ("assist_glued_help_missed", lexicons, 'r"how (?:can|may) i ?(?:help|assist)"', 'r"how (?:can|may) i (?:help|assist)"'),
    ("assist_anything_else_any", lexicons, 'r"anything else i can (?:help|do|assist)"', 'r"anything else i can"'),
    ("hedge_positive_back", lexicons, 'r"(?:wasn\'t|was not|was never|(?:has|have)(?: not| never|n\'t) been|not been) '
     'mentioned"', 'r"(?:was|wasn\'t|was not|has not been|hasn\'t been|haven\'t been) mentioned"'),
    ("self_do_love_dropped", lexicons, 'r"i (?:do|really|also|still|truly|just|absolutely) (?:love|like|enjoy|adore|'
     'prefer)",', ""),
    ("self_spend_dropped", lexicons, 'r"i spend (?:my|most|a lot|time)", ', ""),
    ("self_like_my_own_dropped", lexicons, 'r"like my own"', 'r"zzqq"'),
    ("self_my_own_any", lexicons, 'r"like my own"', 'r"my own"'),
    ("self_past_dropped", lexicons, 'r"i (?:booked|', 'r"zzqq (?:booked|'),
    ("self_called_dropped", lexicons, 'r"i called(?! you(?![a-z]))"', 'r"zzqq"'),
    ("self_called_you_too", lexicons, 'r"i called(?! you(?![a-z]))"', 'r"i called"'),
    ("hedge_no_x_dropped", lexicons, 'r"(?:no|nothing)(?: \\w+){0,3} (?:was|were|has been|had been) (?:ever )?mentioned"',
     'r"zzqq"'),
    ("pronoun_the_one_off", check_events, '|(?<![a-z])the\\s+one(?=\\s+[a-z])", re.I)', '", re.I)'),
    ("pronoun_the_one_any", check_events, '|(?<![a-z])the\\s+one(?=\\s+[a-z])", re.I)',
     '|(?<![a-z])the\\s+one(?![a-z])", re.I)'),
    ("plant_embed_off", check_events, " or L.EMBED_DECL_RE.search(c) or", " or"),
    ("plant_cleft_off", check_events, " or (not q_end and L.PSEUDO_CLEFT_RE.match(c)):", ":"),
    ("plant_cleft_q_ok", check_events, "(not q_end and L.PSEUDO_CLEFT_RE.match(c))", "(L.PSEUDO_CLEFT_RE.match(c))"),
    ("generic_forms_empty", lexicons, 'GENERIC_FORMS = {"noted", "noting"}', "GENERIC_FORMS = set()"),
    ("pronoun_one_the_back", check_events, "(?:which|that|this|each|every|any|other|another|next|last|first|no)\\s+one\"\n"
     '                         r"(?![a-z])|(?<![a-z])the\\s+one(?=\\s+[a-z])", re.I)', "(?:which|that|this|each|every|any|other|"
     'another|the|next|last|first|no)\\s+one"\n                         r"(?![a-z])|(?<![a-z])one(?:\'s|\\s+of)(?![a-z])", re.I)'),
    ("plant_question_ok", check_events, 'and not stated(ctx.text[i], s["value"], fold=ctx.lower_user):', "and False:"),
    ("plant_no_clause_split", check_events, "enumerate(L.PLANT_CLAUSE_SPLIT.split(sent))", "enumerate([sent])"),
    ("plant_no_value_mark", check_events, '.sub(" VALUEMARK ", text)', ".sub(lambda m: m.group(0), text)"),
    ("prev_assist_one_word", check_behav, "PREV_ASSIST_MIN = 2", "PREV_ASSIST_MIN = 1"),
    ("prev_assist_three", check_behav, "PREV_ASSIST_MIN = 2", "PREV_ASSIST_MIN = 3"),
    ("user_label_off", check_behav, 'words = ns + sorted(labs.get(t["i"], ()), key=len, reverse=True)', "words = ns"),
    ("abstain_query_value_off", check_behav, "out += _abstain_query_value(ctx, e)\n", "\n"),
    ("stem_none", check_base, "    s = porter(w)", "    s = w"),
    ("i_strip_off", check_base, "I_STRIP_MIN = 6", "I_STRIP_MIN = 99"),
    ("topic_forms_off", check_behav, ' | content(" ".join(forms))', ""),
    ("forms_no_agent", stemmer, '        out.append(w + "r" if w.endswith("e")', '        0 and out.append(w + "r" if w.endswith("e")'),
    ("forms_no_y_base", stemmer, "            out += [b, b + \"e\"]", "            pass"),
    ("forms_y_base_short", stemmer, 'elif w.endswith("y") and len(w) >= 5:', 'elif w.endswith("y") and len(w) >= 4:'),
    ("forms_no_comparatives", stemmer, "        out += comparatives(w)\n", "\n"),
    ("prev_share_off", check_behav, "PREV_ASSIST_SHARE = 0.5", "PREV_ASSIST_SHARE = 2"),
    ("prev_share_low", check_behav, "PREV_ASSIST_SHARE = 0.5", "PREV_ASSIST_SHARE = 0.2"),
    ("plant_split_at_that", lexicons, r'|\s+(?=that\s)", re.I)', r'|\s+that\s+", re.I)'),
    ("plant_ask_anywhere", check_events, "L.WH_START_RE.match(c)) and (first or q_end))", "L.WH_START_RE.match(c)))"),
    ("plant_aux_ignored", check_events, "return bool((L.AUX_START_RE.match(c) or ", "return bool(("),
    ("generic_forms_ignored", check_base, " and w not in L.GENERIC_FORMS", ""),
    ("porter_cvc_e_dropped", stemmer, "if m > 1 or (m == 1 and not _cvc(stem)):", "if m > 0:"),
    ("porter_1b_no_e", stemmer, "            if _m(w) == 1 and _cvc(w):\n                return w + \"e\"",
     "            if False:\n                return w + \"e\""),
]
