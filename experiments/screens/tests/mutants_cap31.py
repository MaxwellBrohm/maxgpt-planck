"""Mutants for SCREENS.txt AMENDMENT CAP-31 (2026-10-11): the code's 31 h cap, the hours report's default, and the
entry's quotes, decision and numbers. Read by mutation_screens.py (name, file, old, new, test file or file::-k expr);
each must make the named tests fail.
"""
ALIB, SH, TX = "experiments/screens/analyze_lib.py", "experiments/screens/screens_hours.py", "experiments/SCREENS.txt"
C31 = "test_screens_cap31.py"
MUTANTS_CAP31 = [
    ("c31_code_back_to_28", ALIB, "CAP, CUT_ORDER, NEVER_CUT = 31.0,", "CAP, CUT_ORDER, NEVER_CUT = 28.0,", C31),
    ("c31_flag_at_the_cap", ALIB, 'over = prop["total_before"] > cap', 'over = prop["total_before"] >= cap',
     C31 + "::code_cap"),
    ("c31_report_default_not_the_code_cap", SH, "c = AL.CAP if cap is None else cap", "c = 28.0 if cap is None else cap",
     C31 + "::hours_report"),
    ("c31_question_word", TX, "slightly worse scores (~0.9 h)", "slightly worse scores (~0.8 h)", C31 + "::verbatim"),
    ("c31_answer_word", TX, '"do whatever you recommend, however', '"do what you recommend, however', C31 + "::verbatim"),
    ("c31_option_renamed", TX, '"Raise it only as needed", "Run them', '"Raise it as needed", "Run them', C31 + "::verbatim"),
    ("c31_cap_number", TX, "ORDER's cap is 31 GPU-hours from this entry on", "ORDER's cap is 30 GPU-hours from this entry on",
     C31 + "::decision"),
    ("c31_code_line", TX, "analyze_lib.CAP = 31.0 (was 28.0)", "analyze_lib.CAP = 30.0 (was 28.0)", C31 + "::decision"),
    ("c31_replay_hold_typo", TX, "base_s101_rep2 23:32:59 to 23:51:19,", "base_s101_rep2 23:32:59 to 23:51:29,",
     C31 + "::had_wrong"),
    ("c31_c6_seconds_typo", TX, "the C6 queue 674 s, 0.187 h", "the C6 queue 647 s, 0.187 h", C31 + "::had_wrong"),
    ("c31_total_typo", TX, "counted as 0.02 h = 27.112 h", "counted as 0.02 h = 27.121 h", C31 + "::arithmetic"),
    ("c31_headroom_typo", TX, "with 3.888 h of headroom under 31 h", "with 3.388 h of headroom under 31 h",
     C31 + "::arithmetic"),
    ("c31_sha_typo", TX, "dd76ba22c2533d4a..., and", "dd76ba22c2533d4b..., and", C31 + "::arithmetic"),
    ("c31_dating_turned", TX, "Written before any C6\ndiagnostic, BASE replay", "Written after the C6\ndiagnostic, BASE replay",
     C31 + "::place"),
    ("c31_test_count_typo", TX, "tests/test_screens_cap31.py (9: this entry's", "tests/test_screens_cap31.py (8: this entry's",
     C31 + "::counts"),
]
