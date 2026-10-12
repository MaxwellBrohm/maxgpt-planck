"""Mutants for SCREENS.txt AMENDMENT CAP-REMOVED (2026-10-11): the code's infinite cap, cap_check and cap_cut at the
default, screens.py hours printing hours only, screens.py robust without a cut, and the entry's words. Read by
mutation_screens.py (name, file, old, new, test file or file::-k expr); each must make the named tests fail.
"""
SX = "experiments/screens/"
ALIB, SH, GEN, TX = SX + "analyze_lib.py", SX + "screens_hours.py", SX + "screens.py", "experiments/SCREENS.txt"
CR = "test_screens_cap_removed.py"
MUTANTS_CAPREMOVED = [
    ("cr_cap_back_to_31", ALIB, "CAP, CUT_ORDER, NEVER_CUT = INF,", "CAP, CUT_ORDER, NEVER_CUT = 31.0,", CR),
    ("cr_cap_large_not_infinite", ALIB, "CAP, CUT_ORDER, NEVER_CUT = INF,", "CAP, CUT_ORDER, NEVER_CUT = 1e9,", CR),
    ("cr_check_flags_past_28_at_the_default", ALIB, 'over = prop["total_before"] > cap',
     'over = prop["total_before"] > min(cap, 28.0)', CR),
    ("cr_cut_at_the_default", ALIB, "        if total <= cap:\n            break",
     "        if total <= min(cap, 25.0):\n            break", CR),
    ("cr_report_prints_a_cap_line", SH, "    if c == AL.INF:                                    # AMENDMENT CAP-REMOVED",
     "    if False:                                    # AMENDMENT CAP-REMOVED", CR),
    ("cr_report_total_without_other", SH, "        tot = mr + mo + qs\n", "        tot = mr + qs\n", CR),
    ("cr_report_returns_a_cap", SH, 'return {"tally": t, "cap": None, "flag": None,', 'return {"tally": t, "cap": {}, '
     '"flag": None,', CR),
    ("cr_report_line_says_cap", SH, "(AMENDMENT CAP-REMOVED: no cap, cut order or flag): measured",
     "(AMENDMENT CAP-REMOVED: cap 31 h): measured", CR),
    ("cr_robust_takes_a_cut", GEN, '    sub.add_parser("robust")\n',
     '    sub.add_parser("robust").add_argument("--cut", action="append", default=[])\n', CR),
    ("cr_entry_max_word", TX, "so why dont we just drop that gpu hour limit?? you can", "so why dont we just drop the "
     "gpu hour limit?? you can", CR),
    ("cr_entry_dated_after", TX, "nothing committed when written. Written before any run it could affect: no SCREENS queue",
     "nothing committed when written. Written after a run it could affect: no SCREENS queue", CR),
    ("cr_entry_keeps_a_cut", TX, "No cut stands from them (S006's, the only one,", "One cut stands from them (S006's, the only one,",
     CR),
    ("cr_entry_code_line", TX, "analyze_lib.CAP = INF (was 31.0)", "analyze_lib.CAP = INF (was 28.0)", CR),
    ("cr_entry_order_moved", TX, "\nAMENDMENT CAP-REMOVED (2026-10-11", "\nAMENDMENT CAP-REMOVAL (2026-10-11", CR),
]
