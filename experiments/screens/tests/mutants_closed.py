"""Mutants for SCREENS.txt BATCH 1 CLOSED (2026-10-11; E3 Part 3): the closed plan, plans/CURRENT, the acceptance
(closed_lib.closed_state), the entry, and the numbers test's block ends (the verdict entry and E3's PART 4d RESULT end
at the next blank line, so E3's PART 3 REOPENED is not read as PART 4d's). Read by mutation_screens.py (name, file, old,
new, test file); each must make the named test file fail.
"""
SX = "experiments/screens/"
PL, CUR, LIB = SX + "plans/batch1_closed.txt", SX + "plans/CURRENT", SX + "tests/closed_lib.py"
SXT, NT = "experiments/SCREENS.txt", SX + "tests/test_screens_stage2_verdicts_numbers.py"
CL, NUMS = "test_screens_closed.py", "test_screens_stage2_verdicts_numbers.py"
MUTANTS_CLOSED = [
    ("cl_plan_trains", PL, "stop\n", "train base_s101\nstop\n", CL),
    ("cl_plan_marks_instead_of_stop", PL, "stop\n", "mark SCREENS BATCH 1 CLOSED\n", CL),
    ("cl_plan_waits_on_diag", PL, "wait_mark SCREENS SCREENS BASE REPLAY DONE", "wait_mark SCREENS SCREENS BASE DIAG DONE",
     CL),
    ("cl_current_back_to_replay", CUR, "# Seed plans come from screens.py seeds.\nbatch1_closed\n",
     "# Seed plans come from screens.py seeds.\nstage2_replay\n", CL),
    ("cl_current_names_no_plan", CUR, "# Seed plans come from screens.py seeds.\nbatch1_closed\n",
     "# Seed plans come from screens.py seeds.\nbatch2_closed\n", "test_screens_configs.py::registered_order"),
    ("cl_acceptance_any_body", LIB, '    assert lines[1:] == ["wait_mark SCREENS " + END[len("mark "):], "stop"], lines\n',
     "", CL),
    ("cl_acceptance_no_replay_state", LIB, '    assert rep, "the batch closes', '    assert True or rep, "the batch closes',
     CL),
    ("cl_acceptance_no_entry", LIB, "    assert RECORD in section(text),", "    assert True or RECORD in section(text),", CL),
    ("cl_record_line_edited", SXT, "  CLOSED: plans/CURRENT = batch1_closed (stop), after",
     "  CLOSED: plans/CURRENT = batch1_closed (stops), after", CL),
    ("cl_scope_widened", SXT, "result is recorded, never to stage 1 or stage 2.", "result is recorded, also to stage 1 or stage 2.",
     CL),
    ("cl_c6_done_time_typo", SXT, '"c6diag: DONE" at 2026-10-10 22:28:09', '"c6diag: DONE" at 2026-10-10 22:28:90', CL),
    ("cl_e3_block_to_eof", NT, '.split("\\nPART 4d RESULT (2026-10-11", 1)[1].split("\\n\\n", 1)[0]',
     '.split("\\nPART 4d RESULT (2026-10-11", 1)[1]', NUMS),
]
