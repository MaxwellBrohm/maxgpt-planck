"""Mutants for SCREENS.txt CORRECTIONS AFTER THE W1 VERIFY (2026-10-10): ONSET RULE CODE's CORRECTION 1 in analyze.py
(F1, F2), diag_runner.py's 20M budget scale (F3), and the entry's numbers. Read by mutation_screens.py (name, file,
old, new, test file or file::-k expr); each must make the named tests fail.
"""
AN, DRN, TX = "experiments/screens/analyze.py", "experiments/screens/diag_runner.py", "experiments/SCREENS.txt"
MS = "experiments/screens/tests/mutation_screens.py"
OD, VF = "test_screens_onset_diag.py", "test_screens_verify_fixes.py::not results_sentence"   # that one: by hand
MUTANTS_VFIX = [
    ("vf_error_lines_skipped", AN, '        by[r["step"]] = r\n', '        if "error" not in r:\n            by[r["step"]] = r\n',
     OD),
    ("vf_error_line_reads_as_its_ind", AN, 'if "error" in r or type(x)', "if type(x)", OD),
    ("vf_non_finite_ind_kept", AN, "type(x) not in (int, float) or not math.isfinite(x)", "type(x) not in (int, float)",
     OD),
    ("vf_bool_ind_kept", AN, "type(x) not in (int, float) or", "not isinstance(x, (int, float)) or", OD),
    ("vf_missing_grid_steps_ignored", AN, "    if sorted(IND_GRID - set(by)):", "    if False:", OD),
    ("vf_off_grid_steps_ignored", AN, "    if sorted(set(by) - IND_GRID):", "    if False:", OD),
    ("vf_grid_every_typo", AN, "IND_EVERY = 153 ", "IND_EVERY = 152 ", OD),
    ("vf_grid_without_last_step", AN, "IND_EVERY)) | {L.STEPS}", "IND_EVERY))", OD),
    ("vf_nul_bytes_kept", AN, '.replace(b"\\0", b"")', "", OD),
    ("vf_bad_line_ignored", AN, '            probs.append(f"line {i} is not a reading")\n', "", OD),
    ("vf_float_step_accepted", AN, 'type(r.get("step")) is int', 'isinstance(r.get("step"), (int, float))', OD),
    ("vf_first_reading_of_a_step_kept", AN, '        by[r["step"]] = r\n', '        by.setdefault(r["step"], r)\n',
     OD),
    ("vf_incomplete_reads_ok", AN, '("diag incomplete" if probs else "ok")', '"ok"', OD),
    ("vf_problems_not_reported", AN, "probs = {n: diag_why(runs, n) for n in read if", "probs = {n: [] for n in read if",
     OD),
    ("vf_no_file_not_named", AN, 'else ["no diag.jsonl"]', "else []", OD),
    ("vf_arm_runs_not_reported", AN, ' + [n for c in out.values() for n in c.get("runs", [])]', "", OD),
    ("vf_runner_20m_scale_1", DRN, 'scale = 4.0 if r.startswith("e3_20m") else 1.0', "scale = 1.0", VF),
    ("vf_runner_20m_scale_5", DRN, 'scale = 4.0 if r.startswith("e3_20m")', 'scale = 5.0 if r.startswith("e3_20m")', VF),
    ("vf_entry_913_at_0_22", TX, "913.3 s, about 0.25 h", "913.3 s, about 0.22 h", VF),
    ("vf_entry_1290_typo", TX, "1,290.1 s, 0.36 h", "1,290.1 s, 0.35 h", VF),
    ("vf_entry_f5_time", TX, "killed\") was written at 19:04", "killed\") was written at 19:14", VF),
    ("vf_entry_grid_count", TX, "7,630:\n    50 steps;", "7,630:\n    49 steps;", VF),
    ("vf_entry_start_up", TX, "1.9 s of start-up, load and sha256 in a 19 s hold", "2.9 s of start-up, load and "
     "sha256 in a 19 s hold", VF),
    ("vf_not_registered", MS, "MUTANTS += MUTANTS_VFIX\n", "", "test_screens_verify_fixes.py::counts"),
]
