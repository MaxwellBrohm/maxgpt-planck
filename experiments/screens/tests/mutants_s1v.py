"""Mutants for SCREENS.txt STAGE 1 VERDICTS (2026-10-06): the C7 / C4 / Holm / C6 rules the verdicts rest on, the three
screens' results.json and tables.txt, the notes.txt RESULT blocks and the entry. Each must make its test file fail.
Runs on scratch copies through mutation_screens.py's copy_tree and run_tests (the repo is never edited):
  ~/.venvs/planck/bin/python experiments/screens/tests/mutants_s1v.py [--only NAME,...]
Since the 2026-10-06 corrections mutation_screens.py also imports this list, so its whole run includes it; this module
imports mutation_screens only when run itself (an import at the top would be circular).
"""
import sys

V, N = "test_screens_stage1_verdicts.py", "test_screens_stage1_verdicts_numbers.py"
AL = "experiments/screens/analyze_lib.py"
R1, R2, R3 = ("experiments/S001_attn_gate/", "experiments/S002_block_ablations/", "experiments/S003_adamw/")
SXT = "experiments/SCREENS.txt"
ASYM = ('ASYMMETRY (registered; it favours NorMuon; stated beside every S003 verdict): "NorMuon\'s LR5 can be replaced by '
        'E3\'s 7-seed re-check (E3 "E2 re-check", seeds 2..8; SCREENS C1), while AdamW\'s pick rests on E2\'s seed-1 search alone."\n')
CLAUSE = " The asymmetry above could have produced a small REJECT, so the report states it."
MUTANTS_S1V = [
    # the rule (analyze_lib.py): the recorded values, the both-directions fixtures, the noise-check fixtures
    ("s1v_sign_flipped", AL, "else arm[s][key] - base[s][key] for s in used]", "else base[s][key] - arm[s][key] for s in used]", V),
    ("s1v_every_seed_dropped_worse", AL, "worse = dbar > thr and all(x > 0 for x in d)", "worse = dbar > thr", V),
    ("s1v_every_seed_dropped_better", AL, "better = -dbar > thr and all(x < 0 for x in d)", "better = -dbar > thr", V),
    ("s1v_delta_ignored", AL, "equal = abs(dbar) + thr < delta", "equal = not better and not worse", V),
    ("s1v_u_left_at_1", AL, '"SIA": u * n["SD_d_rel"]', '"SIA": n["SD_d_rel"]', V),
    ("s1v_new_init_without_sqrt2", AL, 'math.sqrt(2) * n["sigma_seed_rel"]', 'n["sigma_seed_rel"]', V),
    ("s1v_pooled_sia_blind", AL, '"arms": len(rs), "ok": sd <= lim}', '"arms": len(rs), "ok": True}', V),
    ("s1v_same_init_blind", AL, 'out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "ok": sd <= lim}',
     'out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "ok": True}', V),
    ("s1v_reread_not_underpowered", AL, 'r["labels"] + [why, "underpowered"]', 'r["labels"] + [why]', V),
    ("s1v_holm_one_sided", AL, "else 2 * (1 - t_cdf(abs(dbar) / se, df))", "else (1 - t_cdf(abs(dbar) / se, df))", V),
    ("s1v_onset_abs_own_final", AL, "    L = base_final / 2\n", "    L = final / 2\n", V),
    # the outputs
    ("s1v_results_verdict_typo", R2 + "results.json", '"verdicts": {"S002.nonormscale": "TIE", "S002.noqknorm": "TIE"',
     '"verdicts": {"S002.nonormscale": "TIE", "S002.noqknorm": "INCONCLUSIVE"', V),
    ("s1v_results_thr_typo", R1 + "results.json", '"thr": 0.006965916921273964', '"thr": 0.006965916921273965', V),
    ("s1v_results_clip_count", R3 + "results.json", '"clip_rate_steps_le_160": 0.0625, "clipped": 1',
     '"clip_rate_steps_le_160": 0.0625, "clipped": 2', V),
    ("s1v_results_onset_typo", R2 + "results.json", '"level_abs": 3.7732100857082993, "onset_abs": 1071',
     '"level_abs": 3.7732100857082993, "onset_abs": 918', V),
    ("s1v_tables_row_typo", R1 + "tables.txt", "nogate       CHAT       +0.00366", "nogate       CHAT       +0.00367", V),
    ("s1v_tables_verdict_typo", R3 + "tables.txt", "VERDICT S003.adamw: REJECT", "VERDICT S003.adamw: TIE", V),
    # S003's registered asymmetry beside its verdict and its REJECT clause with the consequence (audit F1, 2026-10-06)
    ("s1v_tables_asymmetry_dropped", R3 + "tables.txt", "(p 4.4e-05)\n" + ASYM, "(p 4.4e-05)\n", V),
    ("s1v_tables_asymmetry_softened", R3 + "tables.txt", "; it favours NorMuon; stated", "; it favours neither; stated", V),
    ("s1v_tables_reject_clause_dropped", R3 + "tables.txt", CLAUSE + " A later optimizer", " A later optimizer", V),
    ("s1v_tables_asymmetry_on_s001", R1 + "tables.txt", "\nVERDICT S001.nogate: TIE", "\n" + ASYM + "VERDICT S001.nogate: TIE", V),
    ("s1v_notes_verdict_typo", R2 + "notes.txt", "VERDICT S002.novres: REJECT.", "VERDICT S002.novres: TIE.", V),
    ("s1v_notes_consequence_changed", R3 + "notes.txt", "parameter. What changes: row 19's 30M check drops its AdamW arm",
     "parameter. What changes: row 19's 30M check keeps its AdamW arm", V),
    ("s1v_notes_number_typo", R2 + "notes.txt", "dbar +0.01721 (1.49% of BASE's mean)", "dbar +0.01712 (1.49% of BASE's mean)", N),
    ("s1v_notes_gate_typo", R1 + "notes.txt", "0.662 / 0.651 at 7,630", "0.662 / 0.615 at 7,630", N),
    ("s1v_entry_line_typo", SXT, "S003.adamw        same init  REJECT  holds          +0.01861",
     "S003.adamw        same init  REJECT  holds          +0.01816", V),
    ("s1v_entry_hours_typo", SXT, "12 runs 3.510 h", "12 runs 3.501 h", N),
]


if __name__ == "__main__":
    import mutation_screens as M
    M.MUTANTS[:] = MUTANTS_S1V
    sys.exit(M.main())
