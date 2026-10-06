"""Mutants for the 2026-10-06 corrections of SCREENS.txt STAGE 1 VERDICTS and of STAGE 2 LAUNCH CHECK (its CORRECTIONS
paragraph): analyze.py's C4 re-label, k_needed, the out/ copies, the privacy check, the BASE level, S003's F(CHAT) per
GPU-hour, the registered wording, tok/s and p, the launch entry's outside-the-lock seconds, the hours tests. Read by
mutation_screens.py (name, file, old, new, test file); each must make its test file fail. No name or address here.
"""
C, T = "test_screens_stage1_corrections.py", "test_screens_stage1_corrections_text.py"
V, S2L, HF = "test_screens_stage1_verdicts.py", "test_screens_stage2_launch.py", "test_screens_hours.py"
AN, PV, HRS = "experiments/screens/analyze.py", "experiments/screens/tests/privacy.py", "experiments/screens/screens_hours.py"
R1, R2, R3 = ("experiments/S001_attn_gate/", "experiments/S002_block_ablations/", "experiments/S003_adamw/")
SXT = "experiments/SCREENS.txt"
MUTANTS_FIX = [
    # A1: C4's "noise check failed later" for verdicts read earlier
    ("fix_later_label_not_used", AN, "LATER if k in earlier else POOLED", "POOLED", C),
    ("fix_later_label_reworded", AN, 'LATER, POOLED = "noise check failed later"', 'LATER, POOLED = "noise check failed (later)"', C),
    ("fix_earlier_ignored", AN, 'return {k for p in paths for k in json.load(open(p))["contrasts"]}', "return set()", C),
    ("fix_s002_later_record_wrong", R2 + "results.json", '"prose_thr": 0.004871752167021515, "verdict": "REJECT"}',
     '"prose_thr": 0.004871752167021515, "verdict": "TIE"}', C),
    ("fix_entry_checked_claim_restored", SXT, "they stay REJECT, TIE, TIE (CHAT thr 0.00697).",
     "they stay REJECT, TIE, TIE (CHAT thr 0.00697) (checked).", C),
    # A2: k_needed per class
    ("fix_k_needed_same_init_typo", R3 + "results.json", '"k_needed_1pct": {"CHAT": 1,', '"k_needed_1pct": {"CHAT": 2,', C),
    ("fix_entry_k_claim_restored", SXT, "Labels: k = max(2, k_needed(1%)) = 2 in every class.",
     "Labels: k 2 = k_needed in every class.", C),
    ("fix_s003_notes_k_claim", R3 + "notes.txt", "1) = 2: k is above k_needed", "1) = 2: k equals k_needed", C),
    # A3: C9's out/<run>/ copies
    ("fix_out_file_byte", R3 + "out/base_s101/log.jsonl", '"step": 7630, "loss": 3.57079', '"step": 7630, "loss": 3.57078', C),
    # A4: the privacy check (no name in the repo; generic shapes, runtime values, the git-ignored local list)
    ("fix_results_private_path", R1 + "results.json", '"gate_note": "in-training hook (eval.induction)',
     '"gate_note": "in-training hook (/home/someone/eval.induction)', V),
    ("fix_privacy_blind", PV, 'return [f"{src} {label}" for src, label, rx in (pats or patterns()) if rx.search(text)]',
     "return []", C),
    ("fix_privacy_runtime_dropped", PV, '    out += [("runtime", k, re.compile(v, re.I)) for k, v in _runtime().items()]\n', "", C),
    # N-B: the BASE level against E3 Part 1 arm A
    ("fix_level_gap_typo", SXT, "gap +0.00755, +0.66%; t 3.26 with E3's", "gap +0.00757, +0.66%; t 3.26 with E3's", T),
    ("fix_level_verdict_at_e3_wrong", R1 + "results.json", '"verdicts_with_sd_ref_and_delta_at_e3_mean": {"S001.nogate": "TIE"}',
     '"verdicts_with_sd_ref_and_delta_at_e3_mean": {"S001.nogate": "INCONCLUSIVE"}', T),
    # N-B at one seed (seed 1: S002's SIA arms minus dbar against E3 A_s1 and its replays), and the pairing's limit
    ("fix_same_seed_shift_typo", R1 + "results.json", '"shift": 0.00220713443487619', '"shift": 0.00230713443487619', T),
    ("fix_same_seed_arm_value", R1 + "results.json", '"S002.novres": 1.1691581223580807', '"S002.novres": 1.1691681223580807', T),
    ("fix_same_seed_entry_typo", SXT, "+0.00221 CHAT (+0.19%)", "+0.00212 CHAT (+0.19%)", T),
    ("fix_spread_claim_restored_notes", R2 + "notes.txt", "would move d itself; the pooled SIA check (it passed) sees it only if "
     "it varies across seeds.", "would show as a spread of d across seeds, which the pooled SIA check reads (it passed).", T),
    ("fix_spread_claim_restored_results", R3 + "results.json", "noise checks (both passed) see it only if it varies across seeds",
     "noise checks (both passed) read it", T),
    # A7 / F5: S003's F(CHAT) per GPU-hour
    ("fix_per_hour_mean_typo", R3 + "notes.txt", "AdamW 4.329 / 4.340, mean 4.334", "AdamW 4.329 / 4.340, mean 4.343", T),
    # A5, A6, A9: the registered wording
    ("fix_s002_paren_dropped_from_entry", SXT, "\n      (row 19's 30M check of the block and optimizer still runs as the ledger states it).\n", ".\n", T),
    ("fix_s003_flip_30m_restored", R3 + "results.json", "A later optimizer flip at larger size is then not looked for (the registered REJECT;",
     "A later optimizer flip at 30M would go unseen (the registered REJECT;", T),
    ("fix_s003_q3_qualifier_dropped", R3 + "results.json", "confirmed E2's pick against E3's arm B only, not against 6e-3 (E3 AUDIT.md Q3; ",
     "confirmed E2's pick (", T),
    ("fix_gate_judgment_restored", SXT, "GATE in BASE, 32-window in-training hook:", "GATE in BASE did not stay open, 32-window in-training hook:", T),
    # N-C, N-D: the p column and the tok/s medians
    ("fix_p_column_back_to_4f", R3 + "tables.txt", "7   4.39e-05  WORSE", "7   0.0000  WORSE", V),
    ("fix_tok_median_all_records_typo", R3 + "results.json", '"s003_adamw_e3_r2_s102": 258544.4', '"s003_adamw_e3_r2_s102": 258551.0', T),
    ("fix_tables_tok_note_wrong", R1 + "tables.txt", "over log.jsonl records 3 to 763:", "over log.jsonl records 1 to 763:", T),
    # the launch entry's CORRECTIONS and its tests
    ("fix_launch_wait_seconds_typo", SXT, "takes 3 s over the 12 runs", "takes 4 s over the 12 runs", S2L),
    ("fix_launch_uncomputed_395", SXT, "was in this file, they passed too.", "was in this file, they passed too (395).", S2L),
    ("fix_hours_loader_accepts_twice", HRS, '            assert name not in runs, f"{name} measured twice"\n', "", HF),
    # a second run on a seen slot absorbed into the slot instead of counted on top (the old dict-overwrite check passed it)
    ("fix_hours_second_run_absorbed", HRS, "if key in reg and key not in seen:", "if key in reg:", HF),
]
