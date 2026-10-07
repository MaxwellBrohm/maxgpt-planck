"""Mutants for SCREENS.txt S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK (2026-10-06): screens.py seeds --cut and
--wait, the acceptance of a cut screen only when SCREENS.txt records the cut (test_screens_configs.stage2_seed_configs),
the stage2_seeds plan and its configs, plans/CURRENT, the extension's measured hours and the entry's numbers. Read by
mutation_screens.py (name, file, old, new, test file); each must make the named test file fail.
"""
S2S, S2E = "test_screens_stage2_seed_state.py", "test_screens_stage2_seeds.py"
GEN, CFT = "experiments/screens/screens.py", "experiments/screens/tests/test_screens_configs.py"
MUTANTS_S2SEEDS_CODE = [
    ("s2s_generator_ignores_cut", GEN, "keep = [sid for sid in L.ORDER[stage] if sid not in cut]",
     "keep = list(L.ORDER[stage])", S2S),
    ("s2s_generator_takes_any_stage_cut", GEN, "any(c not in L.ORDER[a.stage] or c not in AL.CUT_ORDER for c in a.cut)",
     "any(c not in L.ORDER[a.stage] for c in a.cut)", S2S),
    ("s2s_generator_takes_a_cut_twice", GEN, "if len(set(a.cut)) != len(a.cut) or any(", "if any(", S2S),
    ("s2s_generator_drops_the_wait", GEN, 'f"SCREENS STAGE {stage} SEEDS DONE", wait))', 'f"SCREENS STAGE {stage} SEEDS DONE"))',
     S2S),
    ("s2s_header_hides_the_cut", GEN, """    head += f"; cut by ORDER's cap rule (SCREENS.txt), no run: {', '.join(cut)}" if cut else ""\n""", "", S2S),
    ("s2s_acceptance_takes_any_cut", CFT, "assert cuts <= set(L.ORDER[2]) & set(AL.CUT_ORDER), cuts", "assert True, cuts", S2S),
    ("s2s_acceptance_screen_left_out_unrecorded", CFT,
     'for s in L.ORDER[2] if s not in cuts for a in L.SCREENS[s]["arms"]} == set(gs)',
     'for s in L.ORDER[2] if s not in cuts for a in L.SCREENS[s]["arms"]} >= set(gs)', S2S),
    ("s2s_record_unanchored", CFT, """CUT_RE = re.compile(r"^  CUT BY""", """CUT_RE = re.compile(r"  CUT BY""", S2S),
    ("s2s_wait_fixed_to_the_selection_mark", CFT, 'return "wait_mark SCREENS " + last[5:]',
     'return "wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE"', S2S),
]
PL, SXT, CF = "experiments/screens/plans/stage2_seeds.txt", "experiments/SCREENS.txt", "test_screens_configs.py"
TSV, CUR = "experiments/screens/measured_hours.tsv", "experiments/screens/plans/CURRENT"
EXT_LINE = "run\ts006_mtp_g4_s1\t0.35306\t2026-10-06 19:30:49\t2026-10-06 19:52:00\tqueue_screens.txt\n"
STRAY = "run\ts007_smear_g4_s1\t0.30000\t2026-10-06 20:00:00\t2026-10-06 20:18:00\tqueue_screens.txt\n"
MUTANTS_S2SEEDS_STATE = [
    ("s2e_plan_keeps_s006", PL, "train s007_smear_g1_s101\n", "train s007_smear_g1_s101\ntrain s006_mtp_g2_s101\n", S2E),
    ("s2e_plan_waits_on_the_selection_mark", PL, "wait_mark SCREENS SCREENS S006 EXTENSION DONE\n",
     "wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE\n", S2E),
    ("s2e_plan_runs_past_its_mark", PL, "mark SCREENS STAGE 2 SEEDS DONE\n", "mark SCREENS STAGE 2 SEEDS DONE\ntrain base_s101\n",
     S2E),
    ("s2e_plan_priority_swapped", PL, "train s005_forget_g1_s102\ntrain s004_canonac_g1_s102\n",
     "train s004_canonac_g1_s102\ntrain s005_forget_g1_s102\n", S2E),
    ("s2e_plan_drops_a_seed_run", PL, "train s007_smear_g1_s102\n", "", S2E),
    ("s2e_config_at_g2_lrs", "experiments/S004_canon/configs/s004_canonac_g1_s102.yaml",
     "{lr: 0.003, embed_lr: 0.003, scalar_lr: 0.003}", "{lr: 0.006, embed_lr: 0.006, scalar_lr: 0.006}", S2E),
    ("s2e_s005_seed_on_the_shared_engine", "experiments/S005_forget_gate/configs/s005_forget_g1_s101.yaml",
     "train: {doc_attn: mask, micro_batch: 8, grad_accum: 2}\n", "", S2E),
    ("s2e_current_back_to_s006x", CUR, "# Seed plans come from screens.py seeds.\nstage2_seeds\n",
     "# Seed plans come from screens.py seeds.\nstage2_s006x\n", S2E),
    ("s2e_cut_not_recorded", SXT, "  CUT BY ORDER'S CAP RULE BEFORE THE STAGE 2 SEED SETS: S006 (",
     "  CUT, BY ORDER'S CAP RULE BEFORE THE STAGE 2 SEED SETS: S006 (", CF),
    ("s2e_cut_recorded_for_s007_too", SXT, "SEED SETS: S006 (its queued", "SEED SETS: S006, S007 (its queued", CF),
    ("s2e_ind_runs_dropped", GEN, 'if sid in ("S006", "S007") and g != 1.0:', "if False:", S2E),
    ("s2e_cut_order_starts_at_s007", "experiments/screens/analyze_lib.py", '25.0, ["S006", "S003", "S007", "S002"]',
     '25.0, ["S007", "S006", "S003", "S002"]', S2E),
    ("s2e_hours_extension_missing", TSV, EXT_LINE, "", S2E),
    ("s2e_hours_extension_end_typo", TSV, "2026-10-06 19:52:00", "2026-10-06 19:53:00", S2E),
    ("s2e_entry_cap_typo", SXT, "= 25.366 h, 0.366 h over 25 h", "= 25.336 h, 0.366 h over 25 h", S2E),
    ("s2e_entry_after_cut_typo", SXT, "= 23.895 h <= 25 h", "= 23.859 h <= 25 h", S2E),
    ("s2e_entry_pick_row_typo", SXT, "1.17919 / 1.41726  g 2", "1.17919 / 1.41762  g 2", S2E),
    ("s2e_entry_inside_pct_typo", SXT, "headroom (52.8%)", "headroom (58.2%)", S2E),
    ("s2e_notes_after_cut_typo", "experiments/S006_mtp_aux/notes.txt", "leaving 23.895 h", "leaving 23.985 h", S2E),
    # the older tests' changed claims: 81 measured runs with the extension on top; the selection and launch tests read
    # the file as of their entries, the extension's line out, and accept no other later line
    ("s2e_hours_extension_missing_hours_test", TSV, EXT_LINE, "", "test_screens_hours.py"),
    ("s2e_hours_stray_line_selection_test", TSV, EXT_LINE, EXT_LINE + STRAY, "test_screens_stage2_selection.py"),
    ("s2e_hours_stray_line_launch_test", TSV, EXT_LINE, EXT_LINE + STRAY, "test_screens_stage2_launch.py"),
]
MUTANTS_S2SEEDS = MUTANTS_S2SEEDS_CODE + MUTANTS_S2SEEDS_STATE
