"""Mutants for SCREENS.txt STAGE 1 SEEDS DONE / STAGE 2 LAUNCH CHECK (2026-10-06): the stage 2 plan, an S005 g-check
config's micro batch, the cut rule's reading of finished screens, the seed sets' measured hours and the entry's numbers. Read by
mutation_screens.py (name, file, old, new, test file); each must make test_screens_stage2_launch.py fail.
"""
S2L, SP, SXT = ("test_screens_stage2_launch.py", "experiments/screens/plans/stage2_select.txt", "experiments/SCREENS.txt")
TSV = "experiments/screens/measured_hours.tsv"
MUTANTS_S2LAUNCH = [
    ("s2_plan_drops_s006_g2", SP, "train s006_mtp_g2_s1\n", "", S2L),
    ("s2_plan_priority_swapped", SP, "train s005_forget_g2_s1\ntrain s004_canonac_g0.5_s1\n",
     "train s004_canonac_g0.5_s1\ntrain s005_forget_g2_s1\n", S2L),
    ("s2_plan_runs_past_its_mark", SP, "mark SCREENS STAGE 2 SELECTION DONE\n",
     "mark SCREENS STAGE 2 SELECTION DONE\ntrain base_s101\n", S2L),
    ("s2_plan_waits_on_its_own_mark", SP, "wait_mark SCREENS SCREENS STAGE 2 SMOKES RECORDED\n",
     "wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE\n", S2L),
    # S005's micro 8 x accum 2 changed to the shared BASE's 16 x 1, doc_attn mask kept (until the 2026-10-06
    # CORRECTIONS it was named "on the shared engine", which it is not)
    ("s2_s005_config_micro_16x1", "experiments/S005_forget_gate/configs/s005_forget_g0.5_s1.yaml",
     "train: {doc_attn: mask, micro_batch: 8, grad_accum: 2}", "train: {doc_attn: mask, micro_batch: 16, grad_accum: 1}",
     S2L),
    ("s2_cut_lists_finished_screens", "experiments/screens/analyze_lib.py",
     "        if sid in remaining:\n            total -= remaining[sid] - (current or {}).get(sid, 0.0)",
     "        if True:\n            total -= remaining.get(sid, 0.0) - (current or {}).get(sid, 0.0)", S2L),
    ("s2_hours_seed_run_missing", TSV, "run\ts003_adamw_e3_r2_s102\t0.29972\t2026-10-06 00:18:34\t2026-10-06 00:36:33\t"
     "queue_screens.txt\n", "", S2L),
    ("s2_hours_seed_end_time_typo", TSV, "2026-10-06 00:18:33", "2026-10-06 00:19:33", S2L),
    ("s2_entry_hours_row_typo", SXT, "s002_noqknorm_g1_s102      0.262 0.234  302k", "s002_noqknorm_g1_s102      0.262 0.243  302k",
     S2L),
    ("s2_entry_threshold_typo", SXT, "the 9 before S006 over 5.427 h", "the 9 before S006 over 5.472 h", S2L),
]
