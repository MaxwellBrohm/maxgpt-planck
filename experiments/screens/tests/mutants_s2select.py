"""Mutants for SCREENS.txt STAGE 2 SELECTION RESULT (2026-10-06): the extension plan and config, plans/CURRENT, the C3
edge rule and the extension's slot, the 12 g-checks' measured hours and the entry's numbers. Read by mutation_screens.py
(name, file, old, new, test file); each must make test_screens_stage2_selection.py fail.
"""
S2X, PX, SXT = ("test_screens_stage2_selection.py", "experiments/screens/plans/stage2_s006x.txt", "experiments/SCREENS.txt")
TSV, CFG = "experiments/screens/measured_hours.tsv", "experiments/S006_mtp_aux/configs/s006_mtp_g4_s1.yaml"
MUTANTS_S2SELECT = [
    ("s2x_plan_runs_past_its_mark", PX, "mark SCREENS S006 EXTENSION DONE\n",
     "mark SCREENS S006 EXTENSION DONE\ntrain base_s101\n", S2X),
    ("s2x_plan_waits_on_the_smokes_mark", PX, "wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE\n",
     "wait_mark SCREENS SCREENS STAGE 2 SMOKES RECORDED\n", S2X),
    ("s2x_plan_trains_g2_again", PX, "train s006_mtp_g4_s1\n", "train s006_mtp_g2_s1\n", S2X),
    ("s2x_config_at_g2_lrs", CFG, "{lr: 0.012, embed_lr: 0.012, scalar_lr: 0.012}", "{lr: 0.006, embed_lr: 0.006, "
     "scalar_lr: 0.006}", S2X),
    ("s2x_current_back_to_select", "experiments/screens/plans/CURRENT", "# Seed plans come from screens.py seeds.\n"
     "stage2_s006x\n", "# Seed plans come from screens.py seeds.\nstage2_select\n", S2X),
    ("s2x_edge_never_extended", "experiments/screens/analyze.py",
     'want = [] if not res["at_edge"] else [xs[0] / 2 if res["at_edge"] == "low" else xs[-1] * 2]', "want = []", S2X),
    ("s2x_extension_takes_a_g_slot", "experiments/screens/screens_hours.py",
     'return sid, ((sid, m["arm"], "g", g) if g in (0.5, 1.0, 2.0) else None)', 'return sid, (sid, m["arm"], "g", g)', S2X),
    ("s2x_hours_g_check_missing", TSV, "run\ts006_mtp_g2_s1\t0.35611\t2026-10-06 11:06:44\t2026-10-06 11:28:06\t"
     "queue_screens.txt\n", "", S2X),
    ("s2x_hours_lock_start_at_the_wait", TSV, "2026-10-06 05:30:39", "2026-10-06 05:18:06", S2X),
    ("s2x_entry_pick_value_typo", SXT, "F(CHAT) 1.16340 against g 1's 1.16349", "F(CHAT) 1.16340 against g 1's 1.16394", S2X),
    ("s2x_entry_cap_typo", SXT, "(+0.368 h, S006): 24.645 h <= 25 h", "(+0.368 h, S006): 24.654 h <= 25 h", S2X),
    ("s2x_entry_ind_typo", SXT, "= 25.013 h + x, over 25 h", "= 25.031 h + x, over 25 h", S2X),
    ("s2x_entry_g8_threshold_typo", SXT, "under 25 h only while x <= 0.355 h", "under 25 h only while x <= 0.553 h", S2X),
]
