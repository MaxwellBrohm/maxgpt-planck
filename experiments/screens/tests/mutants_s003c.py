"""Mutants for SCREENS.txt S003 STAGE C RESULT (2026-10-05): the stage 1 seed configs and plan, E2's stage C edge
rule, the entry's numbers and the measured hours. Read by mutation_screens.py (name, file, old, new, test file);
kept in their own file so mutation_screens.py stays short. Each must make test_screens_stage1_seeds.py fail.
"""
S1S, SP, SXT = ("test_screens_stage1_seeds.py", "experiments/screens/plans/stage1_seeds.txt", "experiments/SCREENS.txt")
CFG = "experiments/{}/configs/{}.yaml"
MUTANTS_S003C = [
    ("seeds_plan_drops_s003_s102", SP, "train s003_adamw_e3_r2_s102\n", "", S1S),
    ("seeds_plan_runs_past_its_mark", SP, "mark SCREENS STAGE 1 SEEDS DONE\n",
     "mark SCREENS STAGE 1 SEEDS DONE\ntrain s004_canonac_g1_s1\n", S1S),
    ("seeds_plan_waits_on_stage_b", SP, "wait_mark SCREENS SCREENS S003 STAGE C DONE\n",
     "wait_mark SCREENS SCREENS S003 STAGE B DONE\n", S1S),
    ("seeds_plan_not_seed_major", SP, "train s003_adamw_e3_r2_s101\ntrain base_s102\n",
     "train base_s102\ntrain s003_adamw_e3_r2_s101\n", S1S),
    ("seeds_s003_config_at_r1", CFG.format("S003_adamw", "s003_adamw_e3_r2_s102"), "embed_lr: 0.006, scalar_lr: 0.006",
     "embed_lr: 0.003, scalar_lr: 0.003", S1S),
    ("seeds_s002_config_wrong_seed", CFG.format("S002_block_ablations", "s002_noqknorm_g1_s101"), "seed: 101\n",
     "seed: 102\n", S1S),
    ("seeds_s001_flag_dropped", CFG.format("S001_attn_gate", "s001_nogate_g1_s102"), "model: {attn_gate: false}\n", "", S1S),
    ("stage_c_125M_edge_rule_dropped", "experiments/E2_lr_transfer/e2pick.py",
     'if p125.get("ready") and p125["argmin"] in (xs[0], xs[-1]):', "if False:", S1S),
    ("entry_stage_c_value_typo", SXT, "1.16606 / 1.39615", "1.16660 / 1.39615", S1S),
    ("entry_hours_typo", SXT, "0.637 h training", "0.673 h training", S1S),
    ("entry_threshold_typo", SXT, "over 2.019 h", "over 2.018 h", S1S),
    ("entry_config_hash_typo", SXT, "f9e022053c61e183 s003_adamw_e3_r2_s101", "f9e022053c61e138 s003_adamw_e3_r2_s101", S1S),
    ("entry_branch_point_typo", SXT, "6,104 kept). Run directories: no DIVERGED, GAP, HEAT, STOP, crashes.txt or\n    "
     "rc12_eval.jsonl under runs/SCREENS (find, depth 2); score.err empty in all 6 ", "6,014 kept). Run directories: no "
     "DIVERGED, GAP, HEAT, STOP, crashes.txt or\n    rc12_eval.jsonl under runs/SCREENS (find, depth 2); score.err empty "
     "in all 6 ", S1S),
    ("entry_small_count_typo", SXT, '6 "done and scored" (branches)', '5 "done and scored" (branches)', S1S),
    ("entry_cap_row_typo", SXT, "    S003    44             4.468   2           0.625  5.093",
     "    S003    44             4.468   2           0.625  5.039", S1S),
    ("entry_full_run_range_typo", SXT, "held it 0.262 to 0.302 h", "held it 0.262 to 0.320 h", S1S),
    ("hours_stage_c_run_missing", "experiments/screens/measured_hours.tsv", "run\ts003_adamw_e6_r2_b250M\t0.08556\t"
     "2026-10-05 20:15:28\t2026-10-05 20:20:36\tqueue_screens.txt\n", "", S1S),
]
