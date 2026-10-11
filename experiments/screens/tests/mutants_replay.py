"""Mutants for SCREENS.txt AMENDMENT BASE-REPLAY (2026-10-10; cuts audit S7): the run-name rule (screens_lib NAME_RE,
BASE_REPLAYS, check's C2 and C6 for _rep / _hookoff names), screens.py replay, screens_hours.slot, the acceptance
(replay_lib.replay_configs), the plan, the 3 configs, plans/CURRENT, the record and the entry's numbers. Read by
mutation_screens.py (name, file, old, new, test file); each must make the named test file fail. The same-code guard
(test_screens_replay_code.py) needs a full tree, so its mutants are run by hand on an index export (the entry).
"""
SX = "experiments/screens/"
LIB, GEN, HRS = SX + "screens_lib.py", SX + "screens.py", SX + "screens_hours.py"
RLIB, PL, CUR, CFG = SX + "tests/replay_lib.py", SX + "plans/stage2_replay.txt", SX + "plans/CURRENT", SX + "configs/"
PEND = "experiments/SCREENS.txt"     # the entry (written in pending/replay.txt, moved into SCREENS.txt unchanged)
RP, RE, CF = "test_screens_replay.py", "test_screens_replay_entry.py", "test_screens_configs.py"
DIAG_IF = 'if m["diag"] and (sid or arm != "base" or BASE_REPLAYS.get(name) != ("hook-off" if m["diag"] == "hookoff" else "replay")):'
MUTANTS_REPLAY_RULE = [
    ("rp_name_hookoff_unparsed", LIB, "rep[1-9]|hookoff))?", "rep[1-9]))?", RP),
    ("rp_name_any_rep_number", LIB, "rep[1-9]|hookoff))?", "rep\\d+|hookoff))?", RP),
    ("rp_registry_widened", LIB, '"base_s101_rep2": "replay"}', '"base_s101_rep2": "replay", "base_s102_rep1": "replay"}', RP),
    ("rp_registry_unchecked", LIB, DIAG_IF, "if False:", RP),
    ("rp_shared_base_only_dropped", LIB, 'if m["diag"] and (sid or arm != "base" or ', 'if m["diag"] and (', RP),
    ("rp_kind_unmatched", LIB, 'BASE_REPLAYS.get(name) != ("hook-off" if m["diag"] == "hookoff" else "replay")):',
     "name not in BASE_REPLAYS):", RP),
    ("rp_hookoff_keys_not_allowed", LIB, ' | ({"eval.induction"} | set(hook_keys()) if off else set())', "", RP),
    ("rp_hookoff_check_blind", LIB, 'if "eval.induction" not in f or f["eval.induction"] is not None:', "if False:", RP),
    ("rp_hookoff_any_value", LIB, ' or f["eval.induction"] is not None:', ":", RP),
    ("rp_hook_off_by_config", LIB, 'off = BASE_REPLAYS.get(name) == "hook-off"',
     'off = (cfg.get("eval") or {}).get("induction", 0) is None', RP),
    ("rp_hook_off_by_suffix", LIB, 'off = BASE_REPLAYS.get(name) == "hook-off"', 'off = bool(m["diag"])', RP),
    ("rp_gen_hookoff_hook_kept", GEN, '**({"eval.induction": None} if off else {})', "**{}", RP),
    ("rp_gen_header", GEN, '("; eval.induction off" if off else "")', '""', RP),
    ("rp_gen_any_name", GEN, 'if not kind or not m or m["sid"] or m["arm"] != "base":', "if not m:", RP),
    ("rp_slot_takes_base", HRS, '    if m["diag"]:                                      # a BASE replay', "    if False:  # a BASE replay",
     RP),
]
MUTANTS_REPLAY_STATE = [
    ("rp_acc_any_order", RLIB, "and runs == RUNS, lines", "and sorted(runs) == sorted(RUNS), lines", RP),
    ("rp_acc_record_not_needed", RLIB, "assert rec == RUNS and reg == KIND, (rec, reg)", "assert reg == KIND, (rec, reg)", RP),
    ("rp_acc_registry_unchecked", RLIB, "assert rec == RUNS and reg == KIND, (rec, reg)", "assert rec == RUNS, (rec, reg)", RP),
    ("rp_acc_wait_unchecked", RLIB, 'assert lines[0].startswith("# ") and lines[1] == wait and',
     'assert lines[0].startswith("# ") and', RP),
    ("rp_acc_end_unchecked", RLIB, "and lines[-1] == END, lines", ", lines", RP),
    ("rp_acc_check_not_required", RLIB, "assert len(find(r)) == 1 and check(find(r)[0]) == [], r", "assert len(find(r)) == 1, r", RP),
    ("rp_acc_dir_unchecked", RLIB, "assert os.path.dirname(find(r)[0]) == L.config_dir(None), find(r)", "assert True, find(r)", RP),
    ("rp_acc_diff_unchecked", RLIB, 'assert L.diff(f, ref) == want and f["seed"] == 101', 'assert f["seed"] == 101', RP),
    ("rp_acc_kind_unchecked", RLIB,
     'assert (KIND[r] == "hook-off") == ("eval.induction" in f and f["eval.induction"] is None), r', "assert True, r", RP),
    ("rp_acc_diag_not_needed", RLIB, '    assert diag, "BASE-REPLAY runs after', '    assert True or diag, "BASE-REPLAY runs after', RP),
    ("rp_entry_header_renamed", PEND, "\nAMENDMENT BASE-REPLAY (2026-10-10 18:04 EDT", "\nAMENDMENT BASE REPLAY (2026-10-10 18:04 EDT",
     RE),
    ("rp_plan_out_of_order", PL, "train base_s101_rep1\ntrain base_s101_hookoff\n", "train base_s101_hookoff\ntrain base_s101_rep1\n",
     RP),
    ("rp_plan_out_of_order_cf", PL, "train base_s101_rep1\ntrain base_s101_hookoff\n",
     "train base_s101_hookoff\ntrain base_s101_rep1\n", CF + "::registered_order"),
    ("rp_plan_waits_on_s006", PL, "wait_mark SCREENS SCREENS BASE DIAG DONE\n", "wait_mark SCREENS SCREENS STAGE 2 S006 SEEDS DONE\n",
     RP),
    ("rp_plan_runs_past_mark", PL, "mark SCREENS BASE REPLAY DONE\n", "mark SCREENS BASE REPLAY DONE\ntrain base_s101\n", RP),
    ("rp_cfg_rep2_seed", CFG + "base_s101_rep2.yaml", "seed: 101\n", "seed: 102\n", RP),
    ("rp_cfg_hookoff_hook_on", CFG + "base_s101_hookoff.yaml", "eval: {induction: null}\n", "", RP),
    ("rp_cfg_rep1_lr", CFG + "base_s101_rep1.yaml", "{lr: 0.003,", "{lr: 0.006,", CF + "::passes_check"),
    ("rp_current_back_to_diag", CUR, "# Seed plans come from screens.py seeds.\nstage2_replay\n",
     "# Seed plans come from screens.py seeds.\nstage2_base_diag\n", RP),
    ("rp_record_removed", PEND, "  BASE RUNS ADDED BY AMENDMENT BASE-REPLAY: base_s101_rep1,",
     "  BASE RUNS, ADDED BY AMENDMENT BASE-REPLAY: base_s101_rep1,", CF + "::complete"),
    ("rp_entry_total_typo", PEND, "27.240", "27.204", RE),
    ("rp_entry_headroom_typo", PEND, "0.760", "0.706", RE),
    ("rp_entry_t3_typo", PEND, "3.182", "3.128", RE),
    ("rp_entry_bar_typo", PEND, "0.00555", "0.00565", RE),
    ("rp_entry_held_typo", PEND, "6.280", "6.208", RE),
    ("rp_entry_label_reworded", PEND, 'labels it "BASE replay: stable"', 'labels it "BASE replay: robust"', RE),
    ("rp_entry_config_hash_typo", PEND, "9c6a2b46be51a412", "9c6a2b46be51a413", RE),
]
MUTANTS_REPLAY = MUTANTS_REPLAY_RULE + MUTANTS_REPLAY_STATE
