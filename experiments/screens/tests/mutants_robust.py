"""Mutants for SCREENS.txt AMENDMENT SEEDS-103-105 (2026-10-11): screens.py robust, the plan, plans/CURRENT, the configs,
the acceptance (robust_lib.robust_configs), diag_runner.py's robust group, robust_waiter.sh and the entry's words and
numbers. Read by mutation_screens.py (name, file, old, new, test file or file::-k expr); each must make the named tests
fail.
"""
SX = "experiments/screens/"
GEN, PL, CUR, LIB = SX + "screens.py", SX + "plans/robustness_k5.txt", SX + "plans/CURRENT", SX + "tests/robust_lib.py"
DRN, RW, TX = SX + "diag_runner.py", SX + "robust_waiter.sh", "experiments/SCREENS.txt"
R, RD, TRW = "test_screens_robust.py", "test_diag_robust.py", "test_robust_waiter.py"
MUTANTS_ROBUST = [
    # screens.py robust: the picks, the seeds, the order, the IND-only run, the wait and the mark
    ("rb_s006_pick_g1", GEN, '"S006.mtp": 2.0,', '"S006.mtp": 1.0,', R),
    ("rb_s003_at_r1", GEN, "ROBUST_S003 = (3e-3, 2.0)", "ROBUST_S003 = (3e-3, 1.0)", R),
    ("rb_two_seeds", GEN, "ROBUST_SEEDS = (103, 104, 105)", "ROBUST_SEEDS = (103, 104)", R),
    ("rb_stage_2_first", GEN, "        for sid in L.ORDER[1] + L.ORDER[2]:", "        for sid in L.ORDER[2] + L.ORDER[1]:", R),
    ("rb_no_ind_only_run", GEN, 'if g != 1.0 and sid in ("S006", "S007"):', "if False:", R),
    ("rb_own_base_skipped", GEN, "        for sid in L.ORDER[1] + L.ORDER[2]:\n            if L.engine(sid):",
     "        for sid in L.ORDER[1] + L.ORDER[2]:\n            if False:", R),
    ("rb_waits_on_diag", GEN, 'ROBUST_WAIT, ROBUST_MARK = "SCREENS SCREENS BASE REPLAY DONE"',
     'ROBUST_WAIT, ROBUST_MARK = "SCREENS SCREENS BASE DIAG DONE"', R),
    # the plan, CURRENT and one config, edited in place
    ("rb_plan_swapped", PL, "train s004_canonac_g1_s105\ntrain s007_smear_g1_s105\n",
     "train s007_smear_g1_s105\ntrain s004_canonac_g1_s105\n", R),
    ("rb_plan_drops_a_run", PL, "train s002_noqknorm_g1_s104\n", "", R),
    ("rb_plan_no_end_mark", PL, "mark SCREENS ROBUSTNESS K5 DONE\n", "", R),
    ("rb_current_back_to_closed", CUR, "# Seed plans come from screens.py seeds.\nrobustness_k5\n",
     "# Seed plans come from screens.py seeds.\nbatch1_closed\n", R),
    ("rb_config_other_lr", "experiments/S007_smeared_key/configs/s007_smear_g1_s104.yaml", "optim: {lr: 0.003,",
     "optim: {lr: 0.006,", R),
    # the acceptance
    ("rb_acc_any_sequence", LIB, "    assert len(runs) == len(lines) - 3 and runs == SEQUENCE, runs\n",
     "    assert len(runs) == len(lines) - 3, runs\n", R),
    ("rb_acc_any_wait", LIB, "lines[1] == WAIT and lines[-1] == END", "lines[-1] == END", R),
    ("rb_acc_configs_not_diffed", LIB, 'assert L.diff(f, g) == {"name", "out_dir", "seed"} and',
     'assert L.diff(f, g) >= {"name", "out_dir", "seed"} and', R),
    ("rb_acc_not_checked", LIB, "        assert check(p) == [], r\n", "", R),
    ("rb_acc_closed_not_needed", LIB, '    assert (closed_state() if closed is None else closed), "the robustness',
     '    assert True or (closed_state() if closed is None else closed), "the robustness', R),
    ("rb_acc_entry_not_needed", LIB, "    assert RECORD in section(text),", "    assert True or RECORD in section(text),", R),
    # diag_runner.py's robust group
    ("rb_dr_robust_by_default", DRN, "({**GROUPS, **LATER_GROUPS} if groups else GROUPS)", "{**GROUPS, **LATER_GROUPS}", RD),
    ("rb_dr_scores_ind_only_runs", DRN, '          "s006_mtp_g2")]', '          "s006_mtp_g2", "s006_mtp_g1")]', RD),
    ("rb_dr_no_own_base", DRN, '"s003_adamw_e3_r2", "s005_base", "s005_forget_g1"', '"s003_adamw_e3_r2", "s005_forget_g1"',
     RD),
    ("rb_dr_group_refused", DRN, "set(a.groups) - set(GROUPS) - set(LATER_GROUPS):", "set(a.groups) - set(GROUPS):", RD),
    # robust_waiter.sh
    ("rw_e3_never_waits", RW, 'case $? in 0) echo "WAIT an E3 queue', 'case $? in 9) echo "WAIT an E3 queue', TRW),
    ("rw_e3_any_end", RW, '    if printf \'%s\\n\' "$last" | grep -aqE "^$TS plan finished:',
     '    if true || printf \'%s\\n\' "$last" | grep -aqE "^$TS plan finished:', TRW),
    ("rw_e3_any_plan_finished", RW, 'plan finished: ${plan//./\\\\.}$"', 'plan finished: .*$"', TRW),
    ("rw_e3_no_mark_needed", RW, '       printf \'%s\\n\' "$prev" | grep -aqE "^$TS MARK [A-Z0-9 ]+$" && [ "$prev" != "$last" ]',
     '       [ "$prev" != "$last" ]', TRW),
    ("rw_e3_step_0_skipped", RW, "        OK*) break ;;\n        WAIT*) sleep", "        *) break ;;\n        WAIT*) sleep", TRW),
    ("rw_cancel_unchecked_at_step_0", RW, '    [ -e "$CANCEL0" ] && { say "GIVEUP: $CANCEL0 exists"; exit 1; }\n    r=$(e3_state)',
     "    r=$(e3_state)", TRW),
    ("rw_relaunch_any_hold", RW, 'HOLD*"stopped after a 2 h gpu.lock wait"*)\n            hold=', 'HOLD*)\n            hold=', TRW),
    ("rw_maxr_off_by_one", RW, '[ "$n" -lt "$MAXR" ]', '[ "$n" -le "$MAXR" ]', TRW),
    ("rw_no_c6_group", RW, 'bash "$Q/$XREL/diag_queue.sh" --code "$Q" --groups robust 5>&- 6>&-\nrc=$?', "rc=0", TRW),
    ("rw_c6_every_group", RW, '--code "$Q" --groups robust 5>&- 6>&-', '--code "$Q" 5>&- 6>&-', TRW),
    ("rw_c6_exit_ignored", RW, 'if [ "$rc" = 0 ]; then\n    say "DONE:', 'if true; then\n    say "DONE:', TRW),
    ("rw_pin_typo", RW, "SHA=920db4c1", "SHA=920db4c2", TRW),
    ("rw_replay_ref_typo", RW, "REPLAY_REF=2aa37874b34f", "REPLAY_REF=2aa37874b34e", TRW),
    ("rw_step_1_other_plan", RW, '"SCREENS BASE REPLAY DONE" robustness_k5 planck_robust.bundle 5>&-',
     '"SCREENS BASE REPLAY DONE" batch1_closed planck_robust.bundle 5>&-', TRW),
    ("rw_step_1_launch_nothing_kept", RW, 'WAIT*"earlier in the chain"*) say "GIVEUP: step 1 launched nothing"; exit 1 ;;',
     'WAIT*"earlier in the chain"*) hold= ;;', TRW + "::step_1"),
    # the entry
    ("rb_entry_dated_before", TX, "nothing committed when written. Written AFTER the stage 1 verdicts",
     "nothing committed when written. Written before the stage 1 verdicts", R),
    ("rb_entry_verdict_list", TX, "S006 REJECT, S007 TIE), their Holm readings", "S006 TIE, S007 TIE), their Holm readings", R),
    ("rb_entry_sia_threshold", TX, "SIA 0.347% / 0.209%", "SIA 0.348% / 0.209%", R),
    ("rb_entry_k2_threshold", TX, "0.548% / 0.330%, 0.603% /", "0.548% / 0.330%, 0.604% /", R),
    ("rb_entry_pooled_limit", TX, "= 1.215 x u x SD_d", "= 1.251 x u x SD_d", R),
    ("rb_entry_onset_steps", TX, "(147.2 steps at the recorded", "(147.1 steps at the recorded", R),
    ("rb_entry_jobs", TX, "halflife on S005's arm: 102\n    jobs", "halflife on S005's arm: 103\n    jobs", R),
    ("rb_entry_hours", TX, "the plan about 12.4 h", "the plan about 12.3 h", R),
    ("rb_entry_record_line", TX, "36 train lines, 3 seed sets, 33 new runs (BASE lines skipped)",
     "36 train lines, 3 seed sets, 33 runs (BASE lines skipped)", R),
    ("rb_entry_not_adopted_dropped", TX, "Nothing is adopted or dropped on the robustness reading\n    alone",
     "Something is adopted or dropped on the robustness reading\n    alone", R),
]
