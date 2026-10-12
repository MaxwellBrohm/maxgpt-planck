"""Mutants for SCREENS.txt INTEGRATION 2026-10-10: replay_waiter.sh (the replay's launch and its relaunch after a 2 h
gpu.lock wait), the INTEGRATION entry's numbers, and the moved C6-DIAG entry's numbers against the code. Read by
mutation_screens.py (name, file, old, new, test file or file::-k expr); each must make the named tests fail. The
C6-DIAG text mutants name only the c6diag tests, so the moved-entry hash alone cannot kill them.
"""
RW, TX, DQ = "experiments/screens/replay_waiter.sh", "experiments/SCREENS.txt", "experiments/screens/diag_queue.sh"
RT, IT = "test_replay_waiter.py", "test_screens_integration.py"
C6T, CAPT, MVT = IT + "::c6diag", IT + "::cap_position", IT + "::moved_entries"
MUTANTS_INTEG_WAITER = [
    ("rw_sha_pin_typo", RW, "SHA=920db4c14819bec8", "SHA=920db4c14819bec9", RT),
    ("rw_base_ref_typo", RW, "BASE_REF=27d8cb7505248ff4", "BASE_REF=27d8cb7505248ff5", RT),
    ("rw_wrong_bundle", RW, "stage2_replay planck_replay.bundle\nrc=$?", "stage2_replay planck_diag.bundle\nrc=$?", RT),
    ("rw_maxr_default", RW, "MAXR=${CW_MAXR:-24}", "MAXR=${CW_MAXR:-2}", RT),
    ("rw_relaunch_on_any_hold", RW, 'HOLD*"stopped after a 2 h gpu.lock wait"*)', "HOLD*)", RT),
    ("rw_launch_not_done", RW, 'ended cleanly at its mark"; exit 0 ;;', 'ended cleanly at its mark"; exit 1 ;;', RT),
    ("rw_maxr_unbounded", RW, '[ "$n" -lt "$MAXR" ] ||', "[ 0 -lt 1 ] ||", RT),
    ("rw_cancel_ignored", RW, '    [ -e "$LOGD/replay_waiter.CANCEL" ] &&', '    [ -e "$LOGD/no_such.CANCEL" ] &&', RT),
    ("rw_nothing_launched_waits", RW, 'WAIT*"earlier in the chain"*) say "GIVEUP: step 1 launched nothing"; exit 1 ;;',
     "", RT),
    ("rw_clone_unchecked", RW, '"$(git -C "$Q" rev-parse HEAD 2>/dev/null)" != "$REF" ] ||', '"x" != "x" ] ||', RT),
    ("rw_relaunch_reads_current", RW, '--plan "$Q/$XREL/plans/stage2_replay.txt" 6>&-', '--plan "$Q/$XREL/plans/CURRENT" 6>&-',
     RT),
    ("rw_step2_waits_on_diag_mark", RW, '"SCREENS BASE REPLAY DONE" replay_requeue_none',
     '"SCREENS BASE DIAG DONE" replay_requeue_none', RT),
    ("rw_grace_never_ends", RW, 'if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then', "if false; then", RT),
    ("rw_hold_gives_up_at_once", RW, 'if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then', "if true; then", RT + "::kill"),
]
CW, SW = "experiments/screens/chain_waiter.sh", "experiments/screens/s006_waiter.sh"
MUTANTS_INTEG_GRACE = [   # the two older waiters' grace tests, whose lower bound this step moved from 2 s to 1 s
    ("cw_hold_gives_up_at_once", CW, 'if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then', "if true; then",
     "test_chain_waiter_run.py::gives_up_after_the_grace"),
    ("s6w_hold_gives_up_at_once", SW, 'if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then', "if true; then",
     "test_s006_waiter.py::gives_up_after_the_grace"),
]
MUTANTS_INTEG_TEXT = [
    ("ig_cap_total_typo", TX, "= 27.284 h <= 28 h", "= 27.248 h <= 28 h", CAPT),
    ("ig_cap_headroom_typo", TX, "nothing to flag, 0.716 h", "nothing to flag, 0.761 h", CAPT),
    ("ig_cap_rule_a_typo", TX, "gives 27.600 h, 0.400 h", "gives 27.600 h, 0.040 h", CAPT),
    ("ig_cap_estimate_dropped", TX, "counted whole as OTHER", "counted as OTHER only for this batch", CAPT),
    ("ig_moved_sha_typo", TX, "sha256 3d91e9bffb033e01", "sha256 3d91e9bffb033e02", MVT),
    ("ig_moved_c6_text_edited", TX, "Expected holds: two.", "Expected holds: 2.", MVT),
    ("ig_moved_replay_text_edited", TX, "Three short runs (about 0.31 h", "Three short runs (about 0.32 h", MVT),
    ("ig_relaunch_cap_text", TX, "after a 2 h gpu.lock wait, at most 24 times", "after a 2 h gpu.lock wait, at most 20 times",
     IT + "::launch_names"),
    ("c6e_table_jobs_typo", TX, "      stage2    10           38", "      stage2    10           39", C6T),
    ("c6e_table_e3p2_typo", TX, "      e3p2       6           12", "      e3p2       5           12", C6T),
    ("c6e_hold_typo", TX, "hard timeout of 1,980 s", "hard timeout of 1,890 s", C6T),
    ("c6e_floor_typo", TX, "floors (hd 60 s", "floors (hd 50 s", C6T),
    ("c6e_steps_typo", TX, "(7,630 at 5M, 15,259 at 20M)", "(7,630 at 5M, 15,529 at 20M)", C6T),
    ("c6e_mutants_typo", TX, "80 of 80 killed", "79 of 80 killed", C6T),
    ("c6e_tests_typo", TX, "test_diag_queue.py (16:", "test_diag_queue.py (15:", C6T),
    ("c6e_queue_hold_changed", DQ, "HOLD=${C6_HOLD_S:-1980}", "HOLD=${C6_HOLD_S:-2100}", C6T),
    ("c6e_queue_memory_changed", DQ, '[ "$used" -gt 3000 ]', '[ "$used" -gt 4000 ]', C6T),
]
MUTANTS_INTEG = MUTANTS_INTEG_WAITER + MUTANTS_INTEG_GRACE + MUTANTS_INTEG_TEXT
