"""Mutants for SCREENS.txt AMENDMENT S006-REINSTATE (2026-10-06): the waiter's decisions and launch (s006_waiter.sh),
the acceptance of the reinstated state (reinstate_lib.s006_seed_configs), the plan, the 4 configs, plans/CURRENT, the
record, the entry's and the notes' numbers. Read by mutation_screens.py (name, file, old, new, test file); each must make
the named test file fail.
"""
WS, WT = "experiments/screens/s006_waiter.sh", "test_s006_waiter.py"
RT, CF, S2E = "test_screens_s006_reinstate.py", "test_screens_configs.py", "test_screens_stage2_seeds.py"
RLIB, PL6 = "experiments/screens/tests/reinstate_lib.py", "experiments/screens/plans/stage2_s006_seeds.txt"
SXT, CUR, CFG = "experiments/SCREENS.txt", "experiments/screens/plans/CURRENT", "experiments/S006_mtp_aux/configs/"
TSV = "experiments/screens/measured_hours.tsv"
EXT_LINE = "run\ts006_mtp_g4_s1\t0.35306\t2026-10-06 19:30:49\t2026-10-06 19:52:00\tqueue_screens.txt\n"
MUTANTS_S006R_WAITER = [
    ("s6w_running_ignored", WS, '[ "$1" = 1 ] && { echo "WAIT', '[ "$1" = 9 ] && { echo "WAIT', WT),
    ("s6w_pgrep_error_ignored", WS, '[ "$1" = E ] && { echo "GIVEUP', '[ "$1" = X ] && { echo "GIVEUP', WT),
    ("s6w_stop_file_ignored", WS, '[ -e "$P/STOP" ] && {', '[ -e "$P/STOP.none" ] && {', WT),
    ("s6w_cancel_ignored", WS, '[ -e "$CANCEL" ] && {', '[ -e "$CANCEL.none" ] && {', WT),
    ("s6w_stamp_ignored", WS, '[ -e "$STAMP" ] && {', '[ -e "$STAMP.none" ] && {', WT),
    ("s6w_s006_start_line_ignored", WS, 'plan $NEXT_PLAN\\.txt "; then', 'plan NO_SUCH_PLAN\\.txt "; then', WT),
    ("s6w_commit_unpinned", WS, '"), code commit $WANT_REF") ;;', '"), code commit "*) ;;', WT),
    ("s6w_kill_launches", WS, 'echo "HOLD no queue runs and no end line', 'echo "LAUNCH no queue runs and no end line', WT),
    ("s6w_lines_after_end_ignored", WS, '[ "$end" = "$last" ] || {', '[ -n "$end" ] || {', WT),
    ("s6w_mark_line_unchecked", WS, 'if ! printf \'%s\\n\' "$prev" | grep -aqE "^$TS MARK SCREENS STAGE 2 SEEDS DONE$"; then',
     "if false; then", WT),
    ("s6w_mark_file_unchecked", WS, '[ -e "$OUT/marks/$MARK" ] || {', '[ -e "$OUT" ] || {', WT),
    ("s6w_lock_wait_final", WS, 'echo "HOLD $WANT_PLAN stopped after', 'echo "GIVEUP $WANT_PLAN stopped after', WT),
    ("s6w_other_ends_held", WS, 'echo "GIVEUP $WANT_PLAN ended another way:', 'echo "HOLD $WANT_PLAN ended another way:', WT),
    ("s6w_nul_not_stripped", WS, "log_text() { tr -d '\\000' < \"$QLOG\"; }", 'log_text() { cat "$QLOG"; }', WT),
    ("s6w_grace_never_ends", WS, 'if [ $(($(date +%s) - hold)) -ge "$GRACE" ]; then', "if false; then", WT),
    ("s6w_hold_launches_at_grace", WS, 'say "GIVEUP: no new $WANT_PLAN start within $GRACE s ($r)"; exit 1',
     'launch', WT),
    ("s6w_reads_planck_bundle", WS, "B=$KIT/planck_s006.bundle", "B=$KIT/planck.bundle", WT),
    ("s6w_current_unchecked", WS, '[ "$PLAN" = "$NEXT_PLAN" ] || {', '[ -n "$PLAN" ] || {', WT),
    ("s6w_lock_unchecked", WS, "flock -n 7 || {", "flock -n 7 || true || {", WT),
    ("s6w_no_stamp", WS, 'launched $NEXT_PLAN from ${REF:0:12} (pid $$)" > "$STAMP"', 'launched" > /dev/null', WT),
    ("s6w_poll_not_5_min", WS, "POLL=${S006W_POLL:-300}", "POLL=${S006W_POLL:-60}", WT),
    ("s6w_grace_not_45_min", WS, "GRACE=${S006W_GRACE:-2700}", "GRACE=${S006W_GRACE:-600}", WT),
]
MUTANTS_S006R_STATE = [
    ("s6r_plan_ind_first", PL6, "train s006_mtp_g2_s101\ntrain s006_mtp_g1_s101\n",
     "train s006_mtp_g1_s101\ntrain s006_mtp_g2_s101\n", RT),
    ("s6r_plan_waits_on_the_extension_mark", PL6, "wait_mark SCREENS SCREENS STAGE 2 SEEDS DONE\n",
     "wait_mark SCREENS SCREENS S006 EXTENSION DONE\n", RT),
    ("s6r_plan_runs_past_its_mark", PL6, "mark SCREENS STAGE 2 S006 SEEDS DONE\n",
     "mark SCREENS STAGE 2 S006 SEEDS DONE\ntrain base_s101\n", RT),
    ("s6r_config_verdict_run_at_g1_lrs", CFG + "s006_mtp_g2_s102.yaml", "{lr: 0.006, embed_lr: 0.006, scalar_lr: 0.006}",
     "{lr: 0.003, embed_lr: 0.003, scalar_lr: 0.003}", RT),
    ("s6r_config_ind_run_at_g2_lrs", CFG + "s006_mtp_g1_s101.yaml", "{lr: 0.003, embed_lr: 0.003, scalar_lr: 0.003}",
     "{lr: 0.006, embed_lr: 0.006, scalar_lr: 0.006}", RT),
    ("s6r_config_header_not_the_generators", CFG + "s006_mtp_g1_s102.yaml", "IND-only matched-LR run (C6)",
     "IND-only run (C6)", RT),
    ("s6r_current_back_to_seeds", CUR, "# Seed plans come from screens.py seeds.\nstage2_s006_seeds\n",
     "# Seed plans come from screens.py seeds.\nstage2_seeds\n", RT),
    ("s6r_current_back_to_seeds_s2e", CUR, "# Seed plans come from screens.py seeds.\nstage2_s006_seeds\n",
     "# Seed plans come from screens.py seeds.\nstage2_seeds\n", S2E),
    ("s6r_reinstatement_not_recorded", SXT, "  REINSTATED BY AMENDMENT S006-REINSTATE: S006 (",
     "  REINSTATED, BY AMENDMENT S006-REINSTATE: S006 (", CF),
    ("s6r_acceptance_any_order", RLIB, "    assert runs == want, (runs, want)\n",
     "    assert sorted(runs) == sorted(want), (runs, want)\n", RT),
    ("s6r_acceptance_record_not_needed", RLIB, 'assert "S006" in cuts and rein == {"S006"}, (cuts, rein)',
     "assert True, (cuts, rein)", RT),
    ("s6r_acceptance_wait_unchecked", RLIB, 'assert lines[0].startswith("# ") and lines[1] == wait and',
     'assert lines[0].startswith("# ") and', RT),
    ("s6r_acceptance_check_not_required", RLIB, "assert len(find(r)) == 1 and check(find(r)[0]) == [], r",
     "assert len(find(r)) == 1, r", RT),
    ("s6r_acceptance_end_unchecked", RLIB, "and lines[-1] == END, lines", ", lines", RT),
    ("s6r_hours_extension_missing", TSV, EXT_LINE, "", RT),
    ("s6r_entry_total_typo", SXT, "total 25.366 h <= 27 h", "total 25.636 h <= 27 h", RT),
    ("s6r_entry_cap_back_to_25", SXT, "h <= 27 h: nothing", "h <= 25 h: nothing", RT),
    ("s6r_entry_headroom_typo", SXT, "1.634 h of headroom (runs", "1.643 h of headroom (runs", RT),
    ("s6r_entry_pct_typo", SXT, "estimate (39.0%)", "estimate (39.1%)", RT),
    ("s6r_entry_quote_typo", SXT, "maybe its finally", "maybe it's finally", RT),
    ("s6r_notes_cost_typo", "experiments/S006_mtp_aux/notes.txt", "the 4 runs cost 1.471 h", "the 4 runs cost 1.417 h", RT),
]
MUTANTS_S006R = MUTANTS_S006R_WAITER + MUTANTS_S006R_STATE
