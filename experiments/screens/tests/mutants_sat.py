"""Mutants for the 2026-10-10 prep (SCREENS.txt AMENDMENT CAP-FLAG, ONSET RULE CODE, DEVIATION 1; S006 notes
FOLLOW-UP TRIGGER): the cap and its flag, the BASE-DIAG onset rule, the stage 1 --earlier default, the lock-hold
parse, durable scoring and the cut count. Imported by mutation_screens.py (name, file, old, new, test spec)."""
ALIB, AN, HRS = "experiments/screens/analyze_lib.py", "experiments/screens/analyze.py", "experiments/screens/screens_hours.py"
SYNC, QLIB, QUEUE = ("experiments/screens/score_sync.py", "experiments/screens/qlib_screens.sh",
                     "experiments/screens/queue_screens.sh")
AF, OF, HO, QD = "test_analyze.py", "test_screens_onset_rule.py", "test_screens_holds.py", "test_queue_durable.py"
CO, TX = "test_screens_stage1_corrections.py", "test_screens_saturday_text.py"
SXT, S6N, REG = "experiments/SCREENS.txt", "experiments/S006_mtp_aux/notes.txt", "models/registry.jsonl"
NR = ("not read: registered 2026-10-10 before the stage 2 verdict step reads it (SCREENS.txt AMENDMENT BASE-DIAG, ONSET RULE "
      "CODE); the queue scored it (SCORED, bpb.jsonl on the PC); no value of it was read for this entry")
MUTANTS_SAT = [
    # AMENDMENT CAP-FLAG: 28 h, an overrun is a flag, never a cut
    ("cap_left_at_25", ALIB, "CAP, CUT_ORDER, NEVER_CUT = INF,", "CAP, CUT_ORDER, NEVER_CUT = 25.0,", AF),  # INF: CAP-REMOVED
    ("cap_check_cuts_silently", ALIB, '"flag_to_max": over, "cut": [],', '"flag_to_max": over, "cut": prop["cut"],', AF),
    ("cap_check_never_flags", ALIB, '"flag_to_max": over, "cut": [],', '"flag_to_max": False, "cut": [],', AF),
    ("cap_check_flags_at_the_cap", ALIB, 'over = prop["total_before"] > cap', 'over = prop["total_before"] >= cap', AF),
    ("report_prints_no_flag", HRS, '    if not flag["flag_to_max"]:\n', '    if True:\n', OF + "::hours_report"),
    # the ONSET RULE (AMENDMENT BASE-DIAG)
    ("onset_sd_without_sqrt2", ALIB, "    sd = math.sqrt(2) * sigma\n", "    sd = sigma\n", OF),
    ("onset_thr_one_sided", ALIB, "    thr = t_ppf(0.975, df) * se\n", "    thr = t_ppf(0.95, df) * se\n", OF),
    ("onset_earlier_every_seed_dropped", ALIB, "earlier = -dbar > thr and all(x < 0 for x in d)", "earlier = -dbar > thr", OF),
    ("onset_later_every_seed_dropped", ALIB, "later = dbar > thr and all(x > 0 for x in d)", "later = dbar > thr", OF),
    ("onset_equal_label_added", ALIB, 'label = "EARLIER" if earlier else "LATER" if later else "UNRESOLVED"',
     'label = "EARLIER" if earlier else "LATER" if later else "EQUAL" if abs(dbar) <= thr else "UNRESOLVED"', OF),
    ("onset_sigma_population_sd", ALIB, '"sigma_seed": st.stdev(xs)', '"sigma_seed": st.pstdev(xs)', OF),
    ("onset_sigma_zero_p_one", ALIB, "if se > 0 else (0.0 if dbar else 1.0)", "if se > 0 else 1.0", OF),
    ("onset_missing_base_seed_ignored", ALIB,
     '    if miss:\n        return {"sigma_seed": None, "df": None, "why": f"BASE onset_half missing or undefined at seeds '
     '{miss}"}\n    xs = [base_half[s] for s in ONSET_SEEDS]', "    xs = [base_half[s] for s in ONSET_SEEDS if s not in miss]", OF),
    ("onset_sign_flipped", AN, "d = [a[s][1] - base[s][1] for s in seeds", "d = [base[s][1] - a[s][1] for s in seeds", OF),
    ("onset_diverged_base_read", AN, '    if v.get("diverged"):\n        return "diverged", []', '    if False:\n        return "diverged", []', OF),
    ("onset_partial_curve_accepted", AN, "    if sorted(IND_GRID - set(by)):", "    if False:", OF),   # re-anchored 10-10 (vfix)
    ("onset_first_reading_kept", AN, '        by[r["step"]] = r\n', '        by.setdefault(r["step"], r)\n', OF),   # re-anchored
    ("onset_s004_lr_confound_ignored", AN, 'if how == "pick" and g_pick.get(key) != 1.0:', "if False:", OF),
    ("onset_s006_reads_its_pick", AN, "a = {s: onset_half(runs, arm_name(sid, arm, 1.0, s)) for s in seeds}",
     "a = {s: onset_half(runs, arm_name(sid, arm, g_pick.get(key, 1.0), s)) for s in seeds}", OF),
    ("onset_arm_gap_not_descriptive", AN, '        if why:\n            out[key] = {**rec, "reading": "descriptive"',
     '        if sig["sigma_seed"] is None:\n            out[key] = {**rec, "reading": "descriptive"', OF),
    ("onset_holm_family_empty", AN, '        ps[key] = out[key]["p_two_sided"]\n', "", OF),
    ("verdicts_without_onsets", AN, '            "onsets": onset_readings(runs, p, g_pick)}', "}", OF),
    # the stage 1 --earlier default (C4's "noise check failed later" for S002's arms)
    ("earlier_default_dropped", AN, "ear = ([] if a.no_stage1_earlier else STAGE1_RESULTS) + a.earlier", "ear = a.earlier", CO),
    ("earlier_default_s002_missing", AN, 'for s in ("S001", "S002", "S003")]', 'for s in ("S001", "S003")]', CO),
    # SCREENS DEVIATION 1: lock holds counted once each
    ("holds_restart_ends_at_the_restart_line", HRS, "            close(last, SRC_CUT)\n            cur = None\n",
     "            close(t, SRC_CUT)\n            cur = None\n", HO),
    ("holds_unlock_lines_ignored", HRS, "        elif msg.startswith(UNLOCKED):", "        elif False:", HO),
    ("holds_one_per_train_line", HRS, '            cur["run"] = cur["run"] or m[1]',
     '            holds.append((cur["run"], cur["start"], None, SRC_EARLIER)) if cur["run"] else None\n'
     '            cur["run"] = m[1]', HO),
    ("holds_score_line_names_no_run", HRS, '            cur["run"] = cur["run"] or m[1] or m[2]', '            cur["run"] = cur["run"] or m[1]', HO),
    ("holds_nul_tail_not_stripped", HRS, '        ln = ln.lstrip("\\x00")\n', "", HO),
    ("holds_status_outside_a_hold_drops_the_run", HRS, "        else:                    # an earlier hold",
     "        elif False:              # an earlier hold", HO),
    ("load_keeps_the_last_hold_only", HRS, "runs[name] = runs.get(name, 0.0) + hours_between(a, b)",
     "runs[name] = hours_between(a, b)", HO),
    ("load_overlap_unchecked", HRS, 'assert all(b <= a2 or b2 <= a for a2, b2 in seen), f"{name}: two holds overlap"',
     "assert True", HO),
    ("load_repeat_accepted", HRS, 'assert (a, b) not in seen, f"{name} measured twice', 'assert True, f"{name} measured twice', HO),
    # SCREENS DEVIATION 1: durable scoring, the unparsable re-score, the cut count
    ("rescore_never", QLIB, '            $PY "$SX/score_sync.py" whole "$j" && continue\n', "            continue\n", QD),
    ("whole_accepts_a_json_array", SYNC, "return isinstance(json.load(f), dict)", "return json.load(f) is not None", QD),
    ("publish_scored_before_lines", SYNC,
     '    write_durably(os.path.join(out, "bpb.jsonl"), buf.getvalue())\n    write_durably(os.path.join(out, "SCORED"), "")\n',
     '    write_durably(os.path.join(out, "SCORED"), "")\n    write_durably(os.path.join(out, "bpb.jsonl"), buf.getvalue())\n', QD),
    ("publish_no_fsync_of_score_files", SYNC, "    for p in files:\n        durable.fsync_file(p)\n", "", QD),
    ("publish_scored_touched_plainly", SYNC, '    write_durably(os.path.join(out, "SCORED"), "")\n',
     '    open(os.path.join(out, "SCORED"), "w").close()\n', QD),
    ("publish_accepts_a_broken_file", SYNC, "    if not files or bad:\n", "    if not files:\n", QD),
    ("queue_back_to_score_run", QUEUE, 'elif score_screens "$out"; then status',
     'elif score_run "$out"; then touch "$out/SCORED"; status', QD),
    ("queue_cut_never_counted", QUEUE, '    cut_count "$out"\n    [ $? = 2 ]', "    [ 1 = 2 ]", QD),
    ("queue_qrun_never_cleared", QUEUE, '        qrun_clear "$out"\n', "", QD),
    ("cut_gap_at_three", QLIB, '    if [ "$n" -ge 2 ]; then', '    if [ "$n" -ge 3 ]; then', QD),
    ("cut_counted_after_final", QLIB, '    if ls "$out"/final_*.pt >/dev/null 2>&1; then\n        qlog "$(basename "$out"): QRUN',
     '    if false; then\n        qlog "$(basename "$out"): QRUN', QD),
    ("cut_qrun_left_behind", QLIB, '    rm -f "$out/QRUN"\n    n=$(wc -l', "    n=$(wc -l", QD),
    # the entries and the S006 trigger (what a later step reads as the rule)
    ("cap_flag_entry_says_cut", SXT, "would pass 28 h, nothing is cut and", "would pass 28 h, S006 is cut and", TX),
    ("deviation_entry_drops_the_cut_policy", SXT, "  (3) A CUT COUNTS AS THE RUN'S ONE CRASH", "  (3) A CUT IS NOT COUNTED", TX),
    ("onset_entry_without_its_s006_arms", SXT, "never its g 2 verdict runs;", "or its g 2 verdict runs;", TX),
    ("s006_trigger_only_on_a_nomination", S6N, "ONSET RULE CODE), whatever the bpb verdict", "ONSET RULE CODE), if the bpb verdict", TX),
    ("s006_trigger_20m_arm_dropped", S6N, "    (3) 20M (P-105's scale question", "    (3) none (P-105's scale question", TX),
    ("registry_run_moved_out", REG, '"locations": ["pc:~/planck/runs/SCREENS/s007_smear_g1_s102"]',
     '"locations": ["pc:~/planck/runs/SCREENS/s007_smear_g1_s102_old"]', TX + "::registered"),
    ("registry_late_run_marked_read", REG, '"result": "' + NR + '", "status": "success", "run": {"state": "completed", '
     '"started": "2026-10-07T02:27:22-04:00"', '"result": "' + NR[4:] + '", "status": "success", "run": {"state": '
     '"completed", "started": "2026-10-07T02:27:22-04:00"', TX + "::registered"),
]
