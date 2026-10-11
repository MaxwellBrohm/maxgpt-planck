"""Mutants for the C6 diagnostics runner (diag_runner.py, diag_lib.py, diag_queue.sh; SCREENS AMENDMENT C6-DIAG, E3
PART 4d ADDENDUM). Same format as mutation_screens.py (name, file, old, new, test file); each must make the named test
file fail. Run alone, on scratch copies only (the repo is never edited):
  ~/.venvs/planck/bin/python experiments/screens/tests/mutants_c6diag.py [--only NAME,..] [--scratch DIR]
"""
import os
import sys

RN, RD, QU = "test_diag_runner.py", "test_diag_readings.py", "test_diag_queue.py"
DRP, DLP, DQ = "experiments/screens/diag_runner.py", "experiments/screens/diag_lib.py", "experiments/screens/diag_queue.sh"
MUTANTS_C6DIAG_LIB = [
    ("c6_hl_mean_of_half_lives", DLP, '"half_life": (LN2 / mean).tolist(),', '"half_life": hl.mean(dim=0).tolist(),', RD),
    ("c6_hl_pads_counted", DLP, "self.vals[i].append(nlf.transpose(1, 2)[real].double().cpu())",
     "self.vals[i].append(nlf.transpose(1, 2).reshape(-1, nlf.size(1)).double().cpu())", RD),
    ("c6_hl_one_minus_f", DLP, "nlf = -mod.forget_logf(x)", "nlf = 1 - mod.forget_logf(x).exp()", RD),
    ("c6_hl_p10_reads_p90", DLP, '"half_life_p10": torch.quantile(hl, 0.1, dim=0)', '"half_life_p10": torch.quantile(hl, 0.9, dim=0)', RD),
    ("c6_hl_nlf_p90_reads_p10", DLP, '"neg_log_f_p90": torch.quantile(nlf, 0.9, dim=0)', '"neg_log_f_p90": torch.quantile(nlf, 0.1, dim=0)', RD),
    ("c6_hl_log_base", DLP, "LN2 = math.log(2.0)", "LN2 = 1.0", RD),
    ("c6_mod_canon_signed", DLP, "w = sd[k].detach().float().abs()", "w = sd[k].detach().float()", RD),
    ("c6_mod_canon_site_c_dropped", DLP, r'canon_([ac])\.weight$', r'canon_([a])\.weight$', RD),
    ("c6_mod_smear_dropped", DLP, "    if smear:\n", "    if False:\n", RD),
    ("c6_mod_forget_norm_axis", DLP, '"w_norm": sd[k[:-1] + "w"].detach().float().norm(dim=1)',
     '"w_norm": sd[k[:-1] + "w"].detach().float().norm(dim=0)', RD),
    ("c6_hd_off", DLP, '["oasst2"], device, "fp32", BPB_BATCH, None, hd=True)', '["oasst2"], device, "fp32", BPB_BATCH, None, hd=False)', RD),
    ("c6_hd_bf16", DLP, '["oasst2"], device, "fp32", BPB_BATCH, None, hd=True)', '["oasst2"], device, "bf16", BPB_BATCH, None, hd=True)', RD),
    ("c6_attn_windows", DLP, "WINDOWS, BPB_BATCH, ATTN_BATCH = 200, 16384, 8192", "WINDOWS, BPB_BATCH, ATTN_BATCH = 20, 16384, 8192", RD),
    ("c6_attn_check_ignored", DLP, '    if not r["check_ok"]:\n', "    if False:\n", RN),
    ("c6_ind_file_unpinned", DLP, "induction.load_seqs(IND_FILE, IND_SHA)", "induction.load_seqs(IND_FILE, None)", RN),
    ("c6_evalset_pin_value", DLP, 'EVALSET_SHA = "fe1b55bb', 'EVALSET_SHA = "fe1b55bc', RN),
]
MUTANTS_C6DIAG_RUNNER = [
    ("c6_stops_ignored", DRP, 'STOPS = ("DIVERGED", "GAP", "rc12_eval.jsonl")', 'STOPS = ("GAP",)', RN),
    ("c6_second_final_allowed", DRP, "    if len(fin) != 1:\n", "    if len(fin) < 1:\n", RN),
    ("c6_final_name_unchecked", DRP, '    if fin[0] != f"final_{step:08d}.pt":\n', "    if False:\n", RN),
    ("c6_ck_step_unchecked", DRP, 'if ck.get("step") != step or mc.get', "if False or mc.get", RN),
    ("c6_flags_unchecked", DRP, 'mc.get("d_model") != d or got != want:', 'mc.get("d_model") != d:', RN),
    ("c6_d_model_unchecked", DRP, 'or mc.get("d_model") != d or got', "or got", RN),
    ("c6_20m_step", DRP, "(15259, 320, want)", "(15260, 320, want)", RN),
    ("c6_resume_ignored", DRP, "        if not os.path.exists(out):\n", "        if True:\n", RN),
    ("c6_budget_ignored", DRP, "if done_any and time.time() - t0 + need > a.budget_s:", "if False:", RN),
    ("c6_budget_before_any_job", DRP, "if done_any and time.time() - t0 + need > a.budget_s:",
     "if time.time() - t0 + need > a.budget_s:", RN),
    ("c6_replay_same_process", DRP, 'if g == "replay" and (exp, r) in seen:', "if False:", RD),
    ("c6_failed_job_stops_all", DRP, "                status = worst(status, 1)\n                continue\n",
     "                return 1\n", RN),
    ("c6_exit_precedence", DRP, "rank = {0: 0, 3: 1, 1: 2, 2: 3}", "rank = {0: 0, 3: 3, 1: 2, 2: 1}", RN),
    ("c6_cpu_only_builds_model", DRP, "            if not a.cpu_only:\n                import bpb",
     "            if True:\n                import bpb", RN),
    ("c6_cpu_only_runs_all", DRP, 'if a.cpu_only and j != "modules":', "if False:", RN),
    ("c6_tokenizer_unpinned", DRP, "if sha256(DL.TOK_FILE) != DL.TOK_SHA:", "if False:", RN),
    ("c6_evalset_unpinned", DRP, 'if man.get("evalset_sha256") != DL.EVALSET_SHA:', "if False:", RN),
    ("c6_reads_result_files", DRP, "    names = os.listdir(rdir)\n",
     "    names = os.listdir(rdir)\n    [open(os.path.join(rdir, n)).close() for n in names if n.endswith('.jsonl')]\n", RN),
    ("c6_machine_path_written", DRP, '"checkpoint": os.path.basename(ckp),', '"checkpoint": ckp,', RN),
    ("c6_stage2_drops_s007", DRP, '"s004_canonac_g1", "s007_smear_g1", "s006_mtp_g2")', '"s004_canonac_g1", "s006_mtp_g2")', RN),
    ("c6_stage2_ind_only_runs", DRP, '"s007_smear_g1", "s006_mtp_g2")', '"s007_smear_g1", "s006_mtp_g1")', RN),
    ("c6_group_order", DRP, 'GROUPS = {"e3p1": ("E3", E3P1, ("ind", "hd"), ""), "stage2"',
     'GROUPS = {"stage2"', RN),
    ("c6_halflife_on_s005_base", DRP, 'return run.startswith("s005_forget")', 'return run.startswith("s005_")', RN),
    ("c6_stops_gap_dropped", DRP, 'STOPS = ("DIVERGED", "GAP", "rc12_eval.jsonl")', 'STOPS = ("DIVERGED", "rc12_eval.jsonl")', RN),
    ("c6_missing_dir_unchecked", DRP, "    if not os.path.isdir(rdir):\n", "    if False:\n", RN),
    ("c6_unknown_group_accepted", DRP, "if a.groups and set(a.groups) - set(GROUPS):", "if False:", RN),
    ("c6_step_not_recorded", DRP, '"step": ck.get("step"),', '"step": None,', RD),
    ("c6_hl_no_gate_check", DLP, "        if not mod.forget_gate:\n            return\n", "", RD),
]
MUTANTS_C6DIAG_QUEUE = [
    ("c6q_lock_wait_exits", DQ, '        say "gpu.lock not free after $LOCKWAIT s: waiting again"\n        continue\n',
     '        say "gpu.lock not free after $LOCKWAIT s: waiting again"\n        exit 1\n', QU),
    ("c6q_no_hard_timeout", DQ, '    timeout -k 30 "$HOLD" "$PY"', '    "$PY"', QU),
    ("c6q_hold_default_long", DQ, "HOLD=${C6_HOLD_S:-1980}", "HOLD=${C6_HOLD_S:-2400}", QU),
    ("c6q_lock_kept", DQ, '    flock -u 9; exec 9>&-\n    say "gpu.lock released', '    exec 9>&-\n    say "gpu.lock released', QU),
    ("c6q_memory_unchecked", DQ, 'if [ -z "$used" ] || [ "$used" -gt 3000 ]; then', "if false; then", QU),
    ("c6q_heat_unchecked", DQ, '[ "$t" -gt 75 ]', '[ "$t" -gt 99 ]', QU),
    ("c6q_disk_unchecked", DQ, '-lt $MINFREE ]', "-lt 0 ]", QU),
    ("c6q_pause_ignored", DQ, 'if [ -e "$P/PAUSE" ]; then', "if false; then", QU),
    ("c6q_stop_ignored", DQ, '[ -e "$P/STOP" ] && {', '[ -e "$P/STOP.none" ] && {', QU),
    ("c6q_cancel_ignored", DQ, '[ -e "$ST/CANCEL" ] && {', '[ -e "$ST/CANCEL.none" ] && {', QU),
    ("c6q_budget_exit_ends", DQ, "        3) ;;\n", "        3) exit 0 ;;\n", QU),
    ("c6q_failure_retried", DQ, '*) say "runner exit $rc: stopping for a person"; exit 1 ;;', "*) ;;", QU),
    ("c6q_one_kill_stops", DQ, "[ $kills -ge 2 ]", "[ $kills -ge 1 ]", QU),
    ("c6q_kills_never_stop", DQ, "[ $kills -ge 2 ]", "[ $kills -ge 99 ]", QU),
    ("c6q_no_instance_lock", DQ, "flock -n 8 || {", "true || {", QU),
    ("c6q_cpu_failure_ignored", DQ, '[ $rc = 0 ] || { say "CPU pass', 'true || { say "CPU pass', QU),
    ("c6q_planck_bundle_allowed", DQ, 'planck.bundle|*/*|"")', '*/*|"")', QU),
    ("c6q_clone_drops_groups", DQ, 'diag_queue.sh" --code "$Q" "${GA[@]}"', 'diag_queue.sh" --code "$Q"', QU),
    ("c6q_no_cpu_pass", DQ, '"${R[@]}" --cpu-only >> "$LOG" 2>&1', '"${R[@]}" --device "$DEVICE" >> "$LOG" 2>&1', QU),
    ("c6q_done_unmarked", DQ, "0) date '+%F %T' > \"$ST/DONE\"; say \"DONE\"", '0) say "DONE"', QU),
    ("c6q_only_dropped", DQ, '[ -n "$ONLY" ] && GA=("${GA[@]}" --only "$ONLY")', "", QU),
    ("c6q_start_wrong_bundle", "experiments/screens/diag_start.sh", "--bundle planck_c6diag.bundle", "--bundle planck.bundle", QU),
    ("c6q_start_drops_args", "experiments/screens/diag_start.sh", 'planck_c6diag.bundle "$@"', "planck_c6diag.bundle", QU),
    ("c6q_evalset_path", DQ, '--evalset "$P/data/evalsets/v0"', '--evalset "$P/data/evalsets/v1"', QU),
    ("c6q_wait_logged_every_poll", DQ, '[ "$1" = "$LASTW" ] || say "$1"', 'say "$1"', QU),
    ("c6q_lock_wait_short", DQ, "LOCKWAIT=${C6_LOCK_WAIT_S:-7200}", "LOCKWAIT=${C6_LOCK_WAIT_S:-600}", QU),
]
T4, A4 = "test_e3_part4d.py", "experiments/E3_seed_noise/e3analyze_4d.py"
MUTANTS_C6DIAG_E3 = [
    ("e4d_user_reads_assistant", A4, '"HD_user": ("hd", ROLE("user", "HD"))', '"HD_user": ("hd", ROLE("assistant", "HD"))', T4),
    ("e4d_hdn_reads_hd", A4, '"HDn_asst": ("hd", ROLE("assistant", "HD_none"))', '"HDn_asst": ("hd", ROLE("assistant", "HD"))', T4),
    ("e4d_non_finite_kept", A4, "return v if math.isfinite(v) else None", "return v", T4),
    ("e4d_min_pairs_2", A4, "MIN_PAIRS = 6", "MIN_PAIRS = 2", T4),
    ("e4d_sd_of_arm_a", A4, "st.stdev(d), E.factor(n - 1)", "st.stdev(a), E.factor(n - 1)", T4),
    ("e4d_seed1_kept", A4, "for s, a, b in kept if s != 1]", "for s, a, b in kept if s != 0]", T4),
    ("e4d_replay_wrong_run", A4, '"e3_5m_A_s1_R2", "e3_5m_A_s1_R3")', '"e3_5m_A_s2", "e3_5m_A_s1_R3")', T4),
    ("e4d_scorer_replay_signed", A4, "abs(again - once)", "(once - again)", T4),
    ("e4d_tables_to_here", A4, 'open(os.path.join(dest, "tables_part4d.txt"), "w")', 'open(os.path.join(HERE, "tables_part4d.txt"), "w")', T4),
    ("e4d_ub80_dropped", A4, '"unpaired_ub80": f * s_seed * math.sqrt(2 / k) * E.ub(df)}',
     '"unpaired_ub80": f * s_seed * math.sqrt(2 / k)}', T4),
]
MUTANTS_C6DIAG = MUTANTS_C6DIAG_LIB + MUTANTS_C6DIAG_RUNNER + MUTANTS_C6DIAG_QUEUE + MUTANTS_C6DIAG_E3

if __name__ == "__main__":         # these mutants alone; mutation_screens.py also runs them (its tree holds A4)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mutation_screens as M
    M.MUTANTS = MUTANTS_C6DIAG
    sys.exit(M.main())
