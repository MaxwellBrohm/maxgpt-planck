"""Mutation check for the PC kit tests: each mutant breaks one claim on purpose, in a temp
copy of pc/, and the named tests must go red. A mutant that survives means that test
proves nothing.

  <python> pc/tests/mutation_check.py --list
  <python> pc/tests/mutation_check.py --group guard     # guard | runner | start | hb | bench | boot
Killed = pytest exit 1 (a real failure). Exit 2+ (collection or usage error) is INVALID,
never counted as killed. A timeout (hang) is reported separately.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PC)

# (group, name, file under pc/, old, new, tests)
MUTANTS = [
    ("guard", "temp_boundary", "wsl/gpuguard.py", "temp >= c.max_temp_c", "temp > c.max_temp_c",
     "tests/test_gpuguard.py"),
    ("guard", "strikes_never_reset", "wsl/gpuguard.py",
     "self.strikes[key] + 1 if bad else 0", "self.strikes[key] + 1 if bad else self.strikes[key]",
     "tests/test_gpuguard.py"),
    ("guard", "na_temp_not_blind", "wsl/gpuguard.py",
     "blind = sample is None or (c.require_temp and temp is None)", "blind = sample is None",
     "tests/test_gpuguard.py"),
    ("guard", "mem_guard_off", "wsl/gpuguard.py", "mem > c.max_mem_mib", "False",
     "tests/test_gpuguard.py"),
    ("guard", "may_start_ignores_heat", "wsl/gpuguard.py",
     "temp > cfg.resume_temp_c", "temp > 1000", "tests/test_gpuguard.py"),
    ("guard", "no_old_driver_fallback", "wsl/gpuguard.py",
     "for f in (_throttle_field or THROTTLE_FIELDS):",
     "for f in (_throttle_field or THROTTLE_FIELDS[:1]):", "tests/test_gpuguard.py"),
    ("runner", "kill_leader_not_group", "wsl/queue_runner.py",
     "os.killpg(p.pid, sig)", "os.kill(p.pid, sig)", "tests/test_runner.py"),
    ("runner", "heat_fails_not_requeues", "wsl/queue_runner.py",
     'REQUEUE = {"temp", ', "REQUEUE = {", "tests/test_runner.py"),
    ("runner", "runaway_mem_requeued", "wsl/queue_runner.py",
     'REQUEUE = {"temp", ', 'REQUEUE = {"gpu_mem", "temp", ', "tests/test_runner.py"),
    ("runner", "stop_file_ignored", "wsl/queue_runner.py",
     'if trip is None and self.flag("STOP"):', "if False:", "tests/test_runner.py"),
    ("runner", "timeout_ignored", "wsl/queue_runner.py",
     "if trip is None and max_s and", "if False and", "tests/test_runner.py"),
    ("runner", "guard_not_fed", "wsl/queue_runner.py",
     "trip = guard.check(sample, sysinfo.mem_available_mib())", "trip = None",
     "tests/test_runner.py"),
    ("runner", "attempts_not_counted", "wsl/queue_runner.py",
     'job["attempts"] += 1', "pass", "tests/test_runner.py"),
    ("start", "start_ignores_gpu", "wsl/queue_runner.py",
     "ok, why = gpuguard.may_start(gpuguard.query(), gcfg)", "ok, why = True, 'ok'",
     "tests/test_runner_start.py"),
    ("start", "start_ignores_disk", "wsl/queue_runner.py", "if free < need:", "if False:",
     "tests/test_runner_start.py"),
    ("start", "disk_stop_ignored", "wsl/queue_runner.py",
     "if trip is None and sysinfo.free_gb(self.home) < self.stop_gb:", "if False:",
     "tests/test_runner_start.py"),
    ("start", "host_mem_not_fed", "wsl/queue_runner.py",
     "guard.check(sample, sysinfo.mem_available_mib())", "guard.check(sample, None)",
     "tests/test_runner_start.py"),
    ("start", "no_reboot_recovery", "wsl/queue_runner.py",
     "for path in jobq.running(self.home):", "for path in []:", "tests/test_runner_start.py"),
    ("start", "dirty_prereg_passes", "wsl/jobq.py",
     "if clean.returncode != 0:", "if False:", "tests/test_runner_start.py"),
    ("start", "untracked_prereg_passes", "wsl/jobq.py",
     "if tracked.returncode != 0:", "if False:", "tests/test_runner_start.py"),
    ("start", "prereg_optional", "wsl/jobq.py",
     'raise JobError("no prereg and no no_prereg_reason")', 'return "none"',
     "tests/test_runner_start.py"),
    ("start", "malformed_job_overwritten", "wsl/queue_runner.py",
     'jobq.move(todo[0], self.home, "failed")',
     'open(todo[0], "w").write("{}"); jobq.move(todo[0], self.home, "failed")',
     "tests/test_runner_start.py"),
    ("start", "runner_sigterm_ignored", "wsl/queue_runner.py",
     'if trip is None and self.shutdown:', "if False:", "tests/test_runner_start.py"),
    ("hb", "eta_wrong_units", "wsl/heartbeat.py", "/ tps / 3600", "/ tps / 60",
     "tests/test_heartbeat.py"),
    ("hb", "push_never_pushes", "wsl/heartbeat.py", 'r = _git(repo, "push", "-q")',
     'r = _git(repo, "status")', "tests/test_heartbeat.py"),
    ("hb", "throttle_not_shown", "wsl/heartbeat.py",
     'thr = ",".join(g.get("throttle_reasons") or []) or "none"', 'thr = "none"',
     "tests/test_heartbeat.py"),
    ("bench", "docmask_without_docs", "bench_micro.py",   # SPEED V3: the old target also hit the row
     'per_row, g) if mode == "docmask" else None', "per_row, g) if False else None",
     "tests/test_bench_micro.py"),
    ("bench", "oom_not_skipping", "bench_micro.py", "oomed.add(arm)", "pass",   # SPEED V3: per-arm skip
     "tests/test_bench_micro.py"),
    ("bench", "wrong_shape", "bench_micro.py", "cfg = sol.cfg.replace(seq_len=a.seq_len)",
     "cfg = sol.cfg.replace(seq_len=a.seq_len, n_layers=sol.cfg.n_layers + 1)",
     "tests/test_bench_micro.py"),
    # SPEED V3 (--arms, --repeats, --ce-chunk, --dynamo-capture-dynamic)
    ("bench", "arm_not_compiled", "bench_micro.py", "fwd = compile_forward(model, cmode, a.compile_backend)",
     "fwd = model", "tests/test_bench_micro.py::test_arms_interleave_and_compiled_rows_are_one_graph"),
    ("bench", "no_abba_order", "bench_micro.py", "for arm in (arms if rep % 2 == 0 else arms[::-1]):",
     "for arm in arms:", "tests/test_bench_micro.py::test_arms_interleave_and_compiled_rows_are_one_graph"),
    ("bench", "ce_chunk_dropped", "bench_micro.py", 'kw = {"ce_chunk": ce_chunk} if ce_chunk else {}',
     "kw = {}", "tests/test_bench_micro.py::test_ce_chunk_flag_reaches_the_model_and_keeps_the_loss"),
    ("bench", "capture_flag_left_on", "bench_micro.py",
     "torch._dynamo.config.capture_dynamic_output_shape_ops = False", "pass",
     "tests/test_bench_micro.py::test_capture_dynamic_flag_is_recorded_and_reset"),
    ("bench", "ce_arm_suffix_ignored", "bench_micro.py", "a.ce_chunk if (suffix or not a.any_ce) else 0",
     "a.ce_chunk if not a.any_ce else 0", "tests/test_bench_micro.py::test_ce_arms_interleave_with_plain_arms"),
] + [("boot", n, "wsl/" + f, old, new, t if t.startswith("tests/") else "tests/test_boot_resume.py::" + t)
      for n, f, old, new, t in [
    # boot.d (RUNBOOK "RESUME AFTER A REBOOT"): each claim of test_boot_resume.py, broken on purpose
    ("e2_ignores_its_end_lines", "boot.d/10_e2.sh", "ENDS=' (plan finished", "ENDS=' (NEVER",
     "test_e2_run_that_ended_itself_is_left_alone"),
    ("first_start_not_last", "boot.d/bootlib.sh", '"$2" "$1" 2>/dev/null | tail -1', '"$2" "$1" 2>/dev/null | head -1',
     "test_e2_only_the_last_start_counts"),
    ("e006_running_not_checked", "boot.d/20_e006.sh", 'if running "$ANY"; then', "if false; then",
     "test_a_running_queue_is_never_started_twice"),
    ("rc12_running_not_checked", "boot.d/30_rc12.sh", 'if running "$ANY"; then', "if false; then",
     "test_a_running_queue_is_never_started_twice"),
    ("global_stop_ignored", "boot.d/bootlib.sh", '"$P/STOP" "$P/PAUSE" ', "",
     "test_stop_files_hold_every_queue"),
    ("queue_stop_ignored", "boot.d/bootlib.sh", '"$BOOTD/$NAME.STOP" ', "", "test_stop_files_hold_every_queue"),
    ("dry_run_starts", "boot.d/bootlib.sh", 'would start: $*"; return 0; fi', 'would start: $*"; fi',
     "test_cut_off_queue_resumes_and_dry_run_starts_nothing"),
    ("e006_note_in_dry_run", "boot.d/20_e006.sh", "if [ $DRY = 0 ]; then", "if true; then",
     "test_cut_off_queue_resumes_and_dry_run_starts_nothing"),
    ("e006_cut_job_ignored", "boot.d/20_e006.sh", '[ -e "$n.guard.json" ] || cut=', "true || cut=",
     "test_e006_job_cut_mid_run_is_left_for_a_person"),
    ("e006_wrapper_exit_ignored", "boot.d/20_e006.sh", 'if [ -n "$wend" ]; then', "if false; then",
     "test_e006_wrapper_exit_after_last_start_is_left_alone"),
    ("rc12_done_ignored", "boot.d/30_rc12.sh", 'if [ -e "$DONE" ]; then', "if false; then",
     "test_rc12_finished_or_stopped_is_left_alone"),
    ("rc12_own_stop_ignored", "boot.d/30_rc12.sh", 'blocked "$P/logs/rc12_dev_queue.STOP"', "blocked",
     "test_rc12_own_stop_file_holds_it"),
    ("rc12_wrong_code", "boot.d/30_rc12.sh", 'env Q_CODE="$CODE" bash', "env bash",
     "test_cut_off_queue_resumes_and_dry_run_starts_nothing"),
    ("rc12_fixed_code_dir", "boot.d/30_rc12.sh",
     'CODE=${RC12_CODE:-$(sed -n "${start}s/.* queue: start, code \\([^,]*\\), root .*/\\1/p" "$QL")}',
     "CODE=${RC12_CODE:-$HOME/planck/dev/rc12_dg}",
     "tests/test_boot_rc12.py::test_rc12_resumes_from_the_cut_off_runs_code"),
    ("rc12_first_start_code", "boot.d/30_rc12.sh", 'sed -n "${start}s/', 'sed -n "1s/',
     "tests/test_boot_rc12.py::test_rc12_resumes_from_the_cut_off_runs_code"),
    ("rc12_override_ignored", "boot.d/30_rc12.sh", "CODE=${RC12_CODE:-", "CODE=${RC12_UNSET:-",
     "test_cut_off_queue_resumes_and_dry_run_starts_nothing"),
    ("rc12_missing_code_resumed", "boot.d/30_rc12.sh", '[ -f "$CODE/queue_dev_baselines.sh" ] ||', "true ||",
     "tests/test_boot_rc12.py::test_rc12_cut_off_runs_code_missing_is_not_resumed"),
    ("e2_no_fallback", "boot.d/10_e2.sh", 'trying the fallback"\nfallback', 'trying the fallback"\ntrue',
     "test_e2_falls_back_when_launch_e2_never_reaches_the_queue"),
    ("e2_fallback_after_queue_ran", "boot.d/10_e2.sh", "grep -aq '^code .* plan '; then", "false; then",
     "test_e2_no_fallback_when_the_queue_started_and_exited"),
    ("twice_per_boot", "boot.d/boot_resume.sh", 'if [ -n "$bid" ] && [', "if false && [",
     "test_dispatcher_resumes_once_per_boot"),
    ("blind_gpu_resumes", "boot.d/boot_resume.sh", "[ $MODE = dry ] || exit 1", "true",
     "test_dispatcher_resumes_nothing_while_the_gpu_is_blind"),
    ("runs_non_executable", "boot.d/boot_resume.sh", 'if [ ! -x "$s" ]; then', "if false; then",
     "test_dispatcher_dry_run_asks_every_script_and_starts_nothing"),
    ("start_never_resumes", "runner_start.sh", 'if [ -f "$BOOT_RESUME" ]; then', "if false; then",
     "test_runner_start_launches_boot_resume_only_when_installed"),
    ("start_resumes_uninstalled", "runner_start.sh", 'if [ -f "$BOOT_RESUME" ]; then', "if true; then",
     "test_runner_start_launches_boot_resume_only_when_installed"),
]] + [("boot", "screens_" + n, "wsl/boot.d/" + f, old, new, "tests/test_boot_screens.py::" + t)
      for n, f, old, new, t in [
    # boot.d/40_screens.sh: each claim of test_boot_screens.py, broken on purpose
    ("end_plan_finished", "40_screens.sh", "(plan finished: |", "(", "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_plan_says_stop", "40_screens.sh", "|plan says stop|", "|", "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_queue_stops", "40_screens.sh", "|queue stops at: |", "|", "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_stop_waiting", "40_screens.sh", "|STOP while waiting for |", "|",
     "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_refusing", "40_screens.sh", "|refusing: |", "|", "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_bad_plan_line", "40_screens.sh", "|bad plan line: )", ")", "test_every_queue_exit_after_the_start_is_left_alone"),
    ("end_no_time_stamp", "40_screens.sh", 'ENDS="$TS (', 'ENDS=" (',
     "test_end_words_not_at_a_queue_time_stamp_do_not_end_the_run"),
    ("end_searched_from_the_top", "40_screens.sh", 'tail -n +$((start + 1)) "$Q"; }', 'cat "$Q"; }',
     "test_only_the_last_start_counts"),
    ("own_stop_ignored", "40_screens.sh", 'blocked "$OUT/NO_BOOT_RESUME"', "blocked", "test_stop_files_hold_it"),
    ("running_not_checked", "40_screens.sh", 'if running "$ANY"; then', "if false; then",
     "test_nothing_starts_while_any_part_of_screens_runs"),
    ("train_py_not_seen", "40_screens.sh", r"|(train|preflight)\.py", r"|(preflight)\.py",
     "test_nothing_starts_while_any_part_of_screens_runs"),
    ("bpb_py_not_seen", "40_screens.sh", r"|bpb\.py [^ ]*/runs/SCREENS/", "",
     "test_nothing_starts_while_any_part_of_screens_runs"),
    ("newest_clone", "40_screens.sh", 'C=$P/qcode/$(echo "$ref" | cut -c1-12)', 'C=$(ls -d "$P"/qcode/* | tail -1)',
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("plan_from_current", "40_screens.sh", 'plan=$(echo "$sline" | sed -n',
     'plan=$(grep -v "^#" "$P/qcode/$(echo "$ref" | cut -c1-12)/experiments/screens/plans/CURRENT").txt; : $(true',
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("missing_clone_unnamed", "40_screens.sh", '[ -f "$X/queue_screens.sh" ] ||', "true ||",
     "test_missing_or_changed_clone_or_plan_is_refused"),
    ("clone_commit_unchecked", "40_screens.sh", 'case $head in "$ref"*) ;;', "case $head in *) ;;",
     "test_missing_or_changed_clone_or_plan_is_refused"),
    ("plan_sha_unchecked", "40_screens.sh", '[ "$got" = "$psha" ] ||', "true ||",
     "test_missing_or_changed_clone_or_plan_is_refused"),
    ("wrong_queue_args", "40_screens.sh", '--code "$C" --plan "$X/plans/$plan"', '--plan "$X/plans/$plan"',
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("note_in_dry_run", "40_screens.sh", "if [ $DRY = 0 ]; then", "if true; then",
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("note_glued_to_nul_tail", "40_screens.sh", '[ $(tail -c1 "$Q" | wc -l) -eq 1 ] || echo >> "$Q"', "true",
     "test_nul_tailed_log_resumes_and_the_note_starts_a_clean_line"),
    ("no_damage_check", "40_screens.sh", 'bad=$(damage "$OUT"); rc=$?', "bad=; rc=0",
     "test_power_cut_damage_is_left_for_a_person"),
    ("check_failure_resumes", "40_screens.sh", '*) refuse "the damage check did not run ($PY3 exit $rc)";;', "*) ;;",
     "test_a_damage_check_that_cannot_run_resumes_nothing"),
    ("pt_header_only", "40_screens.sh", r'f.read(4) == b"PK\x03\x04" and zipfile.is_zipfile(p)',
     r'f.read(4) == b"PK\x03\x04"', "test_power_cut_damage_is_left_for_a_person"),
    ("pt_tmp_counted", "40_screens.sh", 'if f.endswith(".pt") and not whole', 'if ".pt" in f and not whole',
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("latest_target_unchecked", "40_screens.sh", "if n not in names:", "if n is None:",
     "test_power_cut_damage_is_left_for_a_person"),
    ("scored_trusted", "40_screens.sh", "if scored and (not js or", "if False and (not js or",
     "test_power_cut_damage_is_left_for_a_person"),
    ("scored_json_0_bytes_ok", "40_screens.sh", "if (scored or size(", "if (size(",
     "test_power_cut_damage_is_left_for_a_person"),
    ("unscored_json_0_bytes_damage", "40_screens.sh", "if (scored or size(os.path.join(b, f)) != 0) and", "if True and",
     "test_cut_off_queue_resumes_its_own_plan_and_clone"),
    ("trunk_final_unchecked", "40_screens.sh", 'if "TRUNK_DONE" in names and', "if False and",
     "test_power_cut_damage_is_left_for_a_person"),
    # without -a a log with NULs before its start line reads as binary (GNU grep and, NULs this early, BSD grep)
    ("nul_log_needs_grep_a", "bootlib.sh", 'grep -anE "$2" "$1"', 'grep -nE "$2" "$1"',
     "test_nul_tailed_log_resumes_and_the_note_starts_a_clean_line"),
]] + [("boot", "screens_" + n, "wsl/boot.d/40_screens.sh", old, new, "tests/test_boot_screens_edges.py::" + t)
      for n, old, new, t in [
    # boot.d/40_screens.sh: the review's edge cases (test_boot_screens_edges.py), each survived the main file's tests
    ("start_line_read_whole", """printf '%s\\n' "${sline##* queue start: exp SCREENS, plan }";""",
     """printf '%s\\n' "$sline";""", "test_an_older_end_glued_in_front_of_the_start_line_does_not_end_the_new_run"),
    ("end_at_line_start_only", 'ENDS="$TS (', 'ENDS="^$TS (', "test_a_refusal_glued_after_nuls_still_ends_the_run"),
    ("ref_first_on_the_line", 'ref=$(echo "$sline" | sed -n',
     """ref=$(echo "$sline" | grep -o 'code commit [0-9a-f]*' | head -1 | cut -d' ' -f3); : $(true""",
     "test_a_partial_older_start_glued_in_front_reads_the_newer_one"),
    ("plan_first_on_the_line", """plan=$(echo "$sline" | sed -n 's/.* plan""",
     """plan=$(echo "$sline" | sed -n 's/^[^(]* plan""", "test_a_partial_older_start_glued_in_front_reads_the_newer_one"),
    ("any_experiment_blocks", "experiments/(screens|S00[0-9]_[^/ ]*)/configs/", "experiments/[^/ ]*/configs/",
     "test_other_queues_running_do_not_hold_it_back"),
    ("pt_end_record_only", 'return f.read(4) == b"PK\\x03\\x04" and zipfile.is_zipfile(p)',
     "return zipfile.is_zipfile(p)", "test_a_checkpoint_with_a_nul_head_and_a_whole_end_is_damage"),
    ("bpb_jsonl_one_json", 'parses(os.path.join(d, "bpb.jsonl"), rows=True)', 'parses(os.path.join(d, "bpb.jsonl"))',
     "test_a_healthy_bpb_jsonl_of_many_rows_is_not_damage"),
    # second review: a later start whose line the cut lost (code_sha256_<time>.txt newer than the last start line)
    ("lost_start_unchecked", '[ "$cs" -gt "$st" ]', "false", "test_a_start_line_the_cut_lost_is_left_for_a_person"),
    ("lost_start_oldest_list", "| sort | tail -1)", "| sort | head -1)",
     "test_a_start_line_the_cut_lost_is_left_for_a_person"),
    ("lost_start_same_second", '[ "$cs" -gt "$st" ]', '[ "$cs" -ge "$st" ]',
     "test_each_start_lines_own_code_list_is_not_a_lost_start"),
]]


def run_one(m, python: str) -> tuple[str, float]:
    group, name, rel, old, new, tests = m
    with tempfile.TemporaryDirectory(prefix="pcmut-") as tmp:
        dst = os.path.join(tmp, "pc")
        shutil.copytree(PC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.jsonl"))
        os.symlink(os.path.join(ROOT, "harness"), os.path.join(tmp, "harness"))
        path = os.path.join(dst, rel)
        src = open(path).read()
        if src.count(old) != 1:
            return f"INVALID (pattern found {src.count(old)} times)", 0.0
        open(path, "w").write(src.replace(old, new))
        t0 = time.time()
        try:
            r = subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                tests], cwd=dst, capture_output=True, text=True, timeout=100)
        except subprocess.TimeoutExpired:
            return "killed (TIMEOUT)", time.time() - t0
        dt = time.time() - t0
        if r.returncode == 1:
            return "killed", dt
        if r.returncode == 0:
            return "SURVIVED", dt
        return f"INVALID (pytest exit {r.returncode})", dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    todo = [m for m in MUTANTS if not a.group or m[0] == a.group]
    if a.list:
        for m in todo:
            print(f"{m[0]:>7} {m[1]}")
        return 0
    bad = 0
    for m in todo:
        verdict, dt = run_one(m, sys.executable)
        bad += not verdict.startswith("killed")
        print(f"{m[0]:>7} {m[1]:<26} {verdict:<22} {dt:5.1f}s", flush=True)
    print(f"{len(todo) - bad}/{len(todo)} killed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
