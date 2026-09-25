"""Mutation check for the PC kit tests: each mutant breaks one claim on purpose, in a temp
copy of pc/, and the named tests must go red. A mutant that survives means that test
proves nothing.

  <python> pc/tests/mutation_check.py --list
  <python> pc/tests/mutation_check.py --group guard     # guard | runner | start | hb | bench
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
    ("bench", "docmask_without_docs", "bench_micro.py",
     'if mode == "docmask" else None', "if False else None", "tests/test_bench_micro.py"),
    ("bench", "oom_not_skipping", "bench_micro.py",
     'if r["status"] == "oom":\n                    break', 'if r["status"] == "oom":\n                    pass',
     "tests/test_bench_micro.py"),
    ("bench", "wrong_shape", "bench_micro.py", "cfg = sol.cfg.replace(seq_len=a.seq_len)",
     "cfg = sol.cfg.replace(seq_len=a.seq_len, n_layers=sol.cfg.n_layers + 1)",
     "tests/test_bench_micro.py"),
]


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
