"""Planck PC queue runner: one GPU job at a time, guarded (stdlib only, runs under WSL2).

  python3 queue_runner.py [--home ~/planck] [--poll 10] [--grace 120] [--push]

Layout under --home (default $PLANCK_HOME, else ~/planck); jobq.py documents the job file:
  queue/pending/*.json  started in filename order   queue/running/  at most one job
  queue/done/ queue/failed/                         logs/<job>.log  stdout + stderr
  status/runner.json    live state                   status/jobs.jsonl  one record per attempt
  PAUSE                 start no new job             STOP  checkpoint-and-stop the running job
  runner.lock           one runner per machine (flock)

Stopping a job = SIGTERM to its process group (train.py checkpoints and exits on SIGTERM),
then SIGKILL after --grace seconds. Trips that are about the machine (heat, disk, STOP,
nvidia-smi blind, runner shutdown, reboot) put the job back in pending/ to resume from its
checkpoint, up to max_attempts starts. Runaway GPU or host memory, a timeout or a nonzero
exit send it to failed/ for a person to read. Before any start: >= 25 GB free disk, GPU at
or below resume_temp_c, less than start_max_mem_mib in use, and a committed prereg.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpuguard  # noqa: E402
import jobq  # noqa: E402
import sysinfo  # noqa: E402
from gpuguard import Guard, GuardConfig  # noqa: E402

REQUEUE = {"temp", "disk", "stop_file", "smi", "interrupted", "shutdown"}


class Runner:
    def __init__(self, home: str, poll: float = 10.0, idle: float = 60.0, grace: float = 120.0,
                 min_start_gb: float = 25.0, stop_gb: float = 15.0, guard: GuardConfig | None = None,
                 require_prereg: bool = True, heartbeat_every: float = 1800.0, push: bool = False,
                 exit_when_idle: bool = False, max_attempts: int = 5):
        self.home, self.poll, self.idle, self.grace = home, poll, idle, grace
        self.min_start_gb, self.stop_gb = min_start_gb, stop_gb
        self.guard = guard or GuardConfig()
        self.require_prereg, self.max_attempts = require_prereg, max_attempts
        self.heartbeat_every, self.push, self.exit_when_idle = heartbeat_every, push, exit_when_idle
        self.shutdown = False
        self._hb, self._hb_last = None, 0.0
        self.status_dir = os.path.join(home, "status")
        jobq.ensure_layout(home)

    # ------------------------------------------------------------------ helpers
    def flag(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.home, name))

    def state(self, **kw) -> None:
        rec = {"updated": sysinfo.now_iso(), "runner_pid": os.getpid(), **kw}
        jobq.write_json(os.path.join(self.status_dir, "runner.json"), rec)

    def record(self, path: str, job: dict, outcome: str, **entry) -> None:
        entry = {"attempt": job["attempts"], "end": sysinfo.now_iso(), "outcome": outcome, **entry}
        job["history"].append(entry)
        jobq.save(job, path)
        jobq.move(path, self.home, outcome)
        jobq.append_jsonl(os.path.join(self.status_dir, "jobs.jsonl"), {"name": job["name"], **entry})

    def maybe_heartbeat(self) -> None:
        if self.heartbeat_every <= 0 or time.time() - self._hb_last < self.heartbeat_every:
            return
        if self._hb is not None and self._hb.poll() is None:
            return                                  # the last one is still pushing
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(here, "heartbeat.py"), "--home", self.home]
        self._hb = subprocess.Popen(cmd + (["--push"] if self.push else []),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._hb_last = time.time()

    def _requeue_or_fail(self, job: dict) -> str:
        limit = int(job.get("max_attempts", self.max_attempts))
        return "pending" if job["attempts"] < limit else "failed"

    # ------------------------------------------------------------------ queue steps
    def recover(self) -> None:
        """A job left in running/ was cut off (reboot, crash, wsl --shutdown)."""
        for path in jobq.running(self.home):
            try:
                job = jobq.load(path)
            except jobq.JobError as e:
                jobq.move(path, self.home, "failed")
                print(f"runner: unreadable job in running/ moved to failed: {e}", flush=True)
                continue
            self.record(path, job, self._requeue_or_fail(job), trip="interrupted")

    def preflight(self, path: str) -> tuple[str, str]:
        """-> ("start" | "wait" | "fail", reason)."""
        try:
            job = jobq.load(path)
            note = jobq.check_prereg(job, self.require_prereg)
            gcfg = self.guard.merged(job.get("guard"))
        except (jobq.JobError, ValueError, TypeError) as e:
            return "fail", str(e)
        free, need = sysinfo.free_gb(self.home), max(self.min_start_gb, self.stop_gb)
        if free < need:
            return "wait", f"disk: {free:.1f} GB free < {need:.0f} GB"
        ok, why = gpuguard.may_start(gpuguard.query(), gcfg)
        return ("start", note) if ok else ("wait", why)

    def stop_proc(self, p: subprocess.Popen) -> int:
        for sig, wait in ((signal.SIGTERM, self.grace), (signal.SIGKILL, None)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
            try:
                return p.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        return p.wait()

    def run_job(self, path: str, note: str) -> str:
        job = jobq.load(path)
        path = jobq.move(path, self.home, "running")
        job["attempts"] += 1
        jobq.save(job, path)
        r = jobq.resolved(job)
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               **{k: jobq.expand(str(v)) for k, v in job.get("env", {}).items()}}
        log_path = os.path.join(self.home, "logs", f"{job['name']}.log")
        start, t0 = sysinfo.now_iso(), time.time()
        guard = Guard(self.guard.merged(job.get("guard")))
        max_s = float(job.get("max_hours") or 0) * 3600
        with open(log_path, "a") as log:
            log.write(f"\n=== runner: start {start} attempt {job['attempts']} ({note})\n"
                      f"=== cmd {r['cmd']} cwd {r['cwd']}\n")
            log.flush()
            try:
                p = subprocess.Popen(r["cmd"], cwd=r["cwd"], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            except OSError as e:
                self.record(path, job, "failed", start=start, reason=f"spawn failed: {e}")
                return "failed"
            trip, rc = None, None
            while True:
                rc = p.poll()
                if rc is not None:
                    break
                sample = gpuguard.query()
                trip = guard.check(sample, sysinfo.mem_available_mib())
                if trip is None and sysinfo.free_gb(self.home) < self.stop_gb:
                    trip = "disk"
                if trip is None and self.flag("STOP"):
                    trip = "stop_file"
                if trip is None and self.shutdown:
                    trip = "shutdown"
                if trip is None and max_s and time.time() - t0 > max_s:
                    trip = "timeout"
                self.state(state="running", job=job["name"], pid=p.pid, started=start,
                           elapsed_s=round(time.time() - t0), gpu=sample, strikes=guard.strikes,
                           attempt=job["attempts"], progress_log=r["progress_log"],
                           total_tokens=job.get("total_tokens"))
                self.maybe_heartbeat()
                if trip:
                    break
                time.sleep(self.poll)
            if trip:
                log.write(f"\n=== runner: guard trip '{trip}', stopping (SIGTERM, then SIGKILL "
                          f"after {self.grace:.0f} s)\n")
                log.flush()
                rc = self.stop_proc(p)
            log.write(f"=== runner: end rc={rc} trip={trip}\n")
        if trip in REQUEUE:
            outcome = self._requeue_or_fail(job)
        else:
            outcome = "done" if (trip is None and rc == 0) else "failed"
        self.record(path, job, outcome, start=start, rc=rc, trip=trip,
                    seconds=round(time.time() - t0), peak_temp_c=guard.peak["temp_c"],
                    peak_mem_mib=guard.peak["mem_used_mib"])
        return outcome

    # ------------------------------------------------------------------ main loop
    def loop(self, max_jobs: int = 0) -> int:
        self.recover()
        ran = 0
        while not self.shutdown:
            self.maybe_heartbeat()
            if self.flag("STOP") or self.flag("PAUSE"):
                self.state(state="paused", why="STOP" if self.flag("STOP") else "PAUSE")
                if self.exit_when_idle:
                    return ran
                time.sleep(self.idle)
                continue
            todo = jobq.pending(self.home)
            if not todo:
                self.state(state="idle")
                if self.exit_when_idle:
                    return ran
                time.sleep(self.idle)
                continue
            verdict, why = self.preflight(todo[0])
            if verdict == "fail":
                try:
                    self.record(todo[0], jobq.load(todo[0]), "failed", reason=why)
                except jobq.JobError:            # unreadable: move it untouched, log why
                    jobq.move(todo[0], self.home, "failed")
                    jobq.append_jsonl(os.path.join(self.status_dir, "jobs.jsonl"),
                                      {"name": os.path.basename(todo[0]), "outcome": "failed",
                                       "end": sysinfo.now_iso(), "reason": why})
                continue
            if verdict == "wait":
                self.state(state="waiting", job=os.path.basename(todo[0]), why=why)
                if self.exit_when_idle:
                    return ran
                time.sleep(self.idle)
                continue
            self.run_job(todo[0], why)
            ran += 1
            if max_jobs and ran >= max_jobs:
                return ran
        self.state(state="stopped", why="runner shutdown")
        return ran


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Planck PC GPU job queue runner")
    ap.add_argument("--home", default=os.environ.get("PLANCK_HOME", "~/planck"))
    ap.add_argument("--poll", type=float, default=10.0, help="guard sample interval, s")
    ap.add_argument("--idle", type=float, default=60.0, help="queue check interval, s")
    ap.add_argument("--grace", type=float, default=120.0, help="SIGTERM to SIGKILL, s")
    ap.add_argument("--min-start-gb", type=float, default=25.0)
    ap.add_argument("--stop-gb", type=float, default=15.0)
    ap.add_argument("--heartbeat-every", type=float, default=1800.0, help="0 = off")
    ap.add_argument("--push", action="store_true", help="heartbeat pushes to the status repo")
    ap.add_argument("--no-require-prereg", action="store_true")
    ap.add_argument("--exit-when-idle", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=0)
    a = ap.parse_args(argv)
    home = os.path.abspath(jobq.expand(a.home))
    os.environ["PLANCK_HOME"] = home
    os.makedirs(home, exist_ok=True)
    lock = open(os.path.join(home, "runner.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("runner: another runner holds the lock; exiting", flush=True)
        return 3
    r = Runner(home, a.poll, a.idle, a.grace, a.min_start_gb, a.stop_gb, None,
               not a.no_require_prereg, a.heartbeat_every, a.push, a.exit_when_idle)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: setattr(r, "shutdown", True))
    print(f"runner: home {home}, pid {os.getpid()}", flush=True)
    r.loop(a.max_jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
