"""Job files and queue folders for the PC runner (stdlib only).

A job is one JSON file in queue/pending/, started in filename order (use 010_, 020_ ...):

  {"name": "etok-5m-trunk",                      optional, default = file stem
   "cmd": ["$PLANCK_HOME/venv/bin/python", "train.py", "configs/x/config.yaml",
           "--device", "cuda", "--require-committed"],
   "cwd": "$PLANCK_HOME/repo/harness",
   "env": {"PLANCK_TEST_THREADS": "8"},          optional
   "prereg": "configs/x/prereg.yaml",            relative to cwd; must be committed in git
   "no_prereg_reason": "benchmark",              instead of prereg, for non-experiment jobs
   "progress_log": "../runs/x/log.jsonl",        optional, relative to cwd (heartbeat reads it)
   "total_tokens": 250000000,                    optional, for the heartbeat ETA
   "max_hours": 72,                              optional wall-clock cap
   "max_attempts": 3,                            requeues after reboot/heat/disk/STOP
   "guard": {"max_mem_mib": 11000}}              optional GuardConfig overrides

"~" and $VARS (PLANCK_HOME included) are expanded in cmd, cwd, prereg and progress_log.
The runner owns the "attempts" and "history" keys.
"""
from __future__ import annotations

import json
import os
import subprocess

DIRS = ("pending", "running", "done", "failed")


class JobError(ValueError):
    """The job file itself is wrong: it goes to failed/, never retried."""


def qdir(home: str, which: str) -> str:
    return os.path.join(home, "queue", which)


def ensure_layout(home: str) -> None:
    for d in DIRS:
        os.makedirs(qdir(home, d), exist_ok=True)
    os.makedirs(os.path.join(home, "logs"), exist_ok=True)
    os.makedirs(os.path.join(home, "status"), exist_ok=True)


def expand(s: str) -> str:
    return os.path.expanduser(os.path.expandvars(s))


def load(path: str) -> dict:
    try:
        with open(path) as f:
            job = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise JobError(f"unreadable job file: {e}") from e
    if not isinstance(job, dict):
        raise JobError("job file must hold a JSON object")
    job.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cmd = job.get("cmd")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
        raise JobError("cmd must be a non-empty list of strings")
    job.setdefault("attempts", 0)
    job.setdefault("history", [])
    return job


def save(job: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(job, f, indent=1)
    os.replace(tmp, path)


def pending(home: str) -> list[str]:
    d = qdir(home, "pending")
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json"))


def running(home: str) -> list[str]:
    d = qdir(home, "running")
    return sorted(os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json"))


def move(path: str, home: str, which: str) -> str:
    dst = os.path.join(qdir(home, which), os.path.basename(path))
    os.replace(path, dst)
    return dst


def resolved(job: dict) -> dict:
    """cmd/cwd/prereg/progress_log with ~ and $VARS expanded, relative paths joined to cwd."""
    cwd = expand(job.get("cwd", "~"))
    out = {"cmd": [expand(c) for c in job["cmd"]], "cwd": cwd}
    for k in ("prereg", "progress_log"):
        v = job.get(k)
        out[k] = None if not v else os.path.join(cwd, expand(v))
    return out


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)


def check_prereg(job: dict, require: bool = True) -> str:
    """-> a note for the record. Raises JobError when the job may not start."""
    r = resolved(job)
    if not os.path.isdir(r["cwd"]):
        raise JobError(f"cwd does not exist: {r['cwd']}")
    if r["prereg"] is None:
        if job.get("no_prereg_reason"):
            return f"no prereg: {job['no_prereg_reason']}"
        if require:
            raise JobError("no prereg and no no_prereg_reason")
        return "no prereg (runner not requiring one)"
    p = r["prereg"]
    if not os.path.isfile(p) or os.path.getsize(p) == 0:
        raise JobError(f"prereg missing or empty: {p}")
    d, name = os.path.dirname(p), os.path.basename(p)
    try:
        tracked = _git(["ls-files", "--error-unmatch", name], d)
        clean = _git(["diff", "--quiet", "HEAD", "--", name], d)
        head = _git(["rev-parse", "HEAD"], d)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise JobError(f"git unavailable for the prereg check: {e}") from e
    if tracked.returncode != 0:
        raise JobError(f"prereg not committed (untracked): {p}")
    if clean.returncode != 0:
        raise JobError(f"prereg has uncommitted changes: {p}")
    return f"prereg committed at {head.stdout.strip()[:12]}"


def append_jsonl(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
