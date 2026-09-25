"""Heartbeat: one status line for the PC, written locally and optionally pushed (stdlib only).

  python3 heartbeat.py [--home ~/planck] [--push] [--print]

The queue runner starts this every 30 minutes (queue_runner.py --heartbeat-every). It
writes status/heartbeat.json (latest record) and appends one line to status/heartbeat.log:
time, host, runner state and job, step, loss, tok/s and ETA (read from the job's
progress_log, the harness log.jsonl), GPU temperature, memory, utilization and throttle
reasons, free disk and WSL memory.

--push copies both into a LOCAL CLONE of the private status repo, then commits and pushes.
Clone path: $PLANCK_STATUS_REPO, default ~/planck/status-repo. This script never creates,
clones or authenticates a repo: that is a one-time manual step (RUNBOOK step 10) with a
repo-scoped deploy key in ~/.ssh. No token or key belongs in this file, the job files or
the status repo. Files written in the clone: pc/heartbeat.json and pc/heartbeat.log
(last 500 lines). git runs with prompts disabled, so a missing key fails fast.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpuguard  # noqa: E402
import jobq  # noqa: E402
import sysinfo  # noqa: E402

KEEP_LINES = 500


def last_jsonl(path: str | None) -> dict | None:
    """Last complete JSON line of a file (reads only the tail), or None."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 65536))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            return rec
    return None


def progress(state: dict) -> dict:
    rec = last_jsonl(state.get("progress_log")) or {}
    out = {k: rec.get(k) for k in ("step", "loss", "tok_per_s", "tokens", "phase")}
    total, tps, done = state.get("total_tokens"), rec.get("tok_per_s"), rec.get("tokens")
    if total and tps and done is not None and tps > 0:
        out["eta_h"] = round(max(0.0, total - done) / tps / 3600, 2)
    return out


def build(home: str) -> dict:
    try:
        with open(os.path.join(home, "status", "runner.json")) as f:
            state = json.load(f)
        age = time.time() - os.path.getmtime(os.path.join(home, "status", "runner.json"))
    except (OSError, json.JSONDecodeError):
        state, age = {"state": "unknown (no runner.json)"}, None
    q = {d: len(os.listdir(jobq.qdir(home, d))) if os.path.isdir(jobq.qdir(home, d)) else 0
         for d in jobq.DIRS}
    mem = sysinfo.mem_available_mib()
    return {"time": sysinfo.now_iso(), "host": sysinfo.host(), "runner": state.get("state"),
            "runner_age_s": None if age is None else round(age), "job": state.get("job"),
            "why": state.get("why"), "attempt": state.get("attempt"),
            "progress": progress(state), "gpu": gpuguard.query(with_throttle=True),
            "disk_free_gb": round(sysinfo.free_gb(home), 1),
            "mem_avail_gb": None if mem is None else round(mem / 1024, 1), "queue": q}


def _f(v, fmt: str, none: str = "-") -> str:
    return none if v is None else format(v, fmt)


def line(rec: dict) -> str:
    p, g = rec.get("progress") or {}, rec.get("gpu")
    parts = [rec["time"], rec["host"], str(rec.get("runner"))]
    if rec.get("job"):
        parts.append(f"job {rec['job']}#{rec.get('attempt')}")
    if rec.get("why"):
        parts.append(f"({rec['why']})")
    if p.get("step") is not None:
        parts.append(f"step {p['step']} loss {_f(p.get('loss'), '.4f')} "
                     f"tok/s {_f(p.get('tok_per_s'), ',.0f')} ETA {_f(p.get('eta_h'), '.1f')}h")
    if g is None:
        parts.append("| gpu: nvidia-smi FAILED")
    else:
        thr = ",".join(g.get("throttle_reasons") or []) or "none"
        parts.append(f"| gpu {_f(g.get('temp_c'), '.0f')}C "
                     f"{_f(g.get('mem_used_mib'), '.0f')}/{_f(g.get('mem_total_mib'), '.0f')}MiB "
                     f"{_f(g.get('util_pct'), '.0f')}% {_f(g.get('power_w'), '.0f')}W thr[{thr}]")
    q = rec.get("queue") or {}
    parts.append(f"| disk {rec['disk_free_gb']}GB mem {_f(rec.get('mem_avail_gb'), '')}GB "
                 f"| q {q.get('pending', 0)}p {q.get('running', 0)}r "
                 f"{q.get('done', 0)}d {q.get('failed', 0)}f")
    return " ".join(parts)


def _append_capped(path: str, text: str, keep: int) -> None:
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
    lines = (lines + [text])[-keep:]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_local(home: str, rec: dict, text: str) -> None:
    st = os.path.join(home, "status")
    os.makedirs(st, exist_ok=True)
    jobq.write_json(os.path.join(st, "heartbeat.json"), rec)
    _append_capped(os.path.join(st, "heartbeat.log"), text, 5000)


def _git(repo: str, *args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=20")
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          timeout=timeout, env=env)


def push(rec: dict, text: str, repo: str) -> str:
    """-> 'pushed' | 'no change' | a short failure note. Never raises."""
    if not os.path.isdir(os.path.join(repo, ".git")):
        return f"push skipped: no git clone at {repo}"
    try:
        r = _git(repo, "pull", "--rebase", "--autostash", "-q")
        if r.returncode != 0:
            return f"push failed at pull: {r.stderr.strip()[:200]}"
        d = os.path.join(repo, "pc")
        os.makedirs(d, exist_ok=True)
        jobq.write_json(os.path.join(d, "heartbeat.json"), rec)
        _append_capped(os.path.join(d, "heartbeat.log"), text, KEEP_LINES)
        _git(repo, "add", "pc/heartbeat.json", "pc/heartbeat.log")
        if _git(repo, "diff", "--cached", "--quiet").returncode == 0:
            return "no change"
        r = _git(repo, "commit", "-q", "-m", f"heartbeat pc {rec['time']}")
        if r.returncode != 0:
            return f"push failed at commit: {r.stderr.strip()[:200]}"
        r = _git(repo, "push", "-q")
        return "pushed" if r.returncode == 0 else f"push failed: {r.stderr.strip()[:200]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"push failed: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Planck PC heartbeat")
    ap.add_argument("--home", default=os.environ.get("PLANCK_HOME", "~/planck"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("PLANCK_STATUS_REPO", ""))
    ap.add_argument("--print", action="store_true", help="also print the line")
    a = ap.parse_args(argv)
    home = os.path.abspath(jobq.expand(a.home))
    rec = build(home)
    text = line(rec)
    if a.push:
        rec["push"] = push(rec, text, jobq.expand(a.repo or os.path.join(home, "status-repo")))
    write_local(home, rec, text)
    if a.print:
        print(text + (f"  [{rec['push']}]" if a.push else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
