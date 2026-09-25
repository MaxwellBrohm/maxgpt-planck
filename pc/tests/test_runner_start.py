"""queue_runner: what may start (prereg, disk, GPU), reboot recovery, attempts, the lock."""
import fcntl
import os
import signal
import subprocess

import pytest

import jobq
import queue_runner
from test_runner import add_job, make_runner, records, where

GIT_ENV = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}  # Mac CLT git


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   env=GIT_ENV, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "committed.yaml").write_text("id: X\n")
    (r / "dirty.yaml").write_text("id: Y\n")
    git(r, "add", "committed.yaml", "dirty.yaml")
    git(r, "commit", "-q", "-m", "prereg")
    (r / "dirty.yaml").write_text("id: Y changed\n")
    (r / "untracked.yaml").write_text("id: Z\n")
    return r


def prereg_job(home, repo, name, prereg):
    out = add_job(home, name, "--steps", "1", cwd=str(repo), prereg=prereg)
    p = os.path.join(jobq.qdir(str(home), "pending"), name + ".json")
    job = jobq.load(p)
    job.pop("no_prereg_reason")
    jobq.save(job, p)
    return out


def test_prereg_rules(tmp_path, smi, repo):
    home = tmp_path / "home"
    prereg_job(home, repo, "a_ok", "committed.yaml")
    prereg_job(home, repo, "b_dirty", "dirty.yaml")
    prereg_job(home, repo, "c_untracked", "untracked.yaml")
    prereg_job(home, repo, "d_missing", "nope.yaml")
    add_job(home, "e_none", "--steps", "1")
    p = os.path.join(jobq.qdir(str(home), "pending"), "e_none.json")
    job = jobq.load(p)
    job.pop("no_prereg_reason")
    jobq.save(job, p)
    make_runner(home).loop()
    assert where(home, "a_ok")[0] == "done"
    why = {r["name"]: r.get("reason", "") for r in records(home) if r["outcome"] == "failed"}
    assert set(why) == {"b_dirty", "c_untracked", "d_missing", "e_none"}
    assert "uncommitted" in why["b_dirty"] and "untracked" in why["c_untracked"]
    assert "missing" in why["d_missing"] and "no prereg" in why["e_none"]


def test_malformed_job_file_fails_without_stopping_the_queue(tmp_path, smi):
    jobq.ensure_layout(str(tmp_path))
    open(os.path.join(jobq.qdir(str(tmp_path), "pending"), "010_bad.json"), "w").write("{nope")
    add_job(tmp_path, "020_good", "--steps", "1")
    make_runner(tmp_path).loop()
    bad = os.path.join(jobq.qdir(str(tmp_path), "failed"), "010_bad.json")
    assert open(bad).read() == "{nope"                    # kept as written, for a person to read
    assert "unreadable" in records(tmp_path)[0]["reason"]
    assert where(tmp_path, "020_good")[0] == "done"


def test_start_waits_for_disk_and_gpu(tmp_path, smi):
    add_job(tmp_path, "w", "--steps", "1")
    assert make_runner(tmp_path, min_start_gb=1e9).loop() == 0      # not enough disk
    smi.set(mem_used_mib=5000)
    assert make_runner(tmp_path).loop() == 0                          # someone else on the GPU
    smi.set(mem_used_mib=600, temp_c=80)
    assert make_runner(tmp_path).loop() == 0                          # still hot
    smi.set(temp_na=True, temp_c=40)
    assert make_runner(tmp_path).loop() == 0                          # cannot see temperature
    assert where(tmp_path, "w")[0] == "pending" and records(tmp_path) == []
    smi.set(temp_na=False)
    assert make_runner(tmp_path).loop() == 1


def test_low_disk_during_a_job_checkpoints_and_requeues(tmp_path, smi, monkeypatch):
    calls = []

    def free_gb(_path):                        # 100 GB at the start check, then 10 GB
        calls.append(1)
        return 100.0 if len(calls) == 1 else 10.0
    monkeypatch.setattr(queue_runner.sysinfo, "free_gb", free_gb)
    out = add_job(tmp_path, "d", "--steps", "400")
    make_runner(tmp_path, min_start_gb=25, stop_gb=15).loop()
    assert [r["trip"] for r in records(tmp_path)] == ["disk"]    # and no restart at 10 GB
    assert os.path.exists(os.path.join(out, "checkpointed"))
    assert where(tmp_path, "d")[0] == "pending"
    assert "disk" in open(tmp_path / "status" / "runner.json").read()


def test_start_needs_more_than_the_stop_threshold(tmp_path, smi, monkeypatch):
    monkeypatch.setattr(queue_runner.sysinfo, "free_gb", lambda _p: 20.0)
    add_job(tmp_path, "s", "--steps", "1")
    assert make_runner(tmp_path, min_start_gb=0, stop_gb=30).loop() == 0   # would stop at once


def test_host_memory_floor_fails_the_job(tmp_path, smi, monkeypatch):
    monkeypatch.setattr(queue_runner.sysinfo, "mem_available_mib", lambda: 100.0)
    add_job(tmp_path, "ram", "--steps", "400")
    make_runner(tmp_path).loop()
    assert records(tmp_path)[-1]["trip"] == "host_mem"
    assert where(tmp_path, "ram")[0] == "failed"


def test_reboot_recovery_requeues_then_gives_up(tmp_path, smi):
    add_job(tmp_path, "r", "--steps", "1", max_attempts=2)
    pend = os.path.join(jobq.qdir(str(tmp_path), "pending"), "r.json")
    job = jobq.load(pend)
    job["attempts"] = 1
    jobq.save(job, pend)
    jobq.move(pend, str(tmp_path), "running")                     # as if the PC rebooted
    r = make_runner(tmp_path)
    r.recover()
    d, job = where(tmp_path, "r")
    assert d == "pending" and job["history"][-1]["trip"] == "interrupted"
    job["attempts"] = 2
    jobq.save(job, pend)
    jobq.move(pend, str(tmp_path), "running")
    r.recover()
    assert where(tmp_path, "r")[0] == "failed"


def test_requeue_stops_at_max_attempts(tmp_path, smi):
    add_job(tmp_path, "h", "--steps", "400", "--smi-at", "1", "--smi", '{"temp_c": 95}',
            max_attempts=1)
    make_runner(tmp_path).loop()
    assert records(tmp_path)[-1]["trip"] == "temp"
    assert where(tmp_path, "h")[0] == "failed"


def test_second_runner_refuses_while_the_lock_is_held(tmp_path, smi):
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    home = str(tmp_path)
    os.makedirs(home, exist_ok=True)
    f = open(os.path.join(home, "runner.lock"), "w")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert queue_runner.main(["--home", home, "--exit-when-idle", "--heartbeat-every", "0"]) == 3
    fcntl.flock(f, fcntl.LOCK_UN)
    assert queue_runner.main(["--home", home, "--exit-when-idle", "--heartbeat-every", "0",
                              "--idle", "0.05"]) == 0
    for s, h in saved.items():
        signal.signal(s, h)


def test_runner_sigterm_checkpoints_the_job_and_exits_clean(tmp_path, smi):
    import sys
    import time
    out = add_job(tmp_path, "long", "--steps", "400")
    runner = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "wsl", "queue_runner.py")
    p = subprocess.Popen([sys.executable, runner, "--home", str(tmp_path), "--poll", "0.05",
                          "--idle", "0.05", "--heartbeat-every", "0", "--min-start-gb", "0",
                          "--stop-gb", "0"])
    for _ in range(200):
        if os.path.exists(os.path.join(out, "started")):
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)                   # what WSL or a person sends the runner
    assert p.wait(timeout=30) == 0
    assert records(tmp_path)[-1]["trip"] == "shutdown"
    assert os.path.exists(os.path.join(out, "checkpointed"))
    assert where(tmp_path, "long")[0] == "pending"
