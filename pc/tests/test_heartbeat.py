"""heartbeat: the status line, the ETA, and the optional push (to a throwaway local bare
repo inside pytest's tmp_path; nothing leaves the machine)."""
import json
import os
import subprocess
import time

import pytest

import heartbeat
import jobq
from test_runner import add_job, make_runner

ENV = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}


def git(*args, cwd=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          env=ENV, check=True, capture_output=True, text=True).stdout


def write_state(home, **kw):
    jobq.ensure_layout(str(home))
    jobq.write_json(os.path.join(str(home), "status", "runner.json"), kw)


def test_line_has_progress_eta_gpu_and_disk(tmp_path, smi):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"step": 10, "loss": 3.5, "tok_per_s": 1000.0, "tokens": 1000}) + "\n"
                   + json.dumps({"step": 20, "loss": 3.25, "tok_per_s": 2000.0, "tokens": 2000})
                   + "\n{partial")
    write_state(tmp_path, state="running", job="etok", attempt=2, progress_log=str(log),
                total_tokens=2000 + 2000 * 3600 * 5)
    smi.set(temp_c=71, mem_used_mib=9800, util_pct=98, throttle="0x0000000000000004")
    rec = heartbeat.build(str(tmp_path))
    assert rec["progress"]["step"] == 20 and rec["progress"]["eta_h"] == 5.0
    text = heartbeat.line(rec)
    for s in ("running", "job etok#2", "step 20", "loss 3.2500", "tok/s 2,000", "ETA 5.0h",
              "gpu 71C", "9800/12227MiB", "thr[sw_power_cap]", "disk "):
        assert s in text, (s, text)


def test_line_when_nvidia_smi_fails_and_no_runner(tmp_path, smi):
    smi.set(fail=True)
    text = heartbeat.line(heartbeat.build(str(tmp_path)))
    assert "nvidia-smi FAILED" in text and "unknown" in text


def test_main_writes_local_files(tmp_path, smi):
    write_state(tmp_path, state="idle")
    assert heartbeat.main(["--home", str(tmp_path)]) == 0
    assert heartbeat.main(["--home", str(tmp_path)]) == 0
    assert json.load(open(tmp_path / "status" / "heartbeat.json"))["runner"] == "idle"
    assert len(open(tmp_path / "status" / "heartbeat.log").read().splitlines()) == 2


@pytest.fixture
def status_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    bare = tmp_path / "remote.git"
    git("init", "-q", "--bare", str(bare))
    seed = tmp_path / "seed"
    git("clone", "-q", str(bare), str(seed))
    (seed / "README").write_text("status\n")
    git("add", "README", cwd=seed)
    git("commit", "-q", "-m", "init", cwd=seed)
    git("push", "-q", "origin", "HEAD", cwd=seed)
    clone = tmp_path / "home" / "status-repo"
    git("clone", "-q", str(bare), str(clone))
    return bare, clone


def test_push_commits_to_the_status_repo(tmp_path, smi, status_repo):
    bare, clone = status_repo
    home = tmp_path / "home"
    write_state(home, state="idle")
    heartbeat.main(["--home", str(home), "--push"])
    rec = json.load(open(home / "status" / "heartbeat.json"))
    assert rec["push"] == "pushed"
    assert "heartbeat pc" in git("--git-dir", str(bare), "log", "-1", "--format=%s")
    shown = git("--git-dir", str(bare), "show", "HEAD:pc/heartbeat.log")
    assert "idle" in shown and len(shown.splitlines()) == 1
    heartbeat.main(["--home", str(home), "--push"])
    assert len(git("--git-dir", str(bare), "show", "HEAD:pc/heartbeat.log").splitlines()) == 2


def test_push_without_clone_or_remote_is_reported_not_raised(tmp_path, smi, status_repo):
    home = tmp_path / "home"
    write_state(home, state="idle")
    assert "no git clone" in heartbeat.push({"time": "t"}, "x", str(tmp_path / "nowhere"))
    _, clone = status_repo
    git("remote", "set-url", "origin", str(tmp_path / "gone.git"), cwd=clone)
    assert heartbeat.push({"time": "t"}, "x", str(clone)).startswith("push failed")


def test_runner_starts_heartbeats(tmp_path, smi):
    add_job(tmp_path, "hb", "--steps", "10")
    make_runner(tmp_path, heartbeat_every=0.01).loop()
    p = tmp_path / "status" / "heartbeat.log"
    for _ in range(100):
        if p.exists():
            break
        time.sleep(0.05)
    assert p.exists()
