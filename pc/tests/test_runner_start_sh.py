"""runner_start.sh (the boot task's entry): reads runner.env, logs, exits on rc 0 and rc 3."""
import fcntl
import os
import subprocess

from test_runner import add_job, records

PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PC, "wsl", "runner_start.sh")


def setup_home(tmp_path):
    home = tmp_path / "planck"
    home.mkdir()
    os.symlink(os.path.dirname(PC), home / "repo")
    (home / "runner.env").write_text(
        'RUNNER_ARGS="--exit-when-idle --heartbeat-every 0 --idle 0.05 --poll 0.05 '
        '--min-start-gb 0 --stop-gb 0"\n')
    return home


def run(home):
    env = {**os.environ, "PLANCK_HOME": str(home)}
    return subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)


def test_runs_the_queue_then_exits_clean(tmp_path, smi):
    home = setup_home(tmp_path)
    add_job(home, "010_x", "--steps", "2")
    r = run(home)
    assert r.returncode == 0, r.stderr
    log = (home / "logs" / "runner.log").read_text()
    assert "runner_start: boot" in log and "runner exited rc=0" in log
    assert [x["outcome"] for x in records(home)] == ["done"]


def test_exits_3_when_another_runner_holds_the_lock(tmp_path, smi):
    home = setup_home(tmp_path)
    f = open(home / "runner.lock", "w")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    r = run(home)
    assert r.returncode == 3
    assert "another runner holds the lock" in (home / "logs" / "runner.log").read_text()
