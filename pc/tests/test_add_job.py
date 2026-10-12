"""add_job and the shipped example jobs (validated against this checkout, not run)."""
import os

import add_job
import jobq

PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PC)
JOBS = os.path.join(PC, "wsl", "jobs")


def planck_home(tmp_path, monkeypatch):
    home = tmp_path / "planck"
    home.mkdir()
    os.symlink(ROOT, home / "repo")                    # the PC layout: ~/planck/repo
    monkeypatch.setenv("PLANCK_HOME", str(home))
    return str(home)


def test_examples_resolve_to_real_files(tmp_path, monkeypatch):
    home = planck_home(tmp_path, monkeypatch)
    for name in sorted(os.listdir(JOBS)):
        job = jobq.load(os.path.join(JOBS, name))
        r = jobq.resolved(job)
        assert os.path.isdir(r["cwd"]), name
        assert r["cmd"][0].startswith(home) or r["cmd"][0] == "bash", name
        scripts = [c for c in r["cmd"] if c.endswith(".py")]
        for s in scripts:
            assert os.path.isfile(os.path.join(r["cwd"], s)), (name, s)
        if job.get("prereg"):
            assert os.path.isfile(r["prereg"]), name
        else:
            assert job.get("no_prereg_reason"), name


def test_add_job_queues_and_refuses_duplicates(tmp_path, monkeypatch):
    home = planck_home(tmp_path, monkeypatch)
    src = os.path.join(JOBS, "020_bench_micro.json")
    assert add_job.main([src, "--home", home]) == 0
    queued = os.path.join(jobq.qdir(home, "pending"), "020_bench_micro.json")
    assert jobq.load(queued)["cmd"] == jobq.load(src)["cmd"]
    assert add_job.main([src, "--home", home]) == 1
    assert add_job.main([src, "--home", home, "--as", "021_again"]) == 0
    assert os.path.exists(os.path.join(jobq.qdir(home, "pending"), "021_again.json"))


def test_add_job_refuses_uncommitted_prereg(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    home = tmp_path / "planck"
    home.mkdir()
    repo = tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "cfg" / "prereg.yaml").write_text("id: X\n")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    job = tmp_path / "j.json"
    job.write_text('{"cmd": ["true"], "cwd": "%s", "prereg": "cfg/prereg.yaml"}' % repo)
    assert add_job.main([str(job), "--home", str(home), "--dry-run"]) == 1
    assert "not committed" in capsys.readouterr().out
