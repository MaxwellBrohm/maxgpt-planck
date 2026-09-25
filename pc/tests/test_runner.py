"""queue_runner: job lifecycle and every guard trip, with the fake GPU and a fake job."""
import json
import os
import sys
import time

import pytest

import jobq
from gpuguard import GuardConfig
from queue_runner import Runner

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_JOB = os.path.join(HERE, "fake_job.py")


def make_runner(home, **kw):
    g = GuardConfig(temp_strikes=2, mem_strikes=2, smi_fail_strikes=3, host_strikes=2)
    opts = dict(poll=0.05, idle=0.05, grace=3.0, min_start_gb=0, stop_gb=0, guard=g,
                heartbeat_every=0, exit_when_idle=True)
    opts.update(kw)
    return Runner(str(home), **opts)


def add_job(home, name, *args, **fields):
    out = os.path.join(str(home), "out", name)
    job = {"cmd": [sys.executable, FAKE_JOB, out, *args], "cwd": str(home),
           "no_prereg_reason": "test", **fields}
    jobq.ensure_layout(str(home))
    jobq.save(job, os.path.join(jobq.qdir(str(home), "pending"), f"{name}.json"))
    return out


def where(home, name):
    found = [d for d in jobq.DIRS if os.path.exists(os.path.join(jobq.qdir(str(home), d), name + ".json"))]
    assert len(found) == 1, found
    job = jobq.load(os.path.join(jobq.qdir(str(home), found[0]), name + ".json"))
    return found[0], job


def records(home):
    p = os.path.join(str(home), "status", "jobs.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def test_clean_job_goes_to_done(tmp_path, smi):
    out = add_job(tmp_path, "ok", "--steps", "5")
    assert make_runner(tmp_path).loop() == 1
    d, job = where(tmp_path, "ok")
    assert d == "done" and job["attempts"] == 1
    assert os.path.exists(os.path.join(out, "finished"))
    rec = records(tmp_path)[-1]
    assert rec["rc"] == 0 and rec["trip"] is None and rec["outcome"] == "done"
    log = open(tmp_path / "logs" / "ok.log").read()
    assert "fake job started" in log and "runner: end rc=0" in log


def test_jobs_run_in_filename_order(tmp_path, smi):
    add_job(tmp_path, "020_b", "--steps", "1")
    add_job(tmp_path, "010_a", "--steps", "1")
    make_runner(tmp_path).loop()
    assert [r["name"] for r in records(tmp_path)] == ["010_a", "020_b"]


def test_nonzero_exit_goes_to_failed(tmp_path, smi):
    add_job(tmp_path, "bad", "--steps", "1", "--rc", "3")
    make_runner(tmp_path).loop()
    assert where(tmp_path, "bad")[0] == "failed"
    assert records(tmp_path)[-1]["rc"] == 3


def test_over_temperature_stops_checkpoints_requeues_and_waits(tmp_path, smi):
    out = add_job(tmp_path, "hot", "--steps", "400", "--smi-at", "3", "--smi", '{"temp_c": 95}')
    t0 = time.time()
    make_runner(tmp_path).loop()
    assert time.time() - t0 < 10                     # stopped long before its 20 s of steps
    assert os.path.exists(os.path.join(out, "checkpointed"))   # SIGTERM reached it
    assert not os.path.exists(os.path.join(out, "finished"))
    d, job = where(tmp_path, "hot")
    assert d == "pending" and job["attempts"] == 1   # back in the queue to resume
    rec = records(tmp_path)[-1]
    assert rec["trip"] == "temp" and rec["peak_temp_c"] == 95
    state = json.load(open(tmp_path / "status" / "runner.json"))
    assert state["state"] == "waiting" and "95" in state["why"]   # cool-down before restart


def test_cool_gpu_lets_the_requeued_job_resume(tmp_path, smi):
    add_job(tmp_path, "hot", "--steps", "400", "--smi-at", "2", "--smi", '{"temp_c": 95}')
    make_runner(tmp_path).loop()
    smi.set(temp_c=60)
    job_path = os.path.join(jobq.qdir(str(tmp_path), "pending"), "hot.json")
    job = jobq.load(job_path)
    job["cmd"] = job["cmd"][:3] + ["--steps", "2"]          # the resumed run is short
    jobq.save(job, job_path)
    make_runner(tmp_path).loop()
    d, job = where(tmp_path, "hot")
    assert d == "done" and job["attempts"] == 2
    assert [h["trip"] for h in job["history"]] == ["temp", None]


def test_runaway_gpu_memory_fails_the_job(tmp_path, smi):
    out = add_job(tmp_path, "leak", "--steps", "400", "--smi-at", "2",
                  "--smi", '{"mem_used_mib": 12100}')
    make_runner(tmp_path).loop()
    assert where(tmp_path, "leak")[0] == "failed"
    assert records(tmp_path)[-1]["trip"] == "gpu_mem"
    assert os.path.exists(os.path.join(out, "checkpointed"))


def test_per_job_guard_override(tmp_path, smi):
    add_job(tmp_path, "tight", "--steps", "400", "--smi-at", "2",
            "--smi", '{"mem_used_mib": 5000}', guard={"max_mem_mib": 4000})
    make_runner(tmp_path).loop()
    assert records(tmp_path)[-1]["trip"] == "gpu_mem"


def test_sigkill_after_grace_and_whole_process_group(tmp_path, smi):
    out = add_job(tmp_path, "stubborn", "--steps", "400", "--ignore-term", "--child",
                  "--smi-at", "3", "--smi", '{"temp_c": 99}')
    t0 = time.time()
    make_runner(tmp_path, grace=0.5).loop()
    assert time.time() - t0 < 10
    rec = records(tmp_path)[-1]
    assert rec["trip"] == "temp" and rec["rc"] == -9            # it had to be SIGKILLed
    child = int(open(os.path.join(out, "child.pid")).read())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)                                        # the child died with it


def test_blind_nvidia_smi_stops_the_job(tmp_path, smi):
    add_job(tmp_path, "blind", "--steps", "400", "--smi-at", "2", "--smi", '{"fail": true}')
    make_runner(tmp_path).loop()
    assert records(tmp_path)[-1]["trip"] == "smi"
    assert where(tmp_path, "blind")[0] == "pending"


def test_timeout_fails(tmp_path, smi):
    add_job(tmp_path, "slow", "--steps", "400", max_hours=0.5 / 3600)
    make_runner(tmp_path).loop()
    assert records(tmp_path)[-1]["trip"] == "timeout"
    assert where(tmp_path, "slow")[0] == "failed"


def test_stop_file_checkpoints_and_requeues(tmp_path, smi):
    out = add_job(tmp_path, "stopme", "--steps", "400")
    r = make_runner(tmp_path)
    orig = r.state

    def state(**kw):                                  # drop STOP once the job is running
        orig(**kw)
        if kw.get("state") == "running" and os.path.exists(os.path.join(out, "started")):
            (tmp_path / "STOP").write_text("")
    r.state = state
    r.loop()
    assert records(tmp_path)[-1]["trip"] == "stop_file"
    assert os.path.exists(os.path.join(out, "checkpointed"))
    assert where(tmp_path, "stopme")[0] == "pending"
    assert json.load(open(tmp_path / "status" / "runner.json"))["state"] == "paused"


def test_pause_starts_nothing(tmp_path, smi):
    add_job(tmp_path, "p", "--steps", "1")
    (tmp_path / "PAUSE").write_text("")
    assert make_runner(tmp_path).loop() == 0
    assert where(tmp_path, "p")[0] == "pending"
