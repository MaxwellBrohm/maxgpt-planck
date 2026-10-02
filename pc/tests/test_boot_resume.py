"""boot.d (RUNBOOK "RESUME AFTER A REBOOT"): each queue script resumes its queue only when the queue's own log shows
it was cut off, nothing of it runs and no stop file applies; boot_resume.sh runs them once per boot. The real queues
are replaced by fakes under tmp_path whose command lines match the scripts' process patterns. The scripts call pgrep
from PATH; a wrapper keeps only this test's processes, so another session's shell whose text names a queue script (for
example a monitoring loop) cannot make a test pass or fail."""
import os
import shutil
import subprocess
import time

import pytest

from test_runner_start_sh import run as run_start, setup_home

PC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOTD = os.path.join(PC, "wsl", "boot.d")
REF = "7e7c802c63e0dfec04eb154079f946330a60381f"
E2_START = f"2026-09-27 05:00:00 queue start: exp E2, plan stageA.txt (sha256 0123456789abcdef), code commit {REF}"


def put(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def fake(path, tag, then=""):   # a fake queue: records its start, then stays alive 4 s (so pgrep can see it)
    put(path, f'#!/bin/bash\necho "{tag} $* Q_CODE=${{Q_CODE:-}}" >> "$PLANCK_HOME/started"\n{then}\nsleep 4\n', 0o755)


@pytest.fixture
def box(tmp_path):
    put(str(tmp_path / "bin" / "pgrep"), f'#!/bin/bash\nfor p in $({shutil.which("pgrep")} "$@"); do\n'
        f'  ps -o args= -p "$p" 2>/dev/null | grep -qF "{tmp_path}/" && echo "$p"\ndone\nexit 0\n', 0o755)
    home = tmp_path / "planck"
    b = {"t": tmp_path, "home": home, "kit": tmp_path / "kit", "e006": tmp_path / "e006root",
         "el": home / "e006_run" / "logs", "rc12": tmp_path / "rc12code"}
    fake(str(b["t"] / "fq" / "queue_e2.sh"), "e2queue")
    fake(str(b["kit"] / "launch_e2.sh"), "launch", f'exec bash {b["t"]}/fq/queue_e2.sh --exp E2 --from-launch')
    x = home / "qcode" / REF[:12] / "experiments" / "E2_lr_transfer"
    fake(str(x / "queue_e2.sh"), "fallback")
    put(str(x / "plans" / "stageA.txt"), "train a\n")
    code = b["e006"] / "experiments" / "E006_three_objects" / "code"
    fake(str(code / "queue_e006.sh"), "e006queue")
    fake(str(code / "start_queue_e006.sh"), "e006wrap", f"exec bash {code}/queue_e006.sh")
    fake(str(b["rc12"] / "queue_dev_baselines.sh"), "rc12")
    shutil.copytree(BOOTD, str(home / "boot.d"))
    b["env"] = {**os.environ, "PLANCK_HOME": str(home), "PLANCK_KIT": str(b["kit"]), "E006_ROOT": str(b["e006"]),
                "E006_LOGS": str(b["el"]), "RC12_CODE": str(b["rc12"]), "BOOT_VERIFY_S": "6",
                "PATH": f"{tmp_path}/bin:{os.environ['PATH']}"}
    yield b
    subprocess.run(["pkill", "-f", str(tmp_path)], capture_output=True)
    time.sleep(0.3)


def cut_off(b):   # all three queues as a power cut leaves them
    put(str(b["home"] / "runs" / "E2" / "queue_e2.txt"), f"{E2_START}\n2026-09-27 05:30:39 waiting for gpu.lock\n")
    put(str(b["el"] / "queue.txt"), "2026-09-27 00:00:00 QUEUE E006 START (run x)\n2026-09-27 05:25:18 chat_G1 exit=0\n")
    put(str(b["el"] / "G1.guard.log"), "samples\n")
    put(str(b["el"] / "G1.guard.json"), "{}\n")
    put(str(b["home"] / "logs" / "e006_queue.out"), "start_queue_e006 2026-09-26 14:55:59\n")
    put(str(b["home"] / "logs" / "rc12_dev_queue.log"),
        "2026-09-26T14:47:00 queue: start, code /x/rc12_q9, root /r, pid 1\n2026-09-27T05:34:39 queue: waiting\n")


def script(b, name, *args):
    r = subprocess.run(["bash", str(b["home"] / "boot.d" / name), *args], env=b["env"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode in (0, 1), r.stderr
    return r.stdout


def started(b):
    p = b["home"] / "started"
    return p.read_text() if p.exists() else ""


@pytest.mark.parametrize("name", ["10_e2.sh", "20_e006.sh", "30_rc12.sh"])
def test_cut_off_queue_resumes_and_dry_run_starts_nothing(box, name):
    cut_off(box)
    q006 = (box["el"] / "queue.txt").read_text()
    out = script(box, name, "--dry-run")
    assert "DRY RUN, would start" in out and started(box) == ""
    assert (box["el"] / "queue.txt").read_text() == q006          # a dry run writes nothing
    out = script(box, name)
    assert "running after" in out, out
    want = {"10_e2.sh": "launch", "20_e006.sh": "e006wrap", "30_rc12.sh": f"rc12  Q_CODE={box['rc12']}"}[name]
    assert started(box).startswith(want), started(box)


def test_e006_resume_adds_one_note_to_its_queue_log(box):
    cut_off(box)
    script(box, "20_e006.sh")
    lines = (box["el"] / "queue.txt").read_text().splitlines()
    assert len(lines) == 3 and "boot resume:" in lines[2]


@pytest.mark.parametrize("end", ["plan finished: stageA.txt", "plan says stop", "queue stops at: train a",
                                 "STOP while waiting for E3: x", "refusing: uncommitted changes", "bad plan line: x"])
def test_e2_run_that_ended_itself_is_left_alone(box, end):
    cut_off(box)
    put(str(box["home"] / "runs" / "E2" / "queue_e2.txt"), f"{E2_START}\n2026-09-27 05:31:00 {end}\n")
    assert "ended by itself" in script(box, "10_e2.sh") and started(box) == ""


def test_e2_only_the_last_start_counts(box):
    cut_off(box)
    put(str(box["home"] / "runs" / "E2" / "queue_e2.txt"),
        f"{E2_START}\n2026-09-27 05:10:00 plan finished: stageA.txt\n{E2_START}\n2026-09-27 05:20:00 train a\n")
    assert "cut off" in script(box, "10_e2.sh") and started(box).startswith("launch")


def test_e2_log_with_a_nul_tail_still_resumes(box):
    # 2026-09-29: a power cut left NUL bytes at the end of queue_e2.txt; grep then called the file binary, printed
    # no line, and boot resume said "no queue start line" for a queue that had been cut off mid-run. Only GNU grep
    # (the PC) treats the file as binary: on the Mac (BSD grep) this test passes on the old code too.
    cut_off(box)
    q = box["home"] / "runs" / "E2" / "queue_e2.txt"
    put(str(q), f"{E2_START}\n2026-09-27 05:20:00 train a\n")
    with open(q, "ab") as f:
        f.write(b"\0" * 300)
    assert "cut off" in script(box, "10_e2.sh") and started(box).startswith("launch")


def test_e2_falls_back_when_launch_e2_never_reaches_the_queue(box):
    cut_off(box)
    put(str(box["kit"] / "launch_e2.sh"),
        'echo "no bundle in planck-kit" >> "$PLANCK_HOME/logs/launch_e2.log"; exit 1\n', 0o755)
    out = script(box, "10_e2.sh")
    x = box["home"] / "qcode" / REF[:12]
    assert "trying the fallback" in out
    assert started(box).startswith(f"fallback --exp E2 --code {x} --plan {x}/experiments/E2_lr_transfer/plans/stageA.txt")


def test_e2_no_fallback_when_the_queue_started_and_exited(box):
    cut_off(box)
    put(str(box["kit"] / "launch_e2.sh"),
        f'echo "code {REF} in qcode/{REF[:12]}, plan stageA" >> "$PLANCK_HOME/logs/launch_e2.log"; exit 0\n', 0o755)
    out = script(box, "10_e2.sh")
    assert "reached the queue and the queue exited" in out and started(box) == ""


@pytest.mark.parametrize("stop", ["boot.d/STOP", "boot.d/NAME.STOP", "STOP", "PAUSE"])
@pytest.mark.parametrize("name", ["10_e2.sh", "20_e006.sh", "30_rc12.sh"])
def test_stop_files_hold_every_queue(box, stop, name):
    cut_off(box)
    put(str(box["home"] / stop.replace("NAME", name[3:-3])), "")
    assert "present: not resuming" in script(box, name) and started(box) == ""


def test_rc12_own_stop_file_holds_it(box):
    cut_off(box)
    put(str(box["home"] / "logs" / "rc12_dev_queue.STOP"), "")
    assert "present: not resuming" in script(box, "30_rc12.sh") and started(box) == ""


@pytest.mark.parametrize("name,argv", [
    ("10_e2.sh", ["fq/queue_e2.sh", "--exp", "E2"]),
    ("20_e006.sh", ["e006root/experiments/E006_three_objects/code/queue_e006.sh"]),
    ("30_rc12.sh", ["rc12code/queue_dev_baselines.sh"])])
def test_a_running_queue_is_never_started_twice(box, name, argv):
    cut_off(box)
    p = subprocess.Popen(["bash", str(box["t"] / argv[0]), *argv[1:]], env=box["env"])
    time.sleep(0.5)
    os.remove(box["home"] / "started")
    try:
        assert "running (pids" in script(box, name) and started(box) == ""
    finally:
        p.kill()


@pytest.mark.parametrize("end", ["QUEUE E006 DONE", "QUEUE STOPPED: gpu_mem verdict in e005w1"])
def test_e006_run_that_ended_itself_is_left_alone(box, end):
    cut_off(box)
    with open(box["el"] / "queue.txt", "a") as f:
        f.write(f"2026-09-27 05:26:00 {end}\n")
    assert "ended by itself" in script(box, "20_e006.sh") and started(box) == ""


def test_e006_wrapper_exit_after_last_start_is_left_alone(box):
    cut_off(box)
    with open(box["home"] / "logs" / "e006_queue.out", "a") as f:
        f.write("queue_e006 exit 1 2026-09-27 05:26:00\n")
    assert "wrapper saw the queue exit" in script(box, "20_e006.sh") and started(box) == ""


def test_e006_job_cut_mid_run_is_left_for_a_person(box):
    cut_off(box)
    put(str(box["el"] / "C2.log"), "partial\n")
    put(str(box["el"] / "C2.guard.log"), "samples\n")
    out = script(box, "20_e006.sh")
    assert "cut mid-run: C2" in out and started(box) == ""


@pytest.mark.parametrize("end", ["DONE", "2026-09-27T05:40:00 queue: end; 208 runs",
                                 "2026-09-27T05:40:00 queue: STOP file, exiting"])
def test_rc12_finished_or_stopped_is_left_alone(box, end):
    cut_off(box)
    if end == "DONE":
        put(str(box["home"] / "logs" / "rc12_dev_queue.DONE"), "208 / 208 done\n")
    else:
        with open(box["home"] / "logs" / "rc12_dev_queue.log", "a") as f:
            f.write(end + "\n")
    out = script(box, "30_rc12.sh")
    assert "nothing to resume" in out and started(box) == ""


def dispatch(b, *args, smi="45", boot_id="boot-1"):
    put(str(b["t"] / "smi.sh"), f"#!/bin/bash\necho {smi}\n", 0o755)
    put(str(b["t"] / "boot_id"), boot_id + "\n")
    env = {**b["env"], "PLANCK_NVIDIA_SMI": str(b["t"] / "smi.sh"), "BOOT_ID_FILE": str(b["t"] / "boot_id"),
           "BOOT_DELAY_S": "0", "BOOT_GPU_WAIT_S": "0"}
    return subprocess.run(["bash", str(b["home"] / "boot.d" / "boot_resume.sh"), *args], env=env,
                          capture_output=True, text=True, timeout=120)


def test_dispatcher_dry_run_asks_every_script_and_starts_nothing(box):
    cut_off(box)
    os.chmod(box["home"] / "boot.d" / "20_e006.sh", 0o644)
    put(str(box["home"] / "boot.d" / "rc12.STOP"), "")
    out = dispatch(box, "--dry-run").stdout
    assert out.count("DRY RUN, would start") == 1 and "e2: DRY RUN" in out
    assert "20_e006.sh is not executable: skipped" in out and "rc12.STOP present" in out
    assert started(box) == ""


def test_dispatcher_resumes_once_per_boot(box):
    cut_off(box)
    r = dispatch(box)
    assert r.returncode == 0, r.stdout + r.stderr
    got = sorted(x.split()[0] for x in started(box).splitlines())
    assert got == sorted(["launch", "e2queue", "e006wrap", "e006queue", "rc12"]), got
    subprocess.run(["pkill", "-f", str(box["t"])], capture_output=True)
    time.sleep(0.5)
    os.remove(box["home"] / "started")
    assert "already ran in this WSL boot" in dispatch(box).stdout and started(box) == ""
    assert "running after" in dispatch(box, boot_id="boot-2").stdout and started(box) != ""


def test_dispatcher_resumes_nothing_while_the_gpu_is_blind(box):
    cut_off(box)
    r = dispatch(box, smi="")
    assert r.returncode == 1 and "reads no GPU temperature" in r.stdout and started(box) == ""


def test_runner_start_launches_boot_resume_only_when_installed(tmp_path, smi):
    home = setup_home(tmp_path)
    assert run_start(home).returncode == 0
    assert not (home / "logs" / "boot_resume.log").exists()
    put(str(home / "boot.d" / "boot_resume.sh"), 'echo "boot_resume ran $PLANCK_HOME"\n')
    assert run_start(home).returncode == 0
    for _ in range(50):
        log = home / "logs" / "boot_resume.log"
        if log.exists() and "boot_resume ran" in log.read_text():
            break
        time.sleep(0.1)
    assert (home / "logs" / "boot_resume.log").read_text() == f"boot_resume ran {home}\n"
