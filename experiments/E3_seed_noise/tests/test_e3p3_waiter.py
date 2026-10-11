"""e3p3_waiter.sh (notes.txt PART 3 REOPENED, QUEUE), run whole on a fixture tree: a fixture repo as planck-kit/
planck_e3p3.bundle whose queue_e3.sh is a stub that appends the queue log lines of a clean end, a 2 h gpu.lock wait, a
30 min GPU memory wait, a scoring failure, a kill, a refusal before the start line, or a clean end of another plan file
(macOS has no flock or sha256sum: stubs stand in). Also the sha256 pin against plans/part3.txt. No model, nothing on
the PC. E3_TEST_ROOT (mutation_e3.py) points at a scratch copy of the repo.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_e3p3_waiter.py
"""
import os
import re
import subprocess
import time

import pytest

ROOT = os.environ.get("E3_TEST_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
E3 = os.path.join(ROOT, "experiments", "E3_seed_noise")
W = os.path.join(E3, "e3p3_waiter.sh")
PART3 = os.path.join(E3, "plans", "part3.txt")
T = "2026-10-03 11:3"
PRIOR = [f"{T}8:54 MARK E3 PART 2 DONE", f"{T}8:54 plan finished: part2.txt"]
STUB = r"""#!/bin/bash
echo "$*" >> "$E3W_STUBLOG"
Q=$(dirname "$(dirname "$(dirname "$0")")")
LOG=$E3W_HOME/runs/E3/queue_e3.txt
o=$(head -1 "$E3W_OUTCOMES"); sed -i.bak 1d "$E3W_OUTCOMES"
t() { echo "$(date '+%F %T') $*" >> "$LOG"; }
[ -n "$E3W_SLEEP" ] && sleep "$E3W_SLEEP"
[ "$o" = refuse ] && { t "refusing: uncommitted changes in the code checkout:"; exit 1; }
sha=$(shasum -a 256 "$Q/experiments/E3_seed_noise/plans/part3.txt" | cut -c1-16)
[ "$o" = othersha ] && sha=0123456789abcdef
t "queue start: exp E3, plan part3.txt (sha256 $sha), code commit $(git -C "$Q" rev-parse HEAD)"
t "seen E3: E3 PART 2 DONE"
case $o in
  clean|othersha) t "MARK E3 PART 3 DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_DONE"; t "plan finished: part3.txt"; exit 0 ;;
  nomark) t "MARK E3 PART 3 DONE"; t "plan finished: part3.txt"; exit 0 ;;
  lockwait) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_init2"; exit 1 ;;
  memwait) t "GPU memory in use [8328] MiB > 3000 with the lock held: waiting"; t "queue stops at: train e3_5m_A_s1_P3"; exit 1 ;;
  fail) t "scoring failed for final_00007630 (see score.err)"; t "queue stops at: train e3_5m_A_init2"; exit 1 ;;
  kill) exit 137 ;;
  clean_rc1) t "MARK E3 PART 3 DONE"; touch "$E3W_HOME/runs/E3/marks/E3_PART_3_DONE"; t "plan finished: part3.txt"; exit 1 ;;
  lockwait_rc0) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train e3_5m_A_init2"; exit 0 ;;
esac
"""


def git(cwd, *a):
    env = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c", "init.defaultBranch=main", *a],
                          cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def tree(tmp, done=False):
    p = tmp / "planck"
    os.makedirs(p / "runs" / "E3" / "marks")
    os.makedirs(p / "logs")
    (p / "runs" / "E3" / "queue_e3.txt").write_text("\n".join(PRIOR) + "\n")
    for m in ("E3_PART_1_DONE", "E3_PART_2_DONE") + (("E3_PART_3_DONE",) if done else ()):
        (p / "runs" / "E3" / "marks" / m).touch()
    return p


def kit(tmp, current="part3", plan=None, bundle=True):
    """planck-kit: the waiter, and a fixture repo (CURRENT, part3.txt, the stub queue) as planck_e3p3.bundle."""
    k, src = tmp / "kit", tmp / "src"
    x = src / "experiments" / "E3_seed_noise"
    os.makedirs(x / "plans")
    os.makedirs(k)
    (x / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    (x / "plans" / "part3.txt").write_bytes(open(PART3, "rb").read() if plan is None else plan)
    (x / "queue_e3.sh").write_text(STUB)
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    if bundle:
        git(src, "bundle", "create", "-q", str(k / "planck_e3p3.bundle"), "HEAD", "main")
    (k / "e3p3_waiter.sh").write_text(open(W).read())
    return k, git(src, "rev-parse", "HEAD")


def run(tmp, p, k, outcomes=(), maxr="24", flock_rc="0", pat="e3w_test_matches_no_process_[0-9]{12}", poll="0",
        sleep="", timeout=60):
    os.makedirs(tmp / "bin", exist_ok=True)
    (tmp / "bin" / "flock").write_text(f"#!/bin/sh\nexit {flock_rc}\n")
    (tmp / "bin" / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    for f in ("flock", "sha256sum"):
        os.chmod(tmp / "bin" / f, 0o755)
    (tmp / "outcomes").write_text("".join(o + "\n" for o in outcomes))
    env = {**os.environ, "E3W_HOME": str(p), "E3W_KIT": str(k), "E3W_POLL": poll, "E3W_MAXR": maxr, "E3W_PAT": pat,
           "E3W_STUBLOG": str(tmp / "stub.log"), "E3W_OUTCOMES": str(tmp / "outcomes"), "E3W_SLEEP": sleep,
           "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
    rc = subprocess.run(["/bin/bash", str(k / "e3p3_waiter.sh")], env=env, capture_output=True, text=True,
                        timeout=timeout).returncode
    lp = p / "logs" / "e3p3_waiter.log"
    log = lp.read_text() if lp.exists() else ""
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return rc, log, stub


def call(p, ref):
    return f"--plan {p / 'qcode' / ref[:12]}/experiments/E3_seed_noise/plans/part3.txt"


def test_the_pin_is_part3s_sha256_and_the_script_parses():
    import hashlib
    txt = open(W).read()
    pin = re.search(r"^PLAN_SHA=([0-9a-f]{64})$", txt, re.M)[1]
    assert pin == hashlib.sha256(open(PART3, "rb").read()).hexdigest()
    assert "BUNDLE=planck_e3p3.bundle" in txt and "MAXR=${E3W_MAXR:-24}" in txt and "POLL=${E3W_POLL:-300}" in txt
    assert subprocess.run(["/bin/bash", "-n", W]).returncode == 0
    assert not re.search(r"/(?:home|Users)/[a-z]|\b\d{1,3}(?:\.\d{1,3}){3}\b|@[\w-]+\.(?:com|net|org)", txt)


def test_clean_end_is_done_and_a_second_start_launches_nothing(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 0 and stub == [call(p, ref)] and "DONE part3 ended cleanly at its mark" in log, log
    assert (p / "logs" / "e3p3_waiter.started").exists() and "relaunch" not in log.lower().replace("relaunches", "")
    rc, log2, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 0 and len(stub) == 1 and "DONE: marks/E3_PART_3_DONE exists" in log2[len(log):]


def test_relaunches_after_lock_and_memory_waits_then_done(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait", "memwait", "clean"])
    assert rc == 0 and stub == [call(p, ref)] * 3, (log, stub)
    assert "launch 2:" in log and "launch 3:" in log and "launch 4:" not in log
    assert log.index("after a 2 h gpu.lock wait") < log.index("launch 2:") < log.index("another process's GPU memory")


def test_gives_up_after_maxr_relaunches(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"] * 4, maxr="2")
    assert rc == 1 and stub == [call(p, ref)] * 3 and "GIVEUP: 2 relaunches after 2 h gpu.lock waits" in log, log


@pytest.mark.parametrize("case", ["fail", "kill", "refuse", "othersha", "nomark", "clean_rc1", "lockwait_rc0"])
def test_no_relaunch_after_an_end_it_cannot_name(tmp_path, case):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, [case, "clean"])
    assert rc == 1 and stub == [call(p, ref)] and "launch 2:" not in log, log
    want = {"fail": "GIVEUP the queue ended another way (exit 1)", "kill": "GIVEUP the queue ended another way (exit 137)",
            "refuse": "GIVEUP no part3 start line at", "othersha": "GIVEUP no part3 start line at",
            "nomark": "GIVEUP a clean end but no marks/E3_PART_3_DONE",
            "clean_rc1": "GIVEUP the queue ended another way (exit 1)", "lockwait_rc0": "GIVEUP the queue ended another way (exit 0)"}
    assert want[case] in log, log


def test_a_pgrep_error_gives_up(tmp_path):
    p, (k, _) = tree(tmp_path), kit(tmp_path)
    os.makedirs(tmp_path / "bin")
    (tmp_path / "bin" / "pgrep").write_text("#!/bin/sh\nexit 3\n")       # pgrep's own error, not "none running"
    os.chmod(tmp_path / "bin" / "pgrep", 0o755)
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert (rc, stub) == (1, []) and "GIVEUP: pgrep failed: cannot tell whether an E3 queue runs" in log, log


def test_a_restart_after_a_cut_resumes(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["kill"])               # the queue died with no end line: for a person
    assert rc == 1 and len(stub) == 1
    rc, log, stub = run(tmp_path, p, k, ["clean"])              # the same command again: launches part3, which resumes
    assert rc == 0 and stub == [call(p, ref)] * 2 and log.count("launch 1:") == 2, log


@pytest.mark.parametrize("case", ["cancel", "stop", "done_mark", "other_waiter"])
def test_nothing_is_launched(tmp_path, case):
    p, (k, _) = tree(tmp_path, done=case == "done_mark"), kit(tmp_path)
    if case == "cancel":
        (p / "logs" / "e3p3_waiter.CANCEL").touch()
    if case == "stop":
        (p / "STOP").touch()
    rc, log, stub = run(tmp_path, p, k, ["clean"], flock_rc="1" if case == "other_waiter" else "0")
    want = {"cancel": (1, "e3p3_waiter.CANCEL exists"), "stop": (1, "/STOP exists"),
            "done_mark": (0, "DONE: marks/E3_PART_3_DONE exists"), "other_waiter": (1, "another e3p3_waiter holds")}
    assert (rc, stub) == (want[case][0], []) and want[case][1] in log, log


@pytest.mark.parametrize("case", ["current_names_another_plan", "plan_edited", "no_bundle"])
def test_a_wrong_bundle_launches_nothing(tmp_path, case):
    p = tree(tmp_path)
    k, _ = kit(tmp_path, current="part2" if case == "current_names_another_plan" else "part3",
               plan=b"# edited\ntrain e3_5m_A_init2\n" if case == "plan_edited" else None, bundle=case != "no_bundle")
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    want = {"current_names_another_plan": "names [part2], not part3", "plan_edited": "is missing or not sha256 07018f0f",
            "no_bundle": "GIVEUP: no bundle"}
    assert rc == 1 and stub == [] and want[case] in log, log


def test_waits_while_an_e3_queue_runs(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    sleeper = subprocess.Popen(["/bin/sleep", "2.71828"])
    try:
        t0 = time.time()
        rc, log, stub = run(tmp_path, p, k, ["clean"], pat=r"sleep 2\.71828", poll="1")
    finally:
        sleeper.kill()
    assert rc == 0 and stub == [call(p, ref)] and "WAIT an E3 queue" in log and time.time() - t0 >= 2, log
    assert log.index("WAIT an E3 queue") < log.index("launch 1:") and log.count("WAIT an E3 queue") == 1


def test_reads_only_what_this_queue_wrote(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"], maxr="0")       # a part3 start line is now in the queue log
    assert rc == 1 and "GIVEUP: 0 relaunches" in log
    rc, log, stub = run(tmp_path, p, k, ["refuse"])                   # this queue refused before its own start line
    assert rc == 1 and len(stub) == 2 and "GIVEUP no part3 start line at" in log.splitlines()[-1], log
