"""replay_waiter.sh (SCREENS.txt AMENDMENT BASE-REPLAY, LAUNCH; INTEGRATION 2026-10-10): its arguments against the
entry, the plans and what was read on the PC; its sha256 pin of chain_waiter.sh; and the whole file run on a fixture
tree with a fixture bundle whose queue_screens.sh is a stub that writes the queue log lines of a clean end, a 2 h
gpu.lock wait, a scoring failure or a kill (macOS has no flock or sha256sum: stubs stand in). No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_replay_waiter.py
"""
import os, re, shlex, shutil, subprocess, sys, time  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import privacy, replay_lib as R, replay_record as RR, screens_lib as L  # noqa: E401,E402
from test_chain_waiter_run import git, stubs  # noqa: E402
from closed_lib import CLOSED_PLANS, closed_state  # noqa: E402  (BATCH 1 CLOSED, later)
from robust_lib import ROBUST_PLANS, robust_configs  # noqa: E402  (AMENDMENT SEEDS-103-105, later)

RW, CW = os.path.join(L.HERE, "replay_waiter.sh"), os.path.join(L.HERE, "chain_waiter.sh")
PLANS = os.path.join(L.HERE, "plans")
BREF = "27d8cb7505248ff4391a1dee9a51bad617938195"
T = "2026-10-07 0"
BASE = [f"{T}2:05:00 queue start: exp SCREENS, plan stage2_base_diag.txt (sha256 f2255fcb65aeeb72), code commit {BREF}",
        f"{T}2:05:01 seen SCREENS: SCREENS STAGE 2 S006 SEEDS DONE", f"{T}3:03:50 done base_s105 rc 0",
        f"{T}3:03:58 MARK SCREENS BASE DIAG DONE", f"{T}3:03:58 plan finished: stage2_base_diag.txt"]
STUB = r"""#!/bin/bash
echo "$*" >> "$CW_STUBLOG"
Q=$(dirname "$(dirname "$(dirname "$0")")")
LOG=$CW_HOME/runs/SCREENS/queue_screens.txt
o=$(head -1 "$CW_OUTCOMES"); sed -i.bak 1d "$CW_OUTCOMES"
t() { echo "$(date '+%F %T') $*" >> "$LOG"; }
t "queue start: exp SCREENS, plan stage2_replay.txt (sha256 0123456789abcdef), code commit $(git -C "$Q" rev-parse HEAD)"
t "seen SCREENS: SCREENS BASE DIAG DONE"
case $o in
  clean) t "MARK SCREENS BASE REPLAY DONE"; touch "$CW_HOME/runs/SCREENS/marks/SCREENS_BASE_REPLAY_DONE"
         t "plan finished: stage2_replay.txt"; exit 0 ;;
  lockwait) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train base_s101_rep1"; exit 1 ;;
  fail) t "scoring failed for final_00007630 (see score.err)"; t "queue stops at: train base_s101_rep1"; exit 1 ;;
  kill) exit 137 ;;
esac
"""


def tree(tmp, lines=BASE, mark=True):
    p = tmp / "planck"
    os.makedirs(p / "runs" / "SCREENS" / "marks")
    os.makedirs(p / "logs")
    (p / "runs" / "SCREENS" / "queue_screens.txt").write_text("\n".join(lines) + "\n")
    if mark:
        (p / "runs" / "SCREENS" / "marks" / "SCREENS_BASE_DIAG_DONE").touch()
    return p


def kit(tmp, current="stage2_replay", first="wait_mark SCREENS SCREENS BASE DIAG DONE", bundle=True):
    """planck-kit: the two committed scripts, and a fixture repo (CURRENT, the plan, the stub queue) as its bundle."""
    k, src = tmp / "kit", tmp / "src"
    sx = src / "experiments" / "screens"
    os.makedirs(sx / "plans")
    os.makedirs(k)
    (sx / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    (sx / "plans" / "stage2_replay.txt").write_text(f"# fixture\n{first}\ntrain base_s101_rep1\nmark X\n")
    (sx / "queue_screens.sh").write_text(STUB)
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    if bundle:
        git(src, "bundle", "create", "-q", str(k / "planck_replay.bundle"), "HEAD", "main")
    for f in (RW, CW):
        shutil.copy(f, k / os.path.basename(f))
    return k, git(src, "rev-parse", "HEAD")


def run(tmp, p, k, outcomes=(), maxr="24", grace="2", timeout=60, script="replay_waiter.sh"):
    stubs(tmp)
    (tmp / "outcomes").write_text("".join(o + "\n" for o in outcomes))
    env = {**os.environ, "CW_HOME": str(p), "CW_KIT": str(k), "CW_POLL": "0", "CW_GRACE": grace, "CW_MAXR": maxr,
           "CW_PAT": "rw_test_matches_no_process_[0-9]{12}", "CW_STUBLOG": str(tmp / "stub.log"),
           "CW_OUTCOMES": str(tmp / "outcomes"), "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
           "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
    rc = subprocess.run(["/bin/bash", str(k / script)], env=env, capture_output=True, text=True,
                        timeout=timeout).returncode
    log = (p / "logs" / "replay_waiter.log").read_text() if (p / "logs" / "replay_waiter.log").exists() else ""
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return rc, log, stub


def call(p, ref):
    q = p / "qcode" / ref[:12]
    return f"--code {q} --plan {q}/experiments/screens/plans/stage2_replay.txt"


def test_its_arguments_are_the_entrys_and_the_repos():
    txt = open(RW).read()
    line = next(ln for ln in txt.splitlines() if ln.startswith('bash "$H/chain_waiter.sh" '))
    a = shlex.split(line.replace("$BASE_REF", BREF))[2:]
    sec = " ".join(R.section().split())
    assert "chain_waiter.sh with the arguments " + " ".join(shlex.quote(x) if " " in x else x for x in a).replace(
        "'", '"') + " (stage2_base_diag at 27d8cb7" in sec, a
    assert (a[1], a[2]) == RR.LAST_START[:2] and f"BASE_REF={BREF}" in txt and a[4] == "stage2_replay"
    diag, rep = (open(os.path.join(PLANS, n + ".txt")).read().splitlines() for n in (a[1], a[4]))
    cur = [ln.strip() for ln in open(os.path.join(PLANS, "CURRENT")) if not ln.startswith("#")]
    later = CLOSED_PLANS if closed_state() else []          # BATCH 1 CLOSED moved CURRENT on, later
    later = (ROBUST_PLANS if robust_configs() else []) or later      # and AMENDMENT SEEDS-103-105 after it
    assert diag[-1] == "mark " + a[3] and rep[1] == "wait_mark SCREENS " + a[3] and rep[-1] == R.END and cur == (later or [a[4]])
    assert a[0] == "replay" and a[5] == "planck_replay.bundle" and len(a) == 6
    assert re.search(r"^SHA=([0-9a-f]{64})$", txt, re.M)[1] == L.sha256(CW) and "MAXR=${CW_MAXR:-24}" in txt
    assert 'args replay_requeue stage2_replay "$REF" "SCREENS BASE REPLAY DONE" replay_requeue_none' in txt
    assert R.END == "mark SCREENS BASE REPLAY DONE" and subprocess.run(["/bin/bash", "-n", RW]).returncode == 0


def test_launches_then_done_at_the_clean_end(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 0 and stub == [call(p, ref)], (log, stub)
    assert "LAUNCH stage2_base_diag ended cleanly" in log and "DONE: stage2_replay ended cleanly at its mark" in log
    assert "relaunch" not in log
    rc, log2, stub = run(tmp_path, p, k)                     # a second start: step 1 DONE, step 2 DONE, no launch
    assert rc == 0 and len(stub) == 1 and "DONE already launched" in log2[len(log):]


def test_relaunches_after_a_2h_lock_wait_then_done(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait", "lockwait", "clean"])
    assert rc == 0 and stub == [call(p, ref)] * 3, (log, stub)
    assert "relaunch 1 of stage2_replay" in log and "relaunch 2 of stage2_replay" in log and "relaunch 3" not in log
    assert log.index("stopped after a 2 h gpu.lock wait") < log.index("relaunch 1") < log.index("DONE: stage2_replay")


def test_gives_up_after_maxr_relaunches(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"] * 4, maxr="2")
    assert rc == 1 and stub == [call(p, ref)] * 3 and "GIVEUP: 2 relaunches after 2 h gpu.lock waits" in log, log


@pytest.mark.parametrize("case", ["fail", "kill"])
def test_no_relaunch_after_another_end_or_a_kill(tmp_path, case):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, [case], grace="2", timeout=20)
    assert rc == 1 and stub == [call(p, ref)] and "relaunch" not in log, log
    want = {"fail": "GIVEUP: GIVEUP stage2_replay ended another way", "kill": "GIVEUP: no new stage2_replay start within 2 s"}
    assert want[case] in log, log
    if case == "kill":                  # the grace is held: whole seconds between the HOLD line and the GIVEUP line
        ts = {w: time.mktime(time.strptime(ln[:19], "%Y-%m-%d %H:%M:%S")) for ln in log.splitlines()
              for w in ("HOLD no queue runs", "GIVEUP: no new") if w in ln}
        assert ts["GIVEUP: no new"] - ts["HOLD no queue runs"] >= 2, log


@pytest.mark.parametrize("case", ["base_diag_failed", "no_mark", "current_names_another_plan", "no_bundle"])
def test_step_1_launching_nothing_ends_it(tmp_path, case):
    lines = BASE[:3] + [f"{T}3:00:00 queue stops at: train base_s105"] if case == "base_diag_failed" else BASE
    p = tree(tmp_path, lines, mark=case != "no_mark")
    k, _ = kit(tmp_path, current="stage2_base_diag" if case == "current_names_another_plan" else "stage2_replay",
               bundle=case != "no_bundle")
    rc, log, stub = run(tmp_path, p, k, ["clean"], timeout=20)
    want = {"base_diag_failed": "GIVEUP: step 1 launched nothing", "no_mark": "GIVEUP: step 1 launched nothing",
            "current_names_another_plan": "GIVEUP: step 1 launched nothing", "no_bundle": "GIVEUP: no main in"}
    assert rc == 1 and stub == [] and want[case] in log, log


def test_the_cancel_file_ends_step_2(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"], maxr="0")       # launched, then a 2 h lock wait: given up
    assert rc == 1 and len(stub) == 1 and "GIVEUP: 0 relaunches" in log
    (p / "logs" / "replay_waiter.CANCEL").touch()
    rc, log, stub = run(tmp_path, p, k, ["clean"])                     # would relaunch (MAXR 24) but for the file
    assert rc == 1 and len(stub) == 1 and "replay_waiter.CANCEL exists" in log.splitlines()[-1], log


@pytest.mark.parametrize("case", ["missing", "edited"])
def test_refuses_a_chain_waiter_that_is_not_the_pinned_one(tmp_path, case):
    p, (k, _) = tree(tmp_path), kit(tmp_path)
    if case == "edited":
        (k / "chain_waiter.sh").write_text(open(CW).read() + "# edited\n")
    else:
        os.remove(k / "chain_waiter.sh")
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 1 and stub == [] and f"is missing or not sha256 {L.sha256(CW)[:16]}" in log, log


def test_a_clone_at_another_commit_is_not_relaunched(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"], maxr="0")
    assert rc == 1 and len(stub) == 1 and "GIVEUP: 0 relaunches" in log
    q = p / "qcode" / ref[:12]
    (q / "x").write_text("x")
    git(q, "add", "x")
    git(q, "commit", "-q", "-m", "moved")
    rc, log, stub = run(tmp_path, p, k)
    assert rc == 1 and len(stub) == 1 and "GIVEUP: no clone of " + ref[:12] in log, log


def test_no_private_identifier_in_it():
    txt = open(RW).read()
    assert not [x for s, x, rx in privacy.patterns() if s != "generic" and rx.search(txt)]
    assert not re.search(r"/(?:home|Users)/[a-z]", txt) and "planck.bundle" not in txt.replace("planck_replay.bundle", "")
