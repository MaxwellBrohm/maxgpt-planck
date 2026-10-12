"""robust_waiter.sh (SCREENS.txt AMENDMENT SEEDS-103-105, STARTING IT UNATTENDED): its arguments against the entry, the
plans and the queue log's last start; its sha256 pin of chain_waiter.sh; and the whole file run on a fixture tree with a
fixture bundle whose queue_screens.sh and diag_queue.sh are stubs (macOS has no flock or sha256sum: stubs stand in):
step 0 (E3 running, E3's last queue clean or not, no E3 log), the launch, a clean end, 2 h gpu.lock waits, a failure,
a kill, the C6 group's exit, the cancel file, a restart. Every git call here runs with git's location variables removed
(the 2026-10-11 22:17 incident). No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_robust_waiter.py
"""
import os, re, shlex, shutil, subprocess, sys, threading, time, uuid  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import privacy, robust_lib as RB, screens_lib as L  # noqa: E401,E402

RW, CW = os.path.join(L.HERE, "robust_waiter.sh"), os.path.join(L.HERE, "chain_waiter.sh")
PLANS = os.path.join(L.HERE, "plans")
AREF = "2aa37874b34fc49b6e564e795482b46662212c44"      # stage2_replay's queue commit (INTEGRATION 2026-10-10)
GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR")
T = "2026-10-10 2"
REPLAY = [f"{T}1:32:06 queue start: exp SCREENS, plan stage2_replay.txt (sha256 86bcb6ca35296c2d), code commit {AREF}",
          f"{T}1:32:07 seen SCREENS: SCREENS BASE DIAG DONE", f"{T}3:51:19 MARK SCREENS BASE REPLAY DONE",
          f"{T}3:51:19 plan finished: stage2_replay.txt"]
E3 = ["2026-10-11 13:12:28 queue start: exp E3, plan part3.txt (sha256 07018f0fec572994), code commit "
      "4962426b8f0d17738051bd499f9141a7cbc18c40", "2026-10-11 22:00:56 MARK E3 PART 3 DONE",
      "2026-10-11 22:00:56 plan finished: part3.txt"]
STUB = r"""#!/bin/bash
echo "queue $*" >> "$CW_STUBLOG"
Q=$(dirname "$(dirname "$(dirname "$0")")")
LOG=$CW_HOME/runs/SCREENS/queue_screens.txt
o=$(head -1 "$CW_OUTCOMES"); sed -i.bak 1d "$CW_OUTCOMES"
t() { echo "$(date '+%F %T') $*" >> "$LOG"; }
t "queue start: exp SCREENS, plan robustness_k5.txt (sha256 0123456789abcdef), code commit $(git -C "$Q" rev-parse HEAD)"
t "seen SCREENS: SCREENS BASE REPLAY DONE"
case $o in
  clean) t "MARK SCREENS ROBUSTNESS K5 DONE"; touch "$CW_HOME/runs/SCREENS/marks/SCREENS_ROBUSTNESS_K5_DONE"
         t "plan finished: robustness_k5.txt"; exit 0 ;;
  lockwait) t "waiting for gpu.lock"; t "gpu.lock not free after 2 h"; t "queue stops at: train s001_nogate_g1_s103"; exit 1 ;;
  fail) t "scoring failed for final_00007630 (see score.err)"; t "queue stops at: train s001_nogate_g1_s103"; exit 1 ;;
  kill) exit 137 ;;
esac
"""
DIAG = '#!/bin/bash\necho "diag $*" >> "$CW_STUBLOG"\nexit "${CW_DIAG_RC:-0}"\n'


def env0():
    return {k: v for k, v in os.environ.items() if k not in GIT_ENV}


def git(cwd, *a):
    env = {**env0(), "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c", "init.defaultBranch=main", *a],
                          cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def tree(tmp, screens=REPLAY, e3=E3, mark=True):
    p = tmp / "planck"
    os.makedirs(p / "runs" / "SCREENS" / "marks")
    os.makedirs(p / "logs")
    (p / "runs" / "SCREENS" / "queue_screens.txt").write_text("\n".join(screens) + "\n")
    if e3 is not None:
        os.makedirs(p / "runs" / "E3")
        (p / "runs" / "E3" / "queue_e3.txt").write_text("\n".join(e3) + "\n")
    if mark:
        (p / "runs" / "SCREENS" / "marks" / "SCREENS_BASE_REPLAY_DONE").touch()
    return p


def kit(tmp, current="robustness_k5", first=RB.WAIT, bundle=True):
    k, src = tmp / "kit", tmp / "src"
    sx = src / "experiments" / "screens"
    os.makedirs(sx / "plans")
    os.makedirs(k)
    (sx / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    (sx / "plans" / "robustness_k5.txt").write_text(f"# fixture\n{first}\ntrain s001_nogate_g1_s103\n{RB.END}\n")
    (sx / "queue_screens.sh").write_text(STUB)
    (sx / "diag_queue.sh").write_text(DIAG)
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    if bundle:
        git(src, "bundle", "create", "-q", str(k / "planck_robust.bundle"), "HEAD", "main")
    for f in (RW, CW):
        shutil.copy(f, k / os.path.basename(f))
    return k, git(src, "rev-parse", "HEAD")


def stubs(tmp):
    os.makedirs(tmp / "bin", exist_ok=True)
    (tmp / "bin" / "flock").write_text("#!/bin/sh\nexit 0\n")
    (tmp / "bin" / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    for f in ("flock", "sha256sum"):
        os.chmod(tmp / "bin" / f, 0o755)


def run(tmp, p, k, outcomes=(), maxr="24", grace="2", timeout=60, diag_rc="0", epat=None):
    stubs(tmp)
    (tmp / "outcomes").write_text("".join(o + "\n" for o in outcomes))
    env = {**env0(), "CW_HOME": str(p), "CW_KIT": str(k), "CW_POLL": "0", "CW_GRACE": grace, "CW_MAXR": maxr,
           "CW_PAT": "rbw_test_matches_no_process_[0-9]{12}", "CW_EPAT": epat or "rbw_test_no_e3_[0-9]{12}",
           "CW_STUBLOG": str(tmp / "stub.log"), "CW_OUTCOMES": str(tmp / "outcomes"), "CW_DIAG_RC": diag_rc,
           "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
    rc = subprocess.run(["/bin/bash", str(k / "robust_waiter.sh")], env=env, capture_output=True, text=True,
                        timeout=timeout).returncode
    log = (p / "logs" / "robust_waiter.log").read_text() if (p / "logs" / "robust_waiter.log").exists() else ""
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return rc, log, stub


def qcall(p, ref):
    q = p / "qcode" / ref[:12]
    return f"queue --code {q} --plan {q}/experiments/screens/plans/robustness_k5.txt"


def dcall(p, ref):
    return f"diag --code {p / 'qcode' / ref[:12]} --groups robust"


def test_its_arguments_are_the_entrys_and_the_repos():
    txt = open(RW).read()
    line = next(ln for ln in txt.splitlines() if ln.startswith('bash "$H/chain_waiter.sh" '))
    a = shlex.split(line.replace("$REPLAY_REF", AREF).replace(" 5>&-", ""))[2:]
    f = " ".join(RB.section().split())
    assert a == ["robust", "stage2_replay", AREF, "SCREENS BASE REPLAY DONE", "robustness_k5", "planck_robust.bundle"]
    assert "chain_waiter.sh robust stage2_replay 2aa3787... \"SCREENS BASE REPLAY DONE\" robustness_k5 " \
           "planck_robust.bundle" in f and f"REPLAY_REF={AREF}" in txt
    rep, rob = (open(os.path.join(PLANS, n + ".txt")).read().splitlines() for n in ("stage2_replay", "robustness_k5"))
    assert rep[-1] == "mark " + a[3] and rob[1] == RB.WAIT == "wait_mark SCREENS " + a[3] and rob[-1] == RB.END
    assert re.search(r"^SHA=([0-9a-f]{64})$", txt, re.M)[1] == L.sha256(CW) and "MAXR=${CW_MAXR:-24}" in txt
    assert 'args robust_requeue robustness_k5 "$REF" "SCREENS ROBUSTNESS K5 DONE" robust_requeue_none' in txt
    assert 'bash "$Q/$XREL/diag_queue.sh" --code "$Q" --groups robust' in txt and RB.END == "mark " + \
        "SCREENS ROBUSTNESS K5 DONE"
    assert subprocess.run(["/bin/bash", "-n", RW]).returncode == 0


def test_launches_then_runs_the_c6_group_then_done(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 0 and stub == [qcall(p, ref), dcall(p, ref)], (log, stub)
    assert "step 0: OK E3's last queue (part3.txt) ended cleanly" in log and "LAUNCH stage2_replay ended cleanly" in log
    assert "robustness_k5 ended cleanly at its mark" in log and "DONE: robustness_k5 ended cleanly" in log
    assert "relaunch" not in log
    rc, log2, stub = run(tmp_path, p, k)          # started again: step 1 DONE, no new queue; the C6 group again (skips)
    assert rc == 0 and stub[2:] == [dcall(p, ref)] and "DONE already launched" in log2[len(log):], (log2, stub)


def test_relaunches_after_2h_lock_waits_then_the_c6_group(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait", "lockwait", "clean"])
    assert rc == 0 and stub == [qcall(p, ref)] * 3 + [dcall(p, ref)], (log, stub)
    assert "relaunch 1 of robustness_k5" in log and "relaunch 2 of robustness_k5" in log and "relaunch 3" not in log


def test_gives_up_after_maxr_relaunches(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["lockwait"] * 4, maxr="2")
    assert rc == 1 and stub == [qcall(p, ref)] * 3 and "GIVEUP: 2 relaunches after 2 h gpu.lock waits" in log, log


@pytest.mark.parametrize("case", ["fail", "kill"])
def test_no_relaunch_and_no_c6_after_another_end_or_a_kill(tmp_path, case):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, [case], timeout=20)
    assert rc == 1 and stub == [qcall(p, ref)] and "relaunch" not in log, log
    want = {"fail": "GIVEUP: GIVEUP robustness_k5 ended another way", "kill": "GIVEUP: no new robustness_k5 start within 2 s"}
    assert want[case] in log, log


def test_a_failed_c6_group_is_a_giveup(tmp_path):
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"], diag_rc="1")
    assert rc == 1 and stub == [qcall(p, ref), dcall(p, ref)] and "GIVEUP: the C6 robust group's queue ended, exit 1" in log


@pytest.mark.parametrize("case,want", [
    ("stops", "GIVEUP E3's last queue (part3.txt) did not end cleanly"),
    ("no_mark_line", "GIVEUP E3's last queue (part3.txt) did not end cleanly"),
    ("other_plan_finished", "GIVEUP E3's last queue (part3.txt) did not end cleanly"),
    ("score_before_finish", "GIVEUP E3's last queue (part3.txt) did not end cleanly"),
    ("no_start", "GIVEUP no queue start line in"),
])
def test_step_0_gives_up_unless_e3_ended_cleanly(tmp_path, case, want):
    e3 = {"stops": E3[:1] + ["2026-10-11 20:00:00 queue stops at: train e3_5m_A_data5"],
          "no_mark_line": [E3[0], E3[2]], "other_plan_finished": E3[:2] + ["2026-10-11 22:00:56 plan finished: part2.txt"],
          "score_before_finish": E3[:1] + ["2026-10-11 22:00:18 score e3_5m_A_data5/ckpt_00007497", E3[2]],
          "no_start": E3[1:]}[case]
    p, (k, _) = tree(tmp_path, e3=e3), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"], timeout=20)
    assert rc == 1 and stub == [] and want in log and "GIVEUP: step 0" in log, log


def test_no_e3_log_is_no_e3_queue(tmp_path):
    p, (k, ref) = tree(tmp_path, e3=None), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 0 and stub == [qcall(p, ref), dcall(p, ref)] and "step 0: OK no E3 queue log" in log, log


def test_waits_while_an_e3_queue_runs(tmp_path):
    tok = "rbw_fake_e3_queue_" + uuid.uuid4().hex[:12]
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", tok])
    p, (k, ref) = tree(tmp_path), kit(tmp_path)
    killer = threading.Timer(3.0, proc.kill)
    killer.start()
    try:
        t0 = time.time()
        rc, log, stub = run(tmp_path, p, k, ["clean"], epat=tok)
    finally:
        killer.cancel()
        proc.kill()
    assert rc == 0 and time.time() - t0 >= 2.5 and stub == [qcall(p, ref), dcall(p, ref)], (log, stub)
    assert log.index("step 0: WAIT an E3 queue, launcher or waiter runs") < log.index("step 0: OK E3's last queue")


@pytest.mark.parametrize("case", ["replay_failed", "no_mark", "current_names_another_plan", "no_bundle"])
def test_step_1_launching_nothing_ends_it(tmp_path, case):
    lines = REPLAY[:2] + [f"{T}3:00:00 queue stops at: train base_s101_rep2"] if case == "replay_failed" else REPLAY
    p = tree(tmp_path, lines, mark=case != "no_mark")
    k, _ = kit(tmp_path, current="batch1_closed" if case == "current_names_another_plan" else "robustness_k5",
               bundle=case != "no_bundle")
    rc, log, stub = run(tmp_path, p, k, ["clean"], timeout=20)
    want = "GIVEUP: no main in" if case == "no_bundle" else "GIVEUP: step 1 launched nothing"
    assert rc == 1 and stub == [] and want in log, log


def test_the_cancel_file_ends_it_before_any_launch(tmp_path):
    p, (k, _) = tree(tmp_path), kit(tmp_path)
    (p / "logs" / "robust_waiter.CANCEL").touch()
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 1 and stub == [] and f"GIVEUP: {p / 'logs' / 'robust_waiter.CANCEL'} exists" in log, log
    assert "=== robust_waiter" not in log                     # step 0 ended it: chain_waiter.sh never started


@pytest.mark.parametrize("case", ["missing", "edited"])
def test_refuses_a_chain_waiter_that_is_not_the_pinned_one(tmp_path, case):
    p, (k, _) = tree(tmp_path), kit(tmp_path)
    if case == "edited":
        (k / "chain_waiter.sh").write_text(open(CW).read() + "# edited\n")
    else:
        os.remove(k / "chain_waiter.sh")
    rc, log, stub = run(tmp_path, p, k, ["clean"])
    assert rc == 1 and stub == [] and f"is missing or not sha256 {L.sha256(CW)[:16]}" in log, log


def test_no_private_identifier_in_it():
    txt = open(RW).read()
    assert not [x for s, x, rx in privacy.patterns() if s != "generic" and rx.search(txt)]
    assert not re.search(r"/(?:home|Users)/[a-z]", txt) and "planck.bundle" not in txt.replace("planck_robust.bundle", "")
