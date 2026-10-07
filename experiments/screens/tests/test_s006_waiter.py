"""s006_waiter.sh (SCREENS.txt AMENDMENT S006-REINSTATE): its decision on fixture queue logs, marks and stop files (a
clean end, the mark with the queue still running, a kill, a 2 h gpu.lock wait, a run failure, STOP, another plan or
commit, a double start), and its loop and launch on a fixture tree and a fixture bundle whose queue_screens.sh is a
stub (macOS has no flock: a stub stands in; the PC dry run used the real one). No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_s006_waiter.py
"""
import os, re, subprocess, sys, threading, time  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import privacy, screens_lib as L  # noqa: E401,E402

W = os.path.join(L.HERE, "s006_waiter.sh")
REF = "d21df3de007146c2d619ba7c3f46f264286d318c"          # the commit stage2_seeds was launched from (20:38:01)
T = "2026-10-07 00:37:40 "
OLD = ["2026-10-06 19:30:47 queue start: exp SCREENS, plan stage2_s006x.txt (sha256 7e10048e62c8478e), code commit "
       "53da0f71236782f65bc8a71eb4a74d5e31bb11d2", "2026-10-06 19:52:00 MARK SCREENS S006 EXTENSION DONE",
       "2026-10-06 19:52:00 plan finished: stage2_s006x.txt"]
START = f"2026-10-06 20:38:01 queue start: exp SCREENS, plan stage2_seeds.txt (sha256 ed36242e5b8b45c2), code commit {REF}"
RUN = ["2026-10-06 20:38:01 seen SCREENS: SCREENS S006 EXTENSION DONE", "2026-10-06 20:38:01 skip base_s101 (already SCORED)",
       "2026-10-06 20:38:03 waiting for gpu.lock", "2026-10-06 20:38:03 gpu.lock held; GPU memory in use 600 MiB",
       "2026-10-06 20:38:04 train s005_base_s101.yaml --require-committed"]
LAST = ["2026-10-07 00:36:30 done s007_smear_g1_s102 rc 0", "2026-10-07 00:36:30 score s007_smear_g1_s102/final_00007630"]
CLEAN = [T + "MARK SCREENS STAGE 2 SEEDS DONE", T + "plan finished: stage2_seeds.txt"]
LOCKWAIT = ["2026-10-07 01:00:00 waiting for gpu.lock", "2026-10-07 03:00:00 gpu.lock not free after 2 h",
            "2026-10-07 03:00:00 queue stops at: train s007_smear_g1_s102"]
FAIL = [T + "crash s004_canonac_g1_s102 rc 1: resuming once from its last checkpoint",
        T + "scoring failed for final_00007630 (see score.err)", T + "queue stops at: train s004_canonac_g1_s102"]
STOPPED = [T + "STOP present: queue stops", T + "queue stops at: train s007_smear_g1_s102"]
GAP = [T + "GAP s004_canonac_g1_s102: second crash (rc 1)"]
S6START = T + "queue start: exp SCREENS, plan stage2_s006_seeds.txt (sha256 57ac8f3e5911720b), code commit " + "a" * 40
MID = OLD + [START] + RUN
CASES = {   # name: (log lines, mark file, files under ~/planck, running, the decision's first word, a phrase in it)
    "clean_end": (MID + LAST + CLEAN, True, (), "0", "LAUNCH", "stage2_seeds ended cleanly"),
    "clean_end_with_a_gap_run": (MID + GAP + LAST + CLEAN, True, (), "0", "LAUNCH", "ended cleanly"),
    "clean_end_queue_still_running": (MID + LAST + CLEAN, True, (), "1", "WAIT", "queue or launcher runs"),
    "mark_present_queue_still_running": (MID + LAST + CLEAN[:1], True, (), "1", "WAIT", "runs"),
    "killed_no_end_line": (MID, False, (), "0", "HOLD", "no end line"),
    "mark_present_but_killed_before_its_end": (MID + LAST + CLEAN[:1], True, (), "0", "HOLD", "no end line"),
    "lock_wait_end": (MID + LOCKWAIT, False, (), "0", "HOLD", "2 h gpu.lock wait"),
    "crash_then_scoring_stop": (MID + FAIL, False, (), "0", "GIVEUP", "ended another way"),
    "stop_logged": (MID + STOPPED, False, (), "0", "GIVEUP", "ended another way"),
    "stop_file_while_running": (MID, False, ("STOP",), "1", "GIVEUP", "STOP exists"),
    "stop_while_waiting": (MID[:4] + [T + "STOP while waiting for SCREENS: SCREENS S006 EXTENSION DONE"], False, (), "0",
                           "GIVEUP", "ended another way"),
    "relaunch_refused": (MID + LOCKWAIT + [T + "refusing: uncommitted changes in the code checkout:"], False, (), "0",
                         "GIVEUP", "refusing"),
    "clean_end_no_mark_file": (MID + LAST + CLEAN, False, (), "0", "GIVEUP", "no marks/SCREENS_STAGE_2_SEEDS_DONE"),
    "finished_without_its_mark_line": (MID + LAST + CLEAN[1:], True, (), "0", "GIVEUP", "no MARK line"),
    "a_line_after_the_end": (MID + LAST + CLEAN + [T + "train s006_mtp_g2_s101.yaml"], True, (), "0", "GIVEUP",
                             "lines after the end"),
    "last_start_another_plan": (OLD, True, (), "0", "GIVEUP", "not stage2_seeds"),
    "last_start_another_commit": (OLD + [START.replace(REF, "53da0f71236782f65bc8a71eb4a74d5e31bb11d2")] + RUN[1:] + LAST
                                  + CLEAN, True, (), "0", "GIVEUP", "not stage2_seeds at d21df3de0071"),
    "double_start_stamp": (MID + LAST + CLEAN, True, ("logs/s006_waiter.launched",), "0", "DONE", "already launched"),
    "double_start_log_line": (MID + LAST + CLEAN + [S6START], True, (), "1", "DONE", "stage2_s006_seeds start line"),
    "cancel_file": (MID + LAST + CLEAN, True, ("logs/s006_waiter.CANCEL",), "0", "GIVEUP", "cancel file"),
    "pgrep_failed": (MID + LAST + CLEAN, True, (), "E", "GIVEUP", "pgrep failed"),
}


def tree(tmp, lines, mark=True, files=(), nul=False):
    p = tmp / "planck"
    out = p / "runs" / "SCREENS"
    os.makedirs(out / "marks", exist_ok=True)
    os.makedirs(p / "logs", exist_ok=True)
    text = "\n".join(lines) + "\n"
    if nul:                                     # a power cut's NUL tail glued to the next line (40_screens notes)
        text = text.replace(START, "\0" * 40 + START)
    (out / "queue_screens.txt").write_bytes(text.encode())
    if mark:
        (out / "marks" / "SCREENS_STAGE_2_SEEDS_DONE").touch()
    for f in files:
        (p / f).touch()
    return p


def decide(p, running="0"):
    r = subprocess.run(["/bin/bash", "-c", f'. "{W}"; decide {running}'], env={**os.environ, "S006W_HOME": str(p)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    return r.stdout.strip()


@pytest.mark.parametrize("name", sorted(CASES))
def test_decision(tmp_path, name):
    lines, mark, files, running, word, phrase = CASES[name]
    got = decide(tree(tmp_path, lines, mark, files), running)
    assert got.split()[0] == word and phrase in got, got


def test_nul_bytes_in_the_log_do_not_hide_the_clean_end(tmp_path):
    assert decide(tree(tmp_path, MID + LAST + CLEAN, nul=True)).startswith("LAUNCH")


def test_no_queue_log(tmp_path):
    p = tree(tmp_path, [])
    os.unlink(p / "runs" / "SCREENS" / "queue_screens.txt")
    assert decide(p).startswith("GIVEUP no queue log")


def git(cwd, *a):
    env = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c", "init.defaultBranch=main",
                           *a], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def bundle(tmp, kit, name, current):
    """A fixture repo (CURRENT, both plans, a stub queue_screens.sh that records its arguments) as a git bundle."""
    src = tmp / ("src_" + name)
    sx = src / "experiments" / "screens"
    os.makedirs(sx / "plans")
    (sx / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    for plan in ("stage2_seeds", "stage2_s006_seeds"):
        (sx / "plans" / f"{plan}.txt").write_text(f"# {plan}\nmark X\n")
    (sx / "queue_screens.sh").write_text('#!/bin/bash\necho "$*" >> "$S006W_STUBLOG"\n')
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    os.makedirs(kit, exist_ok=True)
    git(src, "bundle", "create", "-q", str(kit / name), "HEAD", "main")
    return git(src, "rev-parse", "HEAD")


def main_run(tmp, p, poll="0", grace="2", flock_rc="0", timeout=60, **extra):
    os.makedirs(tmp / "bin", exist_ok=True)
    (tmp / "bin" / "flock").write_text(f"#!/bin/sh\nexit {flock_rc}\n")
    os.chmod(tmp / "bin" / "flock", 0o755)
    env = {**os.environ, "S006W_HOME": str(p), "S006W_KIT": str(tmp / "kit"), "S006W_POLL": poll, "S006W_GRACE": grace,
           "S006W_PAT": "s006w_test_matches_no_process_[0-9]{12}", "S006W_STUBLOG": str(tmp / "stub.log"),
           "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", **extra}
    rc = subprocess.run(["/bin/bash", W], env=env, capture_output=True, text=True, timeout=timeout).returncode
    log = (p / "logs" / "s006_waiter.log").read_text()
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return rc, log, stub


def test_launches_once_from_its_own_bundle(tmp_path):
    p, kit = tree(tmp_path, MID + LAST + CLEAN), tmp_path / "kit"
    ref = bundle(tmp_path, kit, "planck_s006.bundle", "stage2_s006_seeds")
    other = bundle(tmp_path, kit, "planck.bundle", "stage2_seeds")       # the autopilot's bundle: never read here
    q = p / "qcode" / ref[:12]
    rc, log, stub = main_run(tmp_path, p)
    assert rc == 0 and stub == [f"--code {q} --plan {q}/experiments/screens/plans/stage2_s006_seeds.txt"], (log, stub)
    assert git(q, "rev-parse", "HEAD") == ref != other and not (p / "qcode" / other[:12]).exists()
    assert "LAUNCH stage2_seeds ended cleanly" in log and f"code {ref} in qcode/{ref[:12]}, plan stage2_s006_seeds" in log
    assert f"launched stage2_s006_seeds from {ref[:12]}" in (p / "logs" / "s006_waiter.launched").read_text()
    rc, log, stub = main_run(tmp_path, p)                               # a second waiter: DONE, no second launch
    assert rc == 0 and len(stub) == 1 and log.rstrip().splitlines()[-1].endswith(
        f"DONE already launched: {(p / 'logs' / 's006_waiter.launched').read_text().strip()}")


@pytest.mark.parametrize("case", ["current_names_another_plan", "no_s006_bundle"])
def test_refuses_a_wrong_or_missing_bundle(tmp_path, case):
    p, kit = tree(tmp_path, MID + LAST + CLEAN), tmp_path / "kit"
    bundle(tmp_path, kit, "planck.bundle", "stage2_s006_seeds")
    if case == "current_names_another_plan":
        bundle(tmp_path, kit, "planck_s006.bundle", "stage2_seeds")
    rc, log, stub = main_run(tmp_path, p)
    want = "names [stage2_seeds], not stage2_s006_seeds" if case.startswith("current") else "GIVEUP: no bundle"
    assert rc == 1 and stub == [] and want in log and not (p / "logs" / "s006_waiter.launched").exists(), log


def test_a_second_waiter_holding_no_lock_exits(tmp_path):
    p = tree(tmp_path, MID + LAST + CLEAN)
    bundle(tmp_path, tmp_path / "kit", "planck_s006.bundle", "stage2_s006_seeds")
    rc, log, stub = main_run(tmp_path, p, flock_rc="1")
    assert rc == 1 and stub == [] and "another s006_waiter holds" in log and "LAUNCH" not in log


def test_giveup_exits_without_a_launch(tmp_path):
    p = tree(tmp_path, MID + FAIL)
    bundle(tmp_path, tmp_path / "kit", "planck_s006.bundle", "stage2_s006_seeds")
    rc, log, stub = main_run(tmp_path, p)
    assert rc == 1 and stub == [] and "GIVEUP stage2_seeds ended another way" in log


def test_a_hold_gives_up_after_the_grace(tmp_path):
    p = tree(tmp_path, MID + LOCKWAIT, mark=False)
    t0 = time.time()
    rc, log, stub = main_run(tmp_path, p, poll="1", grace="2", timeout=30)
    assert rc == 1 and stub == [] and 2 <= time.time() - t0 < 25
    assert "HOLD stage2_seeds stopped after a 2 h gpu.lock wait" in log and "GIVEUP: no new stage2_seeds start within 2 s" in log


def test_a_hold_then_a_relaunch_and_a_clean_end_launches(tmp_path):
    p = tree(tmp_path, MID + LOCKWAIT, mark=False)
    ref = bundle(tmp_path, tmp_path / "kit", "planck_s006.bundle", "stage2_s006_seeds")

    def relaunched():
        time.sleep(1.5)
        (p / "runs" / "SCREENS" / "marks" / "SCREENS_STAGE_2_SEEDS_DONE").touch()     # the mark first: no torn read
        with open(p / "runs" / "SCREENS" / "queue_screens.txt", "a") as f:
            f.write("\n".join([START.replace("2026-10-06 20:38:01", "2026-10-07 03:05:00")] + LAST + CLEAN) + "\n")
    th = threading.Thread(target=relaunched)
    th.start()
    rc, log, stub = main_run(tmp_path, p, poll="1", grace="30", timeout=40)
    th.join()
    assert rc == 0 and len(stub) == 1 and ref[:12] in stub[0] and log.index("HOLD") < log.index("LAUNCH"), log


def test_defaults_and_the_one_bundle_it_reads():
    env = {k: v for k, v in os.environ.items() if not k.startswith("S006W_")}
    out = subprocess.run(["/bin/bash", "-c", f'. "{W}"; echo "$P|$POLL|$GRACE|$PAT|$KIT|$WANT_PLAN|$WANT_REF|$NEXT_PLAN"'],
                         env=env, capture_output=True, text=True).stdout.strip()
    assert out == (f"{os.environ['HOME']}/planck|300|2700|queue_screens\\.sh|launch_screens\\.sh||stage2_seeds|{REF}|"
                   "stage2_s006_seeds")
    code = [re.sub(r"\s+# .*$", "", ln) for ln in open(W) if not ln.lstrip().startswith("#")]   # code, no comments
    assert [ln.strip() for ln in code if ".bundle" in ln] == ["B=$KIT/planck_s006.bundle"]
    assert subprocess.run(["/bin/bash", "-n", W]).returncode == 0


def test_no_private_identifier_in_the_waiter():
    pats = privacy.patterns()
    assert not [x for s, x, rx in pats if s != "generic" and rx.search(open(W).read())]
    assert not re.search(r"/(?:home|Users)/[a-z]", open(W).read())
