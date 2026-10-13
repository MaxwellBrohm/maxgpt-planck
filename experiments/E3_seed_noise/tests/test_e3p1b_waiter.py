"""e3p1b_waiter.sh (notes.txt ADDENDUM 2026-10-12, ORDER AND QUEUE), run whole on tests/w1bfixture.py's fixture tree: it
waits for E3 and SCREENS processes, the robustness mark, the C6 robust group's DONE line and disk; launches once; relaunches
after lock waits without waiting on SCREENS again; never relaunches after an end it cannot name; checks CURRENT, the plan's
sha256 and the five code trees; and its pins equal the files and the repository's trees. No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_e3p1b_waiter.py
"""
import hashlib
import os
import re
import subprocess
import threading
import time

import pytest

import w1bfixture as X
from w1bfixture import call, home, kit, run


def test_pins_and_parse():
    txt = open(X.W).read()
    assert re.search(r"^PLAN_SHA=([0-9a-f]{64})$", txt, re.M)[1] == hashlib.sha256(open(X.PLAN, "rb").read()).hexdigest()
    assert "BUNDLE=planck_e3p1b.bundle; PLAN=part1b_p3at20m" in txt and "MINFREE=${E3W_MINFREE:-41}" in txt
    assert "MAXR=${E3W_MAXR:-24}" in txt and "POLL=${E3W_POLL:-300}" in txt and "QUEUE_MIN_FREE_GB=$MINFREE bash" in txt
    assert subprocess.run(["/bin/bash", "-n", X.W]).returncode == 0
    assert not re.search(r"/(?:home|Users)/[a-z]|\b\d{1,3}(?:\.\d{1,3}){3}\b|@[\w-]+\.(?:com|net|org)", txt)
    assert chr(0x2014) not in txt and chr(0x2013) not in txt


def test_tree_pins_are_the_repositorys():
    """The pinned trees are the ones PART 1b PARITY compared: HEAD's, equal to Part 3's commit 4962426 (git needed; an
    export of the index names the repository's git dir in E3A1_GIT_DIR, passed as an argument only)."""
    gd = os.environ.get("E3A1_GIT_DIR")
    where = ["--git-dir", gd] if gd else ["-C", os.path.dirname(os.path.dirname(X.E3))]
    pins = dict(t.rsplit(":", 1) for t in re.search(r'TREES=\$\{E3W_TREES:-"(.*?)"\}', open(X.W).read(), re.S)[1]
                .replace("\\\n", " ").split())
    assert sorted(pins) == sorted(X.TREES)
    for rev in ("HEAD", "4962426"):
        r = subprocess.run(["git", *where, "rev-parse", *(f"{rev}:{t}" for t in X.TREES)], capture_output=True, text=True,
                           env=X.clean_env(DEVELOPER_DIR="/Library/Developer/CommandLineTools"))
        if r.returncode:
            pytest.skip("git or the repository unavailable")
        assert r.stdout.split() == [pins[t] for t in X.TREES], rev


def test_clean_end_is_done_and_a_second_start_launches_nothing(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])
    assert rc == 0 and stub == [call(p, ref)] and "DONE part1b_p3at20m ended cleanly at its mark" in log, log
    assert (p / "logs" / "e3p1b_waiter.started").exists() and "LAUNCH robustness_k5 and the C6 robust group ended" in log
    rc, log2, stub = run(tmp_path, p, k, tr, ["clean"])
    assert rc == 0 and len(stub) == 1 and "DONE marks/E3_PART_3_AT_20M_DONE exists" in log2[len(log):]


def test_relaunches_after_lock_and_memory_waits_at_both_sizes_then_done(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["lockwait", "lockwait20", "memwait", "clean"])
    assert rc == 0 and stub == [call(p, ref)] * 4, (log, stub)
    assert "launch 4:" in log and "launch 5:" not in log and log.count("RELAUNCH") == 3
    assert log.index("2 h gpu.lock wait: [") < log.index("launch 2:") < log.index("another process's GPU memory")


def test_a_relaunch_does_not_wait_on_screens_again(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    try:     # the stub starts a process the SCREENS pattern matches, then stops after a lock wait
        rc, log, stub = run(tmp_path, p, k, tr, ["lockwait_screens", "clean"], spat=r"sleep 3\.14159", poll="1")
    finally:
        subprocess.run(["pkill", "-f", r"sleep 3\.14159"])
    assert rc == 0 and stub == [call(p, ref)] * 2 and "WAIT a SCREENS" not in log, log


def test_gives_up_after_maxr_relaunches(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["lockwait"] * 4, maxr="2")
    assert rc == 1 and stub == [call(p, ref)] * 3 and "GIVEUP: 2 relaunches after lock waits" in log, log


@pytest.mark.parametrize("case", ["fail", "disk", "kill", "refuse", "othersha", "nomark", "clean_rc1", "lockwait_rc0"])
def test_no_relaunch_after_an_end_it_cannot_name(tmp_path, case):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, [case, "clean"])
    assert rc == 1 and stub == [call(p, ref)] and "launch 2:" not in log, log
    want = {"fail": "ended another way (exit 1)", "disk": "ended another way (exit 1)", "kill": "ended another way (exit 137)",
            "refuse": "GIVEUP no part1b_p3at20m start line at", "othersha": "GIVEUP no part1b_p3at20m start line at",
            "nomark": "GIVEUP a clean end but no marks/E3_PART_3_AT_20M_DONE", "clean_rc1": "ended another way (exit 1)",
            "lockwait_rc0": "ended another way (exit 0)"}
    assert want[case] in log, log


def test_a_pgrep_error_gives_up(tmp_path):
    p, (k, _, tr) = home(tmp_path), kit(tmp_path)
    os.makedirs(tmp_path / "bin")
    (tmp_path / "bin" / "pgrep").write_text("#!/bin/sh\nexit 3\n")
    os.chmod(tmp_path / "bin" / "pgrep", 0o755)
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])
    assert (rc, stub) == (1, []) and "GIVEUP pgrep failed: cannot tell what runs" in log, log


def test_a_restart_after_a_cut_resumes(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["kill"])
    assert rc == 1 and len(stub) == 1
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])          # E3's last start is this plan's, uncleanly ended: resumed
    assert rc == 0 and stub == [call(p, ref)] * 2 and "OK E3's last queue is part1b_p3at20m.txt" in log, log


@pytest.mark.parametrize("case", ["cancel", "stop", "done_mark", "other_waiter"])
def test_nothing_is_launched(tmp_path, case):
    p, (k, _, tr) = home(tmp_path, done=case == "done_mark"), kit(tmp_path)
    if case == "cancel":
        (p / "logs" / "e3p1b_waiter.CANCEL").touch()
    if case == "stop":
        (p / "STOP").touch()
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"], flock_rc="1" if case == "other_waiter" else "0")
    want = {"cancel": (1, "e3p1b_waiter.CANCEL exists"), "stop": (1, "/STOP exists"),
            "done_mark": (0, "DONE marks/E3_PART_3_AT_20M_DONE exists"), "other_waiter": (1, "another e3p1b_waiter holds")}
    assert (rc, stub) == (want[case][0], []) and want[case][1] in log, log


@pytest.mark.parametrize("case", ["current_names_part3", "plan_edited", "no_bundle", "harness_tree", "e2_tree"])
def test_a_wrong_bundle_launches_nothing(tmp_path, case):
    p = home(tmp_path)
    k, _, tr = kit(tmp_path, current="part3" if case == "current_names_part3" else "part1b_p3at20m",
                   plan=b"# edited\ntrain e3_5m_A_s12\n" if case == "plan_edited" else None, bundle=case != "no_bundle")
    if case in ("harness_tree", "e2_tree"):
        t = "harness:" if case == "harness_tree" else "experiments/E2_lr_transfer:"
        tr = re.sub(re.escape(t) + r"[0-9a-f]{40}", t + "0" * 40, tr)
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])
    want = {"current_names_part3": "names [part3], not part1b_p3at20m", "plan_edited": "is missing or not sha256 17c939a5",
            "no_bundle": "GIVEUP: no bundle", "harness_tree": "GIVEUP: harness at", "e2_tree": "GIVEUP: experiments/E2_lr_transfer at"}
    assert rc == 1 and stub == [] and want[case] in log, log


def later(delay, fn):
    th = threading.Thread(target=lambda: (time.sleep(delay), fn()))
    th.start()
    return th


@pytest.mark.parametrize("case", ["e3_process", "screens_process", "robust_mark", "c6_done_line", "disk"])
def test_waits_then_launches(tmp_path, case):
    p, (k, ref, tr) = home(tmp_path, robust=case != "robust_mark", rw_done=case != "c6_done_line"), kit(tmp_path)
    kw = {"poll": "1"}
    sleeper = None
    if case in ("e3_process", "screens_process"):
        sleeper = subprocess.Popen(["/bin/sleep", "2.71828"])
        kw["pat" if case == "e3_process" else "spat"] = r"sleep 2\.71828"
    th = {"robust_mark": lambda: later(2, lambda: (p / "runs" / "SCREENS" / "marks" / "SCREENS_ROBUSTNESS_K5_DONE").touch()),
          "c6_done_line": lambda: later(2, lambda: open(p / "logs" / "robust_waiter.log", "a").write(X.RW_DONE + "\n")),
          "disk": lambda: later(2, lambda: (tmp_path / "avail").write_text("41"))}.get(case, lambda: None)()
    if case == "disk":
        (tmp_path / "avail").write_text("40")
    try:
        t0 = time.time()
        rc, log, stub = run(tmp_path, p, k, tr, ["clean"], **kw)
    finally:
        if sleeper:
            sleeper.kill()
        if th:
            th.join()
    want = {"e3_process": "WAIT an E3 queue", "screens_process": "WAIT a SCREENS queue", "robust_mark": "WAIT no marks/SCREENS_ROBUSTNESS_K5_DONE",
            "c6_done_line": "WAIT robust_waiter.log has no DONE line", "disk": "WAIT disk: [40] GB free (< 41)"}[case]
    assert rc == 0 and stub == [call(p, ref)] and time.time() - t0 >= 2 and log.count(want) == 1, log
    assert log.index(want) < log.index("launch 1:")


@pytest.mark.parametrize("case", ["screens_log_unclean", "e3_other_plan_unclean"])
def test_gives_up_on_a_state_it_cannot_name(tmp_path, case):
    p = home(tmp_path, s_log=X.S_LOG[:2] if case == "screens_log_unclean" else None,
             e3_log=X.E3_PRIOR[:2] if case == "e3_other_plan_unclean" else None)
    k, _, tr = kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])
    want = {"screens_log_unclean": "GIVEUP marks/SCREENS_ROBUSTNESS_K5_DONE exists but the SCREENS log has no clean",
            "e3_other_plan_unclean": "GIVEUP E3's last queue (part3.txt) did not end cleanly"}[case]
    assert (rc, stub) == (1, []) and want in log, log


def test_decide_only_reads(tmp_path):
    p, (k, _, tr) = home(tmp_path), kit(tmp_path)
    rc, out, stub = run(tmp_path, p, k, tr, ["clean"], args=("--decide",))
    assert rc == 0 and stub == [] and out.startswith("LAUNCH robustness_k5 and the C6 robust group ended")
    assert not (p / "logs" / "e3p1b_waiter.log").exists() and not (p / "qcode").exists()


def test_reads_only_what_this_queue_wrote(tmp_path):
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["lockwait"], maxr="0")
    assert rc == 1 and "GIVEUP: 0 relaunches" in log
    rc, log, stub = run(tmp_path, p, k, tr, ["refuse"])
    assert rc == 1 and len(stub) == 2 and "GIVEUP no part1b_p3at20m start line at" in log.splitlines()[-1], log


def test_an_inherited_git_dir_is_never_written(tmp_path, monkeypatch):
    other = tmp_path / "other.git"
    X.git(str(tmp_path), "init", "-q", "--bare", str(other))
    monkeypatch.setenv("GIT_DIR", str(other))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    p, (k, ref, tr) = home(tmp_path), kit(tmp_path)
    rc, log, stub = run(tmp_path, p, k, tr, ["clean"])
    assert rc == 0 and stub == [call(p, ref)]
    refs = subprocess.run(["git", "--git-dir", str(other), "for-each-ref"], capture_output=True, text=True,
                          env=X.clean_env(DEVELOPER_DIR="/Library/Developer/CommandLineTools")).stdout
    assert refs == ""
