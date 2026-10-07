"""chain_waiter.sh's loop and launch on a fixture tree and fixture bundles whose queue_screens.sh is a stub (macOS has no
flock or sha256sum: stubs stand in; the PC dry run used the real ones), and base_diag_waiter.sh, the file pcdetach.sh
starts: its arguments against the repo's plans (SCREENS.txt AMENDMENT BASE-DIAG), its sha256 pin of chain_waiter.sh.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_chain_waiter_run.py
"""
import os, re, shlex, shutil, subprocess, sys, threading, time  # noqa: E401

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import privacy, screens_lib as L  # noqa: E401,E402
from test_chain_waiter import ARGS, CLEAN, FAIL, LAST, LOCKWAIT, MID, REF2, REF6, S2, S2END, START, W, tree  # noqa: E402


WR = os.path.join(L.HERE, "base_diag_waiter.sh")
PLANS = os.path.join(L.HERE, "plans")


def git(cwd, *a):
    env = {**os.environ, "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=fixture", "-c", "init.defaultBranch=main",
                           *a], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def bundle(tmp, kit, name, current, first="wait_mark SCREENS SCREENS STAGE 2 S006 SEEDS DONE"):
    """A fixture repo (CURRENT, three plans, a stub queue_screens.sh that records its arguments) as a git bundle."""
    src = tmp / ("src_" + name)
    sx = src / "experiments" / "screens"
    os.makedirs(sx / "plans")
    (sx / "plans" / "CURRENT").write_text(f"# fixture\n{current}\n")
    for plan in ("stage2_seeds", "stage2_s006_seeds", "stage2_base_diag"):
        (sx / "plans" / f"{plan}.txt").write_text(f"# {plan}\n\n{first}\ntrain base_s103\nmark X\n")
    (sx / "queue_screens.sh").write_text('#!/bin/bash\necho "$*" >> "$CW_STUBLOG"\n')
    git(src, "init", "-q")
    git(src, "add", "-A")
    git(src, "commit", "-q", "-m", "fixture")
    os.makedirs(kit, exist_ok=True)
    git(src, "bundle", "create", "-q", str(kit / name), "HEAD", "main")
    return git(src, "rev-parse", "HEAD")


def stubs(tmp, flock_rc="0"):
    os.makedirs(tmp / "bin", exist_ok=True)
    (tmp / "bin" / "flock").write_text(f"#!/bin/sh\nexit {flock_rc}\n")
    (tmp / "bin" / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    for f in ("flock", "sha256sum"):
        os.chmod(tmp / "bin" / f, 0o755)


def main_run(tmp, p, poll="0", grace="2", flock_rc="0", timeout=60, cmd=None, **extra):
    stubs(tmp, flock_rc)
    env = {**os.environ, "CW_HOME": str(p), "CW_KIT": str(tmp / "kit"), "CW_POLL": poll, "CW_GRACE": grace,
           "CW_PAT": "cw_test_matches_no_process_[0-9]{12}", "CW_STUBLOG": str(tmp / "stub.log"),
           "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "DEVELOPER_DIR": "/Library/Developer/CommandLineTools", **extra}
    rc = subprocess.run(cmd or ["/bin/bash", W, *ARGS], env=env, capture_output=True, text=True, timeout=timeout).returncode
    log = (p / "logs" / "base_diag_waiter.log").read_text() if (p / "logs" / "base_diag_waiter.log").exists() else ""
    stub = (tmp / "stub.log").read_text().splitlines() if (tmp / "stub.log").exists() else []
    return rc, log, stub


def test_launches_once_from_its_own_bundle(tmp_path):
    p, kit = tree(tmp_path, MID + LAST + CLEAN), tmp_path / "kit"
    ref = bundle(tmp_path, kit, "planck_diag.bundle", "stage2_base_diag")
    others = [bundle(tmp_path, kit, "planck.bundle", "stage2_seeds"), bundle(tmp_path, kit, "planck_s006.bundle",
                                                                             "stage2_s006_seeds")]
    q = p / "qcode" / ref[:12]
    rc, log, stub = main_run(tmp_path, p)
    assert rc == 0 and stub == [f"--code {q} --plan {q}/experiments/screens/plans/stage2_base_diag.txt"], (log, stub)
    assert git(q, "rev-parse", "HEAD") == ref and all(o != ref and not (p / "qcode" / o[:12]).exists() for o in others)
    assert "LAUNCH stage2_s006_seeds ended cleanly" in log and f"code {ref} in qcode/{ref[:12]}, plan stage2_base_diag" in log
    assert f"launched stage2_base_diag from {ref[:12]}" in (p / "logs" / "base_diag_waiter.launched").read_text()
    rc, log, stub = main_run(tmp_path, p)                               # a second waiter: DONE, no second launch
    assert rc == 0 and len(stub) == 1 and log.rstrip().splitlines()[-1].endswith(
        f"DONE already launched: {(p / 'logs' / 'base_diag_waiter.launched').read_text().strip()}")


@pytest.mark.parametrize("case", ["current_names_another_plan", "plan_waits_on_another_mark", "no_diag_bundle"])
def test_refuses_a_wrong_or_missing_bundle(tmp_path, case):
    p, kit = tree(tmp_path, MID + LAST + CLEAN), tmp_path / "kit"
    bundle(tmp_path, kit, "planck_s006.bundle", "stage2_base_diag")
    if case == "current_names_another_plan":
        bundle(tmp_path, kit, "planck_diag.bundle", "stage2_s006_seeds")
    elif case == "plan_waits_on_another_mark":
        bundle(tmp_path, kit, "planck_diag.bundle", "stage2_base_diag", "wait_mark SCREENS SCREENS STAGE 2 SEEDS DONE")
    rc, log, stub = main_run(tmp_path, p)
    want = {"current_names_another_plan": "names [stage2_s006_seeds], not stage2_base_diag", "no_diag_bundle":
            "GIVEUP: no bundle", "plan_waits_on_another_mark": "starts [wait_mark SCREENS SCREENS STAGE 2 SEEDS DONE]"}
    assert rc == 1 and stub == [] and want[case] in log and not (p / "logs" / "base_diag_waiter.launched").exists(), log


def test_a_second_waiter_holding_no_lock_exits(tmp_path):
    p = tree(tmp_path, MID + LAST + CLEAN)
    bundle(tmp_path, tmp_path / "kit", "planck_diag.bundle", "stage2_base_diag")
    rc, log, stub = main_run(tmp_path, p, flock_rc="1")
    assert rc == 1 and stub == [] and "another base_diag_waiter holds" in log and "LAUNCH" not in log


def test_giveup_exits_without_a_launch(tmp_path):
    p = tree(tmp_path, MID + FAIL)
    bundle(tmp_path, tmp_path / "kit", "planck_diag.bundle", "stage2_base_diag")
    rc, log, stub = main_run(tmp_path, p)
    assert rc == 1 and stub == [] and "GIVEUP stage2_s006_seeds ended another way" in log


def test_bad_arguments_exit_2_with_a_usage_line(tmp_path):
    p = tree(tmp_path, MID)
    rc, log, stub = main_run(tmp_path, p, cmd=["/bin/bash", W, *ARGS[:5]])
    assert rc == 2 and stub == [] and log == "" and "usage: chain_waiter.sh NAME" in (p / "logs" / "chain_waiter.log").read_text()


def test_a_hold_gives_up_after_the_grace(tmp_path):
    p = tree(tmp_path, MID + LOCKWAIT, mark=False)
    t0 = time.time()
    rc, log, stub = main_run(tmp_path, p, poll="1", grace="2", timeout=30)
    assert rc == 1 and stub == [] and 2 <= time.time() - t0 < 25
    assert "HOLD stage2_s006_seeds stopped after a 2 h gpu.lock wait" in log
    assert "GIVEUP: no new stage2_s006_seeds start within 2 s" in log


def append_later(p, lines, after=1.5, mark=True):
    def go():
        time.sleep(after)
        if mark:                               # the mark first: no torn read
            (p / "runs" / "SCREENS" / "marks" / "SCREENS_STAGE_2_S006_SEEDS_DONE").touch()
        with open(p / "runs" / "SCREENS" / "queue_screens.txt", "a") as f:
            f.write("\n".join(lines) + "\n")
    th = threading.Thread(target=go)
    th.start()
    return th


def test_a_hold_then_a_relaunch_and_a_clean_end_launches(tmp_path):
    p = tree(tmp_path, MID + LOCKWAIT, mark=False)
    ref = bundle(tmp_path, tmp_path / "kit", "planck_diag.bundle", "stage2_base_diag")
    th = append_later(p, [START.replace("2026-10-07 00:42:30", "2026-10-07 05:05:00")] + LAST + CLEAN)
    rc, log, stub = main_run(tmp_path, p, poll="1", grace="30", timeout=40)
    th.join()
    assert rc == 0 and len(stub) == 1 and ref[:12] in stub[0] and log.index("HOLD") < log.index("LAUNCH"), log


RUN_ = [START[:20] + "seen SCREENS: SCREENS STAGE 2 SEEDS DONE"]


def test_waits_through_the_earlier_plan_then_launches(tmp_path):
    p = tree(tmp_path, S2 + S2END, mark=False)          # stage2_seeds ended; S006's queue not started yet
    ref = bundle(tmp_path, tmp_path / "kit", "planck_diag.bundle", "stage2_base_diag")
    th = append_later(p, [START] + RUN_ + LAST + CLEAN, after=2.5)
    rc, log, stub = main_run(tmp_path, p, poll="1", grace="1", timeout=40)
    th.join()
    assert rc == 0 and len(stub) == 1 and ref[:12] in stub[0], log
    assert log.index("WAIT the last start line is stage2_seeds at d21df3de0071") < log.index("LAUNCH")


def wrapper_args():
    txt = open(WR).read()
    tail = txt[txt.index('exec bash "$H/chain_waiter.sh"') + len('exec bash "$H/chain_waiter.sh"'):]
    return shlex.split(tail.replace("\\\n", " ")), re.search(r"^SHA=([0-9a-f]{64})$", txt, re.M)[1]


def test_the_wrappers_arguments_are_the_repos_chain():
    a, sha = wrapper_args()
    assert a == ARGS and sha == L.sha256(W) and subprocess.run(["/bin/bash", "-n", WR]).returncode == 0
    s6 = open(os.path.join(PLANS, "stage2_s006_seeds.txt")).read().splitlines()
    nxt = open(os.path.join(PLANS, a[4] + ".txt")).read().splitlines()
    cur = [ln.strip() for ln in open(os.path.join(PLANS, "CURRENT")) if not ln.startswith("#")]
    assert a[1] == "stage2_s006_seeds" and s6[-1] == "mark " + a[3] and nxt[1] == "wait_mark SCREENS " + a[3]
    assert cur == [a[4]] == ["stage2_base_diag"] and a[0] == "base_diag"
    sw = open(os.path.join(L.HERE, "s006_waiter.sh")).read()                    # stage2_seeds at the s006 waiter's ref
    assert f"WANT_PLAN=stage2_seeds; WANT_REF={REF2}; NEXT_PLAN={a[1]}" in sw and a[6] == f"stage2_seeds@{REF2}"
    assert "B=$KIT/planck_s006.bundle" in sw and a[5] == "planck_diag.bundle" and a[2] == REF6
    if shutil.which("git") and os.path.isdir(os.path.join(L.ROOT, ".git")):   # not in an export of the index
        show = lambda path: git(L.ROOT, "show", f"{REF6}:experiments/screens/{path}")  # noqa: E731
        assert [x for x in show("plans/CURRENT").splitlines() if not x.startswith("#")] == [a[1]]
        assert show(f"plans/{a[1]}.txt").splitlines() == s6


@pytest.mark.parametrize("case", ["missing", "edited"])
def test_the_wrapper_refuses_a_chain_waiter_that_is_not_the_pinned_one(tmp_path, case):
    kit = tmp_path / "kit"
    os.makedirs(kit)
    shutil.copy(WR, kit / "base_diag_waiter.sh")
    if case == "edited":
        (kit / "chain_waiter.sh").write_text(open(W).read() + "# edited\n")
    p = tree(tmp_path, MID + LAST + CLEAN)
    rc, log, stub = main_run(tmp_path, p, cmd=["/bin/bash", str(kit / "base_diag_waiter.sh")])
    assert rc == 1 and stub == [] and f"is missing or not sha256 {L.sha256(W)[:16]}" in log, log


def test_the_wrapper_runs_the_pinned_chain_waiter(tmp_path):
    kit = tmp_path / "kit"
    p = tree(tmp_path, MID + LAST + CLEAN)
    ref = bundle(tmp_path, kit, "planck_diag.bundle", "stage2_base_diag")
    for f in (WR, W):
        shutil.copy(f, kit / os.path.basename(f))
    rc, log, stub = main_run(tmp_path, p, cmd=["/bin/bash", str(kit / "base_diag_waiter.sh")])
    assert rc == 0 and len(stub) == 1 and ref[:12] in stub[0] and "earlier: stage2_seeds@d21df3de" in log, log


def test_no_private_identifier_in_the_waiters():
    pats = privacy.patterns()
    for f in (W, WR):
        assert not [x for s, x, rx in pats if s != "generic" and rx.search(open(f).read())], f
        assert not re.search(r"/(?:home|Users)/[a-z]", open(f).read()), f
    code = [re.sub(r"\s+# .*$", "", ln) for ln in open(W) if not ln.lstrip().startswith("#")]   # code, no comments
    assert [ln.strip() for ln in code if "$KIT/" in ln] == ["B=$KIT/$BUNDLE"]      # the one bundle: its argument
    assert not [ln for ln in code if re.search(r"planck(_s006|_diag)?\.bundle", ln)]
    assert subprocess.run(["/bin/bash", "-n", W]).returncode == 0 and S2 and S2END
