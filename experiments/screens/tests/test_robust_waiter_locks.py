"""robust_waiter.sh's two single-instance locks and its default E3 process pattern (SCREENS.txt AMENDMENT SEEDS-103-105,
STARTING IT UNATTENDED; added 2026-10-12 after the independent check found that test_robust_waiter.py stubs flock to
"always free" and overrides the E3 pattern, so neither was tested). Here flock is a stub with real flock(2) semantics:
it locks the descriptor it inherits from the waiter (the lock stays with that open file until the waiter closes it), so
a lock held by this test is seen as held. The default pattern is read from the file and run through pgrep -f against
real processes. Fixtures from test_robust_waiter.py; every git call without git's location variables. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_robust_waiter_locks.py
"""
import fcntl, os, re, subprocess, sys, threading, time, uuid  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import test_robust_waiter as TW  # noqa: E402

FLOCK = f"""#!{sys.executable}
import fcntl, sys
a = sys.argv[1:]
if len(a) != 2 or a[0] != "-n" or not a[1].isdigit():
    sys.exit("flock stub: only 'flock -n FD' is supported, got %r" % (a,))
try:
    fcntl.flock(int(a[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
"""
E3_LINES = {   # the command lines the PC's E3 processes show (queue_e3.sh and launch_e3.sh exec queue_e2.sh --exp E3)
    "queue_e2": "bash /h/planck/qcode/4962426b8f0d/experiments/E2_lr_transfer/queue_e2.sh --exp E3 --code /h/q --plan p",
    "queue_e3": "bash /h/planck/qcode/4962426b8f0d/experiments/E3_seed_noise/queue_e3.sh --plan part3.txt",
    "launch_e3": "bash /h/planck/repo/experiments/E3_seed_noise/launch_e3.sh",
    "e3p3_waiter": "/bin/bash /k/planck-kit/e3p3_waiter.sh",
}
NOT_E3 = ["bash /h/q/experiments/E2_lr_transfer/queue_e2.sh --exp E2 --code /h/q --plan p",
          "bash /h/q/experiments/screens/queue_screens.sh --code /h/q --plan robustness_k5.txt",
          "/bin/bash /k/planck-kit/robust_waiter.sh", "/bin/bash /k/planck-kit/chain_waiter.sh"]


ALWAYS_FREE_STUBS = TW.stubs


def real_flock(tmp):
    ALWAYS_FREE_STUBS(tmp)                           # sha256sum, and an always-free flock replaced here
    (tmp / "bin" / "flock").write_text(FLOCK)
    os.chmod(tmp / "bin" / "flock", 0o755)


def run_locked(tmp, p, k, outcomes, held):
    """TW.run with the real-semantics flock and this test holding logs/<held> for the whole run."""
    fd = os.open(p / "logs" / held, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if held else None
    try:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        orig = TW.stubs
        TW.stubs = real_flock
        try:
            return TW.run(tmp, p, k, outcomes, timeout=30)
        finally:
            TW.stubs = orig
    finally:
        if fd is not None:
            os.close(fd)


def test_with_no_lock_held_the_real_flock_stub_lets_it_run(tmp_path):
    p, (k, ref) = TW.tree(tmp_path), TW.kit(tmp_path)
    rc, log, stub = run_locked(tmp_path, p, k, ["clean"], None)
    assert rc == 0 and stub == [TW.qcall(p, ref), TW.dcall(p, ref)] and "DONE: robustness_k5" in log, (log, stub)


def test_a_second_waiter_exits_at_robust_waiter_all_lock(tmp_path):
    p, (k, _) = TW.tree(tmp_path), TW.kit(tmp_path)
    rc, log, stub = run_locked(tmp_path, p, k, ["clean"], "robust_waiter_all.lock")
    assert rc == 1 and stub == [] and "another robust_waiter holds" in log and "step 0" not in log, (log, stub)


def test_a_second_watcher_exits_at_robust_requeue_lock_before_the_c6_group(tmp_path):
    p, (k, ref) = TW.tree(tmp_path), TW.kit(tmp_path)
    rc, log, stub = run_locked(tmp_path, p, k, ["clean"], "robust_requeue.lock")
    assert rc == 1 and stub == [TW.qcall(p, ref)], (log, stub)
    assert "another robust_waiter watches robustness_k5: exiting" in log and "step 3" not in log, log


def default_epat():
    m = re.search(r"^EPAT=\$\{CW_EPAT:-'([^']+)'\}$", open(TW.RW).read(), re.M)
    assert m, "robust_waiter.sh has no default E3 pattern line"
    return m[1]


@pytest.mark.parametrize("name", sorted(E3_LINES))
def test_the_default_pattern_finds_each_e3_process(name):
    tok = uuid.uuid4().hex[:12]
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", E3_LINES[name] + " " + tok])
    try:
        time.sleep(0.3)
        out = subprocess.run(["pgrep", "-f", default_epat()], capture_output=True, text=True).stdout.split()
        assert str(proc.pid) in out, (name, out)
    finally:
        proc.kill()


def test_the_default_pattern_finds_no_other_queue():
    rx = re.compile(default_epat())                  # ERE with no ERE-only syntax: Python's re reads it the same
    assert [ln for ln in E3_LINES.values() if rx.search(ln)] == list(E3_LINES.values())
    assert [ln for ln in NOT_E3 if rx.search(ln)] == []


def test_with_the_default_pattern_it_waits_while_an_e3p3_waiter_runs(tmp_path, monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                             E3_LINES["e3p3_waiter"] + " " + uuid.uuid4().hex[:12]])
    p, (k, ref) = TW.tree(tmp_path), TW.kit(tmp_path)
    real_run = subprocess.run

    def no_epat(cmd, *a, **kw):                      # TW.run sets CW_EPAT; drop it so the file's default is used
        if kw.get("env") and "CW_EPAT" in kw["env"]:
            kw["env"] = {x: v for x, v in kw["env"].items() if x != "CW_EPAT"}
        return real_run(cmd, *a, **kw)
    monkeypatch.setattr(subprocess, "run", no_epat)
    killer = threading.Timer(3.0, proc.kill)
    killer.start()
    try:
        t0 = time.time()
        rc, log, stub = TW.run(tmp_path, p, k, ["clean"])
    finally:
        killer.cancel()
        proc.kill()
    assert rc == 0 and time.time() - t0 >= 2.5 and stub == [TW.qcall(p, ref), TW.dcall(p, ref)], (log, stub)
    assert log.index("step 0: WAIT an E3 queue, launcher or waiter runs") < log.index("step 0: OK E3's last queue")
