"""Shared fixtures of the RC-12 lock tests (test_lock_b.py, test_lock_c.py): a FAILS list, file and git helpers,
a fixture git repository with the work-copy paths, and quiet() to keep library prints out of the test output."""
import contextlib
import io
import os
import subprocess

FAILS = []
GIT = dict(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def dies(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit:
        return True
    return False


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True,
                          env=dict(os.environ, **GIT)).stdout.strip()


def repo(tmp):
    r = os.path.join(tmp, "repo")
    for rel in ("rc12/a.py", "oodh/gate.py", "corpus/oodh.py", "experiments/E001_battery_and_probes/code/x.py",
                "experiments/E004_general_updating/code/y.py"):
        write(r, rel, "X = 1\n")
    write(r, "rc12/fail.py", "import sys\nsys.exit(1)\n")
    write(r, "rc12/mark.py", "open('ran', 'w').write('x')\n")
    write(r, ".gitignore", "sealed/\nprivate/\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "fixture")
    return r


def quiet(fn, *a, **kw):
    """the call's result with its printing kept out of this test's output (only this test prints FAIL lines)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def full_gates(fail=None):
    """a gate table with a row for every S4 gate (fail: the one that fails) and its failing-steps line."""
    import sealed_kit as K
    rows = [f"{n:22s} exit {1 if n == fail else 0:3d} {'FAIL' if n == fail else 'PASS'}  x" for n, _ in K.GATES]
    return "\n".join(["commit x split sealed"] + rows + [f"failing steps: {1 if fail else 0}"]) + "\n"


def report(name):
    for f in FAILS:
        print("FAIL", f)
    print(f"{name}: {'PASS' if not FAILS else f'{len(FAILS)} failures'}")
    return 1 if FAILS else 0
