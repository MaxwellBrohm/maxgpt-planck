"""oodh/leaks.py on synthetic OASST2-shaped trees (no real item text). Loads no model.
Run: python3 -B oodh/tests/test_leaks.py"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]

import gzip  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import fixture as FX  # noqa: E402
import leaks as LK  # noqa: E402
from test_build import TESTS, main, test, world  # noqa: E402

TESTS.clear()


@test
def leak_list():
    root = os.path.join(world()[0], "leaks")
    os.makedirs(root)
    res, out = FX.RESERVED, FX.OUTSIDE
    turn = "I would like a gentle plan for learning to swim this summer, please."

    def tree(tid, text):
        return dict(message_tree_id=tid, prompt=dict(role="prompter", text=text, replies=[
            dict(role="assistant", text="Sure.", replies=[dict(role="prompter", text=text + " Thanks!")])]))
    trees = [tree(res[0], turn), tree(out[0], turn), tree(res[1], "A different request that is long enough here."),
             tree(res[2], "Another reserved request, also long enough to count."), tree(out[1], "short")]
    gz, dolly = os.path.join(root, "trees.jsonl.gz"), os.path.join(root, "dolly.jsonl")
    with gzip.open(gz, "wt") as fh:
        fh.write("".join(json.dumps(t) + "\n" for t in trees))
    with open(dolly, "w") as fh:
        fh.write(json.dumps(dict(instruction="another reserved request, ALSO long enough to count.")) + "\n")
    assert LK.compute(gz, dolly) == sorted([res[0], res[2]])
    cmd = [sys.executable, "-B", os.path.join(OODH, "leaks.py"), "--trees", gz, "--dolly", dolly, "--out"]
    bad = subprocess.run(cmd + [os.path.join(tempfile.mkdtemp(), "x.txt")], capture_output=True, text=True, timeout=100)
    assert bad.returncode != 0 and "refusing" in bad.stderr
    good = os.path.join(os.path.dirname(os.path.dirname(root)), "leaked.txt")      # .../sealed/leaked.txt
    r = subprocess.run(cmd + [good], capture_output=True, text=True, timeout=100)
    assert r.returncode == 0 and "2 leaked" in r.stdout and LK.load(good) == {res[0], res[2]}, r.stderr


if __name__ == "__main__":
    sys.exit(main())
