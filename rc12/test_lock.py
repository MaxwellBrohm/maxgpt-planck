"""RC-12 lock tests, part 1: the split configuration, the content hook, the dev bytes, test_gens' RC12_EXPECT hook
and the S1 separation check (rc12/notes.txt STEP 15). Stdlib, no model; fixtures in a temporary folder.
  python3 -B test_lock.py     (exit 1 on any failure; part 2 is test_lock_b.py)"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("RC12_SPLIT", None)
import sealed_overlap as SO                  # noqa: E402
import split_content as SPC                  # noqa: E402

PY = sys.executable
DEV_SHA = "3e543f8d5638e6ab9fd106099fc2584e810dd90544f212146a3cd73bd66689a0"
SEED = "ab" * 16
FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def run(argv, cwd=HERE, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("RC12_")}
    e.update(env)
    return subprocess.run([PY, "-B", *argv], cwd=cwd, env=e, capture_output=True, text=True, timeout=600)


def t_shape():
    d = SPC.shape_diff
    check(d({"a": ["x {v}", "y"]}, {"a": ["p {v}", "q"]}, "m") is None, "shape: same shape refused")
    check(d("x {v}", "x {w}", "m") is not None, "shape: a changed placeholder passes")
    check(d(["a", "b"], ["a"], "m") is not None, "shape: a shorter list passes")
    check(d({"a": 1}, {"b": 1}, "m") is not None, "shape: a renamed key passes")
    check(d({"a": "x"}, {"b": "y"}, "m", free_keys=True) is None, "shape: free keys refused")
    check(d(("a", "b"), ["a", "b"], "m") is not None, "shape: tuple to list passes")
    check(d({"n": 3}, {"n": 4}, "m") is not None, "shape: a changed number passes")
    check(d({"a": ("x", "{k}")}, {"a": ("x", "{j}")}, "m") is not None, "shape: a nested placeholder passes")
    dev = {"door": {"vtype": "colour", "q": "a"}, "day": {"vtype": "weekday", "q": "b"}}
    moved = {"day": {"vtype": "weekday", "q": "y"}, "door": {"vtype": "colour", "q": "x"}}
    check(SPC.pinned("fam_twohop", "TWOHOP_SCHEMAS", dev, moved) is None, "pins: reordered keys refused")
    moved["day"]["vtype"] = "month"
    check(SPC.pinned("fam_twohop", "TWOHOP_SCHEMAS", dev, moved) is not None, "pins: a changed vtype passes")


TOY = '''import split_content as SC
WORDS = ["a {v}", "b"]
TABLE = {"k": "x {v}"}
CONTENT = ("WORDS", "TABLE")
SC.apply(globals(), CONTENT)
DERIVED = WORDS + ["z"]
'''


def toy(content, split="dryrun"):
    """runs the toy module with a content file (None: none); returns (exit code, DERIVED or stderr)."""
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "toymod.py"), "w").write(TOY)
        if content is not None:
            open(os.path.join(d, "content_toymod.py"), "w").write(content)
        code = f"import sys; sys.path[:0] = [{d!r}, {HERE!r}]; import toymod; print(toymod.DERIVED, toymod.TABLE)"
        env = {"RC12_SPLIT": split, "RC12_GEN_SEED": SEED} if split != "dev" else {}
        p = run(["-c", code], cwd=d, **env)
        return p.returncode, (p.stdout.strip() if p.returncode == 0 else p.stderr.strip()[-200:])
    finally:
        shutil.rmtree(d)


def t_apply():
    good = 'WORDS = ["c {v}", "d"]\nTABLE = {"k": "y {v}"}\n'
    rc, out = toy(good)
    check(rc == 0 and out == "['c {v}', 'd', 'z'] {'k': 'y {v}'}", f"apply: content not applied before DERIVED: {out}")
    rc, out = toy(good, split="dev")
    check(rc != 0 and "never graded with dev pools" in out, f"apply: dev ran beside a content file: {out}")
    rc, out = toy(None, split="dev")
    check(rc == 0 and out.startswith("['a {v}', 'b', 'z']"), f"apply: dev changed without a content file: {out}")
    rc, out = toy("import os\n" + good)
    check(rc != 0 and "only NAME = <literal>" in out, f"apply: a content file with code runs: {out}")
    rc, out = toy('WORDS = ["c {v}"] + ["d"]\nTABLE = {"k": "y {v}"}\n')
    check(rc != 0 and "not a plain literal" in out, f"apply: a computed content value passes: {out}")
    rc, out = toy(None)
    check(rc != 0 and "missing" in out, f"apply: a missing content module passes: {out}")
    rc, out = toy(good + 'EXTRA = 1\n')
    check(rc != 0 and "extra" in out, f"apply: an extra name passes: {out}")
    rc, out = toy('WORDS = ["c {w}", "d"]\nTABLE = {"k": "y {v}"}\n')
    check(rc != 0 and "fields" in out, f"apply: a changed placeholder passes: {out}")
    rc, out = toy('WORDS = ["c {v}", "d"]\n')
    check(rc != 0 and "lacks" in out, f"apply: a missing name passes: {out}")


def t_split_env():
    p = run(["-c", "import split_content"], RC12_SPLIT="dryrun", RC12_GEN_SEED="1234")
    check(p.returncode != 0, "split: a short seed accepted")
    p = run(["-c", "import split_content"], RC12_SPLIT="Dry-1", RC12_GEN_SEED=SEED)
    check(p.returncode != 0, "split: a bad split name accepted")
    p = run(["-c", "import common as C; print(C.SPLIT, C.SEED)"])
    check(p.stdout.strip() == "dev 1212", f"split: dev defaults wrong: {p.stdout}")
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "x.jsonl")
        open(path, "w").write('{"split": "sealed", "family": "T0"}\n')
        p = run(["-c", f"import runner; runner.load({path!r})"])
        check(p.returncode != 0 and "records under RC12_SPLIT=dev" in p.stderr, "split: runner loads sealed records "
              "under the dev split")
    finally:
        shutil.rmtree(d)


def t_dev_bytes():
    p = run(["build_dev.py", "--stdout"])
    check(hashlib.sha256(p.stdout.encode()).hexdigest() == DEV_SHA, "dev: the build is not byte-identical (3e543f8d)")
    p = run(["build_dev.py", "--stdout"], RC12_SPLIT="dryrun", RC12_GEN_SEED=SEED)
    check(p.returncode != 0, "dev: a non-dev split built in the repository")


def t_content_propagates():
    """a scratch work copy with dev content but 5 changed constants: each reaches the items or its derived users."""
    d = tempfile.mkdtemp()
    try:
        shutil.copytree(HERE, os.path.join(d, "rc12"), ignore=shutil.ignore_patterns("runs", "logs", "__pycache__"))
        e004 = os.path.join("experiments", "E004_general_updating", "code")      # balance.py imports text_e004
        shutil.copytree(os.path.join(HERE, "..", e004), os.path.join(d, e004),
                        ignore=shutil.ignore_patterns("__pycache__"))
        SO.skeleton(os.path.join(d, "rc12"))
        p = run(["build_dev.py", "--stdout"], cwd=os.path.join(d, "rc12"), RC12_SPLIT="dryrun", RC12_GEN_SEED=SEED)
        check(p.returncode != 0 and "work copy" in p.stdout + p.stderr, "propagate: built without the work-copy marker")
        open(os.path.join(d, ".rc12_workcopy"), "w").write("commit test\n")
        p = run(["build_dev.py", "--stdout"], cwd=os.path.join(d, "rc12"))
        check(p.returncode != 0 and "never graded with dev pools" in p.stdout + p.stderr,
              "propagate: the dev split ran inside a work copy")
        slots = os.path.join(d, "rc12", "content_pools_slots.py")
        keep = open(slots).read()
        open(slots, "w").write(keep.replace("'vtype': 'colour'", "'vtype': 'food'", 1))
        p = run(["build_dev.py", "--stdout"], cwd=os.path.join(d, "rc12"), RC12_SPLIT="dryrun", RC12_GEN_SEED=SEED)
        check(p.returncode != 0 and "pinned to dev's" in p.stdout + p.stderr, "propagate: a pinned vtype changed")
        open(slots, "w").write(keep)
        def sub(m, old, new):
            p = os.path.join(d, "rc12", f"content_{m}.py")
            s = open(p).read()
            check(s.count(old) == 1, f"propagate: fixture text {old!r} not found once in content_{m}")
            open(p, "w").write(s.replace(old, new))
        sub("fam_role_lookup", "ROW = '{k}: {v}'", "ROW = '{k} = {v}'")
        sub("pools_vals", "'teal'", "'zaffre'")
        sub("pools_vals", "'violin'", "'zither'")
        sub("fam_persist", "'groovy'", "'zesty'")
        sub("pools_vals", "'teammate'", "'bandmate'")
        sub("pools_slots", "'Remind me, what color is my bike?'", "'Tell me again, what color is my bike?'")
        env = dict(RC12_SPLIT="dryrun", RC12_GEN_SEED=SEED)
        code = ("import pools_vals as V, grade_text as T, grade_fmt_dyn as F; print('zaffre' in V.ALL_VALUES, "
                "'teal' in V.ALL_VALUES, 'zaffre' in V.POOLS['colour'], 'zither' in T.INFLECT, "
                "'violin' in T.INFLECT, 'zesty' in F.STUFF_POOL, 'bandmate' in V.HOLDERS, "
                "'Tell me again, what color is my bike?' in __import__('pools_slots').SLOT['bike']['q'])")
        p = run(["-c", code], cwd=os.path.join(d, "rc12"), **env)
        check(p.stdout.strip() == "True False True True False True True True",
              f"propagate: derived values stale: {p.stdout}{p.stderr[-300:]}")
        p = run(["build_dev.py", "--stdout"], cwd=os.path.join(d, "rc12"), **env)
        recs = [json.loads(x) for x in p.stdout.splitlines()]
        check(len(recs) == 640, f"propagate: build gave {len(recs)} records: {p.stderr[-300:]}")
        tables = [t["text"] for r in recs if r["family"] == "LOOKUP" for t in r["turns"] if t["kind"] == "T"]
        check(tables and all(" = " in t for t in tables), "propagate: LOOKUP rows do not use the content ROW")
        check(all(r["split"] == "dryrun" and r["id"].startswith("rc12-dryrun-") and r["seed"] == SEED for r in recs),
              "propagate: records do not carry the split, its ids and the seed")
        other = run(["build_dev.py", "--stdout"], cwd=os.path.join(d, "rc12"), RC12_SPLIT="dryruntwo",
                    RC12_GEN_SEED=SEED)
        texts = lambda out: [[t["text"] for t in json.loads(x)["turns"]] for x in out.splitlines()]
        check(texts(other.stdout) != texts(p.stdout), "propagate: the split name does not reach the random streams")
        dev = open(os.path.join(HERE, "dev", "rc12_dev.jsonl")).read().splitlines()
        check(sum(a == json.dumps(r, ensure_ascii=False, sort_keys=True) for a, r in zip(dev, recs)) == 0,
              "propagate: a record equals dev's")
    finally:
        shutil.rmtree(d)


def t_expect_hook():
    import test_gens as T
    recs = [json.loads(x) for x in open(os.path.join(HERE, "dev", "rc12_dev.jsonl"))]
    for r in recs:
        r["split"] = "sealed"
    d = tempfile.mkdtemp()
    try:
        for want, env_file, ok in ((None, False, False), ([3106, 904], True, True), ([3106, 903], True, False)):
            T.FAILS.clear()
            os.environ.pop("RC12_EXPECT", None)
            if env_file:
                path = os.path.join(d, "e.json")
                json.dump({"filler_labels": {"question": 50, "request": 20, "statement": 20}, "asks": want},
                          open(path, "w"))
                os.environ["RC12_EXPECT"] = path
            T.test_asks(recs)
            check((not T.FAILS) == ok, f"expect hook: asks {want} gave {T.FAILS[:2]}")
    finally:
        os.environ.pop("RC12_EXPECT", None)
        shutil.rmtree(d)


def main():
    for t in (t_shape, t_apply, t_split_env, t_dev_bytes, t_content_propagates, t_expect_hook):
        t()
    for f in FAILS:
        print("FAIL", f)
    print(f"test_lock: {'PASS' if not FAILS else f'{len(FAILS)} failures'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
