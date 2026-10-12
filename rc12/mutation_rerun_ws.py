"""Mutation test of test_rerun_wrong_seed.py (rc12/notes.txt STEP 14S; every test claim watched failing). Each mutant
replaces one exact text in a scratch copy of rc12 (*.py, dev/, engines.json), then the test runs there. KILLED = the
test exits non-zero. The unmutated copy must pass; malformed = the text is not found exactly once. No model.
  python3 -B mutation_rerun_ws.py      (exit 1 unless the baseline passes and every mutant is killed)"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.join(os.path.dirname(HERE), "experiments", "E004_general_updating", "code")
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
RT, FF = "rerun_tie.py", "fakes_family.py"
MUTANTS = [
    (RT, 'if getattr(a, "wrong_seed", False):', "if False:", "--wrong-seed ignored"),
    (RT, 'if getattr(a, "wrong_seed", False):', "if True:", "R1w on by default"),
    (RT, "**divergences(other, stored)))", "**divergences(stored, stored)))",
     "R1w compares the stored rows to themselves"),
    (RT, 'tag=f"R1w seed {seed + 100} (wrong seed) replay vs stored seed {seed}"',
     'tag=f"R2w seed {seed + 100} (wrong seed) replay vs stored seed {seed}"', "R1w judged as a gate"),
    (RT, 'ws = ["--wrong-seed"] if getattr(args, "wrong_seed", False) else []', 'ws = ["--wrong-seed"]',
     "verify_dev_runs --rerun gains R1w"),
    (RT, '''    for line in rep["info"]:
        say("INFO", line)''', "    pass", "report drops the INFO lines"),
    (RT, "judge(c, ms[i:i + len(c[\"divergences\"])], say)", "pass", "report prints no verdicts"),
    (RT, "return 1 if bad else 0", "return 0", "report hides a FAIL"),
    (FF, 'return out + (f" Draw {h % 997}." if h % 2 else "")', "return out", "SEEDED ignores the seed"),
]


def copy_tree(dst):
    os.makedirs(dst)
    for f in os.listdir(HERE):
        if f.endswith(".py") or f == "engines.json":
            shutil.copy(os.path.join(HERE, f), dst)
    shutil.copytree(os.path.join(HERE, "dev"), os.path.join(dst, "dev"))


def run(dst):
    r = subprocess.run([sys.executable, "-B", "test_rerun_wrong_seed.py"], cwd=dst, env=ENV, capture_output=True,
                       text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


def main():
    base = tempfile.mkdtemp(prefix="mut_rerun_ws_")
    try:
        copy_tree(os.path.join(base, "clean"))
        code, out = run(os.path.join(base, "clean"))
        print(f"baseline: exit {code}")
        if code != 0:
            print(out[-1500:])
            return 1
        survived = malformed = 0
        for k, (rel, old, new, why) in enumerate(MUTANTS, 1):
            dst = os.path.join(base, f"m{k}")
            copy_tree(dst)
            text = open(os.path.join(dst, rel)).read()
            if text.count(old) != 1:
                malformed += 1
                print(f"#{k} MALFORMED ({text.count(old)} matches): {why}")
                continue
            open(os.path.join(dst, rel), "w").write(text.replace(old, new))
            code, out = run(dst)
            fails = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
            survived += code == 0
            print(f"#{k} {'KILLED' if code else 'SURVIVED'}: {why}" + (f"  [{fails[0][:100]}]" if fails else ""))
        n = len(MUTANTS)
        print(f"{n - survived - malformed} of {n} killed, {survived} survived, {malformed} malformed")
        return 0 if survived == 0 and malformed == 0 else 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
