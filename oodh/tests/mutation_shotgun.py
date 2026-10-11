"""Mutation test of test_shotgun.py (rc12/notes.txt STEP 14S; Max's rule: every test claim is watched failing). Each
mutant replaces one exact text in a scratch copy of rc12/*.py, rc12/dev/, oodh/ and corpus/oodh.py, then runs
oodh/tests/test_shotgun.py there. KILLED = the test exits non-zero. The unmutated copy must pass; malformed = the text
is not found exactly once. No model, no sealed file.
  python3 -B oodh/tests/mutation_shotgun.py     (exit 1 unless the baseline passes and every mutant is killed)"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
G = "rc12/graders.py"
E004 = os.path.join(REPO, "experiments", "E004_general_updating", "code")    # text_e004 (rc12/quick_shortcuts.py)
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
MUTANTS = [
    (G, "shotgun_applies(probe) and len(cands) >= 2", "not probe.get(\"pool_values\") and len(cands) >= 2",
     "the old rule: clause off whenever pool_values is set (OOD-H)"),
    (G, 'return not str(probe.get("pool") or "").startswith("compose_")', "return True", "COMPOSE not exempt"),
    (G, 'return not str(probe.get("pool") or "").startswith("compose_")', "return False", "every probe exempt"),
    (G, '.startswith("compose_")', '.endswith("compose_")', "exemption matches nothing"),
    (G, "shotgun_applies(probe) and len(cands) >= 2", "shotgun_applies(probe) and len(cands) >= 1",
     "fires on one candidate"),
    (G, "all(shotgun_named(text, v, stale, gold_ok) for v in cands)",
     "any(shotgun_named(text, v, stale, gold_ok) for v in cands)", "fires when any candidate is named"),
    # review D1 (notes STEP 14S): a stale value named only in a change phrase, the gold asserted, is not counted
    (G, "if hits and \"stale\" not in T.OFF and v in stale and gold_ok and all(", "if False and all(",
     "D1: a change-phrase stale value counted (the H-CORR right answer fails)"),
    (G, "and v in stale and gold_ok and all(stale_ok", "and v in stale and all(stale_ok", "D1: the gold need not be asserted"),
    (G, "and v in stale and gold_ok and all(stale_ok", "and gold_ok and all(stale_ok", "D1: any candidate exempt, not only stale"),
    (G, "gold_ok and all(stale_ok(text, m) for m in hits)", "gold_ok and any(stale_ok(text, m) for m in hits)",
     "D1: one change-phrase hit exempts a bare one too"),
    (G, "    return bool(hits)\n", "    return True\n", "D1: an unnamed candidate counted"),
    (G, 'fails.append("v3_shotgun")', "pass", "clause never recorded"),
]


def copy_tree(dst):
    os.makedirs(os.path.join(dst, "rc12"))
    for f in glob.glob(os.path.join(REPO, "rc12", "*.py")) + glob.glob(os.path.join(REPO, "rc12", "*.json")):
        shutil.copy(f, os.path.join(dst, "rc12"))
    shutil.copytree(os.path.join(REPO, "rc12", "dev"), os.path.join(dst, "rc12", "dev"))
    shutil.copytree(os.path.join(REPO, "oodh"), os.path.join(dst, "oodh"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(dst, "corpus"))
    shutil.copy(os.path.join(REPO, "corpus", "oodh.py"), os.path.join(dst, "corpus"))


def run(dst):
    r = subprocess.run([sys.executable, "-B", os.path.join(dst, "oodh", "tests", "test_shotgun.py")],
                       capture_output=True, text=True, timeout=300, cwd=dst, env=ENV)
    return r.returncode, r.stdout + r.stderr


def main():
    base = tempfile.mkdtemp(prefix="mut_shotgun_")
    clean = os.path.join(base, "clean")
    copy_tree(clean)
    code, out = run(clean)
    print(f"baseline: exit {code}")
    if code != 0:
        print(out)
        return 1
    survived = malformed = 0
    for k, (rel, old, new, why) in enumerate(MUTANTS, 1):
        dst = os.path.join(base, f"m{k}")
        copy_tree(dst)
        path = os.path.join(dst, rel)
        text = open(path).read()
        if text.count(old) != 1:
            malformed += 1
            print(f"#{k} MALFORMED ({text.count(old)} matches): {why}")
            continue
        open(path, "w").write(text.replace(old, new))
        code, out = run(dst)
        fails = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
        verdict = "KILLED" if code != 0 else "SURVIVED"
        survived += code == 0
        print(f"#{k} {verdict}: {why}" + (f"  [{fails[0][:110]}]" if fails else ""))
    shutil.rmtree(base, ignore_errors=True)
    n = len(MUTANTS)
    print(f"{n - survived - malformed} of {n} killed, {survived} survived, {malformed} malformed")
    return 0 if survived == 0 and malformed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
