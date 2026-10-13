"""Mutation check of test_numbers14.py (rc12/notes.txt STEP 14N): each mutant breaks one recorded number, one
text fragment or one rule of rederive14.py in a scratch copy, and test_numbers14.py must fail on it; the unmutated copy
must pass. Mutants that need the raw copies run only when --raw, --panel and --lik are given (else "not run").
  python3 -B mutation_numbers14.py [--raw R --panel P --lik L]   (exit 1: a mutant survived or the baseline failed)"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COPY = ["test_numbers14.py", "rederive14.py", "tables14.py", "notes.txt", "pins.json"]
LOGS = ["extras14_20261011.jsonl", "panel_20261011.jsonl", "lik_rows_20261011.jsonl", "lik_tokcheck_pc.json"]


def jl(name, pred, fn):
    """edit the first JSONL row of logs/<name> that pred accepts (fn returns the new row, or None to drop it)."""
    def go(d):
        p = os.path.join(d, "logs", name)
        rows, done = [json.loads(x) for x in open(p)], False
        out = []
        for r in rows:
            if not done and pred(r):
                done, r = True, fn(r)
            if r is not None:
                out.append(r)
        assert done, name
        open(p, "w").writelines(json.dumps(r, sort_keys=True) + "\n" for r in out)
    return go


def tx(path, old, new):
    def go(d):
        p = os.path.join(d, path)
        s = open(p).read()
        assert old in s, (path, old)
        open(p, "w").write(s.replace(old, new, 1))
    return go


def setk(r, path, v):
    x = r
    for k in path[:-1]:
        x = x[k]
    x[path[-1]] = v
    return r


def is_(kind, model, render=None):
    return lambda r: r["kind"] == kind and r["model"] == model and (render is None or r["render"] == render)


def lik_is(model, render):
    return lambda r: r["model"] == model and r["render"] == render


MUTANTS = [
    ("R off its families", jl("extras14_20261011.jsonl", is_("mean3", "minimind-3", "template"),
                               lambda r: setk(r, ["R"], r["R"] + 0.01)), False),
    ("CI bound moved", jl("extras14_20261011.jsonl", is_("ci", "cRia-LM-75M-Instruct"),
                          lambda r: setk(r, ["ci", "lo"], r["ci"]["lo"] + 0.01)), False),
    ("CI part flag flipped", jl("extras14_20261011.jsonl", is_("ci", "Swen-28M"),
                                lambda r: setk(r, ["ci_part_met"], False)), False),
    ("Base plain row dropped", jl("extras14_20261011.jsonl", is_("mean3", "Falcon-H1-Tiny-90M-Base", "plain"),
                                  lambda r: None), False),
    ("new extra regrades differently", jl("extras14_20261011.jsonl", is_("mean3", "minimind-3", "plain"),
                                          lambda r: setk(r, ["regraded_rows_differing"], 3)), False),
    ("panel comparator R", jl("panel_20261011.jsonl", is_("mean3", "Qwen2.5-0.5B-Instruct", "template"),
                              lambda r: setk(r, ["R"], r["R"] + 1e-6)), False),
    ("lik matched count", jl("lik_rows_20261011.jsonl", lik_is("Qwen/Qwen3-0.6B", "plain"),
                             lambda r: setk(r, ["families", "ROLE", "n_matched"], 34)), False),
    ("lik self-check failed", jl("lik_rows_20261011.jsonl", lik_is("unsloth/gemma-3-270m-it", "template"),
                                 lambda r: setk(r, ["selfcheck", "ok"], False)), False),
    ("lik model dropped", jl("lik_rows_20261011.jsonl", lik_is("LiquidAI/LFM2-700M", "plain"), lambda r: None), False),
    ("lik common foils", jl("lik_rows_20261011.jsonl", lik_is("SmallDoge/Doge-160M-Instruct", "template"),
                            lambda r: setk(r, ["vs_qwen", "TOPIC", "n_common"], 23)), False),
    ("prereg s12 new extra R", tx("prereg/RC-12.draft.txt", "Curriculum R 0.64 (-6.44", "Curriculum R 0.65 (-6.44"),
     False),
    ("prereg s12 old extra CI", tx("prereg/RC-12.draft.txt", "0.87 (-6.20 to -3.41)", "0.87 (-6.21 to -3.41)"), False),
    ("prereg s10b P bound", tx("prereg/RC-12.draft.txt", "P at most 0.049", "P at most 0.021"), False),
    ("prereg s4 LOOKUP row", tx("prereg/RC-12.draft.txt", "Qwen3-0.6B 0.74 against 0.22",
                                "Qwen3-0.6B 0.73 against 0.22"), False),
    ("prereg s12 clause max S7", tx("prereg/RC-12.draft.txt", "added before the lock at most 1.90",
                                    "added before the lock at most 1.89"), False),
    ("notes table cell", tx("rc12/notes.txt", "cRia-75M              1.19  0.75", "cRia-75M              1.19  0.76"),
     False),
    ("rederive: OWN ungated", tx("rc12/rederive14.py", 'and t["unit"] == 1.0 and r["unit"] == 1.0',
                                 'and r["unit"] == 1.0'), True),
    ("rederive: COMPOSE in TWOHOP", tx("rc12/rederive14.py", 'r["cell"] == "COMPOSE"', 'r["cell"] == "none"'), True),
    ("rederive: BIND as the mean", tx("rc12/rederive14.py", '1.0 if us[0] == 1.0 and us[1] == 1.0 else 0.0',
                                      '(us[0] + us[1]) / 2'), True),
    ("rederive: greedy in the mean", tx("rc12/rederive14.py", 'for s in ("1", "2", "3")) for k in NINE',
                                        'for s in ("greedy", "2", "3")) for k in NINE'), True),
    ("rederive: foils by any count", tx("rc12/rederive14.py", 'and n[c] == n[r["gold"]]]', ']'), True),
    ("rederive: chance 1 / foils", tx("rc12/rederive14.py", "avg(1 / (1 + len(x[\"foils\"]))",
                                      "avg(1 / len(x[\"foils\"])"), True),
    ("prereg s12 role-cut share", tx("prereg/RC-12.draft.txt", "0.593 of all", "0.594 of all"), True),
    ("rederive: p10 rank", tx("rc12/rederive14.py", "math.ceil(0.1 * len(ms)) - 1", "math.ceil(0.1 * len(ms))"), True),
]


def run(d, extra):
    p = subprocess.run([sys.executable, "-B", "test_numbers14.py", "--logs", os.path.join(d, "logs"),
                        "--prereg", os.path.join(d, "prereg", "RC-12.draft.txt"),
                        "--notes", os.path.join(d, "rc12", "notes.txt"), "--pins", os.path.join(d, "rc12", "pins.json")]
                       + extra, cwd=os.path.join(d, "rc12"), capture_output=True, text=True, timeout=900)
    first = next((x for x in p.stdout.splitlines() if x.startswith("FAIL")), p.stdout.strip().splitlines()[-1:])
    return p.returncode, first


def fresh(tmp):
    d = tempfile.mkdtemp(dir=tmp)
    os.makedirs(os.path.join(d, "rc12"))
    os.makedirs(os.path.join(d, "logs"))
    os.makedirs(os.path.join(d, "prereg"))
    for f in COPY:
        shutil.copy(os.path.join(HERE, f), os.path.join(d, "rc12", f))
    for f in LOGS:
        shutil.copy(os.path.join(HERE, "logs", f), os.path.join(d, "logs", f))
    src = next((p for p in (os.path.join(HERE, "..", "prereg", f) for f in ("RC-12.md", "RC-12.draft.txt"))
                if os.path.exists(p)), None)       # the locked file once it exists
    shutil.copy(src, os.path.join(d, "prereg", "RC-12.draft.txt"))     # the scratch name the mutants use
    return d


def main():
    ap = argparse.ArgumentParser()
    for a in ("raw", "panel", "lik"):
        ap.add_argument("--" + a)
    a = ap.parse_args()
    raw = all((a.raw, a.panel, a.lik))
    extra = ["--raw", a.raw, "--panel", a.panel, "--lik", a.lik] if raw else []
    tmp = tempfile.mkdtemp()
    try:
        rc, first = run(fresh(tmp), extra)
        print(f"baseline: exit {rc} {first}")
        killed = survived = skipped = 0
        for name, fn, needs_raw in MUTANTS:
            if needs_raw and not raw:
                skipped += 1
                print(f"NOT RUN  {name} (needs --raw, --panel and --lik)")
                continue
            d = fresh(tmp)
            fn(d)
            rc_m, line = run(d, extra)
            ok = rc_m == 1
            killed, survived = killed + ok, survived + (not ok)
            print(f"{'KILLED' if ok else 'SURVIVED'} {name}: exit {rc_m}; {str(line)[:150]}")
        print(f"mutation_numbers14: {killed} / {killed + survived} killed, {skipped} not run, baseline exit {rc}")
        return 0 if rc == 0 and survived == 0 else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
