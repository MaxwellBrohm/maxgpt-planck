"""Mutation test of rc12/notes.txt STEP 14X (Max's rule: every test claim is watched failing): the shards (shards.py,
dev_batch.py --shard, verify_dev_runs C1), the extras' configs (engines.json, pins.json against test_pins_engines.py)
and dev_extras14.py's model list (test_dev_extras14.py). Each mutant replaces one text in a scratch copy of rc12 (the
*.py and *.json files and dev/), then the three tests run there; KILLED = one of them exits non-zero. The unmutated
copy must pass; MALFORMED = the text is not found exactly once. No model. The queue's shard and run-file mutants are
in mutation_queue.py; the launcher's in extras/mut_rc12x.py.
  python3 -B mutation_step14x.py   (exit 1 unless the baseline passes and every mutant dies)"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # quick_shortcuts' import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
TESTS = ["test_dev_batch_shards.py", "test_pins_engines.py", "test_dev_extras14.py"]
SH, DB, V, E, P, X = "shards.py", "dev_batch.py", "verify_dev_runs.py", "engines.json", "pins.json", "dev_extras14.py"
MUTANTS = [
    (SH, "out.append(parts[k % n][k // n])", "out.append([r for p in parts for r in p][k])", "parts concatenated"),
    (SH, "return recs[i::n]", "return recs[i * len(recs) // n:(i + 1) * len(recs) // n]", "contiguous slices"),
    (SH, 'or [r["id"] for r in rows] != want or bad:', 'or [r["id"] for r in rows] != want:', "config unchecked"),
    (SH, 'shutil.rmtree(final + ".shards", ignore_errors=True)', 'pass', "parts left after the merge"),
    (SH, 'seconds=round(sum(p["seconds"] for p, _ in parts), 1)', 'seconds=parts[0][0]["seconds"]', "seconds of one"),
    (SH, "batches=[b for p, _ in parts for b in p[\"batches\"]]", "batches=parts[0][0][\"batches\"]", "batches of one"),
    (SH, "if n < 2 or not 0 <= i < n:", "if not 0 <= i < n:", "N 1 accepted"),
    (SH, "if type(v) is not int or v < 1:", "if False:", "a string shards value accepted"),
    (DB, "            if SH.all_done(final, nsh):\n                SH.merge(final, nsh, own if own_cf else recs, "
         "publish_run)\n", "            pass\n", "a cut run is never merged"),
    (DB, "(nsh > 1 or shard is not None) and ", "shard is not None and ", "a sharded model runs without --shard"),
    (DB, " if not SH.part_done(t[2], shard[0], nsh)]", "]", "a done part replayed"),
    (DB, "SH.merge(final, nsh, data, publish_run)", "SH.merge(final, nsh, recs, publish_run)", "twin merged on dev"),
    (DB, "        if SH.all_done(final, nsh):\n            SH.merge(final, nsh, data, publish_run)\n", "",
     "the last part does not merge"),
    (V, 'if (m.get("shards") or 1) != nsh:', "if False:", "C1 shards unchecked"),
    (SH, 'meta = dict(p0["config"], model=', 'meta = dict({k: v for k, v in p0["config"].items() if k != "max_batch"}, '
         'model=', "review M13: the merged meta drops max_batch"),
    (E, '"engine": "vllm", "dtype": "float16", "family": "minimind-3"', '"engine": "vllm", "dtype": "bfloat16", '
        '"family": "minimind-3"', "minimind-3 not at its float16"),
    (E, '"family": "minimind-3", "parity": "fp16 NEAR_TIE"', '"family": "Vertex", "parity": "fp16 NEAR_TIE"',
     "minimind-3 given another family"),
    (E, '"shards": 10, "max_batch": 32, ', '"max_batch": 32, ', "cRia not split"),
    (E, '"shards": 10, "max_batch": 32, ', '"shards": 10, ', "cRia without max_batch 32"),
    (P, '"revision": "f92512d4cd6142fa9acc0d6022375049a8974bf6",\n   "license": "apache-2.0"',
     '"revision": "f92512d4cd6142fa9acc0d6022375049a8974bf6",\n   "license": "cc-by-nc-4.0"', "a non-permissive licence"),
    (P, ' "skipped": {\n', ' "skipped": {\n  "jingyaogong/minimind-3": "x",\n', "minimind-3 still skipped"),
    (X, '"Swen-28M", "Veyra2-Mango-30M-Instruct", "minimind-3", ', '"Swen-28M", "Veyra2-Mango-30M-Instruct", ',
     "minimind-3 not scored"),
    (X, '"minimind-3", "cRia-LM-75M-Instruct",', '"cRia-LM-75M-Instruct", "minimind-3",', "not in size order"),
    (X, 'PLAIN_ONLY = {"Falcon-H1-Tiny-90M-Base"}', "PLAIN_ONLY = set()", "the Base scored on template"),
    (X, '"Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO", "Falcon-H1-Tiny-90M-Base"]',
     '"Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO"]', "the Base not scored"),
    (X, 'if r == "plain" or m not in PLAIN_ONLY]', ']', "PLAIN_ONLY ignored by jobs()"),
    (X, "    if card_root:\n        out +=", "    if False:\n        out +=", "the card row dropped"),
]


def run(src=None, old=None, new=None):
    d = tempfile.mkdtemp(prefix="rc12_m14x_")
    for f in os.listdir(HERE):
        if f.endswith(".py") or f.endswith(".json"):
            shutil.copy(os.path.join(HERE, f), d)
    shutil.copytree(os.path.join(HERE, "dev"), os.path.join(d, "dev"))
    try:
        if src:
            text = open(os.path.join(d, src)).read()
            if text.count(old) != 1:
                return "malformed", ""
            open(os.path.join(d, src), "w").write(text.replace(old, new))
        for t in TESTS:
            r = subprocess.run([sys.executable, "-B", t], cwd=d, capture_output=True, text=True, timeout=300, env=ENV)
            if r.returncode != 0:
                lines = [x for x in r.stdout.splitlines() if x.startswith("FAIL")] or r.stderr.splitlines()[-1:]
                return "killed", f"{t}: {lines[0][:120] if lines else r.returncode}"
        return "pass", ""
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    base, _ = run()
    print("baseline:", base, flush=True)
    killed = 0
    for f, old, new, why in MUTANTS:
        v, first = run(f, old, new)
        killed += v == "killed"
        print(f"{'KILLED' if v == 'killed' else v.upper():9s} {f}: {why} | {first}", flush=True)
    print(f"MUTANTS killed {killed}/{len(MUTANTS)}")
    return 0 if base == "pass" and killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
