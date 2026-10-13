"""Mutation test of dev_regrade.py's durable writes (cuts audit row 143, rc12/notes.txt STEP 14L, 2026-10-11) against
test_dev_regrade.py (Max's rule: every test claim is watched failing). Each mutant replaces one text in a scratch copy
of rc12 (the *.py and *.json files and dev/), then the test runs there; KILLED = it exits non-zero. The unmutated copy
must pass; MALFORMED = the text is not found exactly once. No model.
  python3 -B mutation_regrade.py   (exit 1 unless the baseline passes and every mutant dies)"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # text_e004's import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
TESTS = ["test_dev_regrade.py"]
R = "dev_regrade.py"
MUTANTS = [
    (R, "    DB.publish_run(tmp, out)\n", "    os.replace(tmp, out)\n", "renamed with no fsync"),
    (R, "    if os.path.exists(out):                      # a rerun: the earlier copy goes aside in one rename, so a cut "
        "never\n        os.replace(out, aside)                   # leaves a half-deleted run beside its DONE\n"
        "    DB.publish_run(tmp, out)\n",
     "    shutil.rmtree(out, ignore_errors=True)\n    shutil.copytree(tmp, out)\n", "published by a copy, no rename"),
    (R, "    shutil.rmtree(tmp, ignore_errors=True)       # a cut regrade's leftovers\n", "",
     "a cut .partial not cleared"),
    (R, "    shutil.rmtree(aside, ignore_errors=True)\n    os.makedirs(tmp)\n", "    os.makedirs(tmp)\n",
     "a cut .old.partial not cleared"),
    (R, "        os.replace(out, aside)                   # leaves a half-deleted run beside its DONE\n",
     "        pass\n", "a rerun cannot replace"),
    (R, "        os.replace(out, aside)                   # leaves a half-deleted run beside its DONE\n",
     "        shutil.rmtree(out)\n", "the earlier copy deleted in place (a cut can leave DONE without files)"),
    (R, "    DB.publish_run(tmp, out)\n    shutil.rmtree(aside, ignore_errors=True)\n", "    DB.publish_run(tmp, out)\n",
     "the earlier copy left aside"),
    (R, "        f.write(\"regraded by dev_regrade.py\\n\")\n    if os.path",
     "        f.write(\"regraded by dev_regrade.py\\n\")\n    os.remove(os.path.join(tmp, \"DONE\"))\n    if os.path",
     "no DONE"),
    (R, "    DB.publish_run(tmp, out)\n",
     "    os.remove(os.path.join(tmp, \"DONE\"))\n    DB.publish_run(tmp, out)\n"
     "    open(os.path.join(out, \"DONE\"), \"w\").write(\"regraded by dev_regrade.py\\n\")\n",
     "DONE written after the rename"),
    (R, "    DB.fsync_path(path + \".tmp\")\n", "", "regrade.json not fsynced"),
    (R, "    if os.path.exists(stale):", "    if False:", "an earlier regrade.json kept while a rerun runs"),
    (R, "        os.remove(stale)                         # describing a --dst that is half redone\n"
        "        DB.fsync_path(dst, is_dir=True)\n",
     "        os.remove(stale)                         # describing a --dst that is half redone\n",
     "the earlier regrade.json's removal not fsynced"),
    (R, "    os.replace(path + \".tmp\", path)\n    DB.fsync_path(dst, is_dir=True)\n",
     "    os.replace(path + \".tmp\", path)\n", "--dst not fsynced after regrade.json's rename"),
    (R, "    with open(path + \".tmp\", \"w\") as f:", "    with open(path, \"w\") as f:", "regrade.json in place"),
    (R, "if not x.endswith(\".partial\") and os.path.exists", "if os.path.exists", "a .partial source read"),
    (R, "shutil.copy2(os.path.join(src, rel, \"meta.json\"), os.path.join(tmp, \"meta.json\"))",
     "open(os.path.join(tmp, \"meta.json\"), \"w\").write(\"{}\")", "meta.json not the source's"),
    (R, "        f.writelines(json.dumps(x) + \"\\n\" for x in S.score_lines(new_rows))",
     "        f.writelines(json.dumps(x) + \"\\n\" for x in S.score_lines(new_rows)[1:])", "a score line lost"),
]


def run(src=None, old=None, new=None):
    d = tempfile.mkdtemp(prefix="rc12_mrg_")
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
