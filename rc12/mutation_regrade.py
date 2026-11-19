"""Mutation test of dev_regrade.py's durable writes (cuts audit row 143, rc12/notes.txt STEP 14L, 2026-10-11) against
test_dev_regrade.py (Max's rule: every test claim is watched failing). Each mutant replaces one text in a scratch copy
of rc12 (the *.py and *.json files and dev/), then the test runs there; KILLED = it exits non-zero. The unmutated copy
must pass; MALFORMED = the text is not found exactly once. No model.
  python3 -B mutation_regrade.py   (exit 1 unless the baseline passes and every mutant dies)"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = re.compile(r"(?:/private)?" + re.escape(os.path.realpath(tempfile.gettempdir()).replace("/private", "", 1))
                 + r"/[^/\s:']+")
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # text_e004's import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
TESTS = ["test_dev_regrade.py", "test_regrade_expect.py"]    # the second: --expect a2 (lock step A2, 2026-10-19)
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
    # --expect a2 (lock step A2, 2026-10-19; killed by test_regrade_expect.py)
    (R, '"a2_cue", "d4_assert"}', '"a2_cue", "d4_assert", "v1_degen"}', "a2: a degenerate clause may move"),
    (R, 'A2_MOVED = {"v2_gold", ', "A2_MOVED = {", "a2: v2_gold may not move"),
    (R, '"v2_gold", "v3_shotgun", ', '"v2_gold", ', "a2: v3_shotgun may not move (the answer-first waiver)"),
    (R, '"v3_other", "a2_cue"', '"a2_cue"', "a2: v3_other may not move"),
    (R, '"a2_cue", "d4_assert"}', '"d4_assert"}', "a2: a2_cue may not move"),
    (R, ', "d4_assert"}', "}", "a2: d4_assert may not move"),
    (R, 'return set(c["fields"]) <= {"probes", "unit"} and moved <= A2_MOVED',
     'return set(c["fields"]) <= {"probes", "unit", "leaks"} and moved <= A2_MOVED', "a2: the role-leak scan may move"),
    (R, 'return set(c["fields"]) <= {"probes", "unit"} and moved <= A2_MOVED', "return moved <= A2_MOVED",
     "a2: fields not checked"),
    (R, 'if "d0_source" not in set(p["old"]) ^ set(p["new"])\n             for x in set(p["old"]) ^ set(p["new"])}\n'
        '    return set(c["fields"]) <= {"probes", "unit"} and',
     'if True\n             for x in set(p["old"]) ^ set(p["new"])}\n    return set(c["fields"]) <= {"probes", "unit"} and',
     "a2: a flipped OWN source not exempt"),
    (R, '"a2": expected_a2}', '"a2": expected_t0gaps}', "a2: --expect a2 runs another rule"),
    # review r1 (2026-11-19): v2_gold, a2_cue and d4_assert may only be lost (A2_ONLY_LOST; test_regrade_expect)
    (R, " and not gained & A2_ONLY_LOST", "", "a2: a probe may gain v2_gold, a2_cue or d4_assert"),
    (R, 'A2_ONLY_LOST = {"d4_assert", "a2_cue", "v2_gold"}', 'A2_ONLY_LOST = {"a2_cue", "v2_gold"}',
     "a2: d4_assert may be gained"),
    (R, 'A2_ONLY_LOST = {"d4_assert", "a2_cue", "v2_gold"}', 'A2_ONLY_LOST = {"d4_assert", "v2_gold"}',
     "a2: a2_cue may be gained"),
    (R, 'A2_ONLY_LOST = {"d4_assert", "a2_cue", "v2_gold"}', 'A2_ONLY_LOST = {"d4_assert", "a2_cue"}',
     "a2: v2_gold may be gained"),
    (R, '              for x in set(p["new"]) - set(p["old"])}', '              for x in set(p["old"]) - set(p["new"])}',
     "a2: losses read as gains"),
    (R, '    gained = {x for p in c["probes"] if "d0_source" not in set(p["old"]) ^ set(p["new"])\n',
     '    gained = {x for p in c["probes"] if True\n', "a2: a flipped OWN source not exempt from the gain rule"),
    # review r2 (2026-11-19): a change with no changed probe is refused (test_regrade_expect "refuse unit only")
    (R, '    if not c["probes"]:                          # review r2', "    if False:  # review r2",
     "a2: a change with no probe passes"),
    # --expect first and --expect cap (the Nov 3 amendments; tested since 2026-11-19 by test_regrade_expect.py)
    (R, '<= {"v3_other", "v3_shotgun"} and set(p["new"]) <= set(p["old"])', '<= {"v3_other", "v3_shotgun"}',
     "first: a probe may gain a clause"),
    (R, '<= {"v3_other", "v3_shotgun"} and', '<= {"v3_other", "v3_shotgun", "v2_gold"} and', "first: v2_gold may go"),
    (R, '<= {"v3_other", "v3_shotgun"} and', '<= {"v3_other"} and', "first: v3_shotgun may not go"),
    (R, 'return set(c["fields"]) <= {"probes", "unit"} and c["probes"] and all(\n        set(p["old"]) - set(p["new"]) '
        '<= {"v3_other"', 'return set(c["fields"]) <= {"probes", "unit", "flags"} and c["probes"] and all(\n        '
        'set(p["old"]) - set(p["new"]) <= {"v3_other"', "first: a loop flag may move"),
    (R, 'and c["probes"] and all(\n        set(p["old"]) - set(p["new"]) <= {"v3_other"',
     'and all(\n        set(p["old"]) - set(p["new"]) <= {"v3_other"', "first: a change with no probe passes"),
    (R, '"first": expected_first,', '"first": expected_item3,', "first: --expect first runs another rule"),
    (R, '<= CAP_CLAUSES and set(p["new"]) <= set(p["old"]) for p', "<= CAP_CLAUSES for p", "cap: a probe may gain"),
    (R, 'CAP_CLAUSES = {"v1_degen", ', "CAP_CLAUSES = {", "cap: v1_degen may not go"),
    (R, '"d1_degen", "r1_degen", "loop_RUNAWAY"}', '"d1_degen", "r1_degen"}', "cap: loop_RUNAWAY may not go"),
    (R, '"r1_degen", "loop_RUNAWAY"}', '"r1_degen", "loop_RUNAWAY", "v2_gold"}', "cap: v2_gold may go"),
    (R, 'return set(c["fields"]) <= {"probes", "unit"} and c["probes"] and all(\n        set(p["old"]) - set(p["new"]) '
        '<= CAP', 'return set(c["fields"]) <= {"probes", "unit", "flags"} and c["probes"] and all(\n        '
        'set(p["old"]) - set(p["new"]) <= CAP', "cap: a loop flag may move"),
    (R, '"cap": expected_cap,', '"cap": expected_first,', "cap: --expect cap runs another rule"),
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
                why = TMP.sub("<tmp>", lines[0])[:120] if lines else r.returncode     # no machine path in the log
                return "killed", f"{t}: {why}"
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
