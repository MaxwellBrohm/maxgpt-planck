"""Mutation test of the OOD-H wording rule (Max's rule: every test claim is watched failing). Each mutant replaces one
exact text in a scratch copy (rc12/*.py and prereg/RC-12.draft.txt, the file the s1 wording test reads); then
test_oodh_wording.py and test_oodh_pairing.py (STEP 11 FIX ROUND) run there. KILLED = either test exits 1 (its own
FAIL lines; an exception inside a test is reported as a FAIL). The unmutated copy must pass both; malformed = the
text is not found exactly once. No model.
  python3 -B mutation_oodh_wording.py     (writes logs/mutation_oodh_wording.txt; exit 1 unless all are killed)"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(os.path.dirname(HERE), "prereg", "RC-12.draft.txt")
O, SS, T = "oodh_wording.py", "score_stats.py", ("test_oodh_wording.py", "test_oodh_pairing.py")
MUTANTS = [
    (O, "OODH_MARGIN = -5.0", "OODH_MARGIN = -3.0", "bar at -3 points"),
    (O, "OODH_MARGIN = -5.0", "OODH_MARGIN = -7.0", "bar at -7 points"),
    (O, 'd = round(ci["D"], DECIMALS)', 'd = ci["D"]', "float noise decides at exactly -5"),
    (O, "qualified = d < OODH_MARGIN", "qualified = d <= OODH_MARGIN", "exactly -5 counted as worse"),
    (O, "diffs.append(_level(aa, draw, rng) - _level(ab, draw, rng))",
     "diffs.append(_level(aa, draw, rng) - _level(ab, [rng.randrange(m) for _ in range(m)], rng))",
     "unpaired thread draws"),
    (O, "lo=diffs[int(0.025 * n)]", "lo=diffs[0]", "CI lower bound at the minimum"),
    (O, "hi=diffs[min(n - 1, int(0.975 * n))]", "hi=diffs[n - 1]", "CI upper bound at the maximum"),
    (O, "point = _level(aa, idx) - _level(ab, idx)", "point = _level(ab, idx) - _level(aa, idx)",
     "comparator minus Planck"),
    (O, "per_train.append(_mean([_mean([arrs[tr][sd][i] for i in idx]) for sd in seeds]))",
     "per_train += [_mean([arrs[tr][sd][i] for i in idx]) for sd in seeds]", "flat mean over runs, not nested"),
    (O, "for r in S.select(rows):", "for r in rows:", "greedy rows scored beside sampled ones"),
    (O, "if s != union:", "if False:", "unequal thread sets accepted"),
    (O, "if expected is not None and union != set(expected):", "if False:", "item file ids not checked"),
    (O, 'if r["id"] in run:', "if False:", "duplicate thread accepted"),
    (O, 'if r.get("family") != "OODH":', "if False:", "non-OODH rows accepted"),
    (O, 'if r.get("own_cf"):', "if False:", "--own-cf rows accepted"),
    (O, "or not 0 <= u <= 1:", ":", "units outside [0, 1] accepted"),
    (O, "if isinstance(u, bool) or not", "if not", "a bool unit accepted"),
    (O, "if len(renders) != 1:", "if False:", "two renders in one model accepted"),
    (O, "if ra != rb:", "if False:", "renders differing between the models accepted"),
    (O, "provisional = not part2_complete", "provisional = False", "never provisional"),
    (O, "text = text.replace(UNQUALIFIED, QUALIFIED)", "text = text", "qualified text keeps the unqualified wording"),
    (O, "if (float(s) < OODH_MARGIN) == qualified:", "if True:", "shown D on the other side of the bar"),
    (O, 'if dec["decision"] == "qualified" and bare:', "if False:", "unqualified wording never refused"),
    (O, 'if dec["oodh_sentence"] not in text:', "if False:", "claim without the OOD-H result accepted"),
    (O, 'if dec["provisional"] and PROVISIONAL not in text:', "if False:", "provisional note never required"),
    (O, "sealed 12-turn conversation test.", "sealed 11-turn conversation test.", "claim text drifts from s1"),
    (O, "LEVEL_R.format_map(_Keep(fill or {}, oodh=sentence))", "LEVEL_R.format_map(_Keep(oodh=sentence))",
     "fill ignored"),
    (O, "        return 2\n", "        return 0\n", "CLI refusal exits 0"),
    (SS, "import oodh_wording as OW  # noqa", "OW = None  # noqa", "score_stats does not expose the rule"),
    # STEP 11 FIX ROUND (the reviewer's oodh_adv cases)
    (O, "if not is_comparator(nb):", "if False:", "any comparator accepted"),
    (O, "if is_comparator(na):", "if False:", "Planck rows from the comparator accepted"),
    (O, "if len(names) != 1:", "if False:", "two responders in one model accepted"),
    (O, 'name == COMPARATOR or name.endswith(":" + COMPARATOR)', "name == COMPARATOR", "kind-prefixed id refused"),
    (O, "if _greedy(ta) != _greedy(tb):", "if False:", "greedy vs sampled accepted"),
    (O, '([] if ra == "template" else [f"{ra} render (s9: D is the template render)"])', "[]",
     "plain render not labeled diagnostic"),
    (O, '(["greedy rows (s9: D is sampling T 0.6)"] if _greedy(ta) else [])', "[]", "greedy pair not labeled"),
    (O, 'if ci.get("diagnostic"):', "if False:", "a claim worded from a diagnostic CI"),
    (O, 'if part2_complete and not ci.get("part2_threads"):', "if False:", "Part 2 complete without Part 2 threads"),
    (O, "else len(set(ids) - set(part1)))", "else len(ids))", "Part 1 threads counted as Part 2"),
    # the matcher's target since Max, 2026-10-04 (DECISIONS_LEVEL_R_FOR_MAX.md decision 1, D: the new s1 wording)
    (O, r'ON_RC12 = r"\bon\s+RC[\s\-\u2010-\u2015]?12\b"', 'ON_RC12 = r"on RC-12"',
     "only the hyphenated form matched"),
    (O, "re.finditer(ON_RC12, text, re.I)", "re.finditer(ON_RC12, text)", "case-sensitive match"),
    (O, "if text[m.start():m.start() + len(QUALIFIED)] != QUALIFIED]",
     "if \"RC-12's format\" not in text[m.start():].split(\". \")[0]]", "the old sentence rule"),
    (O, "if a.part2_complete and not a.data:", "if False:", "--part2-complete without --data"),
    (O, 'dec = None if ci["diagnostic"] else decide(ci, a.part2_complete, fill)',
     "dec = decide(dict(ci, diagnostic=[]), a.part2_complete, fill)", "CLI words a claim from a diagnostic CI"),
]


def run(src=None, old=None, new=None):
    """-> (exit code or 'malformed', output tail)"""
    d = tempfile.mkdtemp(prefix="rc12_oodhw_mut_")
    try:
        os.makedirs(os.path.join(d, "rc12"))
        os.makedirs(os.path.join(d, "prereg"))
        for f in os.listdir(HERE):
            if f.endswith(".py"):
                shutil.copy(os.path.join(HERE, f), os.path.join(d, "rc12"))
        shutil.copy(PREREG, os.path.join(d, "prereg"))
        if src:
            p = os.path.join(d, "rc12", src)
            text = open(p).read()
            if text.count(old) != 1:
                return "malformed", f"{text.count(old)} matches"
            open(p, "w").write(text.replace(old, new))
        codes, out = [], ""
        for test in T:
            e = subprocess.run([sys.executable, "-B", test], cwd=os.path.join(d, "rc12"), capture_output=True,
                               text=True, timeout=110)
            codes.append(e.returncode)
            out += e.stdout + e.stderr
        fails = [ln for ln in out.splitlines() if ln.startswith("FAIL") or "Error" in ln]
        rc = 1 if 1 in codes else next((c for c in codes if c != 0), 0)
        return rc, (fails[0] if fails else out.strip()[-160:])[:160]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    t0 = time.time()
    rc, tail = run()
    lines, bad = [f"baseline {' + '.join(T)}: {rc}" + ("" if rc == 0 else f"  {tail}")], int(rc != 0)
    for k, (src, old, new, what) in enumerate(MUTANTS, 1):
        rc, tail = run(src, old, new)
        if rc == 1:
            lines.append(f"KILLED   #{k} {src}: {what}  [{tail}]")
        else:
            lines.append(f"PROBLEM  #{k} {src}: {what} (exit {rc}) [{tail}]")
            bad += 1
    lines.append(f"{len(MUTANTS)} mutants, {bad} problems (baseline included), {time.time() - t0:.0f} s. No model.")
    lines.append("ALL MUTANTS KILLED" if not bad else "NOT ALL KILLED")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_oodh_wording.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
