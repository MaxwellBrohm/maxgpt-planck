"""RC-12 near-duplicate report (Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 12)):
grade_loop.near_dup / word_edits and score.near_duplicate_stats, the rate reported beside the loop rate, never gated.
No model. Fixtures:
  unit      one word replaced / added / dropped (true); equal after normalization (false: that is the equality
            clause and the ack-repeat); two edits (false); 12 words one edit (true) and 13 (false); empty reply
            (false); no earlier reply (false); a match two replies back (true); any two different one-word replies
            (true, as the STEP 10e measure Max saw defined it); word_edits on known pairs
  rate      hand-built conversations: the share is over EVERY reply; a near-duplicate the loop rule flags LOOP is
            not counted (a 12-word one-word change is a self-copy LOOP); summarize leaves --own-cf rows out
  real      IDEAL through the real runner and graders on the OWN family (plain, greedy): a reply replaced by an
            earlier one plus a word (the parrot that escapes the equality clause) adds exactly one near-duplicate
            and no LOOP
--mutate: each mutant replaces one text in a scratch copy of rc12 (*.py, dev/) and this test runs there; KILLED =
exit non-zero. The unmutated copy must pass. Writes logs/mutation_near_dup.txt.
Run: python3 -B test_near_dup.py [--mutate]   (exit 1 on any failure)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.join(HERE, "..", "experiments", "E004_general_updating", "code")   # quick_shortcuts' import, by path
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (E004, os.environ.get("PYTHONPATH")) if p))
MUTANTS = [
    ("grade_loop.py", "NEAR = dict(max_words=12,", "NEAR = dict(max_words=13,", "13-word replies counted"),
    ("grade_loop.py", "NEAR = dict(max_words=12,", "NEAR = dict(max_words=11,", "12-word replies dropped"),
    ("grade_loop.py", "max_words=12, edits=1)", "max_words=12, edits=2)", "two word edits instead of one"),
    ("grade_loop.py", 'word_edits(ws, p, NEAR["edits"]) == NEAR["edits"]', 'word_edits(ws, p, NEAR["edits"]) <= NEAR["edits"]',
     "equal replies counted"),
    ("grade_loop.py", 'NEAR["edits"] for p in prior_ws)', 'NEAR["edits"] for p in prior_ws[-1:])',
     "only the previous reply compared"),
    ("grade_loop.py", "prev[j - 1] + (x != y))", "prev[j - 1] + 1)", "a replaced word costs as much as two edits"),
    ("grade_loop.py", 'if not ws or len(ws) > NEAR["max_words"]:', 'if len(ws) > NEAR["max_words"]:',
     "an empty reply can be a near-duplicate"),
    ("score.py", 'hits += "LOOP" not in fl and L.near_dup_ws(ws[i], ws[:i])', "hits += L.near_dup_ws(ws[i], ws[:i])",
     "LOOP replies counted"),
    ("score.py", "return hits / n if n else None", "return hits if n else None", "a count, not a share"),
    ("score.py", "near_duplicate=near_duplicate_stats(rows),", "near_duplicate=near_duplicate_stats(sel),",
     "--own-cf rows counted"),
]
fails = []


def check(name, ok, info=""):
    if not ok:
        fails.append(f"{name} {info}")


def c_unit():
    import grade_loop as L
    cases = [("replace", "I can't answer question 6.", ["I can't answer question 5."], True),
             ("insert", "Your bike is green today.", ["Your bike is green."], True),
             ("drop", "Your bike is green.", ["Your bike is green today."], True),
             ("equal", "Got it!", ["Got it."], False), ("two", "The bike is green.", ["The car is red."], False),
             ("twelve", " ".join(f"w{i}" for i in range(12)), [" ".join(f"w{i}" for i in range(11)) + " x"], True),
             ("thirteen", " ".join(f"w{i}" for i in range(13)), [" ".join(f"w{i}" for i in range(12)) + " x"], False),
             ("empty", "", ["Okay."], False), ("no_prior", "Your bike is green.", [], False),
             ("two_back", "Sure, the bike.", ["Sure, the bike is.", "A long and quite different reply here."], True),
             ("one_word", "Fine.", ["Okay."], True)]   # as defined: two different one-word replies are one edit apart
    for name, reply, prior, want in cases:
        check(f"unit {name}", L.near_dup(reply, prior) is want, f"got {L.near_dup(reply, prior)}")
    for a, b, cap, want in (("a b c", "a c", 1, 1), ("a b", "b a", 2, 2), ("a b c d", "a", 1, 2), ("", "", 1, 0)):
        got = L.word_edits(a.split(), b.split(), cap)
        check(f"word_edits {a!r} {b!r}", got == want, f"got {got} want {want}")


def conv(replies, kinds=None, own_cf=False, rid="c1"):
    import grade_loop as L
    kinds = kinds or ["S"] * len(replies)
    flags = L.conversation_flags(replies, ["eos"] * len(replies), kinds)
    return dict(id=rid, family="OWN", seed=None, own_cf=own_cf, flags=flags,
                turns=[dict(i=i + 1, reply=r) for i, r in enumerate(replies)])


def c_rate():
    import score as S
    a = conv(["Your bike is green.", "Okay.", "Your bike is green today.", "That sounds lovely, thanks for sharing."])
    twelve = " ".join(f"w{i}" for i in range(12))
    b = conv(["Something.", twelve, twelve.replace("w6", "x6"), "Done here now."], rid="c2")
    check("rate a", S.near_duplicate_stats([a]) == 0.25, f"{S.near_duplicate_stats([a])}")
    check("selfcopy LOOP not counted", "LOOP" in b["flags"][2] and S.near_duplicate_stats([b]) == 0.0,
          f"flags {b['flags']} rate {S.near_duplicate_stats([b])}")
    check("rate a + b", S.near_duplicate_stats([a, b]) == 1 / 8, f"{S.near_duplicate_stats([a, b])}")
    check("no rows", S.near_duplicate_stats([]) is None)


def c_real():
    import fakes_family as FF
    import runner as RN
    import score as S
    recs = RN.load(families=["OWN"])
    rows = RN.run(recs, FF.make("IDEAL"), "plain", [None], None, None, "IDEAL")
    cf = RN.run(recs, FF.make("IDEAL"), "plain", [None], None, None, "IDEAL", 0, True)
    base = S.summarize(rows + cf)
    check("IDEAL OWN loop 0", base["loop_rate"] == 0, f"{base['loop_rate']}")
    rec = recs[0]
    turns = [dict(t) for t in rows[0]["turns"]]
    k = next(i for i, t in enumerate(turns) if i >= 2 and t["kind"] in ("S", "L", "D") and len(turns[i - 2]["reply"]
             .split()) <= 10)
    turns[k]["reply"] = turns[k - 2]["reply"].rstrip(".!") + " indeed."
    new = RN.make_row(rec, turns, "plain", None, "IDEAL", 0, False)
    after = S.summarize([new] + rows[1:] + cf)
    n = sum(len(r["flags"]) for r in rows)
    check("parrot plus one word", abs(after["near_duplicate"] - base["near_duplicate"] - 1 / n) < 1e-12
          and after["loop_rate"] == 0, f"before {base['near_duplicate']} after {after['near_duplicate']} "
          f"loop {after['loop_rate']} turn {k + 1} {turns[k]['reply']!r}")
    cfx = [dict(r) for r in cf]
    t2 = [dict(t) for t in cfx[0]["turns"]]
    t2[k]["reply"] = t2[k - 2]["reply"].rstrip(".!") + " indeed."
    cfx[0] = RN.make_row(rec, t2, "plain", None, "IDEAL", 0, True)
    check("cf rows left out", S.summarize(rows + cfx)["near_duplicate"] == base["near_duplicate"])


def mutate():
    lines, bad = [], 0
    for k, (f, old, new, what) in enumerate([(None, None, None, "baseline")] + MUTANTS):
        tmp = tempfile.mkdtemp()
        copy = os.path.join(tmp, "rc12")
        shutil.copytree(HERE, copy, ignore=shutil.ignore_patterns("logs", "runs", "__pycache__"))
        if k:
            p = os.path.join(copy, f)
            src = open(p).read()
            if src.count(old) != 1:
                lines.append(f"MALFORMED #{k} {f}: {what}")
                bad += 1
                shutil.rmtree(tmp)
                continue
            open(p, "w").write(src.replace(old, new))
        r = subprocess.run([sys.executable, "-B", "test_near_dup.py"], cwd=copy, capture_output=True, text=True,
                           timeout=600, env=ENV)
        shutil.rmtree(tmp)
        first = next((x for x in r.stdout.splitlines() if x.startswith("FAIL")), r.stderr.strip()[-120:])
        if not k:
            lines.append(f"baseline: exit {r.returncode}")
            bad += r.returncode != 0
            continue
        lines.append(f"{'KILLED  ' if r.returncode else 'SURVIVED'} #{k} {f}: {what}  [{first}]")
        bad += r.returncode == 0
    lines.append(f"{len(MUTANTS)} mutants, {bad} problems")
    lines.append("ALL MUTANTS KILLED" if not bad else "NOT ALL KILLED")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_near_dup.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


def main():
    if "--mutate" in sys.argv:
        return mutate()
    for c in (c_unit, c_rate, c_real):
        try:
            c()
        except Exception as e:  # noqa: BLE001 (a crash is a failure, reported by name)
            fails.append(f"{c.__name__} crashed: {type(e).__name__}: {e}")
    for f in fails:
        print("FAIL", f)
    print(json.dumps(dict(near_dup_checks="PASS" if not fails else f"{len(fails)} failures")))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
