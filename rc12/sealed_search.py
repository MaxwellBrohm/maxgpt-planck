"""RC-12 sealed-session kit, part 4: the generator-seed draw with redraws (prereg s13 S3, rule of 2026-10-11).

A fresh seed's items can miss the tight item gates by chance alone (G2's 0.40 family bar, G3's shortcut ceiling
below 40, G10, the G9 balance checks of test_gens): a 48-unit family moves by about 0.07 between seeds (s8 margins),
and the dry run of 2026-10-11 and 12 on dev content passed 10 of 42 fresh seeds (rc12/logs/lock_dryrun_seeds.txt).
So the sealed seed is drawn by this rule, fixed before any sealed item exists: draw a fresh seed, build, run the
seed-sensitive gates (test_gens, audit_rules G10, validate_e2e G1-G4, mutation_audit, whose gate detectors are
statistical, and mutation_graders G6, whose fixtures come from the items; fake responders only, never a model),
stopping at the first failure; on any failure discard the seed (its file is deleted) and draw the next, up to --max
draws per run; the first seed that passes is the sealed seed, and then every S4 gate runs on it (sealed_kit.py
gates). Every attempt is logged, one line each with the gate verdicts and the shortcut ceiling and never a seed or
item text, in the draw log LOG (outside the work copy), published with the lock as SEED_ATTEMPTS.txt. The draws
select items on which the fixed-rule cheaters do worse; no model is involved, so this can only make the sealed split
harder to pass by shortcuts.

  python3 -B rc12/sealed_search.py WORK --seed-file F --log LOG --expect-labels q,r,s [--max 200] [--split sealed]
                                   [--supersede REASON]   (a new run on a log with draws; the log continues)
F must not exist; it receives the accepted seed. The asks counts in the expectations are read from each build (the
filler labels are hand-set data, so --expect-labels pins them); WORK/expect.json is written for the accepted seed."""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sealed_crypto as CR                   # noqa: E402
import sealed_kit as K                       # noqa: E402
from sealed_public import content_words, redact              # noqa: E402

# the seed-sensitive gates: the item gates, and mutation_audit, whose gate detectors are statistical (the dry run's
# accepted seed let "CORR gold length rank off" survive: SHORTEST and LONGEST stayed under the bars)
QUICK = {"test_gens", "audit_rules_G10", "validate_e2e_G1_G4", "mutation_audit_1", "mutation_audit_2",
         "mutation_graders_G6"}      # G6's fixtures come from the items too (the singular-form mutant needs one)


def expectations(work, labels):
    recs = [json.loads(x) for x in open(os.path.join(work, K.ITEMS), encoding="utf-8")]
    n = collections.Counter(t["asks"] for r in recs for t in r["turns"] if t["kind"] == "D")
    q, r, s = labels
    return {"filler_labels": {"question": q, "request": r, "statement": s}, "asks": [n[True], n[False]]}


def quiet_text(line, words=frozenset()):
    """a log line for the public draw log: anything quoted and every sealed content word is withheld (gate lines
    name ids and rates, never text)."""
    return redact(line, words)


def summary(work):
    """the attempt's public line: verdict lines of the quick gates and the shortcut ceiling (numbers only)."""
    def log(name):
        path = os.path.join(work, "gate_logs", name + ".txt")
        return open(path).read() if os.path.exists(path) else ""
    tg = re.search(r"^\d+ failures over \d+ records", log("test_gens"), re.M)
    e2e, rules = log("validate_e2e_G1_G4"), log("audit_rules_G10")
    audit = [re.search(r"^killed \d+ of \d+", log(f"mutation_audit_{i}"), re.M) for i in (1, 2)]
    audit = ", ".join(m.group(0) if m else "not run" for m in audit)
    g6 = re.search(r"^MUTANTS killed \d+/\d+", log("mutation_graders_G6"), re.M)
    ceil = re.search(r"shortcut ceiling \(best[^:]*: S7 ([0-9.]+)", e2e)
    words = content_words(work)                   # only the free-text parts are redacted, never the fixed words
    fails = [quiet_text(f, words) for f in re.findall(r"^FAIL (G\d[^\n]*)", e2e, re.M)]
    g10 = re.search(r"^GATE \([^)]*\): (\w+)([^\n]*)", rules, re.M)
    g10 = g10.group(1) + quiet_text(g10.group(2), words) if g10 else "not run"
    return (f"test_gens {tg.group(0) if tg else 'not run'}; ceiling S7 {ceil.group(1) if ceil else '?'}; "
            f"e2e fails: {'; '.join(fails) or 'none'}; G10 {g10}; mutation_audit {audit}; "
            f"G6 {g6.group(0) if g6 else 'not run'}")


REASON = r"[A-Za-z0-9 _.,:;()/-]{3,80}"           # a published restart reason: a step and a module, never item text


def search(work, seed_file, labels, max_draws, split, log_path, supersede=None):
    """draws until a seed passes. The log lives outside the work copy and holds every draw of every run; a run on a
    log that already has draws (a restart after a fix, or after 200 failed draws) must say why (supersede, a short
    plain reason), which voids any earlier acceptance. A draw whose build fails is logged and discarded."""
    if os.path.exists(seed_file):
        K.die(f"{seed_file} exists; the accepted seed is written to a new file")
    if K.inside(log_path, work):
        K.die("the draw log lives outside the work copy, so a restart cannot lose it")
    for x in os.listdir(os.path.dirname(seed_file)):              # a cut search's trial seeds (plaintext)
        if x.startswith(os.path.basename(seed_file) + ".try"):
            os.remove(os.path.join(os.path.dirname(seed_file), x))
    old = open(log_path).read() if os.path.exists(log_path) else ""
    done = len(re.findall(r"^draw \d+:", old, re.M))
    if done and not supersede:
        K.die("the draw log already has draws: a new run needs --supersede REASON")
    if supersede and not re.fullmatch(REASON, supersede):
        K.die(f"--supersede must match {REASON} (it is published)")
    with open(log_path, "a") as log:
        if supersede:
            log.write(f"superseded: {supersede}; any accepted draw above no longer counts\n")
        log.write(f"commit {K.commit_of(work)} split {split}: seed draws (s13 S3 rule, max {max_draws} this run)\n")
        for i in range(done + 1, done + max_draws + 1):
            trial = seed_file + f".try{i}"
            CR.seed(trial)
            try:
                K.build(work, trial, split)
            except (SystemExit, subprocess.TimeoutExpired):
                log.write(f"draw {i}: FAIL; the build refused this seed\n")
                log.flush()
                os.remove(trial)
                continue
            exp = os.path.join(work, "expect.json")
            with open(exp, "w") as f:
                json.dump(expectations(work, labels), f)
            shutil.rmtree(os.path.join(work, "gate_logs"), ignore_errors=True)     # no stale verdicts
            try:
                fails = K.gates(work, trial, exp, split, QUICK, stop=True)
            except subprocess.TimeoutExpired:
                log.write(f"draw {i}: FAIL; a gate timed out\n")
                log.flush()
                os.remove(trial)
                continue
            except SystemExit:                      # the copy changed under the draws: stop, logged
                log.write(f"draw {i}: STOPPED; the kit refused (its message is on the screen)\n")
                os.remove(trial)
                raise
            line = f"draw {i}: {'PASS' if not fails else 'FAIL'}; {summary(work)}"
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
            if not fails:
                os.rename(trial, seed_file)
                log.write(f"accepted draw {i}; seed sha256 {CR.sha(open(seed_file, 'rb').read().strip())}\n")
                return i
            os.remove(trial)
        log.write(f"no seed passed in {max_draws} draws this run\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--seed-file", required=True)
    ap.add_argument("--expect-labels", required=True)
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--split", default="sealed")
    ap.add_argument("--log", required=True, help="the draw log, outside the work copy (published as SEED_ATTEMPTS.txt)")
    ap.add_argument("--supersede", help="why a new run on an existing log (a content or code fix, or 200 failures)")
    a = ap.parse_args()
    labels = [int(x) for x in a.expect_labels.split(",")]
    work = os.path.abspath(a.work)
    K.check_place(work)
    got = search(work, os.path.abspath(a.seed_file), labels, a.max, a.split, os.path.abspath(a.log), a.supersede)
    print(f"accepted draw {got}" if got else f"no seed passed in {a.max} draws: stop and report (s13 S3)")
    sys.exit(0 if got else 1)


if __name__ == "__main__":
    main()
