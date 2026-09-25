"""Mutation test of the render checker (Max's rule: a check never watched failing is not evidence).

  python3 -B mutation_checker.py            full run: suite + every checker mutant (about 90 s)
  python3 -B mutation_checker.py --quick    small corpus, suite + mutants (the unittest wrapper uses this)
  python3 -B mutation_checker.py --no-mutants
  python3 -B mutation_checker.py --quick --mutant-sample 4   what tests/test_checker.py runs (about 30 s)

The suite has five kinds of case, all on FAKE teacher text (no model is loaded or called):
  clean     fake_teacher.render() of every feasible skeleton (RM, RS, RL): must pass
  drift     formatting drift the parser tolerates, applied to clean renders: must pass with identical turns
  hand      hand-written natural renders (must pass) and the measured Gemma habits (must fire their codes)
  defect    one planted defect per case (defects_text, defects_behav): the expected code must fire
  boundary  exactly-at-threshold (must fire) and one-step-inside (must pass) pairs (fixtures_bound)
Then every mutant in checker_mutants must be killed: at least one case must fail while the mutant is applied.
Exit status 0 only if the suite passes, every reason code has a planted fixture, and no mutant survives."""
import argparse
import collections
import sys
import time

sys.dont_write_bytecode = True

import checker  # noqa: E402
import checker_mutants  # noqa: E402
import defects_behav  # noqa: E402
import defects_text  # noqa: E402
import fake_teacher  # noqa: E402
import fixtures_bound  # noqa: E402
import fixtures_hand  # noqa: E402
import parse  # noqa: E402
import render_prompt as R  # noqa: E402
import skeleton  # noqa: E402

DEFECTS = defects_text.DEFECTS + defects_behav.DEFECTS
NOT_PLANTABLE = set()   # every code in checker.ORDER must have a fixture; list exceptions here with a reason


def corpus(n_rm, n_other):
    skels, infeasible = [], collections.Counter()
    for reg, n in (("RM", n_rm), ("RS", n_other), ("RL", n_other)):
        for seed in range(n):
            s = skeleton.build(seed, reg)
            f = R.feasible(s)
            if f:
                infeasible[f[0][1].split(" ")[0] + " " + f[0][1].split(" ")[-1]] += 1
                continue
            skels.append(s)
    return skels, infeasible


def case(name, group, skel, raw, expect, wordlist=None, turns=None):
    return {"name": name, "group": group, "skel": skel, "raw": raw, "expect": expect, "wordlist": wordlist,
            "turns": turns, "built": R.build(skel)}


def build_cases(skels, per_defect, n_drift):
    cases = []
    renders = [(s, fake_teacher.render(s)) for s in skels]
    for s, tx in renders:
        cases.append(case(f"clean {s['register']} {s['seed']}", "clean", s, parse.serialize(s, tx), "ok"))
    for name, fn in fixtures_hand.DRIFT:
        for s, tx in renders[:n_drift]:
            cases.append(case(f"drift {name} {s['seed']}", "drift", s, fn(parse.serialize(s, tx)), "ok", turns=tx))
    for name, s, raw, expect in fixtures_hand.hand_cases():
        cases.append(case(name, "hand", s, raw, expect))
    for name, code, fn in DEFECTS:
        n = 0
        for s, tx in renders:
            if n >= per_defect:
                break
            out = fn(s, tx, R.build(s))
            if out is None:
                continue
            raw = out if isinstance(out, str) else parse.serialize(s, out)
            cases.append(case(f"defect {name} {s['register']} {s['seed']}", "defect", s, raw, {code}))
            n += 1
    seen = collections.Counter()
    for s, tx in renders:
        for b in fixtures_bound.BUILDERS:
            for name, sk, texts, expect, opts in b(s, tx, R.build(s)):
                if seen[name] < per_defect:
                    seen[name] += 1
                    cases.append(case(f"bound {name} {s['seed']}", "boundary", sk, parse.serialize(sk, texts), expect,
                                      wordlist=opts.get("wordlist")))
    return cases


def evaluate(c):
    """None if the case behaves as expected, else a one-line reason."""
    r = checker.run(c["skel"], c["raw"], c["built"], wordlist=c["wordlist"])
    e = c["expect"]
    if e == "ok":
        if not r["ok"]:
            return f"expected pass, got {r['hits'][:2]}"
        if c["turns"] is not None and r["turns"] != c["turns"]:
            return "drift changed the parsed turns"
        return None
    if isinstance(e, tuple):
        return None if r["ok"] and e[1] <= set(r["codes"]) else f"expected report {e[1]}, got {r['codes']} ok={r['ok']}"
    return None if e <= set(r["codes"]) else f"expected {sorted(e)}, got {r['codes']}"


def run_cases(cases, stop_first=False):
    fails = []
    for c in cases:
        why = evaluate(c)
        if why:
            fails.append((c["name"], why))
            if stop_first:
                break
    return fails


def coverage(cases):
    planted = set()
    for c in cases:
        e = c["expect"]
        if e != "ok":
            planted |= e[1] if isinstance(e, tuple) else e
    return [c for c in checker.ORDER if c not in planted and c not in NOT_PLANTABLE]


def _codes(c):
    e = c["expect"]
    return set() if e == "ok" else (e[1] if isinstance(e, tuple) else e)


def prioritized(cases, targets):
    """order only (every case still runs until one fails): fixtures for the mutant's codes, then boundary, hand and
    drift cases, then a slice of clean renders, then the other defects, then the remaining clean renders."""
    first = [c for c in cases if _codes(c) & targets]
    seen = {id(c) for c in first}
    mid = [c for c in cases if id(c) not in seen and c["group"] in ("boundary", "hand", "drift")]
    clean = [c for c in cases if c["group"] == "clean"]
    rest = [c for c in cases if id(c) not in seen and c["group"] == "defect"]
    return first + mid + clean[:25] + rest + clean[25:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--mutant-sample", type=int, default=1, help="run every Nth mutant only (unittest wrapper)")
    a = ap.parse_args(argv)
    t0 = time.time()
    n_rm, n_other, per = (120, 30, 3) if a.quick else (600, 100, 5)
    skels, infeasible = corpus(n_rm, n_other)
    cases = build_cases(skels, per, 6 if a.quick else 20)
    groups = collections.Counter(c["group"] for c in cases)
    print(f"skeletons: {len(skels)} feasible, infeasible {dict(infeasible)}")
    print("cases:", dict(groups), f"({time.time() - t0:.0f} s to build)")
    fails = run_cases(cases)
    by_def = collections.Counter(c["name"].split(" ")[1] for c in cases if c["group"] == "defect")
    missing_defects = [n for n, _, _ in DEFECTS if by_def[n] == 0]
    uncovered = coverage(cases)
    print(f"suite: {len(cases) - len(fails)}/{len(cases)} cases behave as expected")
    for n, why in fails[:20]:
        print("  FAIL", n, "|", why)
    print("defects with no applicable skeleton:", missing_defects or "none")
    print("reason codes with no planted fixture:", uncovered or "none")
    survivors = []
    if not a.no_mutants and not fails:
        muts = checker_mutants.all_mutants()[::a.mutant_sample]
        killed = 0
        slow = []
        for name, apply, targets in muts:
            t1 = time.time()
            undo = apply()
            try:
                k = run_cases(prioritized(cases, targets), stop_first=True)
            finally:
                undo()
            if time.time() - t1 > 2:
                slow.append(f"{name} {time.time() - t1:.1f}s")
            if k:
                killed += 1
            else:
                survivors.append(name)
        assert not run_cases(cases[:50]), "checker not restored after mutants"
        print(f"mutants: {killed}/{len(muts)} killed; slow: {slow or 'none'}")
        for s in survivors:
            print("  SURVIVED", s)
    print(f"total {time.time() - t0:.0f} s")
    ok = not fails and not uncovered and not missing_defects and not survivors and not a.no_mutants
    return 0 if ok or (a.no_mutants and not fails and not uncovered) else 1


if __name__ == "__main__":
    sys.exit(main())
