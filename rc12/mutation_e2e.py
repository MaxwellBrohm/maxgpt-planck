"""RC-12 mutation test of the end-to-end machinery (Max's rule: every test claim is watched failing).
Every mutant of mutants_e2e.py runs in a FRESH process (spawn, one task per child): the mutated module source is
installed in sys.modules before anything imports it, then validate_e2e.gates(write=False) and
validate_machinery.checks() run. Killed = the failure set differs from the unmutated baseline; crashed = an
exception (counted, not a kill). The baseline itself runs in a fresh process too. No model is loaded.
Usage: python3 -B mutation_e2e.py [--part i/n]   (writes logs/mutation_e2e.txt, or ..._part<i>.txt; exit 1 if any
mutant survives, crashes or is malformed, or the baseline has G1-G3 / machinery failures)."""
import argparse
import multiprocessing as mp
import os
import sys
import time
import types

from mutants_e2e import MUTANTS

HERE = os.path.dirname(os.path.abspath(__file__))


def job(k):
    sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True
    if k >= 0:
        mod, old, new, _ = MUTANTS[k]
        path = os.path.join(HERE, mod + ".py")
        src = open(path).read()
        if src.count(old) != 1:
            return k, "malformed", [f"text found {src.count(old)} times"]
        m = types.ModuleType(mod)
        m.__file__ = path
        sys.modules[mod] = m
        exec(compile(src.replace(old, new), path, "exec"), m.__dict__)
    try:
        import validate_e2e as VE
        import validate_machinery as VM
        g = VE.gates(write=False)
        fails = sorted(set(g["core"]) | set(g["items"]) | set(VM.checks(g["recs"])))
    except Exception as e:  # noqa: BLE001
        return k, "crashed", [f"{type(e).__name__}: {e}"[:200]]
    return k, "ran", fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1")
    args = ap.parse_args()
    i, n = (int(x) for x in args.part.split("/"))
    ks = [k for k in range(len(MUTANTS)) if k % n == i - 1]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(9, len(ks) + 1), maxtasksperchild=1) as pool:
        results = dict((k, (v, f)) for k, v, f in pool.map(job, [-1] + ks, chunksize=1))
    _, base = results.pop(-1)
    base_set = set(base)
    lines = [f"RC-12 mutation_e2e.py part {args.part}: {len(ks)} of {len(MUTANTS)} mutants, fresh process each, "
             f"{time.time() - t0:.0f} s. No model.", "baseline failures (expected: only the G4 item lines):"]
    lines += [f"  {f}" for f in base] or ["  none"]
    bad_base = [f for f in base if not f.startswith("G4 ")]
    counts = {"killed": 0, "survived": 0, "crashed": 0, "malformed": 0}
    for k in ks:
        mod, _, _, claim = MUTANTS[k]
        verdict, fails = results[k]
        if verdict == "ran":
            new = sorted(set(fails) - base_set)
            gone = sorted(base_set - set(fails))
            verdict = "killed" if new or gone else "survived"
            why = (new or [f"(baseline line gone) {x}" for x in gone])[:1]
        else:
            why = fails[:1]
        counts[verdict] += 1
        lines.append(f"{verdict.upper():9s} #{k:02d} {mod}: {claim}" + (f"  <- {why[0][:150]}" if why else ""))
    lines.append(" | ".join(f"{k} {v}" for k, v in counts.items()))
    ok = not bad_base and counts["killed"] == len(ks)
    lines.append("ALL MUTANTS KILLED" if ok else "NOT ALL KILLED (or baseline has G1-G3 / machinery failures)")
    text = "\n".join(lines) + "\n"
    name = "mutation_e2e.txt" if n == 1 else f"mutation_e2e_part{i}.txt"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", name), "w") as f:
        f.write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
