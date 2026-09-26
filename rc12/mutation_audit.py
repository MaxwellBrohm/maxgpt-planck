"""RC-12 step 5: mutation test of the audit FIXES (the balance of the dev generators). Each mutant undoes one fix
in the generator source, the whole dev split is rebuilt IN MEMORY (dev/ is never touched), and the detector that
is meant to catch it must go red: the audit_rules.py gate (a cheap rule above the bars) or a named check of
test_gens_fam.py. A kill by another detector does not count; a crash is never a kill.
Writes logs/mutation_audit.txt (or mutation_audit_part<i>.txt); exit 1 unless every mutant is killed.
Run: python3 -B mutation_audit.py [--part i/n]   (--part: every n-th mutant, to keep each run under 2 minutes; each
part re-checks the unmutated build first)"""
import argparse
import os
import sys
import time

import audit_rules as AR
import balance
import build_dev
import fam_corr
import fam_own_topic
import fam_recall
import fam_role_lookup
import fam_twohop
import test_gens_fam as TF

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
MODS = {"balance": balance, "fam_corr": fam_corr, "fam_own_topic": fam_own_topic, "fam_recall": fam_recall,
        "fam_role_lookup": fam_role_lookup, "fam_twohop": fam_twohop}
# (name, module, old text, new text, detector): detector ("gate", rule names) or ("test", message fragment)
MUTANTS = [
    ("echo balance off", "balance", "    if question is not None:\n", "    if False:\n",
     ("gate", ("ANTI_OVERLAP_E", "ANTI_OVERLAP_L", "ANTI_WORDING"))),
    ("length rank off", "balance", "target is None or rank_ok([wc(x) for x in c], gold, target)", "True",
     ("gate", ("LONGEST", "SHORTEST"))),
    ("TWOHOP echo on lures only", "fam_twohop", "C.spread([0, 1, 2], len(idx), rng)", "C.spread([1, 2], len(idx), rng)",
     ("test", "TWOHOP echo chain")),
    ("CORR U-same tails dropped (B-before ends on A)", "fam_corr",
     '"before": [(("Lm",), 3), (("Lm", "Lm"), 2), (("Lp", "Lp"), 2), (("Lm", "Lp"), 1)],', '"before": [((), 8)],',
     ("test", "ends on A's latest")),
    ("CORR lures all plain", "fam_corr", 'form="marked" if mark == "m" else "plain"', 'form="plain"',
     ("test", "tails after A's latest")),
    ("C_twoslot blocks only", "fam_corr", 'for lay in ["blocks", "bfirst"]', 'for lay in ["blocks", "blocks"]',
     ("gate", ("PREV_FRESH", "ANTEPENULT", "PENULT_MARKER"))),
    ("C_noupd echo lure dropped", "fam_corr", "    seq.insert(a0 if eb else", "    0 and seq.insert(a0 if eb else",
     ("test", "C_noupd echo lure")),
    ("C_noupd anchor left last", "fam_corr", "    if tl or form == \"anchor\":", "    if tl:",
     ("test", "C_noupd anchor is the last turn")),
    ("CORR gold length rank off", "fam_corr", "texts = BL.choose(rng, opts, [gold_j], len_rank)",
     "texts = [o[0] for o in opts]", ("gate", ("SHORTEST", "LONGEST"))),
    ("TOPIC no later updates", "fam_own_topic", "rows = [(pos, pat) for pos, pat, c in TOPIC_PLAN",
     "rows = [(pos, ()) for pos, pat, c in TOPIC_PLAN", ("test", "TOPIC updates 16/16/16")),
    ("TOPIC open project never updated", "fam_own_topic", '("open_later" if j == oi else "close_later", nouns[j])',
     '("close_later", nouns[j])', ("test", "later update roles")),
    ("LOOKUP abstain always first", "fam_role_lookup", "p, x = (last, early) if xf[k] else (early, last)",
     "p, x = (last, early)", ("test", "LOOKUP X/P order")),
    ("RECALL abstain X always first", "fam_recall", "            if xfirst[k]:\n", "            if True:\n",
     ("test", "RECALL abstain X/P order")),
]


def with_source(modname, old, new, fn):
    mod = MODS[modname]
    src = open(mod.__file__).read()
    if src.count(old) != 1:
        raise RuntimeError(f"mutant text found {src.count(old)} times in {modname}")
    saved = dict(mod.__dict__)
    try:
        exec(compile(src.replace(old, new), mod.__file__, "exec"), mod.__dict__)
        return fn()
    finally:
        mod.__dict__.clear()
        mod.__dict__.update(saved)


def build():
    """every family from the CURRENT module functions (build_dev.BUILDERS holds the pre-mutation objects)."""
    recs = []
    for _, fn in build_dev.BUILDERS:
        recs += getattr(sys.modules[fn.__module__], fn.__name__)()
    return recs


def detect(recs, det):
    kind, what = det
    if kind == "gate":
        bad = AR.violations(recs)
        hits = [b for b in bad if b[0] in what]
        return bool(hits), (hits or bad)[:3]
    fails = TF.run(recs)
    hits = [f for f in fails if what in f]
    return bool(hits), (hits or fails)[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1")
    i, n = (int(x) for x in ap.parse_args().part.split("/"))
    todo = [m for k, m in enumerate(MUTANTS) if k % n == i - 1]
    t0 = time.time()
    base_ok = not AR.violations(build()) and not TF.run(build())
    lines = ["RC-12 mutation_audit.py: generator mutants that undo a step 5 fix; the named detector must go red.",
             f"part {i}/{n}: {len(todo)} of {len(MUTANTS)} mutants",
             f"unmutated build: gate and family checks {'clean' if base_ok else 'NOT CLEAN'}", ""]
    killed = 0
    for name, mod, old, new, det in todo:
        try:
            ok, why = with_source(mod, old, new, lambda: detect(build(), det))
        except Exception as e:  # a crash is never a kill
            ok, why = None, f"{type(e).__name__}: {e}"
        status = "KILLED " if ok else "CRASHED" if ok is None else "SURVIVE"
        killed += bool(ok)
        lines.append(f"{status} {name} [{det[0]}] <- {why}")
    lines += ["", f"killed {killed} of {len(todo)} in {time.time() - t0:.0f} s"]
    text = "\n".join(lines) + "\n"
    name = "mutation_audit.txt" if n == 1 else f"mutation_audit_part{i}.txt"
    with open(os.path.join(HERE, "logs", name), "w") as f:
        f.write(text)
    print(text)
    return 0 if base_ok and killed == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
