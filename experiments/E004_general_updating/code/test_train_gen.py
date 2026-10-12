"""E004 step 2 self-test of the training generator (train_e004.py). No model, no tokenizer.
Samples 2,000 dialogues from stream(0) and checks: determinism per seed; the declared structures only (kinds,
k, d, fillers, 1-2 objects, one value type, the ideal gold); no held-out object type, value type, eval marker,
alias or name ever appears; answers are sentences that pass the grader's text clauses; 0 frame echoes; every
entry of every phrasing pool is used (in the seed's full 6,400-example stream); shares (kinds, reference forms, markers) near notes.txt.
usage: python3 test_train_gen.py > ../logs/test_train_gen.txt   (exit code 0 = every check passed)"""
import sys
sys.dont_write_bytecode = True
from collections import Counter

import train_e004 as T
from pools_train import POOLS, pool_sizes, MARKERS, NONCORR_MARKERS, REF_TARGET
import checks_train as C
import checks_train_b as CB

N = 2000
FAIL = []


def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def near(x, target, tol):
    return abs(x - target) <= tol


print("== determinism")
a, b, c = T.take(0, 50), T.take(0, 50), T.take(1, 50)
ok(a == b, "seed 0 twice gives identical examples")
ok(a != c, "seed 1 differs from seed 0")
ok(T.take(3, 5) == T.take(3, 5), "seed 3 twice gives identical examples")

print("== pool level")
pp = CB.check_pools()
ok(not pp, f"templates: placeholders, no markers or literal values, pool sizes, 0 pool-level frame echoes "
           f"({len(pp)} problems) {pp[:4]}")
fp = CB.check_fillers()
ok(not fp, f"fillers: >= 40 how-to/why Q&A, no value/digit/object word/marker/name ({len(fp)} problems) {fp[:4]}")
ov = CB.e001_filler_overlap()
ok(not ov, f"0 word 5-grams shared with E001's filler pools ({len(ov)}) {ov[:3]}")

print(f"== {N} dialogues from stream(0)")
exs = T.take(0, N)
checks = [("structure", C.check_structure), ("references", C.check_references), ("values", C.check_values),
          ("answer", CB.check_answer), ("echo", CB.check_echo), ("heldout", CB.check_heldout)]
for name, fn in checks:
    bad = [(i, fn(ex)) for i, ex in enumerate(exs)]
    bad = [(i, p) for i, p in bad if p]
    ok(not bad, f"{name}: {len(bad)} failing dialogues" + (f"; first: #{bad[0][0]} {bad[0][1][:3]}" if bad else ""))

print("== coverage (every pool entry, object, value, kind, marker, reference form used)")
used = Counter(t for ex in exs for t in ex["tpl"])
missing = C.coverage_missing(exs)
print(f"      in these {N}: {len(missing)} pool entries unused {missing[:6]}")
miss64 = C.coverage_missing(T.take(0, 6400))
ok(not miss64, f"every entry of every phrasing pool used in seed 0's full 6,400-example stream "
               f"({sum(pool_sizes().values())} entries; missing {miss64[:6]})")
low = sorted((n, key) for key, n in used.items())[:3]
print(f"      least-used pool entries: {low}")
objs = Counter((ex["vtype"], o) for ex in exs for o in ex["objects"])
ok(all(objs[(vt, o)] for vt, P in POOLS.items() for o in P["objects"]), "every training object used")
vals = Counter((ex["vtype"], s["value"]) for ex in exs for s in ex["stmts"])
ok(all(vals[(vt, v)] for vt, P in POOLS.items() for v in P["values"]), "every training value used")
golds = Counter((ex["vtype"], ex["gold"]) for ex in exs)
ok(all(golds[(vt, v)] for vt, P in POOLS.items() for v in P["values"]), "every training value is a gold somewhere")

print("== shares vs notes.txt")
kc = Counter(ex["kind"] for ex in exs)
for k, p in T.KINDS.items():
    ok(near(kc[k] / N, p, 0.025), f"kind {k:12s} {kc[k] / N:.3f} (target {p:.3f})")
vc = Counter(ex["vtype"] for ex in exs)
ok(all(near(vc[v] / N, 0.25, 0.03) for v in T.VTYPES), f"value types {dict(vc)}")
corrs = [s for ex in exs for s in ex["stmts"] if s["role"] in ("corr", "rev")]
rc = Counter(s["ref"] for s in corrs)
for r, p in REF_TARGET.items():
    ok(near(rc[r] / len(corrs), p, 0.03), f"reference form {r:4s} {rc[r] / len(corrs):.3f} (target {p:.2f}, "
                                           f"n = {len(corrs)} corrections)")
none_corr = sum(1 for s in corrs if s["marker"] is None) / len(corrs)
ok(near(none_corr, 0.35, 0.04), f"corrections without a marker {none_corr:.3f} (target 0.35)")
b_orig = [s for ex in exs for s in ex["stmts"] if s["role"] == "orig" and s["obj"] != ex["stmts"][0]["obj"]]
a_orig = [s for ex in exs for s in ex["stmts"] if s["role"] in ("incid",) or
          (s["role"] == "orig" and s["obj"] == ex["stmts"][0]["obj"])]
rb = sum(1 for s in b_orig if s["marker"]) / len(b_orig)
ra = sum(1 for s in a_orig if s["marker"]) / len(a_orig)
ok(near(rb, 0.30, 0.05), f"second object's first statement marked {rb:.3f} (target 0.30)")
ok(near(ra, 0.10, 0.04), f"first original / incidental marked {ra:.3f} (target 0.10)")
mc = Counter(s["marker"] for ex in exs for s in ex["stmts"] if s["marker"])
ok(set(mc) == set(MARKERS), f"all 7 training markers used {dict(mc)}")
mn = Counter(s["marker"] for ex in exs for s in ex["stmts"] if s["marker"] and s["role"] in ("orig", "incid"))
ok(set(mn) == set(NONCORR_MARKERS), f"non-correction markers {dict(mn)}")
qf = Counter(ex["q_form"] for ex in exs)
ok(near(qf["full"] / N, 0.5, 0.04), f"question names the object by full phrase {qf['full'] / N:.3f}, head noun "
                                     f"{qf['head'] / N:.3f}")
dc = Counter(ex["d"] for ex in exs)
ok(set(dc) == set(range(11)), f"d covers 0-10: {dict(sorted(dc.items()))}")
kd = Counter(ex["k"] for ex in exs)
print(f"      k of the asked object: {dict(sorted(kd.items()))}; objects: "
      f"{dict(Counter(ex['n_obj'] for ex in exs))}")
first = sum(1 for ex in exs if ex["stmts"][0]["value"] == ex["gold"]) / N
last = sum(1 for ex in exs if ex["stmts"][-1]["value"] == ex["gold"]) / N
ok(first <= 0.50 and last <= 0.50, f"pre-check of the stream rule (full oracles in validate_e004.py): "
                                   f"first-mention gold {first:.3f}, last-mention gold {last:.3f} (<= 0.50)")
nans = Counter(ex["tpl"][-1][1] for ex in exs if ex["k"] > 0)
print(f"      answers after a correction drawn from ans / ans_upd: {dict(nans)}")

print("== the scored seeds 1-5 (per-example checks on 2,000 each, coverage on each full 6,400 stream)")
for seed in range(1, 6):
    ex_s = T.take(seed, 6400)
    nbad = {name: sum(1 for ex in ex_s[:N] if fn(ex)) for name, fn in checks}
    miss = C.coverage_missing(ex_s)
    ok(not any(nbad.values()) and not miss, f"seed {seed}: failing dialogues {nbad}; unused pool entries {miss[:4]}")

print("== sample")
for ex in (exs[0], exs[7]):
    print(T.prompt(ex) + ex["answer"].rstrip("\n"))
    print(f"   [{ex['kind']} {ex['vtype']} gold={ex['gold']} k={ex['k']} d={ex['d']} "
          f"refs={[s['ref'] for s in ex['stmts']]}]")
print()
print("RESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
