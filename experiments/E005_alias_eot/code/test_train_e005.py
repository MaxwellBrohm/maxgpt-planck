"""E005 step 2 self-test of the training generator (train_e005.py: changes (a), (b); the render draw of (c)).
No model, no tokenizer. Checks: determinism; the E004 block is train_e004.stream(seed) unchanged and in order;
the alias pools (training-only, disjoint from the E004 eval); every example of seed 0's first 2,000 and every
ALIAS/IND example of seeds 1-5 (first 6,500 draws each) passes E004's per-example checks and the E005 ones;
coverage of every alias pool entry per seed; block, render, case, placement and form shares near notes.txt.
usage: python3 -B test_train_e005.py > ../logs/test_train_e005.txt   (exit code 0 = every check passed)"""
import sys
sys.dont_write_bytecode = True

import train_e004 as T4
import train_e005 as T5
import pools_alias_train as PA
import checks_train_b as CB
import checks_e005 as CE
import checks_e005_b as CB5
import shares_e005 as SH

N0, NS, SEEDS = 2000, 6500, (1, 2, 3, 4, 5)
FAIL = []


def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


print("== determinism")
a, b, c = T5.take(0, 300), T5.take(0, 300), T5.take(1, 300)
ok(a == b, "seed 0 twice gives identical examples")
ok(a != c, "seed 1 differs from seed 0")

print("== pool level")
pp = CE.check_alias_pools(PA)
ok(not pp, f"alias pools: sizes, training-only, disjoint from eval surnames/pairs/joins/templates, no 3-gram with an "
           f"eval alias template, no pronoun/object word/value, no echo of a question/answer ({len(pp)}) {pp[:3]}")
ok(not CB.check_pools() and not CB.check_fillers(), "E004 pools and fillers still pass E004's pool checks")

print(f"== per-example checks: seed 0 first {N0} (all blocks)")
ex0 = T5.take(0, N0)
bad = [(i, CB5.all_problems(ex)) for i, ex in enumerate(ex0)]
bad = [(i, p) for i, p in bad if p]
ok(not bad, f"{len(bad)} failing examples" + (f"; first #{bad[0][0]} {bad[0][1]}" if bad else ""))

print(f"== seeds {SEEDS}, first {NS} draws each")
allx = {}
for s in SEEDS:
    xs = T5.take(s, NS)
    allx[s] = xs
    same, n4 = SH.e004_identity(xs, s)
    ok(same, f"seed {s}: the {n4} E004-block examples are train_e004.stream({s})[:{n4}] unchanged, in order")
    new = [x for x in xs if x["block"] != "e004"]
    bad = [(x["kind"], p) for x in new for p in [CB5.all_problems(x)] if p]
    ok(not bad, f"seed {s}: {len(new)} ALIAS/IND examples, {len(bad)} failing" + (f"; first {bad[0]}" if bad else ""))
    used = {t for x in xs for t in x["tpl"]}
    miss = [key + (j,) for key, n in PA.alias_sizes().items() for j in range(n) if key + (j,) not in used]
    ok(not miss, f"seed {s}: every surname, title, join and alias template used (missing {miss[:4]})")
    objs = {(x["vtype"], o) for x in new for o in x["objects"]}
    ok(len(objs) == sum(len(P["objects"]) for P in T4.POOLS.values()), f"seed {s}: every training object in the "
                                                                       f"new blocks ({len(objs)})")

X = [x for s in SEEDS for x in allx[s]]
AL = [x for x in X if x["block"] == "alias"]
IN = [x for x in X if x["block"] == "ind"]
n = len(X)
print(f"== shares vs notes.txt (seeds {SEEDS} pooled: {n} examples, {len(AL)} ALIAS, {len(IN)} IND)")
for cond, msg in SH.share_checks(X):
    ok(cond, msg)

print("== sample")
for x in (AL[0], IN[0]):
    print(T5.prompt(x) + x["answer"].rstrip("\n"))
    print(f"   [{x['kind']} {x['placement']} both={x['both_aliased']} {x['aliases']} gold={x['gold']} "
          f"refs={[st['ref'] for st in x['stmts']]} render={x['render']}]")
print()
print("RESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
