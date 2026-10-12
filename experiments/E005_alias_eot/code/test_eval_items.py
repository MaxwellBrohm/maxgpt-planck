"""E004 eval item self-test (step 3). No model, no tokenizer. Runs the pool, filler, per-item and draw checks
(checks_eval.py, checks_eval_b.py) on the eval (4004), dev (4104) and probe (4204) draws, prints a summary and
sample items, and exits 0 only if everything passes. Usage: python3 -B test_eval_items.py > ../logs/..."""
import sys
from collections import Counter

import items_e004 as I
from checks_eval import ITEM_CHECKS
from checks_eval_b import check_pools, check_fillers, check_draws

FAILS = []


def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    for name, fn in (("pools", check_pools), ("fillers", check_fillers)):
        r = fn()
        ok(not r, f"{name}: {len(r)} problems" + ("".join("\n      " + x for x in r[:15])))
    draws = {name: I.draw(name) for name in I.DRAWS}
    for name, D in draws.items():
        bad, first = Counter(), {}
        for fam, items in D.items():
            for it in items:
                for cname, fn in ITEM_CHECKS:
                    r = fn(it)
                    if r:
                        bad[(fam, cname)] += 1
                        first.setdefault((fam, cname), f"item {it['idx']}: {r[0]}")
        ok(not bad, f"{name} draw items ({sum(len(v) for v in D.values())}): failing {dict(bad)}"
           + "".join(f"\n      {k}: {v}" for k, v in first.items()))
    r = check_draws(draws)
    ok(not r, "draws: " + ("; ".join(r) if r else "sizes, B balance, no prompt shared between eval/dev/probe"))
    D = draws["eval"]
    print("\nStructure summary, eval draw (per family: k values, latest-correction reference forms, d, objects,"
          " candidates)")
    for fam, items in D.items():
        ks = Counter(it["k"] for it in items)
        refs = Counter([s for s in it["stmts"] if s["obj"] == it["asked"]][-1]["ref"] for it in items)
        nc = Counter(len(it["candidates"]) for it in items)
        print(f"  {fam:10s} k {dict(sorted(ks.items()))}  latest {dict(refs)}  d {sorted({it['d'] for it in items})}"
              f"  objs {sorted({it['n_obj'] for it in items})}  cands {dict(sorted(nc.items()))}")
    print("\nSample items (eval draw, rendered as in the likelihood eval: plain render + forced prefix)")
    for fam in ("H1", "H2", "H6", "C_noupd"):
        it = D[fam][0]
        print(f"\n--- {fam} item 0 cell={it['cell']} gold={it['gold']} candidates={it['candidates']} "
              f"meta={it['meta']}")
        print(I.prompt(it, with_prefix=True))
    print("\nRESULT: " + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
