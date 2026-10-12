"""E005 held-out purity of the training stream (notes.txt WHAT IS RE-CHECKED, "Purity"). Samples the first 6,404
DRAWN examples of seeds 1-5 (every example the trainer can keep before the 768-token rejection moves a few;
validate_e005 repeats the gate on the kept ones) and counts training examples with each held-out axis value.
E004's lines (purity_e004.violations) stay, except its "H1/H2 alias" line (no honorific at all), which the
pre-registered E005 lines replace:
  eval surname        an E004 eval/dev/probe surname (heldout_e004.ALIAS_SURNAMES and every draw's alias)
  eval alias pair     an eval honorific+surname pair
  eval join           an eval join used as a join: a title right after "with", "from" or "run by"
  eval alias 3-gram   an alias correction sharing a word 3-gram with an eval alias template ({a}/{v} placeholders)
  undeclared name     an honorific, role title or name-like word outside the example's declared TRAINING alias,
                      or an alias outside the alias block (checks_e005.check_heldout_e5)
Kept E004 lines: 0 frame echoes, 0 eval markers, k <= 3 (H3), d <= 10 (H4), <= 2 objects (H5), 0 held-out
objects (H6), 0 H6 values or digits, 0 non-training fillers (H7), 0 H7 filler 5-grams in training text.
The alias template pools are checked at pool level too (checks_e005.check_alias_pools).
usage: python3 -B purity_e005.py > ../logs/purity_e005.txt"""
import re
import sys
from collections import Counter

import purity_e004 as P4
import train_e005 as T5
import checks_e005 as CE
import pools_alias_train as PA
from oracles_e004 import strip_marker
from text_e004 import normalize, grams
from pools_train import POOLS

SEEDS, PER_SEED = (1, 2, 3, 4, 5), 6404
NEW_AXES = ["eval surname", "eval alias pair", "eval join", "eval alias 3-gram", "undeclared name"]


def sample():
    out = []
    for s in SEEDS:
        out += T5.take(s, PER_SEED)
    return out


def eval_alias_grams():
    E = CE.eval_vocab()
    return set().union(*(CE._tgrams(t) for t in E["tpls"]))


def violations(ex, eg):
    v = {k: m for k, m in P4.violations(ex).items() if k != "H1/H2 alias"}
    add = lambda axis, msg: v.setdefault(axis, []).append(msg)
    for msg in CE.check_heldout_e5(ex):
        axis = ("eval surname" if msg.startswith("eval surname") else "eval alias pair" if msg.startswith("eval alias")
                else "eval join" if msg.startswith("eval join") else "held-out (E004 lists)"
                if msg.startswith(("held-out object", "eval marker")) else "undeclared name")
        add(axis, msg)
    aliases = list(ex.get("aliases", {}).values())
    pool = POOLS[ex["vtype"]]["values"]
    for s in ex["stmts"]:
        if s["ref"] == "alias":
            n = normalize(strip_marker(ex["turns"][s["turn"]][0]), aliases, pool)
            if grams(n, 3) & eg:
                add("eval alias 3-gram", ex["turns"][s["turn"]][0])
    return v


def main():
    exs = sample()
    eg = eval_alias_grams()
    print(f"E005 purity: {len(exs)} training examples (seeds {SEEDS}, first {PER_SEED} draws of each stream; "
          f"blocks {dict(Counter(x['block'] for x in exs))})\n")
    counts, examples = Counter(), {}
    for ex in exs:
        for axis, msgs in violations(ex, eg).items():
            counts[axis] += 1
            examples.setdefault(axis, msgs[0])
    axes = ["H1/H2 echo", "eval marker", "H3 k45", "H4 d20", "H5 obj3", "H6 object", "H6 value", "H7 filler",
            "held-out (E004 lists)"] + NEW_AXES
    fail = 0
    for axis in axes + sorted(set(counts) - set(axes)):
        n = counts.get(axis, 0)
        fail += n > 0
        print(f"{'PASS' if n == 0 else 'FAIL'}  {axis:22s} training examples with it: {n}"
              + (f"  e.g. {examples[axis]!r}" if n else ""))
    ov = P4.h7_overlap(exs)
    fail += bool(ov)
    print(f"{'PASS' if not ov else 'FAIL'}  H7 genre               H7 filler 5-grams in training text: {len(ov)} {ov[:5]}")
    pp = CE.check_alias_pools(PA)
    fail += bool(pp)
    print(f"{'PASS' if not pp else 'FAIL'}  alias pools            pool-level problems: {len(pp)} {pp[:3]}")
    al = [x for x in exs if x["block"] == "alias"]
    used = Counter(a for x in al for a in x["aliases"].values())
    print(f"\nAlias vocabulary actually used: {len(used)} distinct aliases, "
          f"{len({a.split(' ', 1)[1] for a in used})} surnames, titles "
          f"{dict(Counter(a.split(' ')[0] for x in al for a in x['aliases'].values()))}")
    print("\nTraining ranges actually covered (held-out values must lie outside):")
    for name, c in P4.ranges(exs).items():
        print(f"  {name:32s} {dict(sorted(c.items()))}")
    print("  held out: k 4-5 (H3), d 20 (H4), 3 objects (H5), value types sport/number and 48 H6 objects (H6),"
          " small-talk/creative fillers (H7), frame echoes (H1/H2), eval surnames/pairs/joins/alias templates")
    print("\nRESULT: " + ("ALL PASS" if not fail else f"{fail} FAILED"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
