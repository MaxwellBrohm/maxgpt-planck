"""E006 eval identity (notes.txt WHAT STAYS IDENTICAL 6). No model, no tokenizer, stdlib only.
E006's eval, dev and probe draws (4004/4104/4204, built from E006's copy of E005's code) equal E004's and E005's item
by item: sha256 of (draw, family, index, the LIK prompt with its prefix, gold, candidates), each built in its own
process from its own code directory (python3 -B, so no bytecode is written into E004 or E005); 1,120 of 1,120 each.
eval_identity_e005.hashes (the E005 copy, unedited) does the building.
usage: python3 -B eval_identity_e006.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_identity_e005 as EI

EXPS = os.path.dirname(os.path.dirname(HERE))
DIRS = {"E004": os.path.join(EXPS, "E004_general_updating", "code"), "E005": os.path.join(EXPS, "E005_alias_eot", "code")}


def main():
    fail = []
    mine = EI.hashes(HERE)
    for name, d in DIRS.items():
        other = EI.hashes(d)
        same = sum(1 for x, y in zip(mine, other) if x == y)
        per = {}
        for x, y in zip(mine, other):
            per.setdefault(x[0], [0, 0])
            per[x[0]][0] += x == y
            per[x[0]][1] += 1
        print(f"E006 vs {name}: {same} of {len(other)} identical (E006 built {len(mine)}) "
              + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items()))
        if len(mine) != len(other) or same != len(other) or len(other) != 1120:
            fail.append(f"eval identity vs {name}")
    print("RESULT: " + ("ALL PASS" if not fail else f"FAILED: {fail}"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
