"""dev_extras14.py's model list (notes STEP 14X): every model pinned in rc12/pins.json is scored, by its slug, exactly
once, in size order (pins.json params), nothing else is; jobs() gives every extra both renders except the
Falcon-H1-Tiny-90M-Base (no chat template: plain only), and the Vertex card row on template when a card root is
given. No model, no runs read.
  python -B test_dev_extras14.py [--pins F]   (exit 1 on any failure)"""
import argparse
import json
import os
import sys

import dev_extras14 as X

HERE = os.path.dirname(os.path.abspath(__file__))


def check(pins):
    fails = []
    size = {m.split("/")[-1]: p["params"] for m, p in pins["models"].items()}
    if sorted(X.EXTRAS14) != sorted(size) or len(set(X.EXTRAS14)) != len(X.EXTRAS14):
        fails.append(f"list: missing {sorted(set(size) - set(X.EXTRAS14))}, extra {sorted(set(X.EXTRAS14) - set(size))}"
                     f", {len(X.EXTRAS14)} entries")
    order = [size.get(m, -1) for m in X.EXTRAS14]
    if order != sorted(order):
        fails.append(f"list: not in size order {order}")
    if X.PLAIN_ONLY != {"Falcon-H1-Tiny-90M-Base"}:
        fails.append(f"plain only: {X.PLAIN_ONLY}")
    got = [(m, r) for _, m, _, r, s in X.jobs("R") if s == X.SEEDS]
    want = [(m, r) for m in X.EXTRAS14 for r in ("template", "plain") if r == "plain" or not m.endswith("-Base")]
    if got != want:
        fails.append(f"jobs: {sorted(set(got) ^ set(want))} differ")
    card = [j for j in X.jobs("R", "C") if j[0] == "C"]
    if card != [("C", "Vertex-0.6-15M-Instruct", "Vertex-0.6-15M-Instruct@card", "template", ["1", "2", "3"])] \
            or [j for j in X.jobs("R") if j[0] != "R"]:
        fails.append(f"jobs: card rows {card}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default=os.path.join(HERE, "pins.json"))
    fails = check(json.load(open(ap.parse_args().pins)))
    for f in fails:
        print("FAIL", f)
    print("test_dev_extras14:", "PASS" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
