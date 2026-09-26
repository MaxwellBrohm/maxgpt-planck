"""The leak list for oodh/gate.py: the reserved OASST2 trees whose user turns occur verbatim outside the reserve,
computed by corpus/oodh.leaked_reserve_trees (the ONE rule; nothing re-implemented) over every OASST2 tree and every
Dolly instruction, exactly as oodh/select.py drops them. Runs where the data is (the PC); loads no model. Writes one
tree id per line to --out, which must be under a sealed/ directory, and prints the count only.
Run: python oodh/leaks.py --out ~/planck/sealed/oodh/leaked_reserve_trees.txt"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":               # as a script: oodh/select.py must never shadow the stdlib 'select'
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.append(os.path.join(os.path.dirname(HERE), "corpus"))

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402

from oodh import leaked_reserve_trees  # noqa: E402  (corpus/oodh.py)

TREES = "~/planck/data/raw/starter/OpenAssistant__oasst2/2023-11-05_oasst2_all.trees.jsonl.gz"
DOLLY = "~/planck/data/raw/starter/databricks__databricks-dolly-15k/databricks-dolly-15k.jsonl"


def load(path):
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def compute(trees=TREES, dolly=DOLLY):
    def iter_trees():
        with gzip.open(os.path.expanduser(trees), "rt", encoding="utf-8") as fh:
            yield from (json.loads(line) for line in fh if line.strip())
    with open(os.path.expanduser(dolly), encoding="utf-8") as fh:
        other = [json.loads(line)["instruction"] for line in fh if line.strip()]
    return sorted(leaked_reserve_trees(iter_trees(), other))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", default=TREES)
    ap.add_argument("--dolly", default=DOLLY)
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    if "sealed" not in out.split(os.sep):
        sys.exit(f"refusing --out outside a sealed/ directory: {out}")
    ids = compute(a.trees, a.dolly)
    with open(out, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"{len(ids)} leaked reserved trees written")


if __name__ == "__main__":
    main()
