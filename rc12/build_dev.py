"""Build the RC-12 DEV split (SPEC s9): every family from its own seeded stream Random("RC12:dev:<family>:1212"),
written to dev/rc12_dev.jsonl (one record per line, stable key order). Pure stdlib, no model.

  python3 -B build_dev.py            build and write, print counts per family and cell
  python3 -B build_dev.py --stdout   print the JSONL instead of writing (used by the determinism test)"""
import json
import os
import sys
from collections import Counter

import fam_bind
import fam_corr
import fam_loop_k
import fam_own_topic
import fam_persist
import fam_recall
import fam_role_lookup
import fam_twohop

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dev", "rc12_dev.jsonl")
BUILDERS = [("T0", fam_recall.build_t0), ("RECALL", fam_recall.build_recall), ("CORR", fam_corr.build_corr),
            ("BIND", fam_bind.build_bind), ("TWOHOP", fam_twohop.build_twohop),
            ("PERSIST", fam_persist.build_persist), ("OWN", fam_own_topic.build_own),
            ("TOPIC", fam_own_topic.build_topic), ("ROLE", fam_role_lookup.build_role),
            ("LOOKUP", fam_role_lookup.build_lookup), ("LOOP", fam_loop_k.build_loop), ("K", fam_loop_k.build_k)]


def build_all():
    recs = []
    for fam, fn in BUILDERS:
        got = fn()
        assert all(r["family"] == fam for r in got), fam
        recs += got
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    return recs


def dumps(recs):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs)


def main():
    recs = build_all()
    text = dumps(recs)
    if "--stdout" in sys.argv:
        sys.stdout.write(text)
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    fam = Counter(r["family"] for r in recs)
    cells = Counter((r["family"], r["cell"]) for r in recs)
    print(f"wrote {len(recs)} records ({sum(r['n_turns'] == 12 for r in recs)} twelve-turn) to {OUT}")
    for k, v in fam.items():
        print(f"  {k:8s} {v:4d}   " + ", ".join(f"{c}={n}" for (f2, c), n in sorted(cells.items()) if f2 == k))


if __name__ == "__main__":
    main()
