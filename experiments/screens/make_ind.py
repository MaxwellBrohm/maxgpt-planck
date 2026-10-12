"""The fixed IND sequences (experiments/SCREENS.txt C6 IND): 64 sequences of 128 token ids, each id drawn
uniformly from the NON-CONTROL ids of tok_v0_8k (spec.CONTROL_TOKENS, ids 0..6, left out; the tag tokens 7..14
are not control tokens and stay in), numpy default_rng(SEED). Draws happen here, once; the training hook
(harness/induction.py) reads the committed file and draws nothing. Its sha256 is pinned in
configs/screens_base.yaml (eval.induction.sha256) and in SCREENS.txt's FIXED log.
  ~/.venvs/planck/bin/python make_ind.py [--check]     write ind_v0_8k.json, or check the committed file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "ind_v0_8k.json")
TOK = os.path.join(ROOT, "tokenizer", "v0", "tok_v0_8k.json")
SEED, N, L = 20261003, 64, 128


def build() -> bytes:
    sys.path[:0] = [os.path.join(ROOT, "data_prep")]
    import prep_common as C
    info = C.tokenizer_info(TOK)
    ids = np.array(sorted(set(range(info["vocab"])) - set(info["control_ids"])), dtype=np.int64)
    seqs = np.random.default_rng(SEED).choice(ids, size=(N, L), replace=True)
    doc = {"about": "SCREENS C6 IND: each sequence is fed twice in a row as one document (harness/induction.py)",
           "tokenizer": os.path.basename(TOK), "tokenizer_sha256": info["sha256"], "seed": SEED,
           "rule": "numpy default_rng(seed).choice(non-control ids, (64, 128), replace=True)",
           "excluded_ids": info["control_ids"], "seqs": seqs.tolist()}
    return (json.dumps(doc, separators=(",", ":")) + "\n").encode()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    raw = build()
    if a.check:
        ok = os.path.exists(OUT) and open(OUT, "rb").read() == raw
        print(("ok " if ok else "DIFFERS ") + hashlib.sha256(raw).hexdigest())
        return 0 if ok else 1
    if os.path.exists(OUT) and open(OUT, "rb").read() != raw:
        sys.exit(f"refusing to overwrite {OUT} with different content")
    open(OUT, "wb").write(raw)
    print(hashlib.sha256(raw).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
