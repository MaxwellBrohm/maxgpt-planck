"""Track K shared constants and small helpers (SPEC.txt sections 1, 2, 6, 7, 8). No model, no GPU.

Every number here is copied from ../SPEC.txt; a test (tests/test_pools.py) pins the ones SPEC states as facts
(frame ids, pool sizes, load arithmetic). Seeds: SPEC 8 names the eval draw seeds; the training streams take
tagged seeds under the world seed W = 7 (default_rng([W, tag])), listed in TAGS and written to every manifest.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("K_CODE_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # the code root
HARNESS = os.path.join(ROOT, "harness")
TOK_REL = "tokenizer/v0/tok_v0_8k.json"
TOK_SHA256 = "078b24c4b0755d81985ebc122e912d7c70ff204d335519721235a405756fcfb5"

W = 7                                                                   # world seed (SPEC 1, 11)
# SPEC 1: frame and control ids (single ids in tok_v0_8k, checked by tests/test_pools.py)
Q, EQ, OK, NONE = 2479, 847, 2708, 5151
EOT, SYS, USER, ASST, END, TOOL = 1, 2, 3, 4, 5, 6
LOOKUP, LOOKUP_E, RESULT, RESULT_E = 11, 12, 13, 14
ROLE_IDS = {"system": SYS, "user": USER, "assistant": ASST, "tool": TOOL}
PAD = 0

# SPEC 1 pool sizes, in assignment order
CAP_POOLS = [("F1", 112), ("F2", 112), ("F3", 112), ("Z1", 40), ("Z2", 40), ("Z3", 40),
             ("S1", 48), ("S2", 48), ("S3", 48), ("T1", 24), ("T2", 24), ("T3", 24), ("AL", 32)]
LOW_POOLS = [("A", 6), ("AH", 6), ("V", 2048), ("VH", 256), ("MK", 32)]
CAP_DROP, LOW_DROP = 64, 256
KEVS_FRAC = 0.10                                                        # K-EVAL-S share of S triples [C6]

# SPEC 2 and 7: the fact base and the budget
N_HIGH, N_LOW, N_CAP = 62_000, 15_500, 124_000
E_BIO, E_QA = 100, 10
N_ATTR = 6
BIO_LEN, QA_LEN = 16, 10
B_SLOTS = 7_630 * 32_768                                                # 250.0M
FILL = 0.9885                                                           # drawn tokens per slot (E2 smoke)
SHARES = {"skill": 0.25, "fact": 0.50, "lang": 0.25}
WRITE_MARGIN, FB_REGION = 1.03, 0.95                                    # SPEC 7
E5 = 60                                                                 # K5: answers per attribute (K5 notes)

# tagged training seeds (default_rng([W, tag])); eval draw seeds are SPEC 8's
TAGS = {"pools_cap": 1, "pools_low": 2, "kevs": 3, "fb_names": 10, "fb_values": 11, "qa_split": 12,
        "slot_bio": 20, "slot_qa": 21, "filler_bio": 22, "filler_qa": 23, "bio_order": 24, "qa_order": 25,
        "skill_train": 30, "k5_train": 50, "k5_hits": 51, "k5_slots": 52}
SEEDS = {"dev": 7101, "eval": 7201, "ood": 7202, "evals": 7203, "probe": 7301, "conflict": 7401, "k5": 7501,
         "k2": 7601}


def rng(tag, *extra) -> np.random.Generator:
    t = TAGS[tag] if isinstance(tag, str) else int(tag)
    return np.random.default_rng([W, t, *[int(e) for e in extra]])


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl(path: str, recs) -> int:
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def load_tokenizer(path: str | None = None):
    """tok_v0_8k with encode_special_tokens set (harness/data.py load_tokenizer's rule); refuses a wrong file."""
    from tokenizers import Tokenizer
    p = path or os.path.join(ROOT, TOK_REL)
    if path is None:
        assert sha256_file(p) == TOK_SHA256, f"{TOK_REL}: sha256 is not SPEC 1's"
    tok = Tokenizer.from_file(p)
    tok.encode_special_tokens = True
    return tok


def expected_draw(share: float, slots: int = B_SLOTS) -> float:
    """Tokens a source of this share draws over a run of `slots` slots (SPEC 7)."""
    return share * slots * FILL
