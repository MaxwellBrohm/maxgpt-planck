"""OOD-H reserve pool rule (corpus/oodh.py)."""
from __future__ import annotations

import hashlib
import os
import random
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oodh import in_oodh_reserve  # noqa: E402

# Golden values, derived independently of the implementation: a sha256 digest is 0 mod 8 exactly when
# its last hex digit is 0 or 8. "tree-14" (last digit 4) and "tree-5" (last digit c) are 0 mod 4 but not
# mod 8, so they catch a wrong modulus.
IN_POOL = ["00000000-0000-0000-0000-000000000000", "tree-8", "tree-11"]
NOT_IN_POOL = ["002c4715-b026-48d1-8d19-3f724a9fc1e8", "planck", "a", "tree-1", "tree-4", "tree-5",
               "tree-12", "tree-14"]


def test_golden_values():
    for t in IN_POOL:
        assert in_oodh_reserve(t), t
    for t in NOT_IN_POOL:
        assert not in_oodh_reserve(t), t


def test_matches_last_hex_digit_rule_on_random_ids():
    rng = random.Random(7)
    for _ in range(2000):
        t = str(uuid.UUID(int=rng.getrandbits(128)))
        last = hashlib.sha256(("planck-oodh-v1:" + t).encode()).hexdigest()[-1]
        assert in_oodh_reserve(t) == (last in "08"), t


def test_rate_is_about_one_in_eight():
    rng = random.Random(11)
    n = 40000
    hits = sum(in_oodh_reserve(str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(n))
    # binomial sd = sqrt(n p (1-p)) ~ 66; allow 5 sd
    assert abs(hits - n / 8) < 5 * 66, hits


def test_salt_matters():
    # the same ids under another salt give a different pool
    ids = [f"tree-{i}" for i in range(400)]
    other = [int(hashlib.sha256(("planck-oodh-v2:" + t).encode()).hexdigest(), 16) % 8 == 0 for t in ids]
    assert [in_oodh_reserve(t) for t in ids] != other


@pytest.mark.parametrize("bad", ["", None, 12, b"tree-8"])
def test_rejects_non_ids(bad):
    with pytest.raises(ValueError):
        in_oodh_reserve(bad)
