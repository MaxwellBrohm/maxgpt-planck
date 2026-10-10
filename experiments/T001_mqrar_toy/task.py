"""T001 task generator: multi-query repeated associative recall (PREREG.txt TASK). numpy only, deterministic by rng.

Every event is 3 tokens: "SET k v" (write), "ASK k v" (query; the logits at k are scored on v), "COPY a b" (track
family: b takes a's current value). A sample at size n is BOS, n initial SETs (one per key), then 2n events.
"""
from __future__ import annotations

import numpy as np

BOS, SET, ASK, COPY = 0, 1, 2, 3
K = 160                 # key tokens
V = 64                  # value tokens
KEY0 = 4
VAL0 = KEY0 + K
VOCAB = VAL0 + V
FAMILIES = ("mqrar", "ar", "track")
IGNORE = -100


def seq_len(n: int) -> int:
    return 1 + 9 * n


def make_sample(n: int, rng: np.random.Generator, family: str = "mqrar"):
    """Returns (tokens list, queries list). Each query: pos (index of the key token whose next-token logits are scored),
    ans (value token), key, updated (>= 2 writes before the query), stale (sorted value tokens the key held earlier
    that differ from ans), first (the key's first value token), tracked (latest write was a COPY), dist (events since
    the latest write), outvoted / tied (amendment A1: among the key's earlier explicit "SET k v" and "ASK k v" events,
    some other value occurs more often / as often, but not more often, than the answer)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if not 2 <= n <= K:
        raise ValueError(f"n {n} outside [2, {K}]")
    keys = [int(x) for x in rng.choice(K, size=n, replace=False)]
    toks = [BOS]
    queries = []
    cur: dict[int, int] = {}
    hist: dict[int, list[int]] = {}
    last: dict[int, int] = {}
    copied: dict[int, bool] = {}
    votes: dict[int, dict[int, int]] = {}      # key -> value -> explicit SET/ASK occurrences so far (A1)
    ev = 0

    def vote(k: int, v: int) -> None:
        votes.setdefault(k, {})
        votes[k][v] = votes[k].get(v, 0) + 1

    def write(k: int, v: int, via_copy: bool) -> None:
        cur[k] = v
        hist.setdefault(k, []).append(v)
        last[k] = ev
        copied[k] = via_copy

    for k in rng.permutation(keys):
        k = int(k)
        v = int(rng.integers(V))
        toks += [SET, KEY0 + k, VAL0 + v]
        write(k, v, False)
        vote(k, v)
        ev += 1
    for _ in range(2 * n):
        r = float(rng.random())
        if family == "ar" or r < 0.5:
            k = keys[int(rng.integers(n))]
            toks += [ASK, KEY0 + k, VAL0 + cur[k]]
            h = hist[k]
            mine = votes[k].get(cur[k], 0)
            rival = max((c for x, c in votes[k].items() if x != cur[k]), default=0)
            queries.append(dict(pos=len(toks) - 2, ans=VAL0 + cur[k], key=k, updated=len(h) >= 2,
                                stale=sorted({VAL0 + x for x in h if x != cur[k]}), first=VAL0 + h[0],
                                tracked=copied[k], dist=ev - last[k], outvoted=rival > mine,
                                tied=rival == mine))
            vote(k, cur[k])
        elif family == "track" and r < 0.75:
            i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
            a, b = keys[i], keys[j]
            toks += [COPY, KEY0 + a, KEY0 + b]
            write(b, cur[a], True)
        else:
            k = keys[int(rng.integers(n))]
            v = int(rng.integers(V - 1))
            v = v + (v >= cur[k])          # uniform over the V - 1 values that differ from the current one
            toks += [SET, KEY0 + k, VAL0 + v]
            write(k, v, False)
            vote(k, v)
        ev += 1
    return toks, queries


def make_batch(n: int, batch: int, rng: np.random.Generator, family: str = "mqrar"):
    """(x int64 (B, L), y int64 (B, L) with IGNORE except at answer positions, meta: per row its queries)."""
    L = seq_len(n)
    x = np.zeros((batch, L), dtype=np.int64)
    y = np.full((batch, L), IGNORE, dtype=np.int64)
    meta = []
    for b in range(batch):
        toks, qs = make_sample(n, rng, family)
        x[b] = toks
        for q in qs:
            y[b, q["pos"]] = q["ans"]
        meta.append(qs)
    return x, y, meta


FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}
SPLIT_BASE = {"test": 9_000_000, "dev": 8_000_000}
SPLIT_SIZE = {"test": 256, "dev": 128}


def eval_rng(split: str, family: str, n: int) -> np.random.Generator:
    """Fixed eval stream per (split, family, n); disjoint from training streams default_rng([1000 + seed])."""
    return np.random.default_rng([SPLIT_BASE[split], FAMILY_ID[family], n])


def train_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([1000 + seed])
