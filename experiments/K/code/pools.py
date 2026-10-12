"""SPEC 1: token pools drawn by rule from tok_v0_8k, never by hand.

CAP = vocab entries U+0120 then [A-Z][a-z]{2,}; LOW = U+0120 then [a-z]{3,}. Drop the CAP_DROP lowest-id CAP and
LOW_DROP lowest-id LOW entries, drop the frame tokens, keep entries whose standalone encoding is that single id,
then cut pools in SPEC order from a seeded permutation (W = 7). K-EVAL-S [C6]: each of S1, S2, S3 is paired off by a
seeded permutation into 24 token pairs, a cell is one pair per position (the 8 names {a, a'} x {b, b'} x {c, c'},
a B cube), and K-EVAL-S is a seeded 10% of the 13,824 cells (1,382 cells, 11,056 triples; K3 round 2: B's cube
items need whole cubes of held-out triples, and a random 10% of triples holds about 14 whole cubes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

import kcommon as K

CAP_RE = re.compile("\u0120[A-Z][a-z]{2,}")
LOW_RE = re.compile("\u0120[a-z]{3,}")
FRAME_IDS = {K.Q, K.EQ, K.OK, K.NONE}


@dataclass
class Pools:
    ids: dict                                  # pool name -> np.int64 array of token ids
    kevs: np.ndarray                           # (n, 3) K-EVAL-S triples (S pool token ids)
    sizes: dict = field(default_factory=dict)
    kevs_cells: np.ndarray | None = None       # (m, 3, 2) K-EVAL-S cells: the token pair per position

    def __getitem__(self, k):
        return self.ids[k]

    def triple(self, prefix: str) -> tuple:
        return tuple(self.ids[f"{prefix}{i}"] for i in (1, 2, 3))

    def kevs_set(self) -> set:
        return {tuple(int(x) for x in t) for t in self.kevs}

    def owner(self) -> dict:
        """token id -> pool name (every pooled token sits in exactly one pool)."""
        out = {}
        for name, a in self.ids.items():
            for t in a.tolist():
                assert t not in out, f"token {t} in {out.get(t)} and {name}"
                out[t] = name
        return out


def _single(tok, i: int) -> bool:
    s = tok.id_to_token(i)
    return tok.encode(" " + s[1:], add_special_tokens=False).ids == [i]


def candidates(tok) -> tuple[list[int], list[int]]:
    v = tok.get_vocab()
    cap = sorted(i for s, i in v.items() if CAP_RE.fullmatch(s))[K.CAP_DROP:]
    low = sorted(i for s, i in v.items() if LOW_RE.fullmatch(s))[K.LOW_DROP:]
    cap = [i for i in cap if i not in FRAME_IDS and _single(tok, i)]
    low = [i for i in low if i not in FRAME_IDS and _single(tok, i)]
    return cap, low


def _cut(ids: list[int], spec: list, rng) -> dict:
    perm = rng.permutation(np.asarray(ids, dtype=np.int64))
    need = sum(n for _, n in spec)
    assert len(perm) >= need, f"pools need {need} tokens, the rule leaves {len(perm)} (SPEC 1: shrink pro rata)"
    out, k = {}, 0
    for name, n in spec:
        out[name] = np.sort(perm[k:k + n])
        k += n
    return out


def build(tok) -> Pools:
    cap, low = candidates(tok)
    ids = {**_cut(cap, K.CAP_POOLS, K.rng("pools_cap")), **_cut(low, K.LOW_POOLS, K.rng("pools_low"))}
    r = K.rng("kevs")
    pairs = [np.sort(ids[f"S{i}"][r.permutation(len(ids[f"S{i}"]))[:len(ids[f"S{i}"]) // 2 * 2]].reshape(-1, 2), 1)
             for i in (1, 2, 3)]
    n1, n2, n3 = (len(x) for x in pairs)
    code = np.sort(r.choice(n1 * n2 * n3, int(n1 * n2 * n3 * K.KEVS_FRAC), replace=False))
    cells = np.stack([pairs[0][code // (n2 * n3)], pairs[1][(code // n3) % n2], pairs[2][code % n3]], 1)
    kevs = np.array([[c[0][i], c[1][j], c[2][m]] for c in cells for i in (0, 1) for j in (0, 1) for m in (0, 1)],
                    dtype=np.int64).reshape(-1, 3)
    sizes = {k: int(len(v)) for k, v in ids.items()}
    sizes.update(cap_left=len(cap), low_left=len(low), kevs=int(len(kevs)), kevs_cells=int(len(cells)))
    return Pools(ids, kevs, sizes, cells)


_CACHE: dict = {}


def get(tok=None) -> Pools:
    """The SPEC pools for tok_v0_8k (cached per process)."""
    if "p" not in _CACHE:
        _CACHE["p"] = build(tok or K.load_tokenizer())
    return _CACHE["p"]


class Lex:
    """Token classes the parser and oracles read (built from pools; name pools per position).
    `names` picks which triple family counts as names: ("S", "T") for skill items, ("F",) or ("Z",) for facts."""

    def __init__(self, pools: Pools, name_prefixes=("S", "T", "F", "Z")):
        self.pos = [set(), set(), set()]
        for p in name_prefixes:
            for i in range(3):
                self.pos[i].update(pools[f"{p}{i + 1}"].tolist())
        self.attr = set(pools["A"].tolist()) | set(pools["AH"].tolist())
        self.val = set(pools["V"].tolist()) | set(pools["VH"].tolist())
        self.v_only = sorted(pools["V"].tolist())
        self.alias = set(pools["AL"].tolist())
        self.marker = set(pools["MK"].tolist())
