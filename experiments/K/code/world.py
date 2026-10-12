"""SPEC 2, 3 and 7: the fact base (world W = 7), filler, slot plans and the FACT streams of K1 arms and K0-cap.

One world for every run: N_CAP entities are always drawn, so FB_low (first N_LOW) and FB_high (first N_HIGH) are
prefixes of the K0-cap base and every arm reads the same names and values. Bio doc (tokens, all but the first
supervised): F M L a1 v1 .. a6 v6 <|endoftext|>, the six pairs in a fresh order per exposure. Fact QA (chat):
<|user|> F M L a ? <|end|> <|assistant|> v <|end|>. Filler: the same formats, names from Z1 x Z2 x Z3, values drawn
fresh per document. The FACT share is two sources (bio tokens and QA chat, split by their token ratio per entity);
each is written WRITE_MARGIN longer than its expected draw and holds its FB exposures in its first FB_REGION.
Filler content and attribute orders are a function of the slot index only, so arms that share a slot share it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kcommon as K

ARMS = ("F0", "FL", "FH", "CAP")


@dataclass
class World:
    names: np.ndarray      # (N_CAP, 3) token ids, F1 x F2 x F3, all distinct
    values: np.ndarray     # (N_CAP, 6) token ids from V, one per attribute of A
    attrs: np.ndarray      # (6,) the attribute tokens A
    qa_train: np.ndarray   # (N_CAP,) bool: exactly half of each block [0, N_LOW), [N_LOW, N_HIGH), [N_HIGH, N_CAP)


def make_world(pools) -> World:
    f1, f2, f3 = pools.triple("F")
    n_all = len(f1) * len(f2) * len(f3)
    code = K.rng("fb_names").choice(n_all, K.N_CAP, replace=False)
    names = np.stack([f1[code // (len(f2) * len(f3))], f2[(code // len(f3)) % len(f2)], f3[code % len(f3)]], 1)
    values = pools["V"][K.rng("fb_values").integers(0, len(pools["V"]), (K.N_CAP, K.N_ATTR))]
    qa = np.zeros(K.N_CAP, dtype=bool)
    r = K.rng("qa_split")
    for lo, hi in ((0, K.N_LOW), (K.N_LOW, K.N_HIGH), (K.N_HIGH, K.N_CAP)):
        qa[lo + r.permutation(hi - lo)[:(hi - lo) // 2]] = True
    return World(names.astype(np.int64), values.astype(np.int64), pools["A"].astype(np.int64), qa)


@dataclass
class FactCfg:
    arm: str = "FH"
    slots: int = K.B_SLOTS          # run length in token slots (tests pass a small one)
    e_bio: int = K.E_BIO
    e_qa: int = K.E_QA
    n_high: int = K.N_HIGH
    n_low: int = K.N_LOW
    n_cap: int = K.N_CAP
    shard_docs: int = 1 << 20       # bio docs per .bin shard; QA records per .jsonl shard use the same count

    def __post_init__(self):
        assert self.arm in ARMS, self.arm
        assert self.n_low <= self.n_high <= self.n_cap <= K.N_CAP

    @property
    def share(self) -> float:
        return 1.0 if self.arm == "CAP" else K.SHARES["fact"]

    @property
    def n_plan(self) -> int:
        """Entities in the slot plan the arm is cut from (FH's for F0 and FL: SPEC 3)."""
        return self.n_cap if self.arm == "CAP" else self.n_high

    @property
    def n_fb(self) -> int:
        return {"F0": 0, "FL": self.n_low, "FH": self.n_high, "CAP": self.n_cap}[self.arm]


def layout(c: FactCfg) -> dict:
    """Slot counts and source shares. Tokens per entity: bio e_bio x 16, QA 0.5 x e_qa x 6 x 10 (half QA-train)."""
    bio_t, qa_t = c.e_bio * K.BIO_LEN, 0.5 * c.e_qa * K.N_ATTR * K.QA_LEN
    bio_frac = bio_t / (bio_t + qa_t)
    draw = K.expected_draw(c.share, c.slots)
    out = {"draw_tokens": draw, "tokens_per_entity": bio_t + qa_t,
           "share_bio": c.share * bio_frac, "share_qa": c.share * (1 - bio_frac)}
    out["bio_slots"] = math.ceil(K.WRITE_MARGIN * draw * bio_frac / K.BIO_LEN)
    out["qa_slots"] = math.ceil(K.WRITE_MARGIN * draw * (1 - bio_frac) / K.QA_LEN)
    out["bio_region"] = int(K.FB_REGION * out["bio_slots"])
    out["qa_region"] = int(K.FB_REGION * out["qa_slots"])
    out["bio_exposures"] = c.n_plan * c.e_bio
    out["qa_exposures"] = (c.n_plan // 2) * c.e_qa * K.N_ATTR
    assert out["bio_exposures"] <= out["bio_region"], f"bio exposures {out['bio_exposures']} > region {out}"
    assert out["qa_exposures"] <= out["qa_region"], f"QA exposures {out['qa_exposures']} > region {out}"
    assert out["bio_region"] * K.BIO_LEN <= draw * bio_frac, "FB region reaches past the expected draw"
    return out


def plan(world: World, c: FactCfg) -> dict:
    """-> {"bio": entity per bio slot (-1 filler), "qa_ent", "qa_attr"}: the arm's plan (SPEC 3)."""
    lay = layout(c)
    n = c.n_plan
    bio = np.full(lay["bio_slots"], -1, dtype=np.int32)
    r = K.rng("slot_bio", n)
    pos = r.choice(lay["bio_region"], lay["bio_exposures"], replace=False)
    bio[pos] = r.permutation(np.repeat(np.arange(n, dtype=np.int32), c.e_bio))
    qents = np.flatnonzero(world.qa_train[:n]).astype(np.int32)    # exactly n / 2 at SPEC scale
    pairs_e = np.repeat(qents, K.N_ATTR * c.e_qa)
    pairs_a = np.tile(np.repeat(np.arange(K.N_ATTR, dtype=np.int8), c.e_qa), len(qents))
    qa_e = np.full(lay["qa_slots"], -1, dtype=np.int32)
    qa_a = np.zeros(lay["qa_slots"], dtype=np.int8)
    assert len(pairs_e) <= lay["qa_region"], "QA exposures exceed the QA region"
    r = K.rng("slot_qa", n)
    pos = r.choice(lay["qa_region"], len(pairs_e), replace=False)
    perm = r.permutation(len(pairs_e))
    qa_e[pos], qa_a[pos] = pairs_e[perm], pairs_a[perm]
    keep = c.n_fb
    bio[bio >= keep] = -1                   # FL: FH's plan with every non-FB_low slot turned into filler
    qa_e[qa_e >= keep] = -1                 # F0: keep = 0, all filler
    return {"bio": bio, "qa_ent": qa_e, "qa_attr": qa_a}


def filler_names(pools, r, n: int) -> np.ndarray:
    z = pools.triple("Z")
    return np.stack([z[i][r.integers(0, len(z[i]), n)] for i in range(3)], 1)


def bio_block(world: World, pools, ents: np.ndarray, shard: int) -> np.ndarray:
    """(n, 16) uint16 bio docs for one shard's slots. Filler and orders depend on the shard index only."""
    n = len(ents)
    rf, ro = K.rng("filler_bio", shard), K.rng("bio_order", shard)
    names = filler_names(pools, rf, n)
    vals = pools["V"][rf.integers(0, len(pools["V"]), (n, K.N_ATTR))]
    order = np.argsort(ro.random((n, K.N_ATTR)), axis=1)
    fb = ents >= 0
    names[fb] = world.names[ents[fb]]
    vals[fb] = world.values[ents[fb]]
    doc = np.empty((n, K.BIO_LEN), dtype=np.int64)
    doc[:, :3] = names
    doc[:, 3:15:2] = world.attrs[order]
    doc[:, 4:15:2] = np.take_along_axis(vals, order, 1)
    doc[:, 15] = K.EOT
    return doc.astype(np.uint16)


def qa_block(world: World, pools, ents: np.ndarray, attrs: np.ndarray, shard: int, base: int) -> list[dict]:
    """Fact QA chat records for one shard's slots (filler: Z name, random attribute, fresh value)."""
    n = len(ents)
    rf = K.rng("filler_qa", shard)
    names = filler_names(pools, rf, n)
    fa = rf.integers(0, K.N_ATTR, n)
    fv = pools["V"][rf.integers(0, len(pools["V"]), n)]
    out = []
    for i in range(n):
        e = int(ents[i])
        if e >= 0:
            nm, a, v = world.names[e], int(attrs[i]), int(world.values[e, attrs[i]])
        else:
            nm, a, v = names[i], int(fa[i]), int(fv[i])
        out.append(qa_record([int(x) for x in nm], int(world.attrs[a]), v, f"q{base + i}"))
    return out


def qa_record(name, attr: int, value: int, rid: str) -> dict:
    return {"id": rid, "turns": [{"role": "user", "ids": [*name, attr, K.Q]}, {"role": "assistant", "ids": [value]}]}


def shards(n_slots: int, per: int):
    for s, lo in enumerate(range(0, n_slots, per)):
        yield s, lo, min(n_slots, lo + per)

