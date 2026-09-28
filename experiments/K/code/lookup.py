"""SPEC 6 and K5 DATA: the lookup line and the K5 fact chats.

Hit:  <|tool|> <lookup> F M L </lookup> <result> a1 v1 .. a6 v6 </result> <|end|>   21 tokens (fresh pair order)
Miss: <|tool|> <lookup> F M L </lookup> <result> none </result> <|end|>             10 tokens
A K5 fact chat: the tool line, then two questions about different attributes of one FB_low entity (30 tokens with
a miss, 41 with a hit). Each attribute is answered E5 = 60 times: 60 rounds of a random permutation of the 6
attributes cut into 3 pairs, 180 chats per entity. Chat i is a hit when its uniform draw u_i < p; the draws, the
entities, the pairs and the pair orders are the same for every arm.

Build deviation (logged for the person, K5 notes K5-c): K5's FACT share is ONE chat source (K5 chats plus SPEC 3
fact-QA filler chats), with no bio-doc filler. The loader mixes per source; at p = 1 the K5 chats are 114.4M of the
123.6M FACT draw, so a separate bio filler source would leave no room to hold every K5 chat inside the drawn part
of its own source with SPEC 7's margin (region = FB_REGION of the written stream, written = WRITE_MARGIN x draw).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kcommon as K
import world as Wd

K5_ARMS = {"L0": 0.0, "L50": 0.5, "L100": 1.0}
HIT_LEN, MISS_LEN, Q_LEN = 21, 10, 10


def hit_ids(name, attrs, vals) -> list[int]:
    body = [t for a, v in zip(attrs, vals) for t in (int(a), int(v))]
    return [K.LOOKUP, *[int(x) for x in name], K.LOOKUP_E, K.RESULT, *body, K.RESULT_E]


def miss_ids(name) -> list[int]:
    return [K.LOOKUP, *[int(x) for x in name], K.LOOKUP_E, K.RESULT, K.NONE, K.RESULT_E]


def k5_chat(name, attr_ids, val_ids, a1: int, a2: int, hit: bool, order, rid: str) -> dict:
    """attr_ids, val_ids: the entity's 6 attributes and values (world order); order: the hit line's pair order."""
    tool = hit_ids(name, [attr_ids[j] for j in order], [val_ids[j] for j in order]) if hit else miss_ids(name)
    nm = [int(x) for x in name]
    return {"id": rid, "turns": [
        {"role": "tool", "ids": tool},
        {"role": "user", "ids": [*nm, int(attr_ids[a1]), K.Q]}, {"role": "assistant", "ids": [int(val_ids[a1])]},
        {"role": "user", "ids": [*nm, int(attr_ids[a2]), K.Q]}, {"role": "assistant", "ids": [int(val_ids[a2])]}]}


@dataclass
class K5Cfg:
    arm: str = "L100"
    slots: int = K.B_SLOTS
    n5: int = K.N_LOW
    e5: int = K.E5
    shard_docs: int = 1 << 20

    def __post_init__(self):
        assert self.arm in K5_ARMS, self.arm

    @property
    def p(self) -> float:
        return K5_ARMS[self.arm]


def chats(c: K5Cfg) -> dict:
    """Arm-independent chat draws: entity, the two attribute indices, u, and the hit line's pair order."""
    r = K.rng("k5_train")
    ent, a1, a2 = [], [], []
    for e in range(c.n5):
        perms = np.argsort(r.random((c.e5, K.N_ATTR)), axis=1)      # e5 rounds of a permutation of 6
        a1.append(perms[:, 0::2].ravel())
        a2.append(perms[:, 1::2].ravel())
        ent.append(np.full(c.e5 * 3, e, dtype=np.int32))
    ent, a1, a2 = np.concatenate(ent), np.concatenate(a1), np.concatenate(a2)
    u = K.rng("k5_hits").random(len(ent))
    order = np.argsort(K.rng("k5_train", 1).random((len(ent), K.N_ATTR)), axis=1)
    return {"ent": ent, "a1": a1.astype(np.int8), "a2": a2.astype(np.int8), "u": u, "order": order.astype(np.int8)}


def layout5(c: K5Cfg, ch: dict) -> dict:
    hit = ch["u"] < c.p
    t5 = int(len(hit) * (2 * Q_LEN + MISS_LEN) + hit.sum() * (HIT_LEN - MISS_LEN))
    draw = K.expected_draw(K.SHARES["fact"], c.slots)
    written = math.ceil(K.WRITE_MARGIN * draw)
    region = int(K.FB_REGION * written)
    assert t5 <= region, f"K5 chats {t5} tokens exceed the region {region}"
    nf_region = (region - t5) // K.QA_LEN
    nf_total = (written - t5) // K.QA_LEN
    return {"hit": hit, "k5_tokens": t5, "draw_tokens": draw, "written_tokens": written, "region_tokens": region,
            "filler_records": int(nf_total), "filler_tokens": int(nf_total * K.QA_LEN),
            "records": int(len(hit) + nf_total), "region_records": int(len(hit) + nf_region),
            "share_chat": K.SHARES["fact"]}


def plan5(c: K5Cfg, ch: dict, lay: dict) -> np.ndarray:
    """chat index per record slot (-1 = filler QA). Positions depend on the arm (K5 notes K5-b)."""
    slot = np.full(lay["records"], -1, dtype=np.int64)
    r = K.rng("k5_slots", int(round(c.p * 100)))
    pos = r.choice(lay["region_records"], len(ch["ent"]), replace=False)
    slot[pos] = r.permutation(len(ch["ent"]))
    return slot


def k5_block(world, pools, ch: dict, hit: np.ndarray, slots: np.ndarray, shard: int, base: int) -> list[dict]:
    fill = Wd.qa_block(world, pools, np.full(len(slots), -1), np.zeros(len(slots), np.int8), shard, base)
    out = []
    for i, ci in enumerate(slots.tolist()):
        if ci < 0:
            out.append(fill[i])
            continue
        e = int(ch["ent"][ci])
        out.append(k5_chat(world.names[e], world.attrs, world.values[e], int(ch["a1"][ci]), int(ch["a2"][ci]),
                           bool(hit[ci]), ch["order"][ci].tolist(), f"k{base + i}"))
    return out
