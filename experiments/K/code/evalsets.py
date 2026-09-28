"""SPEC 8: eval, dev and probe files (fixed draws; skill items use T1-T3 names, K-EVAL-S uses its own triples).

Skill records carry "qs" (the final one is scored) and "cands" (SPEC 9 C, shuffled). Fact prompts are either plain
token prompts ("ids", the bio-doc prefix) or chat prompts ("turns", rendered by the scorer up to <|assistant|>).
Entities are allocated once, disjointly, from seeded permutations of FB_low QA-train, FB_low QA-held and
FB_high minus FB_low, in the order FB-PROBE, K-PROBE, K-DEV, K2 split A, K2 split B. CONFLICT and the K5 sets read
FB-PROBE's FB_low entities, so "known" can be read from each arm's own BIO recall of the same key.
"""
from __future__ import annotations

import numpy as np

import items as I
import kcommon as K
import lookup as L
import oracles as O
import skill as S

ALLOC = [("fbprobe", 2048, 2048, 4096), ("kprobe", 256, 256, 512), ("dev", 256, 256, 512),
         ("k2a", 128, 128, 256), ("k2b", 128, 128, 256)]


def allocate(world) -> dict:
    r = np.random.default_rng([K.W, K.SEEDS["eval"], 1])
    low = np.arange(K.N_LOW)
    pools = {"lt": r.permutation(low[world.qa_train[:K.N_LOW]]), "lh": r.permutation(low[~world.qa_train[:K.N_LOW]]),
             "ho": r.permutation(np.arange(K.N_LOW, K.N_HIGH))}
    at, out = {"lt": 0, "lh": 0, "ho": 0}, {}
    for name, *sizes in ALLOC:
        part = {}
        for key, n in zip(("lt", "lh", "ho"), sizes):
            part[key] = pools[key][at[key]:at[key] + n].tolist()
            at[key] += n
        out[name] = part
    return out


def skill_set(pools, lex, split, seed, per_fam, fams="RUBFAP", ood=None, groups=True, tag="") -> list[dict]:
    """groups: B's cube groups (skill.group, K3 round 3; a B block with groups holds at least 32 items)."""
    g = S.Gen(pools, split, ood)
    rc = np.random.default_rng([K.W, seed, 99])
    out = []
    for fam in fams:
        grp = groups and fam == "B"
        its = S.block(g, fam, max(per_fam, 32) if grp else per_fam,
                      (seed, ord(fam), S.OOD.index(ood) + 1 if ood else 0), groups=grp)
        for i, it in enumerate(its):
            rec = I.render(it, f"{fam}.b0.i{i}")
            rec["set"] = tag
            c = O.candidates(rec, rec["qs"][-1]["gold"], lex, rc)
            rec["cands"] = [c[j] for j in rc.permutation(len(c))]
            out.append(rec)
    return out


def mix_set(pools, lex, seed, n, tag) -> list[dict]:
    """A family mix at SPEC 4 shares (K-DEV, K-PROBE), no B groups."""
    per = {f: int(round(n * s)) for f, s in S.FAM_SHARE.items()}
    out = []
    for f, m in per.items():
        out += skill_set(pools, lex, "eval", seed, m, fams=f, groups=False, tag=tag)
    return out


def fact_cands(gold, pools, r, size) -> list[int]:
    v = pools["V"]
    extra = [int(x) for x in r.choice(v[v != gold], size - 1, replace=False)]
    c = [int(gold)] + extra
    return [c[j] for j in r.permutation(size)]


def bio_prompts(world, pools, ents, seed, tag, orders=2, csize=8, qa=True, block="") -> list[dict]:
    """Per entity and attribute: `orders` BIO prompts (F M L + 0-2 other true pairs + a) and one QA prompt."""
    r = np.random.default_rng([K.W, seed, 2])
    out = []
    for e in ents:
        nm = [int(x) for x in world.names[e]]
        for j in range(K.N_ATTR):
            gold = int(world.values[e, j])
            base = {"set": tag, "ent": int(e), "attr": j, "gold": gold, "block": block,
                    "cands": fact_cands(gold, pools, r, csize)}
            for o in range(orders):
                others = [x for x in r.permutation(K.N_ATTR).tolist() if x != j][:int(r.integers(0, 3))]
                ids = nm + [t for x in others for t in (int(world.attrs[x]), int(world.values[e, x]))]
                out.append({**base, "id": f"{tag}.bio.{e}.{j}.{o}", "kind": "bio",
                            "ids": ids + [int(world.attrs[j])]})
            if qa:
                out.append({**base, "id": f"{tag}.qa.{e}.{j}", "kind": "qa",
                            "turns": [{"role": "user", "ids": nm + [int(world.attrs[j]), K.Q]}]})
    return out


def conflict(world, pools, ents, seed, n, tag="CONFLICT") -> list[dict]:
    """n chats stating an FB key with a wrong value then asking it, and n CONSISTENT twins with the true value."""
    r = np.random.default_rng([K.W, seed, 3])
    out = []
    for i in range(n):
        e, j = int(ents[int(r.integers(0, len(ents)))]), int(r.integers(0, K.N_ATTR))
        nm, a, true = [int(x) for x in world.names[e]], int(world.attrs[j]), int(world.values[e, j])
        wrong = int(r.choice(pools["V"][pools["V"] != true]))
        sf, qf = int(r.integers(0, 2)), int(r.integers(0, 2))
        for kind, v in (("conflict", wrong), ("consistent", true)):
            st = nm + [a, v] if sf == 0 else [a] + nm + [v]
            q = nm + [a, K.Q] if qf == 0 else [a] + nm + [K.Q]
            rec = {"id": f"{tag}.{kind}.{i}", "set": tag, "fam": kind, "ent": e, "attr": j, "mem": true,
                   "turns": [{"role": "user", "ids": st}, {"role": "assistant", "ids": [K.OK]},
                             {"role": "user", "ids": q}, {"role": "assistant", "ids": [v]}],
                   "qs": [{"turn": 3, "gold": v}]}
            c = [v] + ([true] if v != true else []) + [int(x) for x in r.choice(pools["V"], 3, replace=False)
                                                       if x not in (v, true)][:3] + [K.NONE]
            rec["cands"] = [c[k] for k in r.permutation(len(c))]
            out.append(rec)
    return out


def k5_sets(world, pools, ents, seed=K.SEEDS["k5"]) -> list[dict]:
    """FB-HIT, UNSEEN-HIT 512 each; CF-HIT 2,048 [C8]; FB-MISS and FB-NOTOOL on `ents` x 6 (SPEC 8)."""
    r = np.random.default_rng([K.W, seed, 5])
    out = []

    def q(nm, a):
        return {"role": "user", "ids": [*nm, int(a), K.Q]}

    def add(tag, i, tool, nm, j, gold, mem, attrs=world.attrs):
        turns = ([{"role": "tool", "ids": tool}] if tool else []) + [q(nm, attrs[j])]
        out.append({"id": f"{tag}.{i}", "set": tag, "kind": "qa", "turns": turns, "gold": int(gold), "mem": int(mem),
                    "attr": int(j), "ent": -1 if tag == "UNSEEN-HIT" else int(ents_i[0]),
                    "cands": fact_cands(gold, pools, r, 8)})
    for tag, n in (("FB-HIT", 512), ("UNSEEN-HIT", 512), ("CF-HIT", 2048)):
        for i in range(n):
            j = int(r.integers(0, K.N_ATTR))
            order = r.permutation(K.N_ATTR).tolist()
            if tag == "UNSEEN-HIT":
                nm = [int(p[int(r.integers(0, len(p)))]) for p in pools.triple("T")]
                vals = [int(x) for x in r.choice(pools["V"], K.N_ATTR, replace=False)]
                ents_i, mem = [-1], -1
            else:
                ents_i = [int(ents[int(r.integers(0, len(ents)))])]
                nm, vals = [int(x) for x in world.names[ents_i[0]]], [int(x) for x in world.values[ents_i[0]]]
                mem = vals[j]
            if tag == "CF-HIT":
                vals = list(vals)
                vals[j] = int(r.choice([v for v in pools["V"].tolist() if v not in vals]))
            tool = L.hit_ids(nm, [world.attrs[k] for k in order], [vals[k] for k in order])
            add(tag, i, tool, nm, j, vals[j], mem)
    for e in ents:
        ents_i = [int(e)]
        nm = [int(x) for x in world.names[e]]
        for j in range(K.N_ATTR):
            v = int(world.values[e, j])
            add("FB-MISS", f"{e}.{j}", L.miss_ids(nm), nm, j, v, v)
            add("FB-NOTOOL", f"{e}.{j}", None, nm, j, v, v)
    return out


def mean_cands(recs) -> int:
    return int(round(float(np.mean([len(r["cands"]) for r in recs]))))


def one_per_entity(prompts: list[dict], seed: int) -> list[dict]:
    """K-DEV and K-PROBE: one seeded attribute per entity (not always the first)."""
    r = np.random.default_rng([K.W, seed, 6])
    pick = {e: int(r.integers(0, K.N_ATTR)) for e in sorted({p["ent"] for p in prompts})}
    return [p for p in prompts if p["attr"] == pick[p["ent"]]]
