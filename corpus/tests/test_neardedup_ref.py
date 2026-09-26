"""neardedup_lsh against slow pure-Python references: the array union-find against a dict
union-find, and the whole pass B against a brute-force banding + union-find."""
import os
import random

import numpy as np

import neardedup as ND
import neardedup_lsh as NL
from nd_fixtures import Gen, write_shard


class RefDSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def test_union_find_matches_reference_in_conflicting_batches():
    r, n = random.Random(3), 3000
    uf, ref = NL.UnionFind(n, np.uint32), RefDSU()
    for batch in range(30):
        hubs = r.sample(range(n), 5)            # many edges share endpoints: hooking conflicts
        e = [(r.choice(hubs), r.randrange(n)) if r.random() < 0.5 else
             (r.randrange(n), r.randrange(n)) for _ in range(150)]
        u = np.array([a for a, _ in e], dtype=np.uint32)
        v = np.array([b for _, b in e], dtype=np.uint32)
        uf.union(u, v)
        for a, b in e:
            ref.union(a, b)
        uf.compress()
        assert uf.p.tolist() == [ref.find(i) for i in range(n)], batch


def reference_keep(stems, tiers, verify):
    sigs = [ND.load_sidecar(s, mmap=False) for s in stems]
    docs = [(t, min(int(d), NL.DAY_CAP), k, i) for k, (a, t) in enumerate(zip(sigs, tiers))
            for i, d in enumerate(a["day"])]
    order = sorted(range(len(docs)), key=lambda g: docs[g])
    rank = {g: r for r, g in enumerate(order)}
    allsig = np.concatenate([a["sig"] for a in sigs])
    bands = ND.band_hashes(allsig)
    dsu = RefDSU()
    for b in range(ND.BANDS):
        buckets = {}
        for g in range(len(docs)):
            buckets.setdefault(int(bands[b, g]), []).append(rank[g])
        for members in buckets.values():
            lead = min(members)
            for m in members:
                j = (allsig[order[m]] == allsig[order[lead]]).mean()
                if m != lead and j >= verify - 1e-9:
                    dsu.union(m, lead)
    keep = [dsu.find(rank[g]) == rank[g] for g in range(len(docs))]
    out, start = [], 0
    for a in sigs:
        out.append(keep[start:start + len(a)])
        start += len(a)
    return out


def test_pass_b_matches_brute_force_reference(tmp_path):
    g, texts = Gen(61), []
    for _ in range(120):
        ws = g.words(g.r.randint(60, 300))
        texts += [" ".join(g.near_copy(ws, t)) for t in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66)]
    for _ in range(4):                          # chains: neighbours similar, ends not
        ws = g.words(600)
        texts += [" ".join(ws[i * 25:i * 25 + 300]) for i in range(10)]
    random.Random(0).shuffle(texts)
    stems = [write_shard(tmp_path, f"s{k}", texts[k::3],
                         [None if i % 7 == 0 else f"20{10 + i % 9}-0{1 + k}-01"
                          for i in range(len(texts[k::3]))]) for k in range(3)]
    tiers = [1, 0, 1]
    for verify in (ND.VERIFY, 0.0):
        rep = NL.find_near_dups([{"stem": s, "tier": t} for s, t in zip(stems, tiers)],
                                str(tmp_path), verify=verify, mem_mb=0.01)
        got = [NL.load_result(s)["keep"].astype(bool).tolist() for s in stems]
        assert got == reference_keep(stems, tiers, verify), verify
        assert rep["partitions"] > 1 and rep["dropped"] > 200
        assert os.path.exists(tmp_path / "neardedup_report.json")
