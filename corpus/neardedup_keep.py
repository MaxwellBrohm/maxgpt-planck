"""The keep rule of neardedup_lsh.py (pass B): the union-find, the end of each round (settle), and
the per-shard .nd.npy files plus the report (finish), one shard at a time so memory stays bounded.

A round links docs (neardedup_lsh) and then settles: every active, non-frozen doc that is not the
root (lowest rank) of its component is dropped, with that root as its lead. With chain_floor > 0 a
non-root whose signature Jaccard with the root is under chain_floor is released instead (kept for
the next round, where it is clustered again among the survivors); rounds repeat until nothing is
released. chain_floor = 0 is plain union-find in one round (the CORPUS 3.2 method), where a chain
A~B~C drops C with A as its lead even when A and C are not similar. Measured on the 10 starter CCCC
files (845,006 gated docs, 2026-09-26): plain drops 213,382 docs (644 MB), and 41 of its 200 audit
pairs sit under 0.7 with their lead; chain_floor 0.7 keeps 16,907 of them (24 MB) and drops 217
others, ran the 10-round cap (9,162 released docs left unsettled, so kept) and took 5x as long.

Report: counts per label, dropped-by-lead label pairs (per source pair, or per snapshot pair when
labels name snapshots), the cluster-size histogram (a cluster is a kept or dropped lead plus the
docs that name it as lead), the largest clusters, and a deterministic audit sample of dropped docs
with the signature Jaccard to their lead (a chained member can sit under the verify threshold:
that is what the sample is for)."""
import numpy as np

import neardedup as ND

VERIFY_CHUNK = 1 << 18         # pairs compared at a time: 2 x 128 MB of signatures


class UnionFind:
    """Array union-find over ids 0..n-1 whose root is always the smallest id in its set."""

    def __init__(self, n, dtype):
        self.p = np.arange(n, dtype=dtype)

    def find(self, x):
        r = self.p[x]
        for _ in range(64):
            nr = self.p[r]
            if np.array_equal(nr, r):
                self.p[x] = r
                return r
            r = nr
        self.compress()
        return self.p[x]

    def compress(self):
        while True:
            q = self.p[self.p]
            if np.array_equal(q, self.p):
                return
            self.p = q

    def union(self, u, v):
        """Joins u[i] with v[i]; -> number of joins that merged two sets."""
        merged = 0
        while len(u):
            ru, rv = self.find(u), self.find(v)
            m = ru != rv
            if not m.any():
                break
            u, v, ru, rv = u[m], v[m], ru[m], rv[m]
            hi = np.maximum(ru, rv)
            before = np.unique(hi).size
            np.minimum.at(self.p, hi, np.minimum(ru, rv))
            merged += before
        return merged


def sig_jaccard(run, a, b):
    """Signature Jaccard of the docs at ranks a[i] and b[i], in chunks."""
    return np.concatenate([ND.est_jaccard(run.sigs(a[i:i + VERIFY_CHUNK]),
                                          run.sigs(b[i:i + VERIFY_CHUNK]))
                           for i in range(0, len(a), VERIFY_CHUNK)] or [np.zeros(0)])


def settle(run, chain_floor=0.0, chunk=1 << 22):
    """End of a round (see the module doc). -> number of docs released."""
    run.uf.compress()
    released = 0
    for c0 in range(0, run.N, chunk):
        r = np.arange(c0, min(run.N, c0 + chunk), dtype=run.idt)
        root = run.uf.p[c0:c0 + len(r)]
        g = run.order[r].astype(np.int64)
        m = (root != r) & run.active[g] & ~run.frozen[g]
        r, root, g = r[m], root[m], g[m]
        if chain_floor > 0 and len(r):
            ok = sig_jaccard(run, r, root) >= chain_floor - 1e-9
            released += int((~ok).sum())
            r, root, g = r[ok], root[ok], g[ok]
        run.lead[r] = root
        run.active[g] = False
    run.st["released"] += released
    return released


def _hist(sizes):
    """Cluster sizes -> {'2': n, '3-4': n, '5-8': n, ...}."""
    out, lo = {}, 2
    while sizes.size and lo <= sizes.max():
        hi = 2 * lo - 1 if lo > 2 else 2
        n = int(((sizes >= lo) & (sizes <= hi)).sum())
        if n:
            out[str(lo) if lo == hi else f"{lo}-{hi}"] = n
        lo = hi + 1
    return out


def finish(run, shards, write, audit_n=200, top_n=20):
    led = run.lead[run.lead != run.none]
    counts = np.zeros(run.N, dtype=np.uint32)              # rank -> docs that name it as lead
    np.add.at(counts, led, 1)
    labels = [str(s.get("label", "")) for s in shards]
    names = sorted(set(labels))
    lab = np.array([names.index(x) for x in labels], dtype=np.int64)
    by = {n: {"docs": 0, "active": 0, "kept": 0, "dropped": 0} for n in names}
    pairs = np.zeros((len(names), len(names)), dtype=np.int64)
    heads, sizes = [], []
    aud_g, aud_key = np.zeros(0, np.int64), np.zeros(0, np.uint64)
    for k, s in enumerate(shards):
        a, b = int(run.off[k]), int(run.off[k + 1])
        rk, inc = run.rank[a:b], run.included[a:b]
        ld = run.lead[rk]
        drop = ld != run.none
        keep = inc & ~drop
        head = np.where(drop, ld, rk)
        csize = 1 + counts[head]
        csize[~inc] = 0
        lk, lr = run.locate(run.order[head].astype(np.int64))
        if write and not s.get("frozen"):
            out = np.zeros(b - a, dtype=ND.ND_DTYPE)
            out["keep"], out["csize"], out["lshard"], out["lrow"] = keep, csize, lk, lr
            ND.save_atomic(s["stem"] + ND.ND_SUFFIX, out)
        st = by[labels[k]]
        for key, m in (("docs", None), ("active", inc), ("kept", keep), ("dropped", drop)):
            st[key] += (b - a) if m is None else int(m.sum())
        pairs[lab[k]] += np.bincount(lab[lk[drop]], minlength=len(names))
        h = counts[rk] > 0
        heads.append(np.flatnonzero(h) + a)
        sizes.append(1 + counts[rk[h]].astype(np.int64))
        g = np.flatnonzero(drop) + a
        aud_g = np.concatenate([aud_g, g])
        aud_key = np.concatenate([aud_key, ND.mix64(g.astype(np.uint64))])
        o = np.argsort(aud_key, kind="stable")[:audit_n]
        aud_g, aud_key = aud_g[o], aud_key[o]
    heads, sizes = np.concatenate(heads), np.concatenate(sizes)
    largest = []
    for i in np.argsort(-sizes, kind="stable")[:top_n]:
        k, r = run.locate(np.array([heads[i]]))
        largest.append({"stem": shards[int(k[0])]["stem"], "row": int(r[0]),
                        "size": int(sizes[i])})
    audit = []
    if aud_g.size:
        pick = np.sort(aud_g)
        lead = run.lead[run.rank[pick]]
        lj = sig_jaccard(run, run.rank[pick], lead)
        (dk, dr), (lk, lr) = run.locate(pick), run.locate(run.order[lead].astype(np.int64))
        for i in range(len(pick)):
            audit.append({"stem": shards[int(dk[i])]["stem"], "row": int(dr[i]),
                          "leader_stem": shards[int(lk[i])]["stem"], "leader_row": int(lr[i]),
                          "csize": int(1 + counts[lead[i]]), "sig_jaccard": round(float(lj[i]), 3)})
    pair_counts = {f"{names[i]} <- {names[j]}": int(pairs[i, j])
                   for i, j in zip(*np.nonzero(pairs))}
    return {"params": ND.PARAMS, "verify": run.verify, "docs": run.N,
            "active": int(run.included.sum()), "dropped": sum(v["dropped"] for v in by.values()),
            "clusters": int(sizes.size), "docs_in_clusters": int(sizes.sum()),
            "edges": dict(run.st), "by_label": by, "dropped_by_leader_label": pair_counts,
            "cluster_size_hist": _hist(sizes), "largest": largest, "audit": audit,
            "shards": [s["stem"] for s in shards]}
