"""MinHash LSH near-dedup, the global half (CORPUS 3.2 step 2, pass B). Pass A is neardedup.py.

    python corpus/neardedup_lsh.py SPEC.json WORK_DIR [--mem-mb 1024] [--workers 8] [--verify 0.7]
                                   [--chain-floor 0.7]

SPEC.json is a list of shards in priority order, each {"stem": path prefix of a stage1 shard whose
sidecar is stem + ".mh.npy" (or ".mh.u32"), "tier": 0-255 (lower wins), "label":
"cccc:CC-MAIN-2019-04" (stats only), "frozen": false, "include": null or a bool .npy path}.
find_near_dups() takes the same list.

Method. Each doc gets a rank from (tier, date, shard order, row): lower tier first, then earlier
date (undated after dated), then file order. For each of the 14 bands, docs are grouped by band key
(neardedup.band_hashes); in every group of 2+ docs, each member is joined to the group's lowest-rank
member by an edge, which is kept only if the two signatures agree on at least `verify` of their 126
positions (skipped when the two are already connected). Edges merge docs in a union-find whose
roots are the lowest rank of their component. The keep rule (neardedup_keep.py): every non-root is
dropped with its root as lead; with --chain-floor, a non-root less similar than that to its root is
released and re-clustered in another round, so long chains through shared templates do not drop
docs that resemble no kept doc. Frozen shards (tiers already finalized) join the graph but are
never dropped; docs outside "include" (dropped upstream) take no part. The result does not depend
on --mem-mb or --workers.

Memory (tracemalloc, 1M docs): about 22 bytes per document held for the run (parent, rank,
order and lead arrays, flags, uint32 cluster counts), 26 at the peak while the rank is built, plus
one band partition at a time sized to --mem-mb (64 bytes per document in it), plus up to 0.3 GB of
signatures while verifying. Disk: band keys, 112 bytes per document, in WORK_DIR/bands (deleted
at the end). Reads: each band row once per partition and round, and the signature rows of
candidate pairs; sidecars are mapped only while read.

Output, for every non-frozen shard: <stem>.nd.npy with one row per document in shard order,
dtype ND_DTYPE: keep (1 = survives, 0 = near-duplicate or excluded), csize (size of its cluster:
its lead plus the docs that name that lead; 0 for excluded docs), lshard and lrow (the lead, as an
index into the shard list and a row; a kept doc names itself). WORK_DIR/neardedup_report.json:
parameters, rounds, per-label counts, dropped-by-label x lead-label counts, cluster-size histogram,
the 20 largest clusters and an audit sample of 200 dropped docs with their lead and signature
Jaccard.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

import neardedup as ND
import neardedup_keep as NK
from neardedup_keep import UnionFind  # noqa: F401  (tests use it from here)
from stream_lock import exit_with_parent

ND_DTYPE, ND_SUFFIX = ND.ND_DTYPE, ND.ND_SUFFIX
BYTES_PER_ENTRY = 64           # peak bytes per partition entry (keys, ranks, sort order, temps)
DAY_CAP = (1 << 20) - 1        # UNDATED and any ordinal past year 2870 sort last within a tier


def _bands_job(job):
    k, stem, path = job
    np.save(path, ND.band_hashes(ND.load_sidecar(stem, mmap=True)["sig"]))
    return k


class _Run:
    def __init__(self, shards, verify):
        self.shards, self.verify = shards, verify
        days = []                       # sidecars are memory-mapped only while read, so file
        for s in shards:                # pages do not pile up in this process's RSS
            a = ND.load_sidecar(s["stem"], mmap=True)
            days.append(np.array(a["day"]))
            del a
        n = np.array([len(d) for d in days], dtype=np.int64)
        self.off = np.concatenate(([0], np.cumsum(n)))
        self.N = int(self.off[-1])
        if self.N >= 1 << 36:
            raise ValueError("more than 2^36 documents")
        self.idt = np.uint32 if self.N < (1 << 32) - 1 else np.uint64
        self.none = np.iinfo(self.idt).max
        self.active = np.ones(self.N, dtype=bool)
        self.frozen = np.zeros(self.N, dtype=bool)
        key = np.empty(self.N, dtype=np.uint64)
        for k, s in enumerate(shards):
            a, b = self.off[k], self.off[k + 1]
            inc = s.get("include")
            if inc is not None:
                m = np.load(inc) if isinstance(inc, str) else np.asarray(inc)
                if m.shape != (b - a,):
                    raise ValueError(f"{s['stem']}: include mask shape {m.shape} != {(b - a,)}")
                self.active[a:b] = m.astype(bool)
            self.frozen[a:b] = bool(s.get("frozen"))
            day = np.minimum(days[k].astype(np.uint64), DAY_CAP)
            if not 0 <= int(s.get("tier", 0)) < 256:
                raise ValueError(f"{s['stem']}: tier must be 0-255")
            tier = np.uint64(int(s.get("tier", 0)))
            key[a:b] = (tier << np.uint64(56)) | (day << np.uint64(36)) | \
                np.arange(a, b, dtype=np.uint64)
        self.order = np.argsort(key, kind="stable").astype(self.idt)   # rank -> gid
        del key
        self.rank = np.empty(self.N, dtype=self.idt)                   # gid -> rank
        self.rank[self.order] = np.arange(self.N, dtype=self.idt)
        self.included = self.active.copy()
        self.lead = np.full(self.N, self.none, dtype=self.idt)          # rank -> lead rank
        self.uf = UnionFind(self.N, self.idt)
        self.st = dict(candidate_edges=0, verified=0, rejected=0, merges=0, rounds=0,
                       released=0, unsettled=0)

    def locate(self, gid):
        k = np.searchsorted(self.off, gid, side="right") - 1
        return k, gid - self.off[k]

    def sigs(self, ranks):
        """Signatures (len(ranks), NPERM) of the docs at these ranks."""
        k, row = self.locate(self.order[ranks].astype(np.int64))
        out = np.empty((len(ranks), ND.NPERM), dtype=np.uint32)
        for kk in np.unique(k):
            m = k == kk
            rows = row[m]
            srt = np.argsort(rows)
            got = np.empty((len(rows), ND.NPERM), dtype=np.uint32)
            a = ND.load_sidecar(self.shards[kk]["stem"], mmap=True)
            got[srt] = a["sig"][rows[srt]]
            del a
            out[m] = got
        return out

    def edges(self, h, r):
        """Band keys h and ranks r of one partition -> (member, leader) rank pairs."""
        o = np.lexsort((r, h))
        h, r = h[o], r[o]
        new = np.empty(len(h), dtype=bool)
        new[:1] = True
        np.not_equal(h[1:], h[:-1], out=new[1:])
        lead = r[np.flatnonzero(new)][np.cumsum(new) - 1]
        m = r != lead
        return r[m], lead[m]

    def link(self, u, v):
        self.st["candidate_edges"] += len(u)
        ru, rv = self.uf.find(u), self.uf.find(v)
        m = ru != rv
        u, v = u[m], v[m]
        if self.verify > 0 and len(u):
            ok = NK.sig_jaccard(self, u, v) >= self.verify - 1e-9
            self.st["verified"] += int(ok.sum())
            self.st["rejected"] += int((~ok).sum())
            u, v = u[ok], v[ok]
        self.st["merges"] += self.uf.union(u, v)


def _parts(n_active, mem_mb):
    need = max(1, n_active * BYTES_PER_ENTRY)
    p = 1
    while p < 1 << 12 and need / p > mem_mb * (1 << 20):
        p *= 2
    return p


def _link_round(run, bands, mem_mb):
    P = _parts(int(run.active.sum()), mem_mb)
    shift = np.uint64(65 - P.bit_length())           # top log2(P) bits pick the partition
    for b in range(ND.BANDS):
        for p in range(P):
            hs, rs = [], []
            for k, path in enumerate(bands):
                hb = np.load(path, mmap_mode="r")[b]
                a, e = run.off[k], run.off[k + 1]
                m = run.active[a:e]
                if P > 1:
                    m = m & ((hb >> shift) == np.uint64(p))
                idx = np.flatnonzero(m)
                hs.append(np.asarray(hb[idx]))
                rs.append(run.rank[a + idx])
            run.link(*run.edges(np.concatenate(hs), np.concatenate(rs)))
        run.uf.compress()
    return P


def find_near_dups(shards, work_dir, verify=ND.VERIFY, mem_mb=1024, workers=1, write=True,
                   audit_n=200, top_n=20, chain_floor=0.0, max_rounds=10):
    t0 = time.time()
    run = _Run(shards, verify)
    bdir = os.path.join(work_dir, "bands")
    shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(bdir)
    jobs = [(k, s["stem"], os.path.join(bdir, f"{k:05d}.npy")) for k, s in enumerate(shards)]
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs)), exit_with_parent,
                                          (os.getpid(),)) as pool:
            list(pool.imap_unordered(_bands_job, jobs))
    else:
        for j in jobs:
            _bands_job(j)
    for rnd in range(1, max_rounds + 1):
        if rnd > 1:
            run.uf = UnionFind(run.N, run.idt)
        P = _link_round(run, [j[2] for j in jobs], mem_mb)
        run.st["rounds"] = rnd
        run.st["unsettled"] = NK.settle(run, chain_floor)
        if not run.st["unsettled"]:
            break
    shutil.rmtree(bdir, ignore_errors=True)
    report = NK.finish(run, shards, write, audit_n, top_n)
    report.update(partitions=P, seconds=round(time.time() - t0, 1), mem_mb=mem_mb,
                  chain_floor=chain_floor)
    with open(os.path.join(work_dir, "neardedup_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def load_result(stem):
    return np.load(stem + ND_SUFFIX)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("work_dir")
    ap.add_argument("--mem-mb", type=float, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify", type=float, default=ND.VERIFY, help="0 disables verification")
    ap.add_argument("--chain-floor", type=float, default=0.0, help="0: plain union-find")
    a = ap.parse_args(argv)
    with open(a.spec) as f:
        shards = json.load(f)
    os.makedirs(a.work_dir, exist_ok=True)
    r = find_near_dups(shards, a.work_dir, verify=a.verify, mem_mb=a.mem_mb, workers=a.workers,
                       chain_floor=a.chain_floor)
    print(json.dumps({k: r[k] for k in ("docs", "active", "dropped", "clusters", "edges",
                                        "partitions", "seconds")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
