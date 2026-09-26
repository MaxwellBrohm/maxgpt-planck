"""Displaced stage1 rows for core_finalize.py: the copies that pass A kept, but that a better-ranked
file committed later (a late file, stream_rank.py) also holds.

A late file's file_done event lists, per lower-ranked owner file, the text keys and id hashes it
took over ("displaced": [{"fid", "keys", "ids"}]). displaced_rows() finds those documents in the
owner's segments of the stage1 shards: text keys through the shard's .keys.u64 rows, id hashes by
reading the segment's lines. core_finalize.py leaves these rows out of near-dedup (an include
mask) and drops them from the final shards as dup_exact_rank, so the final output is the one an
uninterrupted run gives. Every listed key and id must be found, or finalize stops. No separate
input-hash term is needed: a late file changes its own tier's shards (and so every later tier's
input), and run_core_v0.sh seals every shard of a tier (--seal-open) before finalizing it.
"""
import json
import os

import numpy as np

import stream_io as SIO
from stream_ledger import id_hash


def _stem(stage1, rel):
    return os.path.join(stage1, rel.rsplit(".jsonl.", 1)[0])


def _ids_in(path, r0, n, want):
    """-> {offset j < n: id hash} for the lines r0 + j of the shard whose id hash is in want."""
    hit = {}
    with SIO.open_shard(path) as f:
        for i, line in enumerate(f):
            if i >= r0 + n:
                break
            if i >= r0:
                h = id_hash(json.loads(line)["id"])
                if h in want:
                    hit[i - r0] = h
    return hit


def displaced_rows(stage1, events, shards) -> dict:
    """-> {stage1 shard rel: sorted int64 rows} displaced by late files, for the shards in
    `shards` (a set of rels). Ledger events in order; the last file_done of a fid counts."""
    done = {e["fid"]: e for e in events if e["event"] == "file_done"}
    want = {}
    for e in done.values():
        for d in e.get("displaced") or ():
            w = want.setdefault(d["fid"], (set(), set()))
            w[0].update(int(x) for x in d.get("keys") or ())
            w[1].update(int(x) for x in d.get("ids") or ())
    out = {}
    for fid, (keys, ids) in sorted(want.items()):
        if fid not in done:
            raise SystemExit(f"{fid}: displaced by a late file but not done in the ledger")
        segs = done[fid]["segments"]
        if not {s["shard"] for s in segs} & shards:
            continue
        found_k, found_i = set(), set()
        for s in segs:
            r0, n = s["kstart"] // 8, (s["kend"] - s["kstart"]) // 8
            k = np.fromfile(_stem(stage1, s["shard"]) + ".keys.u64", dtype="<i8", count=n,
                            offset=s["kstart"])
            hit = np.isin(k, np.fromiter(keys, np.int64, len(keys)))
            found_k.update(int(x) for x in k[hit])
            if ids:
                got = _ids_in(os.path.join(stage1, s["shard"]), r0, n, ids)
                hit[list(got)] = True
                found_i.update(got.values())
            if s["shard"] in shards:
                out.setdefault(s["shard"], []).extend((r0 + np.flatnonzero(hit)).tolist())
        if keys - found_k or ids - found_i:
            raise SystemExit(f"{fid}: {len(keys - found_k)} displaced keys and "
                             f"{len(ids - found_i)} ids not found in its stage1 rows")
    return {sh: np.unique(np.asarray(r, dtype=np.int64)) for sh, r in out.items() if r}


def include_masks(rows, mine, work) -> dict:
    """Write <work>/displaced/<n>.inc.npy (True = takes part in near-dedup) for every shard of
    `mine` with displaced rows. -> {shard rel: mask path}."""
    out, d = {}, os.path.join(work, "displaced")
    for n, s in enumerate(mine):
        if s["shard"] in rows:
            os.makedirs(d, exist_ok=True)
            m = np.ones(s["docs"], dtype=bool)
            m[rows[s["shard"]]] = False
            out[s["shard"]] = os.path.join(d, f"{n:05d}.inc.npy")
            np.save(out[s["shard"]], m)
    return out
