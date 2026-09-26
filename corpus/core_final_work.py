"""Worker side of core_finalize.py (passes C and D of the core v0 plan), one stage1 shard per job so
the jobs run in spawned processes under the heavy-CPU lock, a chunk at a time.

- bp_pairs: the boilerplate count input of one stage1 shard. For every document that survived
  near-dedup (<stem>.nd.npy keep == 1): its distinct line hashes (boilerplate.doc_hashes) paired
  with the document's group, the hash of its URL key (boilerplate.url_key) or of its text when it
  has no URL. A line is boilerplate when it occurs in at least min_docs distinct groups of the
  source (count_bad), so a page captured in several snapshots counts once, as in the starter, and
  near-duplicates were already removed (CORPUS 3.2: near-dedup before boilerplate). Pairs are
  stored sorted by partition (top bits of the line hash) with offsets, so counting reads each
  partition's slice of every shard once.
- final_write: stage1 shard -> final shard, 1:1 (same index): rows a late file displaced dropped
  (dup_exact_rank, core_displaced.py), near-duplicates dropped (dup_near),
  boilerplate lines stripped and the text re-normalized (a document left under min_bytes is
  dropped as boilerplate_short), meta gains nd_csize (near-dup cluster size) and decon 'pending';
  writes <final stem>.jsonl.<codec> (one frame), .keys.u64 (sha1[:8] of the final text, as
  stream_io) and .info.npy (INFO_DTYPE: stage1 row, changed flag, cluster size per final line).
- drop_rows: pass D rewrite of a final shard without the given lines (exact duplicates created by
  boilerplate stripping), only for the shards that have any.
"""
import hashlib
import json
import os
from array import array

import numpy as np

import boilerplate as BP
import hygiene as H
import stream_io as SIO

INFO_DTYPE = np.dtype([("row", "<u4"), ("changed", "u1"), ("csize", "<u4")])


def group_hash(url, sha1_hex) -> int:
    key = BP.url_key(url)
    b = ("url:" + key).encode("utf-8") if key else bytes.fromhex(sha1_hex)
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


def bp_pairs(job):
    """job {src, nd, out, parts}: writes out.pairs.npy ((n, 2) uint64: line hash, group) sorted by
    partition and out.offs.npy (parts + 1 offsets). -> {pairs, docs}."""
    keep = np.load(job["nd"])["keep"]
    lines, groups, docs, i = array("Q"), array("Q"), 0, -1
    with SIO.open_shard(job["src"]) as f:
        for i, raw in enumerate(f):
            if not keep[i]:
                continue
            d = json.loads(raw)
            hs = BP.doc_hashes(d["text"])
            lines.extend(hs)
            groups.extend([group_hash(d["meta"].get("url"), d["meta"]["sha1"])] * len(hs))
            docs += 1
        if i + 1 != len(keep):
            raise RuntimeError(f"{job['src']}: {i + 1} lines, {len(keep)} near-dedup rows")
    lh, lg = np.frombuffer(lines, np.uint64), np.frombuffer(groups, np.uint64)
    part = part_of(lh, job["parts"])
    o = np.argsort(part, kind="stable")
    offs = np.searchsorted(part[o], np.arange(job["parts"] + 1))
    np.save(job["out"] + ".pairs.npy", np.stack([lh[o], lg[o]], axis=1))
    np.save(job["out"] + ".offs.npy", offs.astype(np.int64))
    return {"pairs": int(len(lh)), "docs": docs}


def part_of(h, parts):
    bits = int(parts).bit_length() - 1
    return (h >> np.uint64(64 - bits)).astype(np.int64) if bits else np.zeros(len(h), np.int64)


def count_bad(prefixes, parts, min_docs):
    """-> sorted uint64 array of the line hashes found in >= min_docs distinct groups."""
    out = []
    for p in range(parts):
        chunks = []
        for pre in prefixes:
            offs = np.load(pre + ".offs.npy")
            a = np.load(pre + ".pairs.npy", mmap_mode="r")
            chunks.append(np.array(a[offs[p]:offs[p + 1]]))
        if not chunks:
            continue
        pr = np.concatenate(chunks)
        if not len(pr):
            continue
        o = np.lexsort((pr[:, 1], pr[:, 0]))
        pr = pr[o]
        new = np.ones(len(pr), dtype=bool)
        new[1:] = (pr[1:, 0] != pr[:-1, 0]) | (pr[1:, 1] != pr[:-1, 1])
        u, c = np.unique(pr[new, 0], return_counts=True)
        out.append(u[c >= min_docs])
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.uint64)


def strip_lines(text, bad):
    """-> text without the lines whose hash is in the sorted array bad, re-normalized (unchanged
    text when none is)."""
    parts = text.split("\n")
    hs = np.array([BP.line_hash(x.strip()) if x.strip() else 0 for x in parts], dtype=np.uint64)
    i = np.minimum(np.searchsorted(bad, hs), len(bad) - 1)
    hit = (bad[i] == hs) & np.array([bool(x.strip()) for x in parts])
    if not hit.any():
        return text
    return H.normalize("\n".join(x for x, h in zip(parts, hit) if not h))


def _drop(st, why, b):
    d = st.setdefault(why, {"docs": 0, "bytes": 0})
    d["docs"] += 1
    d["bytes"] += b


def _finish(dst, fr, fo, h, keys, info):
    b = fr.end()
    fo.write(b)
    h.update(b)
    fo.flush()
    os.fsync(fo.fileno())
    fo.close()
    stem = dst.rsplit(".jsonl.", 1)[0]
    np.asarray(keys, dtype="<u8").tofile(stem + ".keys.u64.part")
    np.save(stem + ".info.part.npy", np.asarray(info, dtype=INFO_DTYPE))
    os.replace(stem + ".keys.u64.part", stem + ".keys.u64")
    os.replace(stem + ".info.part.npy", stem + ".info.npy")
    os.replace(dst + ".part", dst)
    return h.hexdigest()


def final_write(job):
    """job {src, nd, bad (npy path or None), dst, codec, level, min_bytes, excl (displaced
    rows)}. -> stats."""
    nd, excl = np.load(job["nd"]), set(job.get("excl") or ())
    bad = np.load(job["bad"]) if job.get("bad") else None
    bad = bad if bad is not None and len(bad) else None
    st = {"docs_in": 0, "bytes_in": 0, "docs_kept": 0, "bytes_kept": 0, "ubytes": 0,
          "dropped": {}, "stripped": {"docs": 0, "bytes": 0}}
    keys, info = [], []
    os.makedirs(os.path.dirname(job["dst"]), exist_ok=True)
    fr, h = SIO.Frame(job["codec"], job["level"]), hashlib.sha256()
    with SIO.open_shard(job["src"]) as f, open(job["dst"] + ".part", "wb") as fo:
        for i, raw in enumerate(f):
            d = json.loads(raw)
            tb = H.nbytes(d["text"])
            st["docs_in"] += 1
            st["bytes_in"] += tb
            if i in excl:
                _drop(st["dropped"], "dup_exact_rank", tb)
                continue
            if not nd["keep"][i]:
                _drop(st["dropped"], "dup_near", tb)
                continue
            changed = 0
            if bad is not None:
                text = strip_lines(d["text"], bad)
                if text != d["text"]:
                    nb = H.nbytes(text)
                    if nb < job["min_bytes"]:
                        _drop(st["dropped"], "boilerplate_short", tb)
                        continue
                    st["stripped"]["docs"] += 1
                    st["stripped"]["bytes"] += tb - nb
                    d["text"], changed = text, 1
                    d["meta"]["boilerplate_bytes"] = tb - nb
                    d["meta"]["sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
            d["meta"]["nd_csize"] = int(nd["csize"][i])
            d["meta"]["decon"] = "pending"
            line = (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")
            b = fr.compress(line)
            fo.write(b)
            h.update(b)
            keys.append(int(d["meta"]["sha1"][:16], 16))
            info.append((i, changed, int(nd["csize"][i])))
            st["docs_kept"] += 1
            st["bytes_kept"] += H.nbytes(d["text"])
            st["ubytes"] += len(line)
        if st["docs_in"] != len(nd):
            raise RuntimeError(f"{job['src']}: {st['docs_in']} lines, {len(nd)} near-dedup rows")
        st["sha256"] = _finish(job["dst"], fr, fo, h, keys, info)
    st["bytes"] = os.path.getsize(job["dst"])
    return st


def drop_rows(job):
    """job {dst, codec, level, drop (final line numbers)}: rewrite the shard without them.
    -> {docs, ubytes, bytes, sha256, dropped_bytes}."""
    drop, stem = set(job["drop"]), job["dst"].rsplit(".jsonl.", 1)[0]
    info = np.load(stem + ".info.npy")
    keys = SIO.read_keys(stem + ".keys.u64")
    fr, h = SIO.Frame(job["codec"], job["level"]), hashlib.sha256()
    kk, ii, n, ub, gone = [], [], 0, 0, 0
    with SIO.open_shard(job["dst"]) as f, open(job["dst"] + ".part", "wb") as fo:
        for i, line in enumerate(f):
            if i in drop:
                gone += H.nbytes(json.loads(line)["text"])
                continue
            b = fr.compress(line)
            fo.write(b)
            h.update(b)
            kk.append(int(keys[i]))
            ii.append(tuple(info[i]))
            n, ub = n + 1, ub + len(line)
        sha = _finish(job["dst"], fr, fo, h, kk, ii)
    return {"docs": n, "ubytes": ub, "bytes": os.path.getsize(job["dst"]), "sha256": sha,
            "dropped_bytes": gone}
