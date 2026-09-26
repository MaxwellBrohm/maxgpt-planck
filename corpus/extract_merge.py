"""The merge step of extract.py, after the per-file workers:

1. boilerplate line removal (boilerplate.py) for boilerplate.SOURCES, from the line hashes the
   workers wrote (tmp .lh.npy / .ln.npy / .lu); a doc whose sha1 or URL was already seen adds
   nothing to the counts;
2. global exact dedup (sha1 of the final text) and doc id uniqueness, in the order extract.py
   passes the results (source order), so the first copy wins;
3. shards per source, with sha256, docs and bytes.
"""
import hashlib
import json
import os

import numpy as np

import boilerplate as BP
import hygiene as H


def add_drop(st, reason, b):
    d = st["dropped"].setdefault(reason, {"docs": 0, "bytes": 0})
    d["docs"] += 1
    d["bytes"] += b


class ShardWriter:
    def __init__(self, out_dir, source, shard_bytes):
        self.dir, self.source, self.limit = os.path.join(out_dir, source), source, shard_bytes
        os.makedirs(self.dir, exist_ok=True)
        self.n, self.fh, self.outputs = 0, None, []

    def write(self, line: bytes):
        if self.fh is None or self.cur["bytes"] + len(line) > self.limit and self.cur["docs"]:
            self.close()
            path = os.path.join(self.dir, f"{self.source}-{self.n:05d}.jsonl")
            self.n += 1
            self.fh, self.h = open(path, "wb"), hashlib.sha256()
            self.cur = {"path": path, "source": self.source, "docs": 0, "bytes": 0}
        self.fh.write(line)
        self.h.update(line)
        self.cur["docs"] += 1
        self.cur["bytes"] += len(line)

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.cur["sha256"] = self.h.hexdigest()
            self.outputs.append(self.cur)
            self.fh = None


def boilerplate_sets(results, tmp_dir, min_docs):
    """-> {source: frozenset of boilerplate line hashes}; a repeated sha1 or URL counts once."""
    parts, seen = {}, {}
    for idx, (source, _, _, _) in enumerate(results):
        tmp = os.path.join(tmp_dir, f"{idx:05d}")
        if not os.path.exists(tmp + ".lh.npy"):
            continue
        h, n = np.load(tmp + ".lh.npy"), np.load(tmp + ".ln.npy")
        with open(tmp + ".keys") as fk, open(tmp + ".lu", encoding="utf-8") as fu:
            shas = [k.split("\t", 1)[0] for k in fk]
            urls = [u.rstrip("\n") for u in fu]
        assert len(shas) == len(n) == len(urls), f"{tmp}: line-hash sidecar out of step"
        s = seen.setdefault(source, set())
        first = np.zeros(len(shas), dtype=bool)
        for i, (sha, url) in enumerate(zip(shas, urls)):
            key = "url:" + url if url else None
            if sha not in s and key not in s:
                first[i] = True
            s.add(sha)
            if key:
                s.add(key)
        parts.setdefault(source, []).append(h[np.repeat(first, n)])
    return {s: BP.frequent_lines(np.concatenate(a), min_docs) for s, a in parts.items()}


def merge(results, tmp_dir, out_dir, shard_bytes, stats, o):
    """Copies passed lines into shards, updating stats in place; -> the shard list."""
    bad = boilerplate_sets(results, tmp_dir, o["boilerplate_min_docs"])
    seen, ids, writers = set(), set(), {}
    for idx, (source, rel, st, _) in enumerate(results):
        tot = stats[source]
        for k in ("files", "docs_in", "bytes_in"):
            tot[k] += st[k]
        for r, d in st["dropped"].items():
            t = tot["dropped"].setdefault(r, {"docs": 0, "bytes": 0})
            t["docs"] += d["docs"]
            t["bytes"] += d["bytes"]
        for k, v in st["notes"].items():
            tot["notes"][k] = tot["notes"].get(k, 0) + v
        bp = bad.get(source)
        if bp is not None:
            tot["notes"]["boilerplate_lines"] = len(bp)
        w = writers.get(source) or writers.setdefault(source, ShardWriter(out_dir, source,
                                                                          shard_bytes))
        tmp = os.path.join(tmp_dir, f"{idx:05d}")
        with open(tmp + ".jsonl", "rb") as fj, open(tmp + ".keys") as fk:
            for line, key in zip(fj, fk):
                sha, doc_id, kb, rb, und = key.rstrip("\n").split("\t")
                kb, rb, cut = int(kb), int(rb), 0
                if bp:
                    r = json.loads(line)
                    text = BP.strip_boilerplate(r["text"], bp)
                    if text != r["text"]:
                        if H.nbytes(text) < o["min_bytes"]:
                            add_drop(tot, "boilerplate_short", rb)
                            continue
                        cut, kb = kb - H.nbytes(text), H.nbytes(text)
                        r["text"], r["meta"]["sha1"] = text, H.dedup_key(text)
                        r["meta"]["boilerplate_bytes"] = cut
                        sha = r["meta"]["sha1"]
                        line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
                digest = bytes.fromhex(sha)
                if digest in seen or doc_id in ids:
                    add_drop(tot, "dup_exact" if digest in seen else "dup_id", rb)
                    continue
                seen.add(digest)
                ids.add(doc_id)
                w.write(line)
                tot["docs_kept"] += 1
                tot["bytes_kept"] += kb
                tot["bytes_trimmed"] += rb - kb
                tot["bytes_boilerplate"] += cut
                tot["undated_kept"] += int(und)
    outputs = []
    for w in writers.values():
        w.close()
        outputs += [dict(x, path=os.path.relpath(x["path"], out_dir)) for x in w.outputs]
    return outputs
