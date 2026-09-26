"""Optional paragraph-level dedup across documents (off by default; CORPUS 3.2 does not require it).

A paragraph that occurs in several documents is kept in the first document (by the doc ranks of
neardedup_lsh: tier, date, file order) and removed from every later one. Repeats inside one
document are left alone (that is the quality rules' job).

Units: unit="line" (default) splits on newlines, as web and wiki text puts one paragraph per line;
unit="block" splits on blank lines, for hard-wrapped books. A unit is eligible when its normalized
words (neardedup.words: lowercase, no accents, digits as 0) number at least MIN_WORDS, so short
repeated lines ('Thanks!', menus, bylines) stay for boilerplate.py and chat is not touched. The
paragraph key is blake2b-64 of those words joined by spaces.

    pass A:  w = ParaSidecarWriter(stem, unit); w.add(text) per kept doc in shard order; w.close()
             -> <stem>.para.npy, dtype PARA_DTYPE (row, idx, h), idx = the unit's index in the doc
    pass B:  find_para_dups(shards, work_dir, mem_mb) with the same shard list as find_near_dups,
             after it: docs whose <stem>.nd.npy says keep=0 take no part. Writes <stem>.pdrop.npy
             (row, idx), sorted, for every non-frozen shard, and work_dir/para_report.json.
    pass C:  drop_units(text, idxs, unit) on the same stage1 text, before boilerplate removal.
Boilerplate line counts (3.2.3) should skip the units listed in pdrop, as they skip near-dup drops.
"""
import hashlib
import json
import os
import re

import numpy as np

import hygiene as H
import neardedup as ND
import neardedup_lsh as NL

MIN_WORDS = 10
PARA_DTYPE = np.dtype([("row", "<u4"), ("idx", "<u4"), ("h", "<u8")])
DROP_DTYPE = np.dtype([("row", "<u4"), ("idx", "<u4")])
PARA_SUFFIX, PDROP_SUFFIX = ".para.npy", ".pdrop.npy"
BYTES_PER_ENTRY = 64
_BLOCK = re.compile(r"\n[ \t]*\n")


def units(text: str, unit: str = "line") -> list:
    if unit == "line":
        return text.split("\n")
    if unit == "block":
        return _BLOCK.split(text)
    raise ValueError(f"unit must be 'line' or 'block', not {unit!r}")


def para_keys(text: str, unit: str = "line", min_words: int = MIN_WORDS):
    """-> [(idx, 64-bit key)] for the eligible units of text."""
    out = []
    for i, u in enumerate(units(text, unit)):
        ws = ND.words(u)
        if len(ws) >= min_words:
            d = hashlib.blake2b(" ".join(ws).encode("utf-8"), digest_size=8).digest()
            out.append((i, int.from_bytes(d, "little")))
    return out


def drop_units(text: str, idxs, unit: str = "line") -> str:
    """text without the units at these indices, re-normalized (hygiene.normalize)."""
    idxs = set(int(i) for i in idxs)
    if not idxs:
        return text
    kept = [u for i, u in enumerate(units(text, unit)) if i not in idxs]
    return H.normalize(("\n" if unit == "line" else "\n\n").join(kept))


class ParaSidecarWriter:
    def __init__(self, stem, unit="line", min_words=MIN_WORDS):
        self.path, self.unit, self.min_words = stem + PARA_SUFFIX, unit, min_words
        self.recs, self.row = [], 0

    def add(self, text):
        self.recs += [(self.row, i, h) for i, h in para_keys(text, self.unit, self.min_words)]
        self.row += 1

    def close(self):
        ND.save_atomic(self.path, np.array(self.recs, dtype=PARA_DTYPE))
        return self.path


def find_para_dups(shards, work_dir, mem_mb=1024, write=True, top_n=20):
    run = NL._Run(shards, verify=0)
    for k, s in enumerate(shards):                  # near-dup drops take no part
        nd = s["stem"] + NL.ND_SUFFIX
        if not s.get("frozen") and os.path.exists(nd):
            run.active[run.off[k]:run.off[k + 1]] &= np.load(nd)["keep"].astype(bool)
    recs = [np.load(s["stem"] + PARA_SUFFIX, mmap_mode="r") for s in shards]
    total = sum(len(r) for r in recs)
    P = 1
    while P < 1 << 12 and total * BYTES_PER_ENTRY / P > mem_mb * (1 << 20):
        P *= 2
    shift = np.uint64(65 - P.bit_length())
    drops = [[] for _ in shards]
    st = {"paragraphs": 0, "dropped": 0, "partitions": P, "top": []}
    for p in range(P):
        hs, rk, ks, rows, ix = [], [], [], [], []
        for k, r in enumerate(recs):
            m = run.active[run.off[k] + r["row"].astype(np.int64)]
            if P > 1:
                m &= (r["h"] >> shift) == np.uint64(p)
            sel = r[np.flatnonzero(m)]
            hs.append(sel["h"])
            rk.append(run.rank[run.off[k] + sel["row"].astype(np.int64)])
            ks.append(np.full(len(sel), k, dtype=np.int32))
            rows.append(sel["row"])
            ix.append(sel["idx"])
        h, rk, ks, rows, ix = (np.concatenate(a) for a in (hs, rk, ks, rows, ix))
        st["paragraphs"] += len(h)
        o = np.lexsort((ix, rk, h))
        h, rk, ks, rows, ix = h[o], rk[o], ks[o], rows[o], ix[o]
        new = np.empty(len(h), dtype=bool)
        new[:1] = True
        np.not_equal(h[1:], h[:-1], out=new[1:])
        grp = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        dup = rk != rk[first][grp]
        st["dropped"] += int(dup.sum())
        cnt = np.bincount(grp[dup], minlength=len(first))
        for g in np.argsort(-cnt, kind="stable")[:top_n]:
            if cnt[g]:
                f = first[g]
                st["top"].append({"stem": shards[ks[f]]["stem"], "row": int(rows[f]),
                                  "idx": int(ix[f]), "later_copies": int(cnt[g])})
        for k in np.unique(ks[dup]):
            m = dup & (ks == k)
            d = np.zeros(int(m.sum()), dtype=DROP_DTYPE)
            d["row"], d["idx"] = rows[m], ix[m]
            drops[k].append(d)
    st["top"] = sorted(st["top"], key=lambda x: -x["later_copies"])[:top_n]
    for k, s in enumerate(shards):
        if write and not s.get("frozen"):
            d = np.concatenate(drops[k]) if drops[k] else np.zeros(0, DROP_DTYPE)
            ND.save_atomic(s["stem"] + PDROP_SUFFIX, np.sort(d, order=["row", "idx"]))
    with open(os.path.join(work_dir, "para_report.json"), "w") as f:
        json.dump(st, f, indent=1)
    return st


def load_drops(stem) -> dict:
    """<stem>.pdrop.npy -> {row: [unit indices to drop]}."""
    out = {}
    for r, i in np.load(stem + PDROP_SUFFIX):
        out.setdefault(int(r), []).append(int(i))
    return out
