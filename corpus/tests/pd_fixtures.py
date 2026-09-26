"""A fake FINAL core v0 for neardedup_post.py: MANIFEST.json and final shards (JSONL lines, .keys.u64,
.info.npy, as core_final_work writes them) of dolly and foodista (tier 1), gutenberg (tier 2), news
(tier 3 here, so tier and PRIORITY disagree: news is 6th in PRIORITY, gutenberg 11th) and cccc (tier 5,
two shards). Texts are pseudo-words (nd_fixtures), so unrelated documents share no
word 5-gram and only the planted pairs are near-duplicates.

Planted cases (the doc id names them):
- biz_new / biz_old: the bizsugar case. One page in two CCCC snapshots; their stage1 texts differed
  mostly in template lines (a trending sidebar per capture: stage1 Jaccard under 0.6, so pass B kept
  both), and the final texts, stripped, differ only in a view counter and a few words (Jaccard about
  0.9). biz_new (cccc-00000, created 2020) goes, biz_old (cccc-00001, created 2019) stays.
- dt_new / dt_old: the same with meta.date, and a meta.created in the opposite order (2018 on dt_new,
  2021 on dt_old): meta.date wins over meta.created, so dt_new goes.
- gut / gutcc: a Gutenberg book (tier 2, undated) and a CCCC copy dated 2019: the page goes.
- dolly / food: a dolly row (undated, first in PRIORITY) and a foodista page dated 2018: food goes.
  food (dropped) and u1 (kept) hold multibyte UTF-8, so text bytes differ from characters.
- gut2 / nw: a Gutenberg text (tier 2) and a news copy (tier 3, earlier in PRIORITY): tier wins over
  PRIORITY, so nw goes.
- far0 / far1: two pages sharing part of their text (Jaccard about 0.4): both stay.
- vf0 / vf1: a pair at Jaccard about 0.6 that shares an LSH band (a candidate) but agrees on under
  0.7 of its signature, so verification rejects it: both stay.
"""
import hashlib
import json
import os

import numpy as np

import stream_io as SIO
from nd_fixtures import Gen

INFO_DTYPE = np.dtype([("row", "<u4"), ("changed", "u1"), ("csize", "<u4")])
TIERS = {"dolly": 1, "foodista": 1, "gutenberg": 2, "news": 3, "cccc": 5}
DROPPED = {"biz_new": "biz_old", "dt_new": "dt_old", "gutcc": "gut", "food": "dolly", "nw": "gut2"}


def texts():
    g = Gen(20260926)
    body = g.words(220)
    t = {"biz_old": "BizSugar\nViews: 1234\n" + " ".join(body),
         "biz_new": "BizSugar\nViews: 98765\n" + " ".join(g.near_copy(body, 0.9))}
    t["stage1_biz_old"] = t["biz_old"] + "\n" + g.text(260)
    t["stage1_biz_new"] = t["biz_new"] + "\n" + g.text(260)
    for a, b, n in (("dt_old", "dt_new", 180), ("gut", "gutcc", 400), ("dolly", "food", 150)):
        ws = g.words(n)
        t[a], t[b] = " ".join(ws), " ".join(g.near_copy(ws, 0.92))
    shared = g.words(120)
    t["far0"], t["far1"] = " ".join(shared + g.words(90)), " ".join(shared + g.words(90))
    for k in range(12):
        t[f"u{k}"] = g.text(100 + 10 * k)
    v = Gen(777)                                  # searched: the 2nd pair of this stream collides
    for target in (0.5, 0.55):
        ws = v.words(160)
        cp = v.near_copy(ws, target)
    t["vf0"], t["vf1"] = " ".join(ws), " ".join(cp)
    ws = g.words(160)                             # drawn last, so the texts above stay as they were
    t["gut2"], t["nw"], t["u8"] = " ".join(ws), " ".join(g.near_copy(ws, 0.92)), g.text(140)
    t["food"] += " crème brûlée"
    t["u1"] += " naïve café"
    return t


def rec(i, source, text, **meta):
    meta["sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return {"id": i, "source": source, "text": text, "meta": meta}


def docs():
    t = texts()
    r = lambda i, src, **m: rec(i, src, t[i], **m)                        # noqa: E731
    return {
        ("dolly", 0): [r("dolly", "dolly"), r("u0", "dolly")],
        ("foodista", 0): [r("u1", "foodista", created="2017-01-01"),
                          r("food", "foodista", created="2018-01-01T00:00:00")],
        ("gutenberg", 0): [r("u2", "gutenberg"), r("gut", "gutenberg"), r("u3", "gutenberg"),
                           r("gut2", "gutenberg")],
        ("news", 0): [r("u8", "news", date="2019-01-01", created="January 1, 2019"),
                      r("nw", "news", date="2018-05-01", created="May 1, 2018")],
        ("cccc", 0): [r("u4", "cccc", created="2019-02-11", url="https://a.example/4"),
                      r("biz_new", "cccc", created="2020-10-05T00:00:00Z",
                        url="https://www.bizsugar.com/o/x"),
                      r("dt_new", "cccc", date="2020-10-05", created="2018-01-01",
                        url="https://d.example/p"),
                      r("gutcc", "cccc", created="2019-02-11", url="https://g.example/b"),
                      r("far0", "cccc", created="2019-02-11", url="https://f.example/0"),
                      r("vf0", "cccc", created="2019-02-11", url="https://v.example/0"),
                      r("u5", "cccc", created="2019-02-11", url="https://a.example/5")],
        ("cccc", 1): [r("u6", "cccc", created="2019-02-11", url="https://a.example/6"),
                      r("biz_old", "cccc", created="2019-02-11T05:33:58Z",
                        url="https://www.bizsugar.com/o/x"),
                      r("dt_old", "cccc", date="2019-02-11", created="2021-01-01",
                        url="https://d.example/p"),
                      r("far1", "cccc", created="2020-10-05", url="https://f.example/1"),
                      r("vf1", "cccc", created="2020-10-05", url="https://v.example/1"),
                      r("u7", "cccc", created="2020-10-05", url="https://a.example/7")],
    }


def write_shard(root, source, idx, rows, codec):
    rel = f"{source}/{source}-{idx:05d}.jsonl.{codec}"
    lines = [(json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8") for d in rows]
    fr = SIO.Frame(codec, 3)
    data = b"".join(fr.compress(x) for x in lines) + fr.end()
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    stem = p.rsplit(".jsonl.", 1)[0]
    keys = [int(d["meta"]["sha1"][:16], 16) for d in rows]
    np.asarray(keys, dtype="<u8").tofile(stem + ".keys.u64")
    info = np.zeros(len(rows), dtype=INFO_DTYPE)
    info["row"], info["changed"], info["csize"] = np.arange(len(rows)) * 3 + 1, 1, 1
    np.save(stem + ".info.npy", info)
    return {"source": source, "shard": rel, "stage1": rel, "docs": len(rows),
            "ubytes": sum(map(len, lines)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "keys": stem[len(root) + 1:] + ".keys.u64",
            "keys_sha256": SIO.sha256_file(stem + ".keys.u64")}


def make_final(root, codec=None):
    """Writes the fake FINAL under root; -> root."""
    codec = codec or SIO.best_codec()
    shards = [write_shard(root, s, i, rows, codec) for (s, i), rows in docs().items()]
    tiers = {}
    for s, t in TIERS.items():
        tiers.setdefault(str(t), {"sources": {}})["sources"][s] = {}
    man = {"updated": "2026-09-26T10:09:37", "status": {"near_dedup": "done"},
           "totals": {"docs": sum(s["docs"] for s in shards), "shards": len(shards)},
           "tiers": tiers, "shards": shards[::-1]}
    with open(os.path.join(root, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return root
