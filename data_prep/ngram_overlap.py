"""Near-duplicate measure: token n-gram overlap of the eval sets with the training shards (E2 notes, FIXED 4).

    python data_prep/ngram_overlap.py EVALSET_DIR SHARDS_DIR --tokenizer tok.json [--n 13] [--stride 2]
        [--workers 2] [--out result.json]

Eval side: each text doc, and each user and assistant turn of a chat doc, is tokenized alone (the harness
tokenizer rule, prep_common.load_tokenizer); every stride-th n-gram start is sampled. Training side: every
n-gram of every .bin shard (EOT-separated docs) and of every chat record rendered <|role|> ids <|end|> per
turn. Eval n-grams never hold a control id, so none can match across a training doc or turn seam. N-grams
are compared by a 64-bit hash (the chance of a false match is about 1e-12 per sampled n-gram).
Per set: hit_share (sampled n-grams found anywhere in training), hit_share_by_train_source,
hit_share_of_once_seen (only n-grams sampled once over all eval sets, so repeated boilerplate does not
count), docs with at least 50% and 90% of their n-grams found and their byte share, and per chat role.
It measures; it removes nothing and decides nothing. No torch.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

import prep_common as C

CHUNK = 8 << 20
MUL = np.uint64(0x9E3779B97F4A7C15)
BITS = 27
G: dict = {}


def hashes(a: np.ndarray, n: int) -> np.ndarray:
    """64-bit hash of every n-gram of a -> len(a) - n + 1 values (none when a is shorter than n)."""
    m = len(a) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    x = np.asarray(a).astype(np.uint64) + np.uint64(1)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * MUL + x[j:j + m]
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
    return h


def _found(a: np.ndarray) -> np.ndarray:
    """Sampled eval hashes that occur in the token stream a (bitmap prefilter, then an exact lookup)."""
    n, S, out = G["n"], G["sample"], []
    for st in range(0, max(1, len(a) - n + 1), CHUNK):
        h = hashes(a[st:st + CHUNK + n - 1], n)
        c = h[G["bitmap"][(h >> np.uint64(64 - BITS)).astype(np.int64)]]
        if len(c):
            k = np.minimum(np.searchsorted(S, c), len(S) - 1)
            out.append(c[S[k] == c])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.uint64)


def scan_file(job: tuple) -> tuple:
    src, path, kind = job
    if kind == "tokens":
        return src, _found(np.memmap(path, dtype="<u2", mode="r"))
    out, buf = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for t in json.loads(line)["turns"]:
                buf += [G["role_ids"][t["role"]], *t["ids"], G["end_id"]]
            if len(buf) > CHUNK:
                out.append(_found(np.array(buf, dtype=np.int64)))
                buf = []
    if buf:
        out.append(_found(np.array(buf, dtype=np.int64)))
    return src, np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.uint64)


def eval_units(evaldir: str, man: dict, tok, n: int, stride: int) -> list[tuple]:
    """-> [(set, doc id, role or None, utf-8 bytes, sampled hashes)] per text doc or scored chat turn."""
    units = []
    for name, st in sorted(man["sets"].items()):
        with open(os.path.join(evaldir, st["files"]["docs"]["path"]), encoding="utf-8") as f:
            docs = [json.loads(x) for x in f]
        items = []
        for d in docs:
            if "text" in d:
                items.append((d["id"], None, d["text"]))
            else:
                items += [(d["id"], t["role"], t["text"]) for t in d["turns"] if t["role"] in ("user", "assistant")]
        encs = tok.encode_batch([x[2] for x in items], add_special_tokens=False)
        units += [(name, i, role, len(text.encode("utf-8")), hashes(np.array(e.ids, dtype=np.int64), n)[::stride])
                  for (i, role, text), e in zip(items, encs)]
    return units


def summarize(units: list[tuple], hit_src: dict) -> dict:
    allh = np.concatenate([u[4] for u in units]) if units else np.zeros(0, dtype=np.uint64)
    off = np.cumsum([0] + [len(u[4]) for u in units])
    hit_any = np.unique(np.concatenate(list(hit_src.values()))) if hit_src else np.zeros(0, dtype=np.uint64)
    f_any = np.isin(allh, hit_any)
    f_src = {s: np.isin(allh, h) for s, h in hit_src.items()}
    uq, cnt = np.unique(allh, return_counts=True)
    once = cnt[np.searchsorted(uq, allh)] == 1 if len(allh) else np.zeros(0, dtype=bool)
    out = {}
    for name in sorted({u[0] for u in units}):
        ix = [k for k, u in enumerate(units) if u[0] == name]
        seg = np.concatenate([np.arange(off[k], off[k + 1]) for k in ix]).astype(np.int64)
        tot = max(1, len(seg))
        per_doc: dict = {}
        for k in ix:
            a = per_doc.setdefault(units[k][1], [0, 0, 0])
            a[0] += int(off[k + 1] - off[k])
            a[1] += int(f_any[off[k]:off[k + 1]].sum())
            a[2] += units[k][3]
        fr = [(v[1] / v[0], v[2], i) for i, v in per_doc.items() if v[0]]
        nb = sum(b for _, b, _ in fr) or 1
        r = {"sampled": len(seg), "hit_share": round(float(f_any[seg].sum()) / tot, 5),
             "hit_share_of_once_seen": round(float((f_any & once)[seg].sum()) / max(1, int(once[seg].sum())), 5),
             "hit_share_by_train_source": {s: round(float(f[seg].sum()) / tot, 5) for s, f in sorted(f_src.items())},
             "docs": len(fr), "docs_hit_ge_50pct": sum(f >= 0.5 for f, _, _ in fr),
             "docs_hit_ge_90pct": sum(f >= 0.9 for f, _, _ in fr),
             "bytes_share_docs_ge_50pct": round(sum(b for f, b, _ in fr if f >= 0.5) / nb, 5),
             "examples_ge_90pct": sorted(i for f, _, i in fr if f >= 0.9)[:5]}
        for role in sorted({units[k][2] for k in ix if units[k][2]}):
            kr = [k for k in ix if units[k][2] == role and off[k + 1] > off[k]]
            tf = [(float(f_any[off[k]:off[k + 1]].mean()), units[k][3]) for k in kr]
            r[f"{role}_hit_share"] = round(sum(int(f_any[off[k]:off[k + 1]].sum()) for k in kr) / max(
                1, sum(int(off[k + 1] - off[k]) for k in kr)), 5)
            r[f"{role}_turns_hit_ge_50pct"] = sum(f >= 0.5 for f, _ in tf)
            r[f"{role}_bytes_share_turns_ge_50pct"] = round(sum(b for f, b in tf if f >= 0.5) / max(
                1, sum(b for _, b in tf)), 5)
        out[name] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evalset_dir")
    ap.add_argument("shards_dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    man = C.read_json(os.path.join(a.evalset_dir, "manifest.json"))
    top = C.read_json(os.path.join(a.shards_dir, "manifest.json"))
    tok = C.load_tokenizer(a.tokenizer)
    units = eval_units(a.evalset_dir, man, tok, a.n, a.stride)
    allh = np.concatenate([u[4] for u in units])
    G.update(n=a.n, sample=np.unique(allh), role_ids=top["tokenizer"]["role_ids"], end_id=top["tokenizer"]["end_id"])
    bm = np.zeros(1 << BITS, dtype=bool)
    bm[(G["sample"] >> np.uint64(64 - BITS)).astype(np.int64)] = True
    G["bitmap"] = bm
    jobs = []
    for s in sorted(top["sources"]):
        m = C.read_json(os.path.join(a.shards_dir, s, "manifest.json"))
        jobs += [(s, os.path.join(a.shards_dir, x["path"]), m["kind"]) for x in m["shards"]]
    got: dict = {}
    with mp.get_context("fork").Pool(max(1, a.workers)) as pool:
        for src, found in pool.imap_unordered(scan_file, jobs):
            got.setdefault(src, []).append(found)
    hit_src = {s: np.unique(np.concatenate(v)) for s, v in got.items()}
    res = {"format": "planck-ngram-overlap-v1", "n": a.n, "stride": a.stride,
           "evalset_sha256": man["evalset_sha256"],
           "shard_manifest_sha256": C.sha256_file(os.path.join(a.shards_dir, "manifest.json")),
           "tokenizer": {"file": os.path.basename(a.tokenizer), "sha256": C.sha256_file(a.tokenizer)},
           "sampled_total": int(len(allh)), "sets": summarize(units, hit_src)}
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
