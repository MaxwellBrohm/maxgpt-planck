"""Controlled word list, step 1: counting over core v0 (BANKPASS s4; SPEC 4 "computed from our allowed corpora").
PC CPU, read-only over ~/planck/data/core_v0 (MANIFEST.json lists the 870 shards and their sha256). Output goes under
~/planck/runs/bankpass/, never under data/.

    nice -n 10 python wordlist.py CORE_DIR OUT_DIR --samples A,B [--workers 7] [--lock ~/planck/locks/pc_heavy.lock]
    nice -n 10 python wordlist.py CORE_DIR OUT_DIR --samples full

Samples: a doc (shard path, line index) falls in sample A when blake2b("bankpass-wl-v0|shard|i") mod 20 == 0, in B
when it is 1 (two disjoint seeded 5% doc samples, counted in ONE read); "full" takes every doc. Unsampled lines are
never JSON-parsed. Per doc, after quote straightening and lowercase, words are ASCII letter runs with inner
apostrophes and no letter of any other script touching them. Counted per source: DF, TF; left-context counts for
the POS proxy (after the/a/an/my/your/this: noun; after to/will/can/don't/should: verb; after very/so/too/quite/
more/most, or between is/was and punctuation: adjective); "a" vs "an" before the word; plural (after many/several/
these/those/few) and mass (after much) contexts; mid-sentence uses (after a lowercase letter or , ; : and a space)
and how many of those are capitalized (the proper-noun filter). With --vocab FILE, only the listed words are kept
(the full read counts the union of the two sample tables, so memory stays bounded). OASST2 docs re-assert corpus/oodh.in_oodh_reserve on
their tree id (core v0 already left the reserve out; a hit is counted and skipped). The lock, when given, is held
per chunk of shards so other heavy CPU jobs can interleave."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]   # run as a script: keep bankpass/ off the path
_PIPE = os.path.dirname(_HERE)
for _p in (_PIPE, os.path.join(os.path.dirname(_PIPE), "corpus")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FEATS = ("df", "tf", "noun", "verb", "adj", "a", "an", "pl", "mass", "mid", "capmid")
SEED = "bankpass-wl-v0"
_Q = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "′": "'", "“": '"', "”": '"'})
WORD = re.compile(r"(?<![^\W\d_])[a-z]+(?:'[a-z]+)*(?![^\W\d_])")
_W = r"([a-z]+(?:'[a-z]+)*)(?![^\W\d_])"
CTX = {"noun": re.compile(r"(?<![a-z'])(?:the|a|an|my|your|this) (?=" + _W + ")"),
       "verb": re.compile(r"(?<![a-z'])(?:to|will|can|don't|should) (?=" + _W + ")"),
       "adj": re.compile(r"(?<![a-z'])(?:very|so|too|quite|more|most) (?=" + _W + ")"),
       "a": re.compile(r"(?<![a-z'])a (?=" + _W + ")"), "an": re.compile(r"(?<![a-z'])an (?=" + _W + ")"),
       "pl": re.compile(r"(?<![a-z'])(?:many|several|these|those|few) (?=" + _W + ")"),
       "mass": re.compile(r"(?<![a-z'])much (?=" + _W + ")")}
ADJ_BE = re.compile(r"(?<![a-z'])(?:is|was) ([a-z]+)[.,!?;:]")
# a mid-sentence use: the word right after an all-lowercase word (optionally with , ; :), so Title Case runs
# ("General Public License") do not count as capitalized mid-sentence uses
MID = re.compile(r"(?<![A-Za-z'])[a-z][a-z']*[,;:]? (?=([A-Za-z][a-z']*)(?![^\W\d_]))")


def sample_of(shard, i, samples):
    if samples == ("full",):
        return "full"
    h = int.from_bytes(hashlib.blake2b(f"{SEED}|{shard}|{i}".encode(), digest_size=8).digest(), "big") % 20
    lab = "AB"[h] if h < 2 else None
    return lab if lab in samples else None


def count_doc(text, acc):
    """add one doc's counts to acc[feat] (Counters)."""
    t = text.translate(_Q)
    low = t.lower()
    toks = WORD.findall(low)
    acc["tf"].update(toks)
    acc["df"].update(set(toks))
    for f, rx in CTX.items():
        acc[f].update(rx.findall(low))
    acc["adj"].update(ADJ_BE.findall(low))
    for w in MID.findall(t):
        lw = w.lower()
        acc["mid"][lw] += 1
        if w[0].isupper():
            acc["capmid"][lw] += 1


def _open(path):
    if path.endswith(".zst"):
        from compression import zstd           # Python 3.14 on the PC
        return zstd.open(path, "rb")
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rb")
    return open(path, "rb")


_VOCAB = {}


def _vocab(path):
    if path and path not in _VOCAB:
        with open(path, encoding="utf-8") as f:
            _VOCAB[path] = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    return _VOCAB.get(path)


def count_shard(job):
    """job = (core_dir, rel shard path, source, samples[, vocab path]) -> (rel, source, {sample: (docs, {feat:
    Counter})}, notes)."""
    core, rel, source, samples = job[:4]
    keep = _vocab(job[4]) if len(job) > 4 else None
    out, notes = {}, collections.Counter()
    oodh = None
    if source == "oasst2":
        from oodh import in_oodh_reserve as oodh
    with _open(os.path.join(core, rel)) as fh:
        for i, line in enumerate(fh):
            lab = sample_of(rel, i, samples)
            if lab is None:
                continue
            r = json.loads(line)
            if oodh is not None and oodh(r["meta"]["tree_id"]):
                notes["oodh_reserve_hit"] += 1
                continue
            docs, acc = out.setdefault(lab, [0, {f: collections.Counter() for f in FEATS}])
            out[lab][0] += 1
            count_doc(r["text"], acc)
    return rel, source, {k: (v[0], {f: ({w: n for w, n in c.items() if w in keep} if keep else dict(c))
                                    for f, c in v[1].items()}) for k, v in out.items()}, dict(notes)


def merge(total, source, part, src_docs):
    """pool a shard's counts straight into its s4 group (loc keeps DF only), so memory holds one table per group."""
    from bankpass.wordtable import GROUP_OF, VOTING
    g = GROUP_OF[source]
    for lab, (docs, feats) in part.items():
        t = total.setdefault(lab, {}).setdefault(g, {"docs": 0, "feats": {f: collections.Counter() for f in FEATS}})
        t["docs"] += docs
        src_docs[lab][source] += docs
        for f, c in feats.items():
            if f == "df" or g in VOTING:
                t["feats"][f].update(c)


def run(core, out_dir, samples, workers=7, lock=None, chunk=40, sources=None, max_shards=None, vocab=None):
    import multiprocessing as mp
    with open(os.path.join(core, "MANIFEST.json"), "rb") as f:
        raw = f.read()
    man = json.loads(raw)
    jobs = [(core, s["shard"], s["source"], samples) + ((vocab,) if vocab else ()) for s in man["shards"]
            if not sources or s["source"] in sources]
    jobs = jobs[:max_shards] if max_shards else jobs
    total, notes, t0 = {}, collections.Counter(), time.time()
    src_docs = collections.defaultdict(collections.Counter)
    lk = open(os.path.expanduser(lock), "a") if lock else None
    with mp.get_context("spawn").Pool(workers) as pool:
        for k in range(0, len(jobs), chunk):
            if lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                for rel, source, part, nt in pool.imap_unordered(count_shard, jobs[k:k + chunk]):
                    merge(total, source, part, src_docs)
                    notes.update(nt)
            finally:
                if lk:
                    fcntl.flock(lk, fcntl.LOCK_UN)
            print(f"{time.strftime('%H:%M:%S')} shards {min(k + chunk, len(jobs))}/{len(jobs)} "
                  f"{time.time() - t0:.0f}s", flush=True)
    meta = {"input_manifest_sha256": hashlib.sha256(raw).hexdigest(), "shards": len(jobs), "samples": list(samples),
            "seed": SEED, "seconds": round(time.time() - t0, 1), "workers": workers, "notes": dict(notes),
            "docs_by_source": {k: dict(v) for k, v in src_docs.items()},
            "vocab": {"path": vocab, "sha256": hashlib.sha256(open(vocab, "rb").read()).hexdigest(),
                      "words": len(_vocab(vocab))} if vocab else None,
            "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    return total, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("core_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--samples", default="A,B")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--lock", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--vocab", default=None)
    a = ap.parse_args(argv)
    from bankpass import wordtable
    samples = tuple(a.samples.split(","))
    total, meta = run(a.core_dir, a.out_dir, samples, a.workers, a.lock, max_shards=a.max_shards, vocab=a.vocab)
    os.makedirs(a.out_dir, exist_ok=True)
    for lab, by_group in sorted(total.items()):
        n = wordtable.write(os.path.join(a.out_dir, f"counts_{lab}.tsv.gz"), by_group, dict(meta, sample=lab))
        print(f"sample {lab}: {n} rows", flush=True)
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
