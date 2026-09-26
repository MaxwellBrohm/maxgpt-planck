"""Common English word list for the IRC nickname guard (irc_clean.py).

    python corpus/make_common_words.py EXTRACTED_DIR OUT_TXT [--workers 8]
        [--min-df-frac 5e-4] [--min-docs 5] [--min-sources 2]

CORPUS 3.2 step 5 maps IRC nicknames to stable speaker labels. A speaker's nickname is also
replaced where it is mentioned inside messages, except when the nickname is an ordinary English
word ('help', 'mike', 'ubuntu'): replacing every 'help' in a channel log would damage the text.
This script builds that word list from extract.py's output of the allowed prose sources.

A word (a run of ASCII letters, lowercased, 2+ letters) is common when its document frequency
reaches max(min_docs, min_df_frac x docs in that source) in at least min_sources of: gutenberg,
wikimedia (namespace 0 only), stackexchange, dolly. Needing two sources keeps out names that live
in one source only (a StackExchange user's handle, a book's characters). IRC is never an input.
The output is sorted, one word per line, after '#' header lines naming the rule and the sha256 of
the input's extract_stats.json. PLAN Phase 1 allows "word-frequency lists we compute from our own
allowed corpora".
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
from collections import Counter

SOURCES = ("gutenberg", "wikimedia", "stackexchange", "dolly")
_WORD = re.compile(r"[a-z]{2,}")


def shard_df(job):
    """-> (source, docs, Counter word -> docs containing it) for one shard."""
    source, path = job
    df, n = Counter(), 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if source == "wikimedia" and str(r["meta"].get("namespace")) != "0":
                continue
            n += 1
            df.update(set(_WORD.findall(r["text"].lower())))
    return source, n, df


def common_words(jobs, workers=1, min_df_frac=5e-4, min_docs=5, min_sources=2):
    docs, df = Counter(), {s: Counter() for s in SOURCES}
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(shard_df, jobs, chunksize=1)
    else:
        results = [shard_df(j) for j in jobs]
    for s, n, c in results:
        docs[s] += n
        df[s].update(c)
    votes = Counter()
    for s in SOURCES:
        floor = max(min_docs, min_df_frac * docs[s])
        votes.update(w for w, k in df[s].items() if k >= floor)
    return sorted(w for w, v in votes.items() if v >= min_sources), dict(docs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("extracted_dir")
    ap.add_argument("out_txt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-df-frac", type=float, default=5e-4)
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--min-sources", type=int, default=2)
    a = ap.parse_args(argv)
    with open(os.path.join(a.extracted_dir, "extract_stats.json"), "rb") as f:
        raw = f.read()
    stats = json.loads(raw)
    jobs = [(o["source"], os.path.join(a.extracted_dir, o["path"])) for o in stats["outputs"]
            if o["source"] in SOURCES]
    words, docs = common_words(jobs, a.workers, a.min_df_frac, a.min_docs, a.min_sources)
    head = [f"# common English words for the IRC nickname guard: corpus/make_common_words.py",
            f"# rule: document frequency >= max({a.min_docs}, {a.min_df_frac} x docs) in >= "
            f"{a.min_sources} of {', '.join(SOURCES)} (wikimedia namespace 0 only)",
            f"# input extract_stats.json sha256 {hashlib.sha256(raw).hexdigest()}",
            f"# docs per source {json.dumps(docs, sort_keys=True)}; words {len(words)}"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out_txt)), exist_ok=True)
    with open(a.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(head + words) + "\n")
    print("\n".join(head))


if __name__ == "__main__":
    sys.exit(main())
