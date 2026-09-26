"""Starter extractor: raw starter files -> plain-text JSONL shards + stats (tokenizer v0 input).

    python corpus/extract.py RAW_DIR OUT_DIR [--workers 24] [--max-docs-per-file N]

RAW_DIR is laid out as the starter fetch leaves it: RAW_DIR/<dataset with / as __>/<path>.
OUT_DIR/<source>/<source>-NNNNN.jsonl holds one {"id", "source", "text", "meta"} per line, and
OUT_DIR/extract_stats.json has docs and bytes in, kept and dropped by source and reason, the
undated count, the per-source notes (OOD-H reserved trees, OASST2 message census) and the sha256
of every shard.

Per-source rules: readers_cp.py (Common Pile), irc_clean.py (IRC cleanup and nickname labels) and
readers_chat.py (OASST2, Dolly). Gates, hygiene and the AI-ism filter: hygiene.py. OOD-H reserve:
oodh.py. Byte accounting: bytes_in and dropped bytes are the raw text bytes; bytes_kept is the
final text; bytes_trimmed is what cleaning, the Gutenberg cap and boilerplate removal took from
kept docs (bytes_boilerplate is the boilerplate part). The merge (extract_merge.py) removes
boilerplate lines (boilerplate.py; CCCC and Wikimedia), then runs global exact dedup (sha1 of the
final text) in source order oasst2, dolly, irc, cccc, stackexchange, gutenberg, wikimedia, so the
first copy wins and chat copies beat prose copies. A repeated doc id is dropped as dup_id.
Not implemented in v0 (CORPUS 3.2): MinHash near-dedup (step 2), quality heuristics (step 4) and
13-gram decontamination (step 6).

The English stopword gate (hygiene.MIN_EN_STOPWORDS = 0.10) was added after a probe of the real
files (timestamps stripped first): of the first 4,000 Ubuntu IRC docs, 1,214 had 200+ bytes; the
30% non-ASCII rule caught 81 of them, and 283 of the rest scored under 0.10. The docs under 0.12
were mostly local-team channels (#ubuntu-it, -de, -hr, -se, -es, -si, -br, -nl ...). In the first
2,000 CCCC 2022-05 docs, 49 .de pages scored under 0.12; SE english had none under 0.25.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys

from array import array

import numpy as np

import boilerplate as BP
import hygiene as H
from extract_merge import add_drop, merge
from readers_chat import read_dolly, read_oasst
from readers_cp import read_common_pile

SOURCE_ORDER = ["oasst2", "dolly", "irc", "cccc", "stackexchange", "gutenberg", "wikimedia"]
PREFIX = {"OpenAssistant__oasst2/": "oasst2", "databricks__databricks-dolly-15k/": "dolly",
          "common-pile__ubuntu_irc/": "irc", "common-pile__cccc/": "cccc",
          "common-pile__stackexchange/": "stackexchange",
          "common-pile__project_gutenberg/": "gutenberg", "common-pile__wikimedia/": "wikimedia"}
READERS = {"oasst2": read_oasst, "dolly": read_dolly}
DEFAULTS = dict(cutoff=H.DATE_CUTOFF, min_bytes=H.MIN_BYTES, max_non_ascii=H.MAX_NON_ASCII,
                min_stopwords=H.MIN_EN_STOPWORDS, gutenberg_cap=200_000,
                oasst_states=["ready_for_export"], max_docs=0, boilerplate_min_docs=BP.MIN_DOCS)
HERE = os.path.dirname(os.path.abspath(__file__))


def discover(raw_dir):
    """-> sorted [(source, relpath)] of the raw files this extractor understands."""
    found = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), raw_dir).replace(os.sep, "/")
            src = next((s for p, s in PREFIX.items() if rel.startswith(p)), None)
            if src is None or not rel.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
                continue
            if src == "oasst2" and not rel.endswith(".trees.jsonl.gz"):
                continue
            found.append((src, rel))
    return sorted(found, key=lambda x: (SOURCE_ORDER.index(x[0]), x[1]))


def new_stats():
    return {"files": 0, "docs_in": 0, "bytes_in": 0, "docs_kept": 0, "bytes_kept": 0,
            "bytes_trimmed": 0, "bytes_boilerplate": 0, "undated_kept": 0, "dropped": {},
            "notes": {}}


def process_file(job):
    """Worker: one raw file -> tmp JSONL of the docs that passed its gates, a keys sidecar, and for
    boilerplate sources each doc's distinct line hashes (.lh.npy) with per-doc counts (.ln.npy)
    and URL keys (.lu, one line per doc)."""
    source, raw_dir, rel, tmp, o = job
    reader = READERS.get(source, read_common_pile)
    st, passed = new_stats(), 0
    bp = source in BP.SOURCES and o["boilerplate_min_docs"] > 0
    lh, ln, lu = array("Q"), array("q"), []
    with open(tmp + ".jsonl", "w", encoding="utf-8") as out, open(tmp + ".keys", "w") as keys:
        for ev in reader(source, os.path.join(raw_dir, rel), rel, o):
            if ev[0] == "note":
                st["notes"][ev[1]] = st["notes"].get(ev[1], 0) + ev[2]
                continue
            st["docs_in"] += 1
            st["bytes_in"] += ev[2]
            if ev[0] == "drop":
                add_drop(st, ev[1], ev[2])
                continue
            rec = ev[1]
            rec["meta"]["sha1"] = H.dedup_key(rec["text"])
            undated = int(rec["meta"].get("date_status") == "undated")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            keys.write(f"{rec['meta']['sha1']}\t{rec['id']}\t{H.nbytes(rec['text'])}\t{ev[2]}"
                       f"\t{undated}\n")
            if bp:
                h = BP.doc_hashes(rec["text"])
                lh.extend(h)
                ln.append(len(h))
                lu.append(BP.url_key(rec["meta"].get("url")))
            passed += 1
    if bp:
        np.save(tmp + ".lh.npy", np.frombuffer(lh, dtype=np.uint64))
        np.save(tmp + ".ln.npy", np.frombuffer(ln, dtype=np.int64))
        with open(tmp + ".lu", "w", encoding="utf-8") as f:
            f.write("".join(u.replace("\n", " ") + "\n" for u in lu))
    st["files"] = 1
    return source, rel, st, passed


def code_hashes():
    """sha256 of every corpus/*.py and corpus/data/* file (the IRC common-word list lives there)."""
    files = [f for f in sorted(os.listdir(HERE)) if f.endswith(".py")]
    data = os.path.join(HERE, "data")
    files += [f"data/{f}" for f in sorted(os.listdir(data))] if os.path.isdir(data) else []
    return {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest() for f in files}


def run_extract(raw_dir, out_dir, workers=1, shard_bytes=256 << 20, overwrite=False, **opts):
    o = dict(DEFAULTS, **opts)
    stats_path = os.path.join(out_dir, "extract_stats.json")
    if os.path.exists(stats_path) and not overwrite:
        raise SystemExit(f"{stats_path} exists; pass --overwrite to rebuild")
    for s in SOURCE_ORDER:
        shutil.rmtree(os.path.join(out_dir, s), ignore_errors=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    files = discover(raw_dir)
    jobs = [(s, raw_dir, rel, os.path.join(tmp_dir, f"{i:05d}"), o)
            for i, (s, rel) in enumerate(files)]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(process_file, jobs, chunksize=1)
    else:
        results = [process_file(j) for j in jobs]
    stats = {s: new_stats() for s in SOURCE_ORDER}
    outputs = merge(results, tmp_dir, out_dir, shard_bytes, stats, o)
    shutil.rmtree(tmp_dir)
    report = {"config": dict(o, shard_bytes=shard_bytes), "code_sha256": code_hashes(),
              "files_in": [{"source": s, "path": rel,
                            "bytes": os.path.getsize(os.path.join(raw_dir, rel))}
                           for s, rel in files],
              "sources": stats, "outputs": outputs,
              "totals": {k: sum(v[k] for v in stats.values())
                         for k in ("docs_in", "bytes_in", "docs_kept", "bytes_kept")}}
    with open(stats_path, "w") as f:
        json.dump(report, f, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-docs-per-file", type=int, default=0, help="smoke runs only")
    ap.add_argument("--gutenberg-cap", type=int, default=DEFAULTS["gutenberg_cap"])
    ap.add_argument("--min-bytes", type=int, default=H.MIN_BYTES)
    ap.add_argument("--max-non-ascii", type=float, default=H.MAX_NON_ASCII)
    ap.add_argument("--min-stopwords", type=float, default=H.MIN_EN_STOPWORDS)
    ap.add_argument("--cutoff", default=H.DATE_CUTOFF)
    ap.add_argument("--shard-mb", type=int, default=256)
    ap.add_argument("--boilerplate-min-docs", type=int, default=BP.MIN_DOCS, help="0 disables")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    r = run_extract(a.raw_dir, a.out_dir, workers=a.workers, shard_bytes=a.shard_mb << 20,
                    overwrite=a.overwrite, max_docs=a.max_docs_per_file,
                    gutenberg_cap=a.gutenberg_cap, min_bytes=a.min_bytes,
                    max_non_ascii=a.max_non_ascii, min_stopwords=a.min_stopwords, cutoff=a.cutoff,
                    boilerplate_min_docs=a.boilerplate_min_docs)
    for s, st in r["sources"].items():
        drops = {k: v["docs"] for k, v in sorted(st["dropped"].items())}
        print(f"{s:14s} in {st['docs_in']:>9,} kept {st['docs_kept']:>9,} "
              f"({st['bytes_kept'] / 1e6:,.1f} MB) undated {st['undated_kept']:,} drop {drops}")
    print("totals", r["totals"])


if __name__ == "__main__":
    sys.exit(main())
