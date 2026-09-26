"""Run readers_se_dump over Stack Exchange site dumps: one stage-1 JSONL per site, plus its stats.

    python corpus/se_dump_run.py OUT_DIR DUMP [DUMP ...] [--dataset archive.org/stackexchange_20221005]
        [--dump-date 2022-10-05] [--no-comments] [--tmp-dir DIR] [--max-docs N] [--zstd]

DUMP is a <site>.7z or a directory with Posts.xml and Comments.xml. OUT_DIR/<site host>.jsonl (or
.jsonl.zst with --zstd: stdlib compression.zstd, Python 3.14+, level 3) holds one record per kept
thread; OUT_DIR/<site host>.stats.json holds docs and bytes in and kept, drops by reason, the reader's
notes, the config, the sha256 of the input, the output and the reader code, and the statement that
this source replaces the Common Pile StackExchange subset in core v0. Outputs are written under a
temporary name and renamed when complete, so a present .stats.json means a finished site.
"""
import argparse
import hashlib
import json
import os
import sys

import hygiene as H
from readers_se_dump import REPLACES, SOURCE, read_se_dump, site_host

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ("readers_se_dump.py", "se_stage.py", "se_html.py", "sevenzip_min.py", "sevenzip_hdr.py",
        "hygiene.py", "se_dump_run.py")
NOTE = ("Core v0 StackExchange comes from the official Stack Exchange data dump with per-post "
        "CreationDate gating (readers_se_dump.py). It replaces the Common Pile StackExchange subset "
        f"({REPLACES}) that the starter v0 extract used.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def input_sha256(path):
    if os.path.isdir(path):
        return {m: sha256_file(os.path.join(path, m)) for m in sorted(os.listdir(path))
                if m.endswith(".xml")}
    return sha256_file(path)


def open_out(path, zstd):
    if zstd:
        from compression import zstd as Z          # Python 3.14+
        return Z.open(path, "wb", level=3)
    return open(path, "wb")


def run_site(dump, out_dir, o, zstd=False):
    host = site_host(dump)
    out = os.path.join(out_dir, host + (".jsonl.zst" if zstd else ".jsonl"))
    st = {"source": SOURCE, "note": NOTE, "replaces": REPLACES, "site": host,
          "input": {"path": os.path.abspath(dump), "sha256": input_sha256(dump)},
          "config": o, "docs_in": 0, "bytes_in": 0, "docs_kept": 0, "bytes_kept": 0,
          "dropped": {}, "notes": {}}
    with open_out(out + ".tmp", zstd) as fh:
        rel = "/".join(os.path.normpath(os.path.abspath(dump)).split(os.sep)[-2:])
        for ev in read_se_dump(SOURCE, dump, rel, o):       # rel: '<dataset dir>/<site>.7z'
            if ev[0] == "note":
                st["notes"][ev[1]] = st["notes"].get(ev[1], 0) + ev[2]
                continue
            st["docs_in"] += 1
            st["bytes_in"] += ev[2]
            if ev[0] == "drop":
                d = st["dropped"].setdefault(ev[1], {"docs": 0, "bytes": 0})
                d["docs"] += 1
                d["bytes"] += ev[2]
                continue
            rec = ev[1]
            rec["meta"]["sha1"] = H.dedup_key(rec["text"])
            fh.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            st["docs_kept"] += 1
            st["bytes_kept"] += H.nbytes(rec["text"])
    os.replace(out + ".tmp", out)
    st["output"] = {"path": os.path.basename(out), "bytes": os.path.getsize(out),
                    "sha256": sha256_file(out)}
    st["code_sha256"] = {f: sha256_file(os.path.join(HERE, f)) for f in CODE}
    stats = os.path.join(out_dir, host + ".stats.json")
    with open(stats + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(stats + ".tmp", stats)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--dataset", default=None, help="e.g. archive.org/stackexchange_20221005")
    ap.add_argument("--dump-date", default=None, help="e.g. 2022-10-05")
    ap.add_argument("--no-comments", action="store_true")
    ap.add_argument("--min-answers", type=int, default=1)
    ap.add_argument("--min-answer-score", type=int, default=0)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--max-docs", type=int, default=0, help="smoke runs only")
    ap.add_argument("--cutoff", default=H.DATE_CUTOFF)
    ap.add_argument("--zstd", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    o = dict(cutoff=a.cutoff, min_bytes=H.MIN_BYTES, max_non_ascii=H.MAX_NON_ASCII,
             min_stopwords=H.MIN_EN_STOPWORDS, max_docs=a.max_docs, se_dataset=a.dataset,
             se_dump_date=a.dump_date, se_comments=not a.no_comments,
             se_min_answers=a.min_answers, se_min_answer_score=a.min_answer_score,
             se_tmp_dir=a.tmp_dir)
    for d in a.dumps:
        st = run_site(d, a.out_dir, o, zstd=a.zstd)
        drops = {k: v["docs"] for k, v in sorted(st["dropped"].items())}
        print(f"{st['site']}: in {st['docs_in']:,} kept {st['docs_kept']:,} "
              f"({st['bytes_kept'] / 1e6:,.1f} MB) drop {drops}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
