"""Corpus JSONL -> harness training shards, tokenized with a tokenizer v0 member.

    python data_prep/pretokenize.py INPUT_DIR OUT_DIR (--heldout DIR | --no-heldout)
        [--vocab 8192 | --tokenizer tok.json] [--sources a,b] [--workers 16] [--chunk-mb 64]
        [--shard-mb 128] [--seed 0] [--overwrite]

INPUT_DIR/<source>/*.jsonl[.gz] holds one record per line: {"id", "text", "meta"...} for a text doc, or
with "turns": [{"role", "text"(, "loss")}] (and optional "system") for a chat doc. That is the
corpus/extract.py shard format; any later corpus (core v0) in the same layout works unchanged. A source
must be all text or all chat.

Out, per source (OUT_DIR/<source>/):
  text  <source>-NNNNN.bin  headerless little-endian uint16, each doc's ids then EOT (<|endoftext|>, id 1)
        = harness sources.TokenShardSource (config kind: tokens, eot_id 1).
  chat  <source>-NNNNN.jsonl  {"id", "source", "turns": [{"role", "ids"(, "loss")}]} per line
        = harness sources.ChatJsonlSource (config kind: chat). The harness ChatTemplate renders them as
        <|role|> ids <|end|> per turn; loss "assistant" supervises assistant content + its <|end|> only.
  <shard>.ids   one line per doc in shard order: id TAB split keys (|-joined) TAB sha1[:16] of the text.
  manifest.json per shard sha256, bytes, docs, tokens (+ supervised tokens for chat); per source counts.
OUT_DIR/manifest.json joins the per-source manifests: tokens per source, totals and a harness `data:`
block (paths relative to OUT_DIR). Running one source at a time (one lock hold each) is supported: every
run rewrites it from the per-source manifests, which must agree on tokenizer, seed and held-out input.

Rules. Held-out docs are excluded by id, meta.split_key and meta.tree_id (so a Dolly row that shares a
context with a held-out row, or any record of a held-out OASST2 tree, stays out). An OASST2 record from
the OOD-H reserve stops the run (corpus/oodh.py). Control and markup token strings in text become one
space (the tokenizer-sample contract, corpus/sample_plan.py), counted per source. Every doc goes to
shard key % n_shards and is ordered in it by key, key = sha256(format, seed, source, id): the output is
a pure function of (input records, tokenizer, seed, shard count), whatever the workers or chunks.
n_shards = ceil(input bytes of the source / shard-mb). Memory: one chunk per worker plus one index.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import sys

import prep_common as C
import pretok_worker as W

CODE = ["data_prep/prep_common.py", "data_prep/pretok_worker.py", "data_prep/pretokenize.py",
        "harness/chat_template.py", "corpus/oodh.py", "corpus/sample_plan.py"]


def input_files(in_dir: str, source: str) -> list[str]:
    fs = glob.glob(os.path.join(in_dir, source, "*.jsonl")) + glob.glob(os.path.join(in_dir, source, "*.jsonl.gz"))
    return sorted(fs)


def list_sources(in_dir: str) -> list[str]:
    return sorted(d for d in os.listdir(in_dir)
                  if os.path.isdir(os.path.join(in_dir, d)) and input_files(in_dir, d))


def plan_chunks(files: list[str], chunk_bytes: int) -> list[tuple[str, int, int | None]]:
    out = []
    for p in files:
        size = os.path.getsize(p)
        if p.endswith(".gz") or size <= chunk_bytes:
            out.append((p, 0, None))
            continue
        starts = list(range(0, size, chunk_bytes))
        out += [(p, s, min(s + chunk_bytes, size)) for s in starts]
    return out


def heldout_keys(heldout_dir: str | None, source: str) -> tuple[set, dict | None]:
    if heldout_dir is None:
        return set(), None
    p = os.path.join(heldout_dir, f"{source}.jsonl")
    if not os.path.exists(p):
        return set(), {"file": None}
    keys, docs = set(), 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                keys.update(C.record_keys(json.loads(line)))
                docs += 1
    return keys, {"file": os.path.basename(p), "sha256": C.sha256_file(p), "docs": docs, "keys": len(keys)}


def _init():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"


def run_source(a, source: str, tok_path: str, tok_info: dict) -> dict:
    files = input_files(a.input_dir, source)
    total = sum(os.path.getsize(p) for p in files)
    nb = max(1, math.ceil(total / (a.shard_mb * (1 << 20))))
    final = os.path.join(a.out_dir, source)
    if os.path.exists(final):
        if not a.overwrite:
            raise SystemExit(f"{final} exists (use --overwrite)")
        shutil.rmtree(final)
    work = os.path.join(a.out_dir, ".work", source)
    part = final + ".partial"
    for d in (work, part):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    excl, held = heldout_keys(a.heldout, source)
    jobs = []
    for i, (p, s, e) in enumerate(plan_chunks(files, max(1, int(a.chunk_mb * (1 << 20))))):
        stem = os.path.join(work, f"c{i:05d}")
        jobs.append({"index": i, "path": p, "start": s, "end": e, "blob": stem + ".blob", "npz": stem + ".npz",
                     "ids": stem + ".ids", "tokenizer": tok_path, "tok_info": tok_info, "seed": a.seed,
                     "source": source, "n_buckets": nb, "exclude": sorted(excl)})
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, len(jobs)), initializer=_init) as pool:
        res = pool.map(W.run_chunk, jobs, chunksize=1)
        cnt = W.new_counts()
        for r in res:
            W.add_counts(cnt, r["counts"])
        if cnt["text_docs"] and cnt["chat_docs"]:
            raise SystemExit(f"{source}: {cnt['text_docs']} text and {cnt['chat_docs']} chat docs; "
                             "a source must be one kind")
        kind = "chat" if cnt["chat_docs"] else "tokens"
        ext = "jsonl" if kind == "chat" else "bin"
        chunks = [(j["blob"], j["npz"], j["ids"]) for j in jobs]
        ajobs = [{"bucket": b, "chunks": chunks, "out": os.path.join(part, f"{source}-{b:05d}.{ext}"),
                  "ids_out": os.path.join(part, f"{source}-{b:05d}.ids")} for b in range(nb)]
        shards = pool.map(W.assemble, ajobs, chunksize=1)
    shards = [s for s in shards if s["docs"]]
    assert sum(s["docs"] for s in shards) == cnt["docs"] and sum(s["tokens"] for s in shards) == cnt["tokens"]
    man = {"format": C.FORMAT_VERSION, "source": source, "kind": kind, "tokenizer": tok_info, "seed": a.seed,
           "dtype": "uint16" if kind == "tokens" else None, "n_shards": nb, "shard_mb": a.shard_mb,
           "input": {"dir": C.home_rel(a.input_dir), "files": [os.path.relpath(p, a.input_dir) for p in files],
                     "bytes": total, "extract_stats_sha256": _stats_sha(a.input_dir)},
           "heldout": {"dir": C.home_rel(a.heldout) if a.heldout else None,
                       "manifest_sha256": _stats_sha(os.path.dirname(os.path.abspath(a.heldout)), "manifest.json")
                       if a.heldout else None, "source_file": held},
           "counts": cnt, "bytes_per_token": round(cnt["text_bytes"] / max(1, cnt["tokens"] - (
               cnt["docs"] if kind == "tokens" else 0)), 4),
           "shards": [{"path": f"{source}/{source}-{s['bucket']:05d}.{ext}",
                       "ids": f"{source}/{source}-{s['bucket']:05d}.ids",
                       **{k: s[k] for k in ("sha256", "bytes", "docs", "tokens", "sup_tokens", "text_bytes")}}
                      for s in shards],
           "code_sha256": C.code_hashes(CODE)}
    C.write_json(os.path.join(part, "manifest.json"), man)
    os.replace(part, final)
    shutil.rmtree(work, ignore_errors=True)
    return man


def _stats_sha(d: str, name: str = "extract_stats.json"):
    p = os.path.join(d, name)
    return C.sha256_file(p) if os.path.exists(p) else None


def write_top(out_dir: str) -> dict:
    mans = [C.read_json(p) for p in sorted(glob.glob(os.path.join(out_dir, "*", "manifest.json")))]
    assert mans, f"no per-source manifests under {out_dir}"
    same = {"tokenizer": mans[0]["tokenizer"], "seed": mans[0]["seed"], "format": mans[0]["format"],
            "heldout_manifest_sha256": mans[0]["heldout"]["manifest_sha256"]}
    for m in mans[1:]:
        got = {"tokenizer": m["tokenizer"], "seed": m["seed"], "format": m["format"],
               "heldout_manifest_sha256": m["heldout"]["manifest_sha256"]}
        assert got == same, f"{m['source']}: manifest disagrees with {mans[0]['source']} on {same} vs {got}"
    ti = same["tokenizer"]
    srcs = {m["source"]: {"kind": m["kind"], "docs": m["counts"]["docs"], "tokens": m["counts"]["tokens"],
                          "sup_tokens": m["counts"]["sup_tokens"], "text_bytes": m["counts"]["text_bytes"],
                          "bytes_per_token": m["bytes_per_token"], "shards": m["n_shards"],
                          "excluded_heldout_docs": m["counts"]["excluded_docs"],
                          "manifest_sha256": C.sha256_file(os.path.join(out_dir, m["source"], "manifest.json"))}
            for m in mans}
    hs = [{"name": s, "kind": v["kind"], "paths": [f"{s}/{s}-*.{'jsonl' if v['kind'] == 'chat' else 'bin'}"],
           **({"eot_id": ti["eot_id"]} if v["kind"] == "tokens" else {})} for s, v in srcs.items()]
    top = {**same, "sources": srcs,
           "totals": {k: sum(v[k] for v in srcs.values()) for k in ("docs", "tokens", "sup_tokens", "text_bytes")},
           "harness_data": {"pad_id": ti["pad_id"], "sources": hs,
                            "chat": {"loss": "assistant", "role_ids": ti["role_ids"], "end_id": ti["end_id"]}}}
    C.write_json(os.path.join(out_dir, "manifest.json"), top)
    return top


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--heldout", help="dir of <source>.jsonl held-out records to exclude")
    g.add_argument("--no-heldout", action="store_true")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=float, default=64)
    ap.add_argument("--shard-mb", type=float, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    a.input_dir, a.out_dir = os.path.abspath(a.input_dir), os.path.abspath(a.out_dir)
    a.heldout = os.path.abspath(a.heldout) if a.heldout else None
    tok_path = C.tokenizer_path(a.vocab, a.tokenizer)
    tok_info = C.tokenizer_info(tok_path)
    sources = a.sources.split(",") if a.sources else list_sources(a.input_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    for s in sources:
        assert input_files(a.input_dir, s), f"no input files for source {s}"
        try:
            m = run_source(a, s, tok_path, tok_info)
        except C.Refusal as e:                       # OOD-H, bad role: nothing is written for this source
            for d in (f"{s}.partial", os.path.join(".work", s)):
                shutil.rmtree(os.path.join(a.out_dir, d), ignore_errors=True)
            print(f"[pretokenize] refused: {e}", file=sys.stderr, flush=True)
            return 3
        c = m["counts"]
        print(f"[pretokenize] {s}: {c['docs']:,} docs, {c['tokens']:,} tokens ({m['kind']}), "
              f"{c['excluded_docs']:,} held-out docs excluded, {m['n_shards']} shards", flush=True)
    shutil.rmtree(os.path.join(a.out_dir, ".work"), ignore_errors=True)
    top = write_top(a.out_dir)
    print(json.dumps({"totals": top["totals"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
