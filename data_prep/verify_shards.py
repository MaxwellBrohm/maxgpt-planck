"""Independent check of a pretokenize.py output against its manifest and its input corpus.

    python data_prep/verify_shards.py SHARDS_DIR INPUT_DIR [--heldout DIR] [--sample 200] [--sources a,b]

Per source: every shard's sha256 and size; text shards: EOT count == docs, no control or tag id inside a
doc, max id < vocab, token count; chat shards: every record renders through harness ChatTemplate, token
and supervised counts; ids unique; docs_in == docs + held-out excluded + empty; with --heldout, no doc
shares an id or split key with the held-out file; OASST2 trees never in the OOD-H reserve; and a sample
of docs (every k-th in shard order) decoded or rendered and compared with the input record read back from
INPUT_DIR. Prints a JSON report; exit 1 if any check fails.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

import prep_common as C
import pretokenize as P
from chat_template import ChatTemplate


def load_ids(out: str, man: dict) -> list[list[str]]:
    return [[x.rstrip("\n").split("\t") for x in open(os.path.join(out, sh["ids"]), encoding="utf-8")]
            for sh in man["shards"]]


def find_records(in_dir: str, source: str, want: set) -> dict:
    got = {}
    for p in P.input_files(in_dir, source):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith('{"id": "'):
                    i = line[8:line.index('"', 8)]
                    if i not in want:
                        continue
                r = json.loads(line)
                if r["id"] in want:
                    got[r["id"]] = r
    return got


def verify_source(out: str, in_dir: str, source: str, tok, info: dict, sample: int, held: set | None) -> dict:
    man = C.read_json(os.path.join(out, source, "manifest.json"))
    enc = C.encoder(tok)
    tmpl = ChatTemplate(dict(info["role_ids"]), info["end_id"], "assistant", "tool")
    fails, rep = [], {"docs": 0, "tokens": 0, "sup_tokens": 0}
    ids = load_ids(out, man)
    flat = [row for rows in ids for row in rows]
    if len({r[0] for r in flat}) != len(flat):
        fails.append("repeated doc id")
    step = max(1, len(flat) // max(1, sample))
    pick = {flat[j][0] for j in range(0, len(flat), step)}
    recs = find_records(in_dir, source, pick)
    if len(recs) != len(pick):
        fails.append(f"sample ids missing from input: {len(pick) - len(recs)}")
    checked = 0
    for sh, rows in zip(man["shards"], ids):
        p = os.path.join(out, sh["path"])
        if C.sha256_file(p) != sh["sha256"] or os.path.getsize(p) != sh["bytes"]:
            fails.append(f"hash or size: {sh['path']}")
        if man["kind"] == "tokens":
            a = np.fromfile(p, dtype="<u2")
            ends = np.flatnonzero(a == info["eot_id"])
            bad = int(np.count_nonzero(a < info["n_special"])) - len(ends)
            if len(ends) != len(rows) or bad or int(a.max()) >= info["vocab"] or len(a) != sh["tokens"]:
                fails.append(f"token shard {sh['path']}: eot {len(ends)} docs {len(rows)} bad {bad}")
            starts = np.concatenate([[0], ends[:-1] + 1])
            docs = {rows[j][0]: a[starts[j]:ends[j]].tolist() for j in range(len(rows)) if rows[j][0] in recs}
            for i, got in docs.items():
                checked += 1
                if got != enc(C.clean_text(recs[i]["text"])[0]):
                    fails.append(f"round trip {i}")
            rep["tokens"] += len(a)
        else:
            lines = open(p, encoding="utf-8").read().splitlines()
            if len(lines) != len(rows):
                fails.append(f"chat shard {sh['path']}: {len(lines)} lines, {len(rows)} ids")
            for j, line in enumerate(lines):
                r = json.loads(line)
                r_ids, fl = tmpl.render(r)
                rep["tokens"] += len(r_ids)
                rep["sup_tokens"] += int(fl.sum())
                if r["id"] != rows[j][0]:
                    fails.append(f"id order {sh['path']}:{j}")
                if r["id"] in recs:
                    checked += 1
                    o = recs[r["id"]]
                    tr = [{"role": t["role"], "text": C.clean_text(t["text"])[0]} for t in o["turns"]]
                    if o.get("system"):
                        tr = [{"role": "system", "text": C.clean_text(o["system"])[0], "loss": False}] + tr
                    t_ids, t_fl = tmpl.render({"turns": tr}, enc)
                    if t_ids.tolist() != r_ids.tolist() or t_fl.tolist() != fl.tolist():
                        fails.append(f"chat render {r['id']}")
        rep["docs"] += len(rows)
    c = man["counts"]
    if rep["tokens"] != c["tokens"] or rep["docs"] != c["docs"] or (man["kind"] == "chat" and rep["sup_tokens"] != c["sup_tokens"]):
        fails.append(f"counts {rep} vs manifest {c['docs']} docs {c['tokens']} tokens")
    if c["docs_in"] != c["docs"] + c["excluded_docs"] + c["empty_docs"]:
        fails.append("docs_in != docs + excluded + empty")
    if held is not None:
        hit = sum(1 for r in flat if r[0] in held or any(k in held for k in r[1].split("|") if k))
        if hit:
            fails.append(f"{hit} docs share a key with the held-out split")
    reserved = sum(C.in_oodh_reserve(k) for r in flat for k in r[1].split("|") if k and source == "oasst2")
    if reserved:
        fails.append(f"{reserved} OOD-H reserved trees")
    return {"source": source, "kind": man["kind"], "docs": rep["docs"], "tokens": rep["tokens"],
            "sup_tokens": rep["sup_tokens"], "sample_checked": checked, "fails": fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards_dir")
    ap.add_argument("input_dir")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--sources", default=None)
    a = ap.parse_args(argv)
    top = C.read_json(os.path.join(a.shards_dir, "manifest.json"))
    tok_path = C.tokenizer_path(path=os.path.join(C.TOK_DIR, top["tokenizer"]["file"]))
    info = C.tokenizer_info(tok_path)
    assert info == top["tokenizer"], "tokenizer file differs from the one the shards were made with"
    tok = C.load_tokenizer(tok_path)
    res = []
    for s in (a.sources.split(",") if a.sources else sorted(top["sources"])):
        held = P.heldout_keys(a.heldout, s)[0] if a.heldout else None
        res.append(verify_source(a.shards_dir, a.input_dir, s, tok, info, a.sample, held))
    ok = all(not r["fails"] for r in res)
    print(json.dumps({"ok": ok, "sources": res}, indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
