"""pretokenize.py work units, run in a process pool. No torch.

run_chunk   one byte range of one input JSONL file -> CHUNK.blob (each kept doc's output bytes, in input
            order), CHUNK.npz (per doc: bucket, key, offset and length in the blob, tokens, supervised
            tokens, text bytes) and CHUNK.ids (per doc: id TAB split keys joined by | TAB text hash). Returns counts.
assemble    one output shard: every chunk's docs whose bucket is this shard, sorted by (key, id), their
            blobs concatenated -> the shard file, its .ids file, sha256 and counts.

A text doc's blob is its token ids then EOT, as little-endian uint16 (harness TokenShardSource format).
A chat doc's blob is one JSON line {"id", "source", "turns": [{"role", "ids"(, "loss")}]}; a top-level
"system" text becomes a first system turn with "loss": false (as harness/from_pipeline.py does), so
the record renders with no tokenizer in the harness.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os

import numpy as np

import prep_common as C
from chat_template import ChatTemplate

LONG = (2048, 4096)          # chat docs whose rendered length exceeds these are counted
BATCH_DOCS = 256              # encode_batch flushes at this many docs or BATCH_BYTES of text,
BATCH_BYTES = 2 << 20         # which bounds a worker's token lists (Gutenberg docs are 200 KB each)
_TOK: dict = {}


def _tok(path: str):
    if path not in _TOK:
        _TOK[path] = C.load_tokenizer(path)
    return _TOK[path]


def iter_lines(path: str, start: int, end: int | None):
    """Lines whose first byte lies in [start, end). A .gz file is one unit (start 0, end None)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            yield from f
        return
    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()                    # finish the line that started before this range
        while end is None or f.tell() < end:
            line = f.readline()
            if not line:
                return
            yield line


def new_counts() -> dict:
    return {"docs_in": 0, "docs": 0, "text_docs": 0, "chat_docs": 0, "excluded_docs": 0,
            "excluded_text_bytes": 0, "empty_docs": 0, "special_replaced_docs": 0, "tokens": 0,
            "sup_tokens": 0, "text_bytes": 0, "max_doc_tokens": 0,
            **{f"chat_over_{n}": 0 for n in LONG}, **{f"chat_over_{n}_tokens": 0 for n in LONG}}


def add_counts(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = max(a.get(k, 0), v) if k.startswith("max_") else a.get(k, 0) + v
    return a


def _prepare(rec: dict):
    """-> (kind, texts to encode, turn roles/loss flags, cleaned whole text, changed?)"""
    if C.is_chat(rec):
        turns = []
        if rec.get("system"):
            turns.append({"role": "system", "text": rec["system"], "loss": False})
        turns += list(rec["turns"])
        texts, spec, changed = [], [], False
        for t in turns:
            role = t.get("role")
            if role not in C.CHAT_ROLES:
                raise C.Refusal(f"{rec.get('id')}: turn role {role!r} is not one of {C.CHAT_ROLES}")
            s, ch = C.clean_text(t.get("text") or "")
            texts.append(s)
            spec.append((role, t.get("loss")))
            changed |= ch
        whole, ch = C.clean_text(C.doc_text(rec))
        return "chat", texts, spec, whole, changed or ch
    s, ch = C.clean_text(rec.get("text") or "")
    return "text", [s], None, s, ch


def run_chunk(job: dict) -> dict:
    tok = _tok(job["tokenizer"])
    info = job["tok_info"]
    tmpl = ChatTemplate(dict(info["role_ids"]), info["end_id"], "assistant", "tool")
    eot, vocab, seed, src, nb = info["eot_id"], info["vocab"], job["seed"], job["source"], job["n_buckets"]
    exclude = set(job["exclude"])
    cnt = new_counts()
    rows: list[tuple] = []
    blob = open(job["blob"], "wb")
    idsf = open(job["ids"], "w", encoding="utf-8")
    pos = 0
    pending: list = []
    pend_bytes = 0

    def flush():
        nonlocal pos
        texts = [s for p in pending for s in p[2]]
        encs = tok.encode_batch(texts, add_special_tokens=False) if texts else []
        at = 0
        for rec, kind, parts, spec, whole in pending:
            ids_list = [e.ids for e in encs[at:at + len(parts)]]
            at += len(parts)
            if kind == "text":
                ids = ids_list[0]
                if not ids:
                    cnt["empty_docs"] += 1
                    continue
                arr = np.empty(len(ids) + 1, dtype=np.uint16)
                arr[:-1] = ids
                arr[-1] = eot
                assert max(ids) < vocab and eot not in ids, f"{rec['id']}: bad ids"
                data, n_tok, n_sup = arr.tobytes(), len(arr), 0
            else:
                turns = []
                for (role, loss), ids in zip(spec, ids_list):
                    t = {"role": role, "ids": ids}
                    if loss is not None:
                        t["loss"] = bool(loss)
                    turns.append(t)
                out = {"id": rec["id"], "source": src, "turns": turns}
                r_ids, flags = tmpl.render(out)
                data = (json.dumps(out, separators=(",", ":")) + "\n").encode("utf-8")
                n_tok, n_sup = len(r_ids), int(flags.sum())
                for n in LONG:
                    if n_tok > n:
                        cnt[f"chat_over_{n}"] += 1
                        cnt[f"chat_over_{n}_tokens"] += n_tok
            key = C.doc_key(seed, src, rec["id"])
            tb = len(whole.encode("utf-8"))
            blob.write(data)
            rows.append((key % nb, key, pos, len(data), n_tok, n_sup, tb))
            pos += len(data)
            keys = C.record_keys(rec)
            idsf.write(f"{rec['id']}\t{'|'.join(keys[1:])}\t{C.text_hash(whole)}\n")
            cnt["docs"] += 1
            cnt[f"{kind}_docs"] += 1
            cnt["tokens"] += n_tok
            cnt["sup_tokens"] += n_sup
            cnt["text_bytes"] += tb
            cnt["max_doc_tokens"] = max(cnt["max_doc_tokens"], n_tok)
        pending.clear()

    for line in iter_lines(job["path"], job["start"], job["end"]):
        if not line.strip():
            continue
        rec = json.loads(line)
        cnt["docs_in"] += 1
        C.check_oodh(rec)
        if any(k in exclude for k in C.record_keys(rec)):
            cnt["excluded_docs"] += 1
            cnt["excluded_text_bytes"] += len(C.doc_text(rec).encode("utf-8"))
            continue
        kind, parts, spec, whole, changed = _prepare(rec)
        cnt["special_replaced_docs"] += int(changed)
        pending.append((rec, kind, parts, spec, whole))
        pend_bytes += len(line)
        if len(pending) >= BATCH_DOCS or pend_bytes >= BATCH_BYTES:
            flush()
            pend_bytes = 0
    flush()
    blob.close()
    idsf.close()
    a = np.array(rows, dtype=np.uint64).reshape(-1, 7)
    np.savez(job["npz"], bucket=a[:, 0], key=a[:, 1], off=a[:, 2], length=a[:, 3],
             n_tok=a[:, 4], n_sup=a[:, 5], text_bytes=a[:, 6])
    return {"job": job["index"], "counts": cnt}


def assemble(job: dict) -> dict:
    """job: chunks [(blob, npz, ids)], bucket, out, ids_out."""
    b = job["bucket"]
    entries = []
    for ci, (blob, npz, idsp) in enumerate(job["chunks"]):
        z = np.load(npz)
        sel = np.flatnonzero(z["bucket"] == b)
        if not len(sel):
            continue
        with open(idsp, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for j in sel:
            doc_id = lines[j].split("\t", 1)[0]
            entries.append((int(z["key"][j]), doc_id, ci, int(z["off"][j]), int(z["length"][j]),
                            int(z["n_tok"][j]), int(z["n_sup"][j]), int(z["text_bytes"][j]), lines[j]))
    entries.sort(key=lambda e: (e[0], e[1]))
    if not entries:                   # no file: an empty .bin cannot be memory-mapped by the harness
        return {"bucket": b, "sha256": None, "bytes": 0, "docs": 0, "tokens": 0, "sup_tokens": 0,
                "text_bytes": 0}
    h = hashlib.sha256()
    handles: dict = {}
    n_bytes = n_tok = n_sup = tb = 0
    with open(job["out"] + ".tmp", "wb") as out, open(job["ids_out"] + ".tmp", "w", encoding="utf-8") as idf:
        for key, doc_id, ci, off, ln, nt, ns, t, _ in entries:
            if ci not in handles:
                handles[ci] = open(job["chunks"][ci][0], "rb")
            f = handles[ci]
            f.seek(off)
            data = f.read(ln)
            assert len(data) == ln, f"short read in chunk {ci}"
            out.write(data)
            h.update(data)
            n_bytes += ln
            n_tok += nt
            n_sup += ns
            tb += t
        for e in entries:
            idf.write(e[8] + "\n")
    for f in handles.values():
        f.close()
    os.replace(job["out"] + ".tmp", job["out"])
    os.replace(job["ids_out"] + ".tmp", job["ids_out"])
    return {"bucket": b, "sha256": h.hexdigest(), "bytes": n_bytes, "docs": len(entries),
            "tokens": n_tok, "sup_tokens": n_sup, "text_bytes": tb}
