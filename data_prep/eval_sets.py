"""Fixed-window bits-per-byte eval sets from held-out splits, stored as raw text plus byte offsets.

    python data_prep/eval_sets.py HELDOUT_DIR OUT_DIR [--sources a,b] [--window-bytes 2048]
        [--context-bytes 1024] [--chat-context-bytes 3072] [--lookback 256]
        [--train-shards SHARDS_DIR ...] [--overwrite]

HELDOUT_DIR/<source>.jsonl is a held-out split (corpus/make_tok_sample.py heldout/: records as the
extractor writes them). One set per source, named after it:
  OUT_DIR/<set>.docs.jsonl     text: {"i", "id", "text"}; chat: {"i", "id", "turns": [{"role", "text"}]}
                               (+ "tree_id" for OASST2). Text is the record's text after the tokenizer-
                               sample special-string rule (idempotent on held-out records).
  OUT_DIR/<set>.windows.jsonl  one scored window per line, in doc order. text: {"d", "k", "c", "s", "e",
                               "b", "h"}; chat: {"d", "k", "t", "role", "ct", "cc", "s", "e", "b", "h"}.
                               d = doc line index, k = window index in the doc, b = target bytes, h =
                               32-bit subsample rank. Rules and meaning: evalwin.py.
  OUT_DIR/manifest.json        config, per set counts (docs, windows, scored bytes, per role for chat,
                               skipped docs), file sha256s, input sha256s, the checks, evalset_sha256.
Checks (the build refuses and writes nothing when one fails): no OASST2 tree of the OOD-H reserve
(corpus/oodh.py) in any set; with --train-shards, no eval doc shares an id, a split key (Dolly context,
OASST2 tree) or an exact text (sha1 of the cleaned text) with any doc in any training shard, read from
the shards' .ids files (pretokenize.py). The OASST2 set is the (b) dev slice: held-out threads, never
OOD-H. OOD-H itself is never built here.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import evalwin as EW
import prep_common as C

CODE = ["data_prep/prep_common.py", "data_prep/evalwin.py", "data_prep/eval_sets.py", "corpus/oodh.py",
        "corpus/sample_plan.py"]
CORPUS_METRIC = {"oasst2": "b: held-out human conversational turns (OASST2 dev slice, disjoint from OOD-H)",
                 "irc": "b: held-out human conversational text (IRC)",
                 "dolly": "held-out human instruction/response turns",
                 "cccc": "d: held-out pre-2023 web text", "gutenberg": "d: held-out strict-open prose",
                 "wikimedia": "d: held-out strict-open prose", "stackexchange": "held-out Q&A prose"}


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_index(dirs: list[str]) -> tuple[set, set, set, int]:
    ids, keys, hashes, n = set(), set(), set(), 0
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*", "*.ids")))
        assert files, f"no .ids files under {d}"
        for p in files:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    i, k, h = line.rstrip("\n").split("\t")
                    ids.add(i)
                    keys.update(x for x in k.split("|") if x)
                    hashes.add(h)
                    n += 1
    return ids, keys, hashes, n


def build_set(name: str, recs: list[dict], a) -> tuple[list[dict], list[dict], dict]:
    chat = any(C.is_chat(r) for r in recs)
    assert not chat or all(C.is_chat(r) for r in recs), f"{name}: mixes chat and text records"
    docs, wins = [], []
    st = {"kind": "chat" if chat else "text", "docs_in": len(recs), "docs": 0, "skipped_docs": 0,
          "windows": 0, "scored_bytes": 0, "special_replaced_docs": 0, "doc_bytes": 0}
    for r in recs:
        C.check_oodh(r)
        if chat:
            turns, changed = [], False
            src = ([{"role": "system", "text": r["system"]}] if r.get("system") else []) + list(r["turns"])
            for t in src:
                s, ch = C.clean_text(t.get("text") or "")
                turns.append({"role": t["role"], "text": s})
                changed |= ch
            ws = EW.chat_windows(turns, a.window_bytes, a.chat_context_bytes, a.lookback)
            doc = {"i": len(docs), "id": r["id"], "turns": turns}
            tid = (r.get("meta") or {}).get("tree_id")
            if tid:
                doc["tree_id"] = tid
            nbytes = sum(len(t["text"].encode("utf-8")) for t in turns)
        else:
            text, changed = C.clean_text(r.get("text") or "")
            ws = EW.text_windows(text, a.window_bytes, a.context_bytes, a.lookback)
            doc = {"i": len(docs), "id": r["id"], "text": text}
            nbytes = len(text.encode("utf-8"))
        if not ws:
            st["skipped_docs"] += 1
            continue
        st["special_replaced_docs"] += int(changed)
        for k, w in enumerate(ws):
            w = {"d": doc["i"], "k": k, **w, "h": EW.window_hash(name, r["id"], k)}
            wins.append(w)
            st["scored_bytes"] += w["b"]
            if chat:
                rk = f"scored_bytes_{w['role']}"
                st[rk] = st.get(rk, 0) + w["b"]
        docs.append(doc)
        st["docs"] += 1
        st["doc_bytes"] += nbytes
    st["windows"] = len(wins)
    return docs, wins, st


def check_train(name: str, recs: list[dict], index) -> dict:
    ids, keys, hashes, _ = index
    ov = {"id": 0, "split_key": 0, "text_sha1": 0}
    for r in recs:
        ks = C.record_keys(r)
        ov["id"] += r["id"] in ids
        ov["split_key"] += any(k in keys or k in ids for k in ks)
        ov["text_sha1"] += C.text_hash(C.clean_text(C.doc_text(r))[0]) in hashes
    return ov


def write_jsonl(path: str, rows: list[dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return C.sha256_file(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("heldout_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--sources", default=None)
    ap.add_argument("--window-bytes", type=int, default=2048)
    ap.add_argument("--context-bytes", type=int, default=1024)
    ap.add_argument("--chat-context-bytes", type=int, default=3072)
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--train-shards", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out_dir)
    if os.path.exists(out) and not a.overwrite:
        raise SystemExit(f"{out} exists (use --overwrite)")
    names = a.sources.split(",") if a.sources else sorted(
        os.path.basename(p)[:-6] for p in glob.glob(os.path.join(a.heldout_dir, "*.jsonl")))
    index = train_index(a.train_shards) if a.train_shards else None
    part = out + ".partial"
    shutil.rmtree(part, ignore_errors=True)
    os.makedirs(part)
    sets, inputs, checks = {}, {}, {"oodh_trees_checked": 0, "oodh_reserved_found": 0, "train_overlap": {}}
    try:
        for name in names:
            p = os.path.join(a.heldout_dir, f"{name}.jsonl")
            recs = read_records(p)
            inputs[name] = {"file": f"{name}.jsonl", "sha256": C.sha256_file(p), "records": len(recs)}
            docs, wins, st = build_set(name, recs, a)          # refuses an OOD-H tree (C.Refusal)
            checks["oodh_trees_checked"] += sum(bool((r.get("meta") or {}).get("tree_id")) for r in recs)
            if index is not None:
                ov = check_train(name, recs, index)
                checks["train_overlap"][name] = ov
                if any(ov.values()):
                    raise C.Refusal(f"{name}: eval docs overlap the training shards {ov}")
            st["files"] = {"docs": {"path": f"{name}.docs.jsonl", "sha256": write_jsonl(
                os.path.join(part, f"{name}.docs.jsonl"), docs)},
                "windows": {"path": f"{name}.windows.jsonl", "sha256": write_jsonl(
                    os.path.join(part, f"{name}.windows.jsonl"), wins)}}
            st["corpus_metric"] = CORPUS_METRIC.get(name)
            sets[name] = st
            print(f"[eval_sets] {name}: {st['docs']:,} docs, {st['windows']:,} windows, "
                  f"{st['scored_bytes']:,} scored bytes ({st['kind']})", flush=True)
    except C.Refusal as e:
        shutil.rmtree(part, ignore_errors=True)
        print(f"[eval_sets] refused: {e}", file=sys.stderr, flush=True)
        return 3
    cfg = {"window_bytes": a.window_bytes, "context_bytes": a.context_bytes,
           "chat_context_bytes": a.chat_context_bytes, "lookback": a.lookback,
           "scored_chat_roles": list(EW.SCORED_ROLES), "offsets": "UTF-8 byte offsets, at character starts"}
    heldman = os.path.join(os.path.dirname(os.path.abspath(a.heldout_dir)), "manifest.json")
    sig = hashlib.sha256(json.dumps([cfg, {n: s["files"] for n, s in sets.items()}], sort_keys=True).encode())
    man = {"format": "planck-evalset-v1", "config": cfg, "sets": sets, "inputs": inputs,
           "heldout_dir": C.home_rel(a.heldout_dir),
           "heldout_manifest_sha256": C.sha256_file(heldman) if os.path.exists(heldman) else None,
           "train_shards": [{"dir": C.home_rel(d), "manifest_sha256": C.sha256_file(os.path.join(d, "manifest.json"))
                             if os.path.exists(os.path.join(d, "manifest.json")) else None} for d in a.train_shards],
           "train_docs_checked": index[3] if index else 0, "checks": checks,
           "evalset_sha256": sig.hexdigest(), "code_sha256": C.code_hashes(CODE)}
    C.write_json(os.path.join(part, "manifest.json"), man)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(part, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
