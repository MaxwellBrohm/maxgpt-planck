"""A small hand-made corpus in the extractor's layout, with held-out splits that exercise every rule:
held-out ids, a Dolly row that shares a held-out context under another id, OASST2 trees (never from the
OOD-H reserve unless asked), control and markup strings in text, multi-byte text, a doc with no word
boundary, an empty doc, long docs (many windows, several chunks) and a system prompt."""
from __future__ import annotations

import hashlib
import json
import os
import random

from oodh import in_oodh_reserve

WORDS = ("the dog Pearl is twelve and likes long walks near the river while it rains a little "
         "café naïve résumé 東京 はい 🙂 👍🏽 é 1234 5,678.90 don't she'll we're (paren) [x] {y} "
         "http://example.org/a?b=c#d tab\tsep new\nline double  space end.").split(" ")
SPECIALS = ["<|end|>", "<|endoftext|>", "<think>", "</lookup>", "<|user|>"]


def tree_ids(reserved: bool, n: int, seed: int = 0) -> list[str]:
    out, i = [], 0
    while len(out) < n:
        tid = hashlib.md5(f"tree-{seed}-{i}".encode()).hexdigest()
        tid = f"{tid[:8]}-{tid[8:12]}-{tid[12:16]}-{tid[16:20]}-{tid[20:]}"
        if in_oodh_reserve(tid) == reserved:
            out.append(tid)
        i += 1
    return out


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def text_docs(source: str, n: int, seed: int, long_every: int = 7) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        k = rng.randint(3, 60) if i % long_every else rng.randint(900, 2500)
        text = sentence(rng, k)
        if i % 5 == 1:
            text += " " + rng.choice(SPECIALS) + " after"
        if i % 11 == 3:
            text = "\n\n".join([text, sentence(rng, 40)])
        docs.append({"id": f"{source}:{i}", "source": source, "text": text,
                     "meta": {"dataset": f"fake/{source}", "n": i}})
    docs.append({"id": f"{source}:noboundary", "source": source, "text": source[0] * 50, "meta": {}})
    docs.append({"id": f"{source}:empty", "source": source, "text": "", "meta": {}})
    return docs


def chat_docs(source: str, n: int, seed: int, tids: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        turns = []
        for j in range(2 * rng.randint(1, 3)):
            k = rng.randint(2, 40) if (i + j) % 9 else rng.randint(500, 900)
            t = sentence(rng, k)
            if (i + j) % 6 == 2:
                t += " " + rng.choice(SPECIALS)
            turns.append({"role": "user" if j % 2 == 0 else "assistant", "text": t})
        rec = {"id": f"{source}:{i}", "source": source, "text": "\n".join(t["text"] for t in turns),
               "turns": turns, "meta": {"dataset": f"fake/{source}"}}
        if tids is not None:
            rec["meta"].update({"dataset": "OpenAssistant/oasst2", "tree_id": tids[i], "split_key": tids[i]})
            rec["id"] = f"oasst2:{tids[i]}"
        else:
            rec["meta"]["split_key"] = f"ctx:{i // 2}"          # rows 2m and 2m+1 share a context
        if i == 4:
            rec["system"] = "Be brief. " + sentence(rng, 5)
        docs.append(rec)
    return docs


def write_jsonl(path: str, recs: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(root: str, reserved_tree: bool = False) -> dict:
    """-> {input, heldout, records: {source: [...]}, held: {source: [...]}, excluded_ids: {source: set}}"""
    inp, held = os.path.join(root, "input"), os.path.join(root, "heldout")
    tids = tree_ids(False, 30)
    if reserved_tree:
        tids[7] = tree_ids(True, 1)[0]
    recs = {"web": text_docs("web", 60, 1), "books": text_docs("books", 12, 2, long_every=1),
            "oasst2": chat_docs("oasst2", 30, 3, tids), "dolly": chat_docs("dolly", 24, 4)}
    web = recs["web"]
    write_jsonl(os.path.join(inp, "web", "web-00000.jsonl"), web[:35])
    write_jsonl(os.path.join(inp, "web", "web-00001.jsonl"), web[35:])
    for s in ("books", "oasst2", "dolly"):
        write_jsonl(os.path.join(inp, s, f"{s}-00000.jsonl"), recs[s])
    hsel = {"web": [0, 1, 7, 14, 21, 30, 33], "books": [0, 3], "oasst2": [2, 5, 11, 12], "dolly": [4, 9]}
    hrecs = {s: [recs[s][i] for i in idx] for s, idx in hsel.items()}
    hrecs["web"].append(recs["web"][-2])                     # the no-boundary doc: never scorable
    for s, rs in hrecs.items():
        write_jsonl(os.path.join(held, f"{s}.jsonl"), rs)
    excluded = {s: {r["id"] for r in rs} for s, rs in hrecs.items()}
    excluded["dolly"] |= {"dolly:5", "dolly:8"}               # share ctx:2 and ctx:4 with held-out rows
    return {"input": inp, "heldout": held, "records": recs, "held": hrecs, "excluded_ids": excluded}
