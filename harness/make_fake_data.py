"""Synthetic data for tests and smoke runs (no tokenizer, no downloads).

Token ids: 0 pad, 1 eot, 2 <|system|>, 3 <|user|>, 4 <|assistant|>, 5 <|end|>, 6 <|tool|>,
content ids from 8 up. General text: documents that count upward from a random start
(mod the content range), so the next token is predictable. Chat: the assistant repeats
the user's tokens (a copy task, so assistant-only loss has something to learn).

  python make_fake_data.py OUT_DIR [--vocab 256] [--docs 400] [--chats 300] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SPECIAL = {"pad_id": 0, "eot_id": 1, "role_ids": {"system": 2, "user": 3, "assistant": 4, "tool": 6},
           "end_id": 5}
FIRST_CONTENT = 8


def make(out_dir: str, vocab: int = 256, docs: int = 400, chats: int = 300, seed: int = 0,
         n_shards: int = 2, doc_len=(5, 300), chat_turn_len=(2, 20)) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    span = vocab - FIRST_CONTENT
    per = [[] for _ in range(n_shards)]
    for d in range(docs):
        n = int(rng.integers(*doc_len))
        start = int(rng.integers(0, span))
        per[d % n_shards].extend(list(FIRST_CONTENT + (start + np.arange(n)) % span) + [SPECIAL["eot_id"]])
    tok_paths = []
    for s, ids in enumerate(per):
        p = os.path.join(out_dir, f"text_{s:03d}.bin")
        np.asarray(ids, dtype=np.uint16).tofile(p)
        tok_paths.append(p)
    chat_paths = [os.path.join(out_dir, f"chat_{s:03d}.jsonl") for s in range(n_shards)]
    files = [open(p, "w", encoding="utf-8") for p in chat_paths]
    try:
        for c in range(chats):
            turns = []
            for _ in range(int(rng.integers(1, 4))):
                u = [int(x) for x in rng.integers(FIRST_CONTENT, vocab, int(rng.integers(*chat_turn_len)))]
                turns.append({"role": "user", "ids": u})
                turns.append({"role": "assistant", "ids": list(u)})
            rec = {"id": f"fake-{c}", "turns": turns}
            if c % 5 == 0:
                rec["system"] = None
            files[c % n_shards].write(json.dumps(rec) + "\n")
            if c % 7 == 0:
                files[c % n_shards].write("\n")          # blank lines must be skipped
    finally:
        for f in files:
            f.close()
    return {"tokens": tok_paths, "chat": chat_paths, **SPECIAL}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--chats", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    info = make(a.out_dir, a.vocab, a.docs, a.chats, a.seed)
    print(json.dumps(info, indent=1))
