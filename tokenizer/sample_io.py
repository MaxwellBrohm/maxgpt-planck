"""Tokenizer sample layout on disk, as corpus/make_tok_sample.py writes it, read by train_bpe.py and health.py.

  SAMPLE/train/<source>.jsonl     one record per line; only its "text" field is read here
  SAMPLE/heldout/<source>.jsonl   same, disjoint from train
  SAMPLE/manifest.json            the builder's record; an optional "weights" {source: repeats} is honored
The trainer repeats each source's train file `weight` times (default 1; train_bpe.py --weight overrides).
Repetition in a BPE sample is the same as multiplying that source's word counts.
Special-token strings are removed from every text before training (strip_specials).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

from spec import SPECIALS

SPECIAL_RE = re.compile("|".join(re.escape(s) for s in sorted(SPECIALS, key=len, reverse=True)))


def strip_specials(text: str) -> str:
    """Special-token strings never enter the tokenizer sample (they would only teach merges like '<|')."""
    return SPECIAL_RE.sub(" ", text)


def source_files(sample_dir: str, split: str) -> dict[str, str]:
    paths = sorted(glob.glob(os.path.join(sample_dir, split, "*.jsonl")))
    return {os.path.basename(p)[: -len(".jsonl")]: p for p in paths}


def iter_file(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["text"]


def load_weights(sample_dir: str) -> dict[str, int]:
    p = os.path.join(sample_dir, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).get("weights", {}).items()}


def iter_texts(sample_dir: str, split: str = "train", weights: dict[str, int] | None = None):
    """Every document of the split, each source repeated weights[source] times (default 1)."""
    w = load_weights(sample_dir) if weights is None else weights
    files = source_files(sample_dir, split)
    unknown = set(w) - set(files)
    assert not unknown, f"weights name sources with no {split} file: {sorted(unknown)}"
    for name, path in files.items():
        for _ in range(int(w.get(name, 1))):
            yield from iter_file(path)


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path, self.f, self.n, self.bytes = path, open(path, "w", encoding="utf-8"), 0, 0

    def write(self, text: str) -> None:
        self.f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.n += 1
        self.bytes += len(text.encode("utf-8"))

    def close(self) -> None:
        self.f.close()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
