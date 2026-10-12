"""Item sources for the Planck loader. An item is one document or conversation:
(ids int64 array, flags bool array), flags[i] True = ids[i] is a supervised target.

TokenShardSource   headerless uint16 .bin shards (the format of MaxGPT-Ultra's
                   data/prepare.py: every document followed by an EOT id). Items are
                   documents cut at EOT (EOT kept as the last token); a document longer
                   than max_len is cut into max_len chunks, each its own item.
ChatJsonlSource    jsonl conversation shards (pipeline/SPEC.txt section 11), rendered with
                   chat_template.ChatTemplate. A conversation longer than max_len is
                   dropped and counted (never cut: a cut conversation loses its start).

Both loop forever: after the last shard they start the next epoch, with the shard order
permuted by (seed, epoch) when shuffle is on. state_dict() is a small JSON-able cursor.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _order(n: int, seed: int, epoch: int, shuffle: bool) -> list[int]:
    if not shuffle:
        return list(range(n))
    return [int(i) for i in np.random.default_rng([seed, epoch, 7]).permutation(n)]


class TokenShardSource:
    def __init__(self, paths: list[str], eot_id: int, max_len: int, seed: int = 0,
                 shuffle: bool = True):
        assert paths, "no token shards"
        self.paths, self.eot, self.max_len = list(paths), int(eot_id), int(max_len)
        self.seed, self.shuffle = seed, shuffle
        self.epoch, self.k, self.offset = 0, 0, 0     # k = position in this epoch's order
        self._mm: dict[int, np.memmap] = {}

    def _shard(self, si: int) -> np.memmap:
        if si not in self._mm:
            if len(self._mm) > 8:
                self._mm.clear()
            self._mm[si] = np.memmap(self.paths[si], dtype=np.uint16, mode="r")
        return self._mm[si]

    def next_item(self):
        for _ in range(2 * len(self.paths) + 2):
            si = _order(len(self.paths), self.seed, self.epoch, self.shuffle)[self.k]
            mm = self._shard(si)
            if self.offset >= len(mm):
                self.k, self.offset = self.k + 1, 0
                if self.k >= len(self.paths):
                    self.epoch, self.k = self.epoch + 1, 0
                continue
            window = np.asarray(mm[self.offset:self.offset + self.max_len])
            hits = np.flatnonzero(window == self.eot)
            n = int(hits[0]) + 1 if len(hits) else len(window)
            ids = window[:n].astype(np.int64)
            self.offset += n
            flags = np.ones(n, dtype=bool)
            flags[0] = False
            return ids, flags
        raise RuntimeError("token shards are empty")

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "k": self.k, "offset": self.offset}

    def load_state_dict(self, st: dict) -> None:
        self.epoch, self.k, self.offset = int(st["epoch"]), int(st["k"]), int(st["offset"])


class ChatJsonlSource:
    def __init__(self, paths: list[str], template, max_len: int, encode=None, seed: int = 0,
                 shuffle: bool = True):
        assert paths, "no conversation shards"
        self.paths, self.template, self.max_len = list(paths), template, int(max_len)
        self.encode, self.seed, self.shuffle = encode, seed, shuffle
        self.epoch, self.k, self.offset = 0, 0, 0     # offset = byte offset in the file
        self.dropped_long = 0

    def _read_line(self):
        """Next non-empty line from the cursor, advancing it; None at end of file."""
        fi = _order(len(self.paths), self.seed, self.epoch, self.shuffle)[self.k]
        with open(self.paths[fi], "rb") as f:
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line:
                    return None
                self.offset = f.tell()
                if line.strip():
                    return line

    def next_item(self):
        empty_passes = 0
        while True:
            line = self._read_line()
            if line is None:
                self.k, self.offset = self.k + 1, 0
                if self.k >= len(self.paths):
                    self.epoch, self.k = self.epoch + 1, 0
                    empty_passes += 1
                    assert empty_passes < 3, "conversation shards hold no usable record"
                continue
            ids, flags = self.template.render(json.loads(line), self.encode)
            if len(ids) > self.max_len:
                self.dropped_long += 1
                continue
            if len(ids) >= 2:
                return ids, flags

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "k": self.k, "offset": self.offset,
                "dropped_long": self.dropped_long}

    def load_state_dict(self, st: dict) -> None:
        self.epoch, self.k, self.offset = int(st["epoch"]), int(st["k"]), int(st["offset"])
        self.dropped_long = int(st.get("dropped_long", 0))


def expand(patterns, base_dir: str = ".") -> list[str]:
    """Glob patterns (relative to base_dir) -> sorted existing paths."""
    import glob
    out: list[str] = []
    for p in ([patterns] if isinstance(patterns, str) else patterns):
        full = p if os.path.isabs(p) else os.path.join(base_dir, p)
        out.extend(sorted(glob.glob(full)))
    return out
