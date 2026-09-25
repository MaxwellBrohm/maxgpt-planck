"""The Planck training loader: mixed sources, windows, packing or bucketing, exact resume.

Mixing is by TOKENS and deterministic: the next item comes from the source furthest below
its target share (share * total drawn - drawn from it), so a 25% chat share is 25% of
tokens, not of items. Items are drawn in windows of about window_tokens; each window is
packed (pack mode) or bucketed (bucket mode) with an rng seeded by (seed, window index),
and emitted as a list of units (rows, or whole micro-batches).

Resume is exact: state = window index, units already emitted from it, and the source
cursors and mix counters at the window's start. Loading replays the window's draw and
skips the emitted units. See packing.py for the two modes and chat_template.py for the
template and loss flags.
"""
from __future__ import annotations

import copy

import numpy as np

from chat_template import ChatTemplate
from packing import bucket_batches, collate_rows, pack_rows, to_torch_batch
from sources import ChatJsonlSource, TokenShardSource, expand


class Loader:
    def __init__(self, sources: list, shares: list[float], names: list[str], *, mode: str,
                 seq_len: int, micro_batch: int, pad_id: int, seed: int = 0,
                 window_tokens: int | None = None, buckets: list[int] | None = None,
                 tokens_per_micro: int | None = None):
        assert mode in ("pack", "bucket"), mode
        assert len(sources) == len(shares) == len(names) and sources
        tot = float(sum(shares))
        assert tot > 0
        self.sources, self.names = sources, names
        self.shares = [s / tot for s in shares]
        self.mode, self.L, self.B, self.pad = mode, seq_len, micro_batch, pad_id
        self.seed = seed
        self.window_tokens = window_tokens or 64 * seq_len
        self.buckets = sorted(buckets or [seq_len])
        assert self.buckets[-1] <= seq_len
        self.tokens_per_micro = tokens_per_micro or micro_batch * seq_len
        self.counts = [0] * len(sources)
        self._w, self._u, self._units, self._start = -1, 0, [], None

    # ---------------- windows ----------------
    def _snapshot(self) -> dict:
        return {"sources": [copy.deepcopy(s.state_dict()) for s in self.sources],
                "counts": list(self.counts)}

    def _pick(self) -> int:
        total = sum(self.counts)
        deficits = [sh * total - c for sh, c in zip(self.shares, self.counts)]
        return int(np.argmax(deficits))

    def _build_window(self) -> None:
        self._w += 1
        self._start = self._snapshot()
        items, got = [], 0
        while got < self.window_tokens:
            i = self._pick()
            ids, flags = self.sources[i].next_item()
            self.counts[i] += len(ids)
            items.append((ids, flags))
            got += len(ids)
        rng = np.random.default_rng([self.seed, self._w, 11])
        if self.mode == "pack":
            self._units = pack_rows(items, self.L, self.pad, rng)
        else:
            self._units = bucket_batches(items, self.buckets, self.tokens_per_micro, self.pad, rng)
        self._u = 0

    def _next_unit(self):
        while self._u >= len(self._units):
            self._build_window()
        unit = self._units[self._u]
        self._u += 1
        return unit

    def next_batch(self) -> dict:
        """-> {idx, tgt, doc, pos} torch int64 tensors on CPU (doc/pos None in bucket mode)."""
        if self.mode == "pack":
            return collate_rows([self._next_unit() for _ in range(self.B)])
        return to_torch_batch(self._next_unit())

    # ---------------- state ----------------
    def state_dict(self) -> dict:
        return {"window": self._w, "unit": self._u, "start": copy.deepcopy(self._start),
                "counts_now": list(self.counts)}

    def load_state_dict(self, st: dict) -> None:
        if st["window"] < 0:
            return
        for s, ss in zip(self.sources, st["start"]["sources"]):
            s.load_state_dict(ss)
        self.counts = list(st["start"]["counts"])
        self._w = st["window"] - 1
        self._build_window()
        self._u = st["unit"]
        assert self.counts == st["counts_now"], "resume replay drew different data"

    def stats(self) -> dict:
        # tokens read into windows (runs slightly ahead of what training has consumed)
        out = {f"drawn_{n}": c for n, c in zip(self.names, self.counts)}
        for n, s in zip(self.names, self.sources):
            if hasattr(s, "dropped_long"):
                out[f"dropped_long_{n}"] = s.dropped_long
        return out


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def build_loader(dcfg: dict, seq_len: int, micro_batch: int, base_dir: str = ".",
                 seed: int = 0) -> Loader:
    """dcfg = the config's `data:` block, see configs/sample_tiny/config.yaml."""
    tok = load_tokenizer(_abs(dcfg["tokenizer"], base_dir)) if dcfg.get("tokenizer") else None
    encode = (lambda s: tok.encode(s, add_special_tokens=False).ids) if tok else None
    chat = dcfg.get("chat", {})
    max_len = int(dcfg.get("max_item_len", seq_len))
    if dcfg.get("mode", "pack") == "bucket":
        max_len = min(max_len, max(dcfg.get("buckets", [seq_len])))
    sources, shares, names = [], [], []
    for k, sc in enumerate(dcfg["sources"]):
        paths = expand(sc["paths"], base_dir)
        assert paths, f"source {sc.get('name', k)}: no files match {sc['paths']}"
        s_seed = seed * 1000 + k
        if sc["kind"] == "tokens":
            src = TokenShardSource(paths, sc["eot_id"], max_len, s_seed, sc.get("shuffle", True))
        elif sc["kind"] == "chat":
            if "role_ids" in chat:
                tmpl = ChatTemplate({r: int(i) for r, i in chat["role_ids"].items()},
                                    int(chat["end_id"]), chat.get("loss", "assistant"),
                                    chat.get("tool_role", "tool"))
            else:
                assert tok is not None, "chat source needs chat.role_ids or a tokenizer"
                tmpl = ChatTemplate.from_tokenizer(tok, chat.get("loss", "assistant"))
            src = ChatJsonlSource(paths, tmpl, max_len, encode, s_seed, sc.get("shuffle", True))
        else:
            raise ValueError(f"unknown source kind {sc['kind']!r}")
        sources.append(src)
        shares.append(float(sc.get("share", 1.0)))
        names.append(sc.get("name", sc["kind"]))
    return Loader(sources, shares, names, mode=dcfg.get("mode", "pack"), seq_len=seq_len,
                  micro_batch=micro_batch, pad_id=int(dcfg["pad_id"]), seed=seed,
                  window_tokens=dcfg.get("window_tokens"), buckets=dcfg.get("buckets"),
                  tokens_per_micro=dcfg.get("tokens_per_micro"))


def _abs(p: str, base: str) -> str:
    import os
    return p if os.path.isabs(p) else os.path.join(base, p)
