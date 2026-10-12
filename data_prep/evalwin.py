"""Fixed-window bits per byte: where windows are cut and how a window becomes model input.

Everything is in BYTES of the UTF-8 text, never in tokens, so every tokenizer is scored on exactly the
same bytes with exactly the same context bytes (CORPUS 6.1: "fixed-window bits per byte, which does not
depend on the tokenizer"). All offsets are byte offsets into a doc's (or a turn's) UTF-8 encoding and
always fall on character starts.

A BOUNDARY is a byte position p >= 1 where byte p is ASCII whitespace and byte p-1 is not. Cutting there
splits a word from the whitespace before the next word, which is a pre-token boundary of the v0
pre-tokenizer (and of GPT-style regexes generally), so tok(context) + tok(target) matches tok(whole) at
the seam.

Text doc (one window list per doc):
  scored region  [h, len) with h = the doc's first boundary (the first word is context only: the harness
                 never predicts a doc's first token, so no window may score it). A doc with no boundary
                 is not scored.
  target windows tile [h, len): from s, the end is the last boundary in [s+W-lookback, s+W], else the
                 last character start <= s+W. W = window bytes (default 2048).
  context        [c, s): c = 0 if s-C <= 0, else the first boundary in [s-C, s) (else the first character
                 start >= s-C). C = context bytes (default 1024). Context never crosses the doc start.
  model input    tok(doc[c:s]) + tok(doc[s:e]); only the target tokens are scored, each predicted from
                 everything before it. Nothing is prepended: a harness text doc starts with its own first
                 token (EOT only ever ends a doc).
Chat doc (turns, each rendered <|role|> text <|end|> as in harness/chat_template.py):
  target windows tile the text [0, len) of every user and assistant turn with the same cut rule (the
                 role token precedes a turn, so its first token is predictable). System and tool turns
                 are context only. Results are kept per role.
  context        up to C_chat bytes of text (default 3072): first the turn's own bytes before s (from a
                 boundary >= s-C_chat when they alone exceed it), then earlier turns, newest first, whole
                 while they fit; the first one that does not fit enters from its first boundary >= its
                 length minus the bytes left, or not at all. Stored as (ct, cc): first context turn and
                 the byte where it enters. Every included turn keeps its role token; closed turns keep
                 their <|end|>.
  model input    for j in ct..t-1: <|role_j|> tok(turn_j[cc if j == ct else 0:]) <|end|>, then
                 <|role_t|> tok(turn_t[(cc if ct == t else 0):s]) and the scored tok(turn_t[s:e]).
Length: if an input exceeds max_len tokens, context tokens are dropped from the LEFT (counted as
truncated; a target that alone exceeds max_len - 1 tokens is an error). Only then does the context
depend on the tokenizer. It is rare but real: digits are one token each and some windows fall under
1.5 bytes per token. Measured on evalsets/v0 at max_len 2048 (2026-09-26): 24 of 23,844 windows with
tok_v0_8k (wikimedia 20, oasst2 4) and 49 with tok_v0_2k (wikimedia 32, oasst2 10, dolly 7); the target
bytes are never cut. bpb.py reports the count per set.
bits per byte of a set = (sum over its windows of the target tokens' NLL in nats) / ln 2 / (sum of target
bytes). Target bytes are the same for every tokenizer; the <|end|> after a turn is never scored (it has
no bytes).
"""
from __future__ import annotations

import hashlib

import numpy as np

WS = np.zeros(256, dtype=bool)
WS[[9, 10, 11, 12, 13, 32]] = True


def boundaries(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.uint8)
    if len(a) < 2:
        return np.zeros(0, dtype=np.int64)
    ws = WS[a]
    return (np.flatnonzero(ws[1:] & ~ws[:-1]) + 1).astype(np.int64)


def char_floor(b: bytes, x: int) -> int:
    while 0 < x < len(b) and (b[x] & 0xC0) == 0x80:
        x -= 1
    return x


def char_ceil(b: bytes, x: int) -> int:
    while x < len(b) and (b[x] & 0xC0) == 0x80:
        x += 1
    return x


def cut(b: bytes, start: int, W: int, lookback: int, bnd: np.ndarray) -> list[tuple[int, int]]:
    out, s, n = [], start, len(b)
    while s < n:
        x = s + W
        if x >= n:
            e = n
        else:
            i = int(np.searchsorted(bnd, x, side="right")) - 1
            if i >= 0 and bnd[i] > s and bnd[i] >= x - lookback:
                e = int(bnd[i])
            else:
                e = char_floor(b, x)
                if e <= s:
                    e = char_ceil(b, s + 1)
        out.append((s, e))
        s = e
    return out


def ctx_start(b: bytes, s: int, C: int, bnd: np.ndarray) -> int:
    lo = s - C
    if lo <= 0:
        return 0
    i = int(np.searchsorted(bnd, lo, side="left"))
    if i < len(bnd) and bnd[i] < s:
        return int(bnd[i])
    return min(char_ceil(b, lo), s)


def text_windows(text: str, W: int, C: int, lookback: int = 256) -> list[dict]:
    b = text.encode("utf-8")
    bnd = boundaries(b)
    if not len(bnd):
        return []
    out = []
    for s, e in cut(b, int(bnd[0]), W, lookback, bnd):
        c = ctx_start(b, s, C, bnd)
        if c >= s:                                   # never a window without context
            c = char_floor(b, s - 1)
        out.append({"c": c, "s": s, "e": e, "b": e - s})
    return out


SCORED_ROLES = ("user", "assistant")


def chat_windows(turns: list[dict], W: int, C: int, lookback: int = 256,
                 roles: tuple = SCORED_ROLES) -> list[dict]:
    tb = [(t.get("text") or "").encode("utf-8") for t in turns]
    bnds = [boundaries(b) for b in tb]
    out = []
    for t, b in enumerate(tb):
        if turns[t]["role"] not in roles:
            continue
        for s, e in cut(b, 0, W, lookback, bnds[t]):
            if s >= C:
                ct, cc = t, ctx_start(b, s, C, bnds[t])
            else:
                ct, cc, rem = t, 0, C - s
                for j in range(t - 1, -1, -1):
                    L = len(tb[j])
                    if L <= rem:
                        ct, cc, rem = j, 0, rem - L
                        continue
                    k = int(np.searchsorted(bnds[j], L - rem, side="left"))
                    if k < len(bnds[j]):
                        ct, cc = j, int(bnds[j][k])
                    break
            out.append({"t": t, "role": turns[t]["role"], "ct": ct, "cc": cc, "s": s, "e": e, "b": e - s})
    return out


def window_hash(set_name: str, doc_id: str, k: int) -> int:
    """32-bit rank for a deterministic subsample (bpb.py --max-windows keeps the lowest)."""
    return int(hashlib.sha256(f"{set_name}\t{doc_id}\t{k}".encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------- model input
def text_item(doc_bytes: bytes, w: dict, encode) -> tuple[list[int], int]:
    """-> (ids, n_ctx): ids[n_ctx:] are the scored target tokens."""
    ctx = encode(doc_bytes[w["c"]:w["s"]].decode("utf-8"))
    tgt = encode(doc_bytes[w["s"]:w["e"]].decode("utf-8"))
    return ctx + tgt, len(ctx)


def chat_item(turn_bytes: list[bytes], roles: list[str], w: dict, encode, role_ids: dict,
              end_id: int) -> tuple[list[int], int]:
    t, ct, cc = w["t"], w["ct"], w["cc"]
    ids: list[int] = []
    for j in range(ct, t):
        ids.append(role_ids[roles[j]])
        ids += encode(turn_bytes[j][(cc if j == ct else 0):].decode("utf-8"))
        ids.append(end_id)
    ids.append(role_ids[roles[t]])
    ids += encode(turn_bytes[t][(cc if ct == t else 0):w["s"]].decode("utf-8"))
    n_ctx = len(ids)
    ids += encode(turn_bytes[t][w["s"]:w["e"]].decode("utf-8"))
    return ids, n_ctx


def fit(ids: list[int], n_ctx: int, max_len: int) -> tuple[list[int], int, bool]:
    """Drop context tokens from the left until the input fits max_len. -> (ids, n_ctx, truncated)."""
    if len(ids) <= max_len:
        return ids, n_ctx, False
    drop = len(ids) - max_len
    if drop >= n_ctx:
        raise ValueError(f"target of {len(ids) - n_ctx} tokens does not fit max_len {max_len} with context")
    return ids[drop:], n_ctx - drop, True
