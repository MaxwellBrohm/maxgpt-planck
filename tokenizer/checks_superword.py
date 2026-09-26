"""P-101: what would a few hundred chat-phrase superword tokens add to the 8k vocabulary?

Pieces are the family pre-tokenizer's pieces (spec.PRETOKENIZE_REGEX): a word piece is a letter run
with at most one leading space (" how", "Hi"); a contraction piece is "'m", "'t", "'ll" ... A WORD
UNIT is a word piece plus the contraction piece right after it, if any (" don" + "'t"). A PHRASE is
2-4 consecutive word units where every unit after the first starts with a space, taken as its exact
surface string, so " how are you" (mid-text) and "How are you" (after a newline) are different
phrases, as they would be different tokens. Phrases never cross punctuation, digits or newlines.

Mining: count every phrase occurrence (overlapping, all of k = 2, 3, 4) in the TRAINING chat files;
rank by count (ties by string). Ranking "freq" is the specified method; ranking "saved" orders by
count * (tokens the phrase takes under the base - 1), an alternative that favors long phrases. It
ignores overlaps between phrases, so it is not an upper bound (on the trial sample it did worse).
Scoring on the HELD-OUT chat files, for each N: the base tokenizer (the family member of size V,
default 8192) is truncated to V - N (the last N merge-made tokens go, the same as the family member
of that size) and the top N phrases become one token each, so the total stays V. Text is tokenized
left to right: at each word unit the longest listed phrase starting there (4, then 3, then 2 units)
becomes one token; every other unit costs what the V - N BPE gives its pieces.
  gain_pct = (base_tokens / variant_tokens - 1) * 100 = relative rise in bytes/token (same bytes)
The ledger bar: kept only if gain_pct >= 5 on held-out chat (pooled over the chat sources).
Two counts per arm. The greedy count above is the specified method, but it assumes a phrase token
fires only at word-unit boundaries. Real added tokens (tokenizers AddedToken, normalized=False) are
matched anywhere in the raw text before the pre-tokenizer, so a phrase also fires inside a longer
word ('all of these' -> 'all', ' of the', 'se') and splits it. real_tokens / real_gain_pct measure
that realizable mechanism on the same truncated BPE; the greedy gain is usually the higher,
optimistic figure. passes_bar is on greedy (as specified), passes_bar_real on the real count.
Special-token strings are stripped (sample_io.strip_specials) from every text first, so piece-wise
counting equals Tokenizer.encode exactly; p101() asserts that on the base.
"""
from __future__ import annotations

import collections
import json
import sys
import unicodedata

import sample_io
import spec
from checks_bpt import POOLED, count_tokens, utf8_len
from health import byte_decoder
from truncate import truncate_json

MIN_K, MAX_K = 2, 4
DEFAULT_NS = (100, 300, 500)
BAR_PCT = 5.0
APOSTROPHES = "'’"


def make_splitter():
    from tokenizers import Regex, pre_tokenizers
    return pre_tokenizers.Split(Regex(spec.PRETOKENIZE_REGEX), behavior="isolated")


def is_word_piece(p: str) -> bool:
    q = p[1:] if p.startswith(" ") else p
    return bool(q) and unicodedata.category(q[0])[0] in "LM"


def is_contraction(p: str) -> bool:
    return len(p) > 1 and p[0] in APOSTROPHES and p[1:].isalpha()


class Units:
    """Word-unit view of a list of texts: per doc (surfaces, run), where run[j] is how many phrase-
    joinable word units start at unit j (0 for a non-word unit). parts maps a surface to its pieces."""

    def __init__(self, texts, splitter=None):
        self.splitter = splitter or make_splitter()
        self.parts: dict[str, tuple] = {}
        self.docs = [self.split(t) for t in texts]

    def split(self, text: str):
        pcs = [p for p, _ in self.splitter.pre_tokenize_str(text)]
        surf, word, cont = [], [], []
        i = 0
        while i < len(pcs):
            p = pcs[i]
            grp = (p,)
            if is_word_piece(p) and i + 1 < len(pcs) and is_contraction(pcs[i + 1]):
                grp = (p, pcs[i + 1])
            s = sys.intern("".join(grp))
            self.parts.setdefault(s, grp)
            surf.append(s)
            word.append(is_word_piece(p))
            cont.append(word[-1] and p[0] == " ")
            i += len(grp)
        run = [0] * len(surf)
        for j in range(len(surf) - 1, -1, -1):
            if word[j]:
                run[j] = 1 + (run[j + 1] if j + 1 < len(surf) and cont[j + 1] else 0)
        return surf, run


def mine(units: Units) -> collections.Counter:
    """Occurrences of every 2-4-unit phrase (overlapping)."""
    c = collections.Counter()
    for surf, run in units.docs:
        for j, r in enumerate(run):
            for k in range(MIN_K, min(MAX_K, r) + 1):
                c["".join(surf[j: j + k])] += 1
    return c


class PieceCost:
    """Tokens a piece takes under one BPE model (the model's own tokenize on its byte-level form)."""

    def __init__(self, tok):
        self.model = tok.model
        self.enc = {b: c for c, b in byte_decoder().items()}
        self.cache: dict[str, int] = {}

    def __call__(self, piece: str) -> int:
        n = self.cache.get(piece)
        if n is None:
            n = len(self.model.tokenize("".join(self.enc[b] for b in piece.encode("utf-8"))))
            self.cache[piece] = n
        return n


def greedy_tokens(units: Units, cost: PieceCost, phrases: set) -> tuple[int, int]:
    """(total tokens, phrase tokens used) with left-to-right longest-phrase replacement."""
    total = hits = 0
    ucost: dict[str, int] = {}
    for surf, run in units.docs:
        j, n = 0, len(surf)
        while j < n:
            k = 0
            if phrases and run[j] >= MIN_K:
                for kk in range(min(MAX_K, run[j]), MIN_K - 1, -1):
                    if "".join(surf[j: j + kk]) in phrases:
                        k = kk
                        break
            if k:
                total, hits, j = total + 1, hits + 1, j + k
                continue
            s = surf[j]
            c = ucost.get(s)
            if c is None:
                c = ucost[s] = sum(cost(p) for p in units.parts[s])
            total, j = total + c, j + 1
    return total, hits


def phrase_cost(phrase: str, splitter, cost: PieceCost) -> int:
    return sum(cost(p) for p, _ in splitter.pre_tokenize_str(phrase))


def rank(counts: collections.Counter, how: str, splitter=None, cost: PieceCost | None = None) -> list:
    """[(phrase, count, score)] best first. freq: score = count. saved: count * (base tokens - 1)."""
    rows = []
    for p, c in counts.items():
        if c < 2:
            continue
        rows.append((p, c, c if how == "freq" else c * (phrase_cost(p, splitter, cost) - 1)))
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def _gain(base: int, var: int) -> float | None:
    return round((base / var - 1.0) * 100.0, 4) if var else None


def real_tokens(small_json: dict, phrases, held: dict[str, list[str]]) -> dict:
    """Token counts with the phrases added as real AddedTokens to the truncated BPE."""
    from tokenizers import AddedToken, Tokenizer
    tok = Tokenizer.from_str(json.dumps(small_json))
    if phrases:
        tok.add_tokens([AddedToken(p, normalized=False) for p in sorted(phrases)])
    out = {s: count_tokens(tok, ts) for s, ts in held.items()}
    out[POOLED] = sum(out.values())
    return out


def p101(base_json: dict, train_texts, heldout_chat: dict[str, list[str]], ns=DEFAULT_NS,
         rankings=("freq", "saved"), n_examples: int = 20) -> dict:
    from tokenizers import Tokenizer
    v = len(base_json["model"]["vocab"])
    splitter = make_splitter()
    train_u = Units((sample_io.strip_specials(t) for t in train_texts), splitter)
    counts = mine(train_u)
    held = {s: [sample_io.strip_specials(t) for t in ts] for s, ts in sorted(heldout_chat.items())}
    held_u = {s: Units(ts, splitter) for s, ts in held.items()}
    base_tok = Tokenizer.from_str(json.dumps(base_json))
    base_cost = PieceCost(base_tok)
    base = {s: greedy_tokens(u, base_cost, set())[0] for s, u in held_u.items()}
    for s, ts in held.items():
        enc = count_tokens(base_tok, ts)
        assert base[s] == enc, f"piece-wise count {base[s]} != encode count {enc} on {s}"
    base[POOLED] = sum(base.values())
    nbytes = {s: utf8_len(ts) for s, ts in held.items()}
    nbytes[POOLED] = sum(nbytes.values())
    ranked = {how: rank(counts, how, splitter, base_cost) for how in rankings}
    out = {"base_vocab": v, "train_docs": len(train_u.docs), "distinct_phrases_mined": len(counts),
           "heldout_bytes": nbytes, "base_tokens": base,
           "base_bpt": {s: round(nbytes[s] / base[s], 4) if base[s] else None for s in base},
           "bar_pct": BAR_PCT, "arms": []}
    for n in ns:
        small_json = truncate_json(base_json, v - n)
        small = Tokenizer.from_str(json.dumps(small_json))
        cost = PieceCost(small)
        plain = {s: greedy_tokens(u, cost, set())[0] for s, u in held_u.items()}
        plain[POOLED] = sum(plain.values())
        for how in rankings:
            chosen = ranked[how][:n]
            phrases = {p for p, _, _ in chosen}
            res = {s: greedy_tokens(u, cost, phrases) for s, u in held_u.items()}
            tok = {s: r[0] for s, r in res.items()}
            tok[POOLED] = sum(tok.values())
            hits = sum(r[1] for r in res.values())
            real = real_tokens(small_json, phrases, held)
            out["arms"].append({
                "n": n, "ranking": how, "n_phrases": len(phrases), "truncated_to": v - n,
                "tokens": tok, "tokens_without_phrases": plain, "phrase_tokens_used": hits,
                "gain_pct": {s: _gain(base[s], tok[s]) for s in tok},
                "real_tokens": real, "real_gain_pct": {s: _gain(base[s], real[s]) for s in real},
                "drop_merges_gain_pct": {s: _gain(base[s], plain[s]) for s in plain},
                "token_reduction_pct": round((1 - tok[POOLED] / base[POOLED]) * 100, 4) if base[POOLED] else None,
                "examples": [{"phrase": p, "count": c} for p, c, _ in chosen[:n_examples]]})
    freq = [a["gain_pct"][POOLED] for a in out["arms"] if a["ranking"] == "freq" and a["gain_pct"][POOLED] is not None]
    anyr = [a["gain_pct"][POOLED] for a in out["arms"] if a["gain_pct"][POOLED] is not None]
    out["best_gain_pct_freq"] = max(freq) if freq else None
    out["best_gain_pct_any"] = max(anyr) if anyr else None
    out["passes_bar"] = bool(freq) and max(freq) >= BAR_PCT
    real = [a["real_gain_pct"][POOLED] for a in out["arms"]
            if a["ranking"] == "freq" and a["real_gain_pct"][POOLED] is not None]
    out["best_real_gain_pct_freq"] = max(real) if real else None
    out["passes_bar_real"] = bool(real) and max(real) >= BAR_PCT
    return out
