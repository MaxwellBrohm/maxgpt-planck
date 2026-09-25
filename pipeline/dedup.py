"""Exact and near-duplicate detection for accepted conversations (SPEC codes DUP_EXACT and DUP_NEAR).

DUP_EXACT: sha256 of the normalized conversation (NFKC, lowercase, straight quotes, punctuation dropped, whitespace
  collapsed, role letter per turn) equals one already accepted.
DUP_NEAR:  MinHash over word 5-gram shingles of the SLOT-MASKED conversation (every slot value replaced by its type
  tag, so a copy that differs only in slot values, which R2 re-draw produces for free, counts as the same chat).
  128 permutations, LSH with 32 bands of 4 rows (candidate threshold about 0.42), then a candidate is a duplicate
  when the estimated Jaccard is >= NEAR_T (0.7 default; std of the estimate about 0.04 at 0.7).
The index lives in memory and is rebuilt from the accepted shards on resume (add() in file order)."""
import hashlib
import re
import unicodedata

import numpy as np

import golds

NUM_PERM = 128
BANDS = 32
ROWS = NUM_PERM // BANDS
NEAR_T = 0.7
SHINGLE = 5
_P = (1 << 31) - 1
_rng = np.random.default_rng(20260925)
_A = _rng.integers(1, _P, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _P, NUM_PERM, dtype=np.uint64)
_PUNCT = re.compile(r"[^\w\s<>]+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
LETTER = {"user": "u", "assistant": "a", "tool": "t", "system": "s"}


def norm(text):
    t = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    return " ".join(_PUNCT.sub(" ", t).split())


def exact_key(turns):
    """turns: [{role, text}] -> hex digest of the normalized conversation."""
    body = "\n".join(LETTER.get(t["role"], "?") + ": " + norm(t["text"]) for t in turns)
    return hashlib.sha256(body.encode()).hexdigest()


def mask(text, slots):
    """replace every slot value (longest first, golds.value_re matching) with <type>."""
    for value, typ in sorted(slots, key=lambda s: -len(s[0])):
        text = golds.value_re(value).sub(f" <{typ}> ", text)
    return text


def shingles(turns, slots):
    words = []
    for t in turns:
        words.append(f"<{LETTER.get(t['role'], '?')}>")
        words.extend(norm(mask(t["text"], slots)).split())
    if len(words) < SHINGLE:
        return {" ".join(words)}
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def _h32(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")


def signature(sh):
    h = np.fromiter((_h32(s) for s in sh), dtype=np.uint64, count=len(sh)).reshape(-1, 1)
    return ((h * _A + _B) % _P).min(axis=0).astype(np.uint32)


def jaccard_est(a, b):
    return float(np.mean(a == b))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def slot_list(slots):
    """record or skeleton slots {id: {type, value}} -> [(value, type)]."""
    return [(s["value"], s["type"]) for s in slots.values() if s.get("value")]


class Index:
    def __init__(self, near_t=NEAR_T):
        self.near_t = near_t
        self.exact = {}
        self.sigs, self.ids = [], []
        self.bands = [dict() for _ in range(BANDS)]

    def __len__(self):
        return len(self.ids)

    def keys(self, turns, slots):
        sig = signature(shingles(turns, slot_list(slots)))
        return exact_key(turns), sig

    def check(self, turns, slots):
        """-> (code, matched conv id, jaccard estimate) or None, plus the keys to pass to add()."""
        ek, sig = self.keys(turns, slots)
        if ek in self.exact:
            return ("DUP_EXACT", self.exact[ek], 1.0), (ek, sig)
        best = None
        for b in range(BANDS):
            bk = sig[b * ROWS:(b + 1) * ROWS].tobytes()
            for j in self.bands[b].get(bk, ()):
                est = jaccard_est(sig, self.sigs[j])
                if est >= self.near_t and (best is None or est > best[1]):
                    best = (j, est)
        if best:
            return ("DUP_NEAR", self.ids[best[0]], round(best[1], 3)), (ek, sig)
        return None, (ek, sig)

    def add(self, conv_id, keys):
        ek, sig = keys
        self.exact.setdefault(ek, conv_id)
        j = len(self.ids)
        self.ids.append(conv_id)
        self.sigs.append(sig)
        for b in range(BANDS):
            self.bands[b].setdefault(sig[b * ROWS:(b + 1) * ROWS].tobytes(), []).append(j)
