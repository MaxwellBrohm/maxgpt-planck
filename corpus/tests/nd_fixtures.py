"""Synthetic documents for the near-dedup tests: random pseudo-words (letters only, so digit
normalization does not merge them), and edited copies with a chosen shingle Jaccard."""
import os
import random

import neardedup as ND

SYL = [c + v for c in "bcdfghjklmnprstvwz" for v in "aeiou"]


def vocab(seed=7, n=8000):
    r = random.Random(seed)
    return sorted({"".join(r.choice(SYL) for _ in range(r.randint(2, 4))) for _ in range(n)})


class Gen:
    def __init__(self, seed=1):
        self.r, self.v = random.Random(seed), vocab()

    def words(self, n):
        return [self.r.choice(self.v) for _ in range(n)]

    def text(self, n):
        return " ".join(self.words(n))

    def edit(self, ws, k):
        """ws with k words (at distinct random positions) replaced by random words."""
        ws = list(ws)
        for i in self.r.sample(range(len(ws)), min(k, len(ws))):
            ws[i] = self.r.choice(self.v)
        return ws

    def near_copy(self, ws, target):
        """An edited copy whose shingle Jaccard with ws is roughly target (edits spread out:
        each replaced word changes up to 5 shingles)."""
        s = len(ws) - ND.NGRAM + 1
        k = max(1, round(s * (1 - target) / (1 + target) / ND.NGRAM)) if target < 1 else 0
        return self.edit(ws, k)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(ND.shingles(a).tolist()), set(ND.shingles(b).tolist())
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def write_shard(d, name, texts, created=None):
    """Pass A for a list of texts -> the shard stem (d/name, sidecar d/name.mh.npy)."""
    stem = os.path.join(d, name)
    w = ND.SidecarWriter(stem)
    for i, t in enumerate(texts):
        w.add(t, created[i] if created is not None else None)
    w.close()
    return stem
