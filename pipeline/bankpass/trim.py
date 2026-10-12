"""Dedup, diversity caps, author thirds and p_exact (BANKPASS s2f, s2g). Input: the kept item records of one bank
(after gates and judges). Every removal is a status change to "dropped" with a code, never a deletion, so the bank
file still holds the full record (s2h).

  exact dedup     NFKC, straight quotes, lowercase, normalized holes (store.norm_key): DUP_EXACT
  near dedup      word 3-gram Jaccard >= 0.6 against a kept item: DUP_NEAR (the earlier, better-judged item stays)
  frame cap       frame signature (function words and holes kept, content words as their POS) <= 5% of a bank:
                  FRAME_CAP (a bank under 40 items keeps at least 2 per frame)
  author thirds   each pinned teacher keeps at most the smallest teacher share x 1.1 (seeded): AUTHOR_TRIM
  p_exact         0.7 x kept / target, at most 0.7, for a bank short of its target (s2g)"""
import random
import re

import lexicons as L
from bankpass import store

NEAR_J, FRAME_MAX, AUTHOR_SLACK, P_EXACT = 0.6, 0.05, 1.1, 0.7
_W = re.compile(r"\{\w+\}|[a-z']+")


def grams3(text):
    w = _W.findall(store.straight(text).lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)} or {tuple(w)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def frame(text, pos_of=None):
    """function words and holes as written, every other word as its POS (word list) or 'w'."""
    out = []
    for w in _W.findall(store.straight(text).lower()):
        if w.startswith("{") or w in L.FUNCTION_WORDS or w in L.STOPWORDS:
            out.append(w)
        else:
            out.append((pos_of(w) if pos_of else None) or "w")
    return " ".join(out)


def _rank(r):
    """better-judged first: more keep verdicts, then the likelihood tail (rare before common), then id."""
    keeps = sum(j.get("verdict") == "keep" for j in r.get("judges") or [])
    tail = {"rare": 0, "uncommon": 1, "common": 2}.get(r.get("features", {}).get("likelihood"), 3)
    return (-keeps, tail, r["id"])


def _drop(r, code):
    r["status"], r["drop"] = "dropped", code


def dedup(recs):
    kept, seen, grams = [], {}, []
    for r in sorted((r for r in recs if r["status"] == "kept"), key=_rank):
        k = store.norm_key(r["text"])
        if k in seen:
            _drop(r, "DUP_EXACT")
            r["dedup"] = {"cluster": seen[k]}
            continue
        g = grams3(r["text"])
        near = next((kid for kid, kg in grams if jaccard(g, kg) >= NEAR_J), None)
        if near:
            _drop(r, "DUP_NEAR")
            r["dedup"] = {"cluster": near}
            continue
        seen[k] = r["id"]
        grams.append((r["id"], g))
        r["dedup"] = {"cluster": r["id"], "rank": len(kept)}
        kept.append(r)
    return kept


def frame_cap(recs, pos_of=None):
    kept = [r for r in recs if r["status"] == "kept"]
    cap = max(2, int(FRAME_MAX * len(kept)))
    count = {}
    for r in sorted(kept, key=_rank):
        f = frame(r["text"], pos_of)
        count[f] = count.get(f, 0) + 1
        r.setdefault("features", {})["frame"] = f
        if count[f] > cap:
            _drop(r, "FRAME_CAP")
    return [r for r in recs if r["status"] == "kept"]


def author_thirds(recs, seed="bankpass"):
    """trim teacher surplus at random (seeded) to the smallest teacher count x AUTHOR_SLACK."""
    kept = [r for r in recs if r["status"] == "kept" and r["author"].get("kind") == "teacher"]
    by = {}
    for r in kept:
        by.setdefault(r["author"]["model"], []).append(r)
    if len(by) < len(store.TEACHERS):
        return [r for r in recs if r["status"] == "kept"]
    limit = int(min(len(v) for v in by.values()) * AUTHOR_SLACK)
    for model, rs in sorted(by.items()):
        rng = random.Random(f"{seed}:{model}:{rs[0]['bank']}")
        extra = sorted(rs, key=lambda r: r["id"])
        rng.shuffle(extra)
        for r in extra[limit:]:
            _drop(r, "AUTHOR_TRIM")
    return [r for r in recs if r["status"] == "kept"]


def p_exact(kept, target):
    return round(min(P_EXACT, P_EXACT * kept / target), 4) if target else P_EXACT


def report(recs):
    kept = [r for r in recs if r["status"] == "kept"]
    frames = {}
    for r in kept:
        f = r.get("features", {}).get("frame") or frame(r["text"])
        frames[f] = frames.get(f, 0) + 1
    words = [w for r in kept for w in _W.findall(r["text"].lower())]
    bi = {tuple(words[i:i + 2]) for i in range(len(words) - 1)}
    return {"kept": len(kept), "frames": len(frames), "top_frame_share": round(max(frames.values()) / len(kept), 4)
            if kept else 0.0, "distinct_2": round(len(bi) / max(1, len(words) - 1), 4),
            "authors": store.author_shares(recs)}


def run(recs, target, pos_of=None, seed="bankpass"):
    dedup(recs)
    frame_cap(recs, pos_of)
    author_thirds(recs, seed)
    rep = report(recs)
    rep["p_exact"] = p_exact(rep["kept"], target)
    return rep
