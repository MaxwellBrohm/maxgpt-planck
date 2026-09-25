"""RC-12 DEV builders, step 5 audit: statement balance, so a gold is not findable by the SHAPE of the statement that
carries it. A builder gives every value statement a list of candidate texts (frames, optionally with a neutral
tail); choose() returns one text per statement such that
  length   the GOLD statements sit at the requested length rank among all value statements:
             long   the longest statement is a gold one (strictly, unique) and the shortest is not
             short  the shortest statement is a gold one (strictly, unique) and the longest is not
             mid    neither the longest nor the shortest statement is a gold one (both strict and unique)
  echo     (with a question) the gold is neither the UNIQUE most nor the UNIQUE least echoing statement, both for
           content words shared with the question (OVERLAP / ANTI_OVERLAP) and for the longest shared word run
           (WORDING / ANTI_WORDING); preferring combinations where every statement ties
uniformly among the combinations that qualify. Builders spread the three length targets evenly over each cell, so
LONGEST and SHORTEST land on the gold in about a third of the items; echo rules fall back to position (balanced).
audit_rules.py measures all of it on the built split."""
import itertools
import os
import re
import sys

TAILS = [", by the way.", ", just so you know.", ", in case it matters.", ", for what it's worth."]
TARGETS = ["long", "mid", "short"]
_TE = None


def _te():
    global _TE
    if _TE is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments",
                                        "E004_general_updating", "code"))
        import text_e004
        sys.path.pop(0)
        _TE = text_e004
    return _TE


def wc(text):
    return len(re.findall(r"[A-Za-z0-9']+", text))


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def content(text, values=()):
    te = _te()
    low = {v.lower() for v in values}
    return {w for w in te.words(text) if w not in te.STOPWORDS and w not in te.OBJECT_FREE_VERBS and w not in low}


def run(text, question):
    tw, qw = words(text), words(question)
    best = 0
    for i in range(len(tw)):
        for j in range(len(qw)):
            k = 0
            while i + k < len(tw) and j + k < len(qw) and tw[i + k] == qw[j + k]:
                k += 1
            best = max(best, k)
    return best


def tailed(text, tails=TAILS):
    """the text plus one variant per neutral tail (the final period replaced)."""
    base = text.rstrip(".")
    return [text] + [base + t for t in tails]


def rank_ok(lens, gold, target):
    top, low = max(lens), min(lens)
    at_top = [j for j, x in enumerate(lens) if x == top]
    at_low = [j for j, x in enumerate(lens) if x == low]
    long_gold = len(at_top) == 1 and at_top[0] in gold
    short_gold = len(at_low) == 1 and at_low[0] in gold
    if target == "long":
        return long_gold and not any(j in gold for j in at_low)
    if target == "short":
        return short_gold and not any(j in gold for j in at_top)
    if target == "none":
        return not long_gold and not short_gold
    return len(at_top) == 1 and len(at_low) == 1 and at_top[0] not in gold and at_low[0] not in gold


def choose(rng, options, gold, target, question=None, values=(), cap=20000):
    """options[j]: candidate texts of statement j; gold: indices of gold statements; target: a length rank, or None
    for no length constraint. With a question the echo balance applies too. None if no combination fits."""
    gold = set(gold)
    size = 1
    for o in options:
        size *= len(o)
    if size > cap:
        raise ValueError(f"{size} combinations; give fewer options")
    combos = [c for c in itertools.product(*options) if target is None or rank_ok([wc(x) for x in c], gold, target)]
    if question is not None:
        qc = content(question, values)
        keys = {x: (len(content(x, values) & qc), run(x, question)) for o in options for x in o}
        ok = [c for c in combos if all(rank_ok([keys[x][m] for x in c], gold, "none") for m in (0, 1))]
        tie = [c for c in ok if all(len({keys[x][m] for x in c}) == 1 for m in (0, 1))]
        combos = tie or ok
    return list(rng.choice(combos)) if combos else None
