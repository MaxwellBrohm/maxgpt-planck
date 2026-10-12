"""RC-12 generator core: the conversation builder, placement helpers and the record format (SPEC s1).

Record (one JSON line): id, split, family, cell, seed, knowledge, n_turns, turns [{i, kind, text, ideal, facts,
vals, asks (D turns only)}], probes [{turn, kind, grader, question, gold, gold_fn, candidates, pool, stale, holder, object_words,
pair_id, d, dc, src, prefix, ideal, ...}], meta {...}.
  kind   S L C I O Q T D P X (SPEC s1 turn kinds). ideal = the IDEAL reply for that turn (never shown to a model).
  facts  statement annotations [{holder, object, value, role}] (role: gold, lure, stale, orig, corr, ...).
  vals   every RC-12 pool value in the user text, in order (strict-case whole words).
  asks   D turns only (F2, decided 2026-09-25): False when the filler is a plain statement that asks nothing (its
         hand label in pools_fill_a / _b is "statement"), else True. The loop rule's equality clause reads it
         (grade_loop.turn_kinds); for every other purpose the turn stays kind D. Non-D turns carry no asks field.
  src    user turns that carry the probe's gold (for the L2/G8 checks); d = probe turn - latest src turn."""
import random
import re

import pools_vals as V
from pools_fill_a import FILLERS_A
from pools_fill_b import FILLERS_B

FILLERS = FILLERS_A + FILLERS_B          # (user turn, IDEAL reply, label) entries
ASKING = {"question": True, "request": True, "statement": False}     # filler label -> the D turn's asks field (F2)
SEED = 1212
N_TURNS = 12
REPLY_WORDS = 50          # SPEC s1 budget: every reply counted as 50 words
TOKENS_PER_WORD = 1.35
TOKENS_PER_MSG = 4
MAX_TOKENS = 1800


def stream(family, seed=SEED):
    return random.Random(f"RC12:dev:{family}:{seed}")


def nwords(text):
    return len(text.split())


def token_proxy(user_texts):
    n = len(user_texts)
    words = sum(nwords(t) for t in user_texts) + REPLY_WORDS * n
    return TOKENS_PER_WORD * words + TOKENS_PER_MSG * 2 * n


def spread(values, n, rng=None):
    """n items cycling through values (exact balance when n is a multiple), shuffled when rng is given."""
    out = [values[i % len(values)] for i in range(n)]
    if rng is not None:
        rng.shuffle(out)
    return out


def probe_turns(rng, n):
    """main-probe turn per conversation: u12 in 3/4, u9-u11 in 1/4 (cycled), shuffled (SPEC s1)."""
    late = n // 4
    out = [12] * (n - late) + [9 + (i % 3) for i in range(late)]
    rng.shuffle(out)
    return out


def place_blocks(rng, lengths, lo, hi, ok=None, tries=4000):
    """start turns for consecutive blocks (kept in order, no overlap) inside [lo, hi]; ok(starts) filters."""
    m, total = len(lengths), sum(lengths)
    slack = (hi - lo + 1) - total
    if slack < 0:
        raise ValueError("blocks do not fit")
    for _ in range(tries):
        cuts = sorted(rng.sample(range(slack + m), m))
        starts, used = [], 0
        for j, p in enumerate(cuts):
            starts.append(lo + (p - j) + used)
            used += lengths[j]
        if ok is None or ok(starts):
            return starts
    raise ValueError("no placement satisfies the constraint")


class Conv:
    """one scripted conversation under construction."""

    def __init__(self, family, cell, rid, rng, n_turns=N_TURNS, knowledge=False):
        self.family, self.cell, self.rid, self.rng = family, cell, rid, rng
        self.n = n_turns
        self.knowledge = knowledge
        self.turns = {}
        self.probes = []
        self.meta = {}
        self.avoid = set()            # words a filler must not contain (object words of this conversation)
        self._acks = list(V.ACKS)
        rng.shuffle(self._acks)

    def ack(self):
        return self._acks.pop()

    def free(self):
        return [i for i in range(1, self.n + 1) if i not in self.turns]

    def put(self, i, kind, text, ideal=None, facts=None, asks=None):
        text = articles(text)
        ideal = articles(ideal) if ideal is not None else None
        if i in self.turns:
            raise ValueError(f"{self.rid}: turn {i} used twice")
        if not 1 <= i <= self.n:
            raise ValueError(f"{self.rid}: turn {i} out of range")
        self.turns[i] = dict(i=i, kind=kind, text=text, ideal=ideal if ideal is not None else self.ack(),
                             facts=facts or [])
        if (kind == "D") != (asks is not None):
            raise ValueError(f"{self.rid}: turn {i}: asks is set on D turns only, and on every one")
        if asks is not None:
            self.turns[i]["asks"] = asks

    def add_probe(self, turn, kind, grader, question, ideal, **kw):
        self.put(turn, kind, question, ideal=ideal)
        p = dict(turn=turn, kind=kind, grader=grader, question=question, gold=None, gold_fn=None, candidates=[],
                 pool=None, stale=[], holder=None, object_words=[], pair_id=None, d=None, dc=None, src=[],
                 prefix=None, ideal=ideal)
        p.update(kw)
        if p["src"] and p["d"] is None:
            p["d"] = turn - max(p["src"])
        self.probes.append(p)
        return p

    def filler_pool(self):
        bad = {w.lower() for w in self.avoid}
        out = []
        for q, a, label in FILLERS:
            low = (q + " " + a).lower()
            if not any(_has_word(low, w) for w in bad):
                out.append((q, a, label))
        return out

    def fill(self, transform=None):
        pool = self.filler_pool()
        self.rng.shuffle(pool)
        for i in self.free():
            q, a, label = pool.pop()
            self.put(i, "D", q, ideal=transform(a) if transform else a, asks=ASKING[label])

    def record(self, seed=SEED):
        assert not self.free(), f"{self.rid}: unfilled turns {self.free()}"
        turns = []
        for i in range(1, self.n + 1):
            t = dict(self.turns[i])
            t["vals"] = V.scan(t["text"], V.ALL_VALUES)
            turns.append(t)
        return dict(id=self.rid, split="dev", family=self.family, cell=self.cell, seed=seed,
                    knowledge=self.knowledge, n_turns=self.n, turns=turns,
                    probes=sorted(self.probes, key=lambda p: p["turn"]), meta=self.meta)


_AN = re.compile(r"\b([Aa]) (?=[aeioAEIO])(?![Oo]n(?:e|ce)\b|[Oo]ne-)")


def articles(text):
    """a -> an before a vowel-initial word (not u: "a ukulele"); templates put values after "a"."""
    return _AN.sub(lambda m: m.group(1) + "n ", text)


def _has_word(low_text, w):
    return re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low_text) is not None


def pick(rng, seq, k=None, exclude=()):
    cand = [x for x in seq if x not in exclude]
    return rng.choice(cand) if k is None else rng.sample(cand, k)


def rid(family, n, suffix=""):
    return f"rc12-dev-{family.lower()}-{n:03d}{suffix}"
