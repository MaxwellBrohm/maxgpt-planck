"""E004 cheap oracles O1-O8 and IDEAL (notes "CHEAP ORACLES"). Each picks one in-context value with no model,
from the TEXT only: the turns, the question, the forced answer prefix, the item's value pool and object names
(phrase and head noun, which a model sees in the text). None reads the builder's annotation except IDEAL.
A statement = a user turn that mentions a value of the item's value type. Mentions = every value occurrence in
user and assistant turns, in order. Works on eval items (items_e004.py) and on training examples adapted by
from_train() (the prefix of a training example is the answer's words before the value)."""
import re

from text_e004 import words, content_words, value_re, normalize, echo_runs
from heldout_e004 import EVAL_MARKER_FMT

TRAIN_MARKER_STARTS = ["actually,", "wait,", "change of plans:", "sorry, i meant:", "update:", "oh,", "never mind,"]
MARKER_STARTS = TRAIN_MARKER_STARTS + [f.split("{s}")[0].strip().lower() for f in EVAL_MARKER_FMT.values()]
_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def strip_marker(text):
    t = text.lower()
    for m in sorted(MARKER_STARTS, key=len, reverse=True):
        if t.startswith(m):
            return text[len(m):].strip()
    return text


def has_marker(text):
    return strip_marker(text) != text


def names(text, term):
    return re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"s?(?![a-z])", text.lower()) is not None


class View:
    """text-level view of one item: statements [(turn index, text, value)], mentions [value], tokens."""

    def __init__(self, it):
        self.it = it
        self.pool = it["values"]
        self.turns = it["turns"]
        self.q, self.p = it["question"], it.get("prefix", "")
        self.asked = it["objects"][it["asked"]]
        self.others = [o for i, o in enumerate(it["objects"]) if i != it["asked"]]
        self.stmts, self.mentions = [], []
        for ti, (u, a) in enumerate(self.turns):
            vu = self.values_in_order(u)
            if vu:
                self.stmts.append((ti, u, vu[0]))
            self.mentions += vu + self.values_in_order(a)
        self.tokens = []                       # (lowercased word, value or None) over the whole dialogue
        for u, a in self.turns:
            for text in (u, a):
                for m in _TOK.finditer(text):
                    w = m.group(0)
                    val = next((v for v in self.pool if (w == v if v[:1].isupper() else w.lower() == v)), None)
                    self.tokens.append((w.lower(), val))

    def values_in_order(self, text):
        hits = []
        for v in self.pool:
            hits += [(m.start(), v) for m in value_re(v).finditer(text)]
        return [v for _, v in sorted(hits)]

    def names_asked(self, text, full_only=False):
        ph, h = self.asked
        return names(text, ph) or (not full_only and names(text, h))

    def names_other(self, text):
        return any(names(text, ph) or names(text, h) for ph, h in self.others)

    def content(self, text):
        return set(content_words(strip_marker(text), extra_exclude=[v.lower() for v in self.pool]))

    def latest(self, pred, fallback):
        for ti, text, v in reversed(self.stmts):
            if pred(ti, text, v):
                return v
        return fallback

    def best_latest(self, score):
        """value of the statement with the highest score; ties go to the latest."""
        best = max(score(text) for _, text, _ in self.stmts)
        return self.latest(lambda ti, t, v: score(t) == best, self.mentions[-1])

    def terms(self):
        return [t for ph, h in self.it["objects"] for t in (ph, h)] + ([self.it["alias"]] if self.it.get("alias")
                                                                        else [])

    def echoes(self, text):
        n = lambda s: normalize(s, self.terms(), self.pool)
        return bool(echo_runs(n(self.q), n(text)) or (self.p and echo_runs(n(self.p), n(text))))


def o1_first(V):
    return V.mentions[0]


def o2_last(V):
    return V.mentions[-1]


def o3_marker(V):
    return V.latest(lambda ti, t, v: has_marker(t), o1_first(V))


def o4_prefix_copy(V):
    pw = words(V.p)
    toks = [w for w, _ in V.tokens]
    for n in (3, 2):
        if len(pw) < n:
            continue
        pat = pw[-n:]
        occ = [i for i in range(len(toks) - n + 1) if toks[i:i + n] == pat]
        if occ:
            after = [val for _, val in V.tokens[occ[-1] + n:] if val is not None]
            if after:
                return after[0]
    return o2_last(V)


def o5_content(V):
    qs = V.content(V.q)
    return V.best_latest(lambda t: len(V.content(t) & qs))


def o5_content_qp(V):
    qs = V.content(V.q) | V.content(V.p)
    return V.best_latest(lambda t: len(V.content(t) & qs))


def o5_allwords(V):
    qs = set(words(V.q))
    return V.best_latest(lambda t: len(set(words(strip_marker(t))) & qs))


def o5_runs(V):
    """most shared word 3-grams with the question and prefix (raw words), latest wins."""
    g = lambda s: {tuple(words(s)[i:i + 3]) for i in range(len(words(s)) - 2)}
    qs = g(V.q) | g(V.p)
    return V.best_latest(lambda t: len(g(strip_marker(t)) & qs))


def o6_names_asked(V):
    return V.latest(lambda ti, t, v: V.names_asked(t), o1_first(V))


def o7_no_other(V):
    return V.latest(lambda ti, t, v: not V.names_other(t), o1_first(V))


def o8_frequent(V):
    counts = {}
    for v in V.mentions:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    last_pos = {v: i for i, v in enumerate(V.mentions)}
    return max((v for v in counts if counts[v] == best), key=lambda v: last_pos[v])


def ideal(V):
    return [s for s in V.it["stmts"] if s["obj"] == V.it["asked"]][-1]["value"]


ORACLES = [("O1 first mention", o1_first), ("O2 last mention", o2_last), ("O3 last marked, else first", o3_marker),
           ("O4 prefix copy (3/2 words)", o4_prefix_copy), ("O5 content words vs question", o5_content),
           ("O5 content words vs q+prefix", o5_content_qp), ("O5 all words vs question", o5_allwords),
           ("O5 3-word runs vs q+prefix", o5_runs), ("O6 latest naming asked", o6_names_asked),
           ("O7 latest naming no other", o7_no_other), ("O8 most frequent", o8_frequent)]


def from_train(ex):
    """a training example (train_e004.gen) as an oracle-ready item: prefix = the answer's words before the
    value (notes O4); incidental statements have obj None and are never the asked object."""
    from pools_train import POOLS
    return dict(values=POOLS[ex["vtype"]]["values"], turns=ex["turns"], question=ex["question"],
                prefix=ex["answer_pre"], objects=ex["objects"], asked=ex["asked"], stmts=ex["stmts"], alias=None)
