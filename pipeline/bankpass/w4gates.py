"""W4 gates (BANKPASS s0, s2d; the W4 task: held-out E004 / RC-12 dev / OOD-H, safety, 6-word example copies). They run
on every item W3 kept, before dedup and the trim; a hit drops the item with its code (the W3 item checks and the E004
gate already ran at itemize time and are re-checked by admit).

  RC12DEV_ECHO  a word 8-gram shared with an RC-12 dev text (rc12/dev/rc12_dev.jsonl: user turns, ideal replies, probe
                questions, prefixes with golds, probe ideals). Dev is never training text (rc12/SPEC s0, whose own
                rule is 13-grams on whole chats); a bank line is reused across many chats, so it gets the stricter
                8-gram. 13-gram hits are counted too (n13).
  OODH_ECHO     a word 8-gram shared with a user turn of a reserved OASST2 tree (corpus/oodh.in_oodh_reserve, imported,
                never re-implemented), the pool OOD-H Part 1 draws from.
  SAFETY_LIST   an LDNOOBW term (human_v0 rubric.safety; human.unsafe_re), the real list beside the checker's FAKE one.
  MINED_COPY    a word 6-gram shared with ANY mined example sentence (mined_v0), not only the one its prompt showed
                (D8 as designed: a bank line copying a 6-word run from an example is dropped).
Line templates are checked on their hole-free runs and on N_FILLS seeded fills (n-grams form across hole edges);
values (pools, topics, labels, word set words) on their own text."""
import gzip
import json
import os
import re
import sys

import banks as B
import pools as P
from bankpass import gates, specs, store

N_ECHO, N_MINED, N_DEC13, N_FILLS = 8, 6, 13, 5
_HOLE = re.compile(r"\{\w+\}")


def words(s):
    return gates.words(store.straight(s))


def grams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


class Ngrams:
    """word n-grams of a text collection; hits(text) -> the first shared n-gram, or None."""

    def __init__(self, texts, n):
        self.n, self.set, self.n_texts = n, set(), 0
        for t in texts:
            self.set |= grams(words(t), n)
            self.n_texts += 1

    def hit(self, text):
        """the first shared n-gram in text order (never set order: a record must hash the same in every run)."""
        ws = words(text)
        for i in range(len(ws) - self.n + 1):
            if tuple(ws[i:i + self.n]) in self.set:
                return " ".join(ws[i:i + self.n])
        return None


def rc12_dev_texts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for t in r.get("turns") or []:
                out += [t.get("text") or "", t.get("ideal") or ""]
            for p in r.get("probes") or []:
                out += [p.get("question") or "", (p.get("prefix") or "") + " " + str(p.get("gold") or ""),
                        p.get("ideal") or ""]
    return [t for t in out if t.strip()]


def oodh_reserved_turns(trees_gz):
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.join(here, "corpus") not in sys.path:
        sys.path.insert(0, os.path.join(here, "corpus"))
    import oodh
    out, n_trees = [], 0
    with gzip.open(os.path.expanduser(trees_gz), "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if oodh.in_oodh_reserve(t["message_tree_id"]):
                n_trees += 1
                out += list(oodh._user_turns(t["prompt"]))
    return out, n_trees


def mined_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln)["text"] for ln in f if ln.strip()]


def safety_terms(rubric_path):
    return [r["text"] for r in store.read_jsonl(rubric_path, tolerate_torn_tail=False) if r["status"] == "kept"]


class Gates:
    """the four W4 gates; any part may be None (then that gate is reported as not run)."""

    def __init__(self, rc12=None, oodh=None, mined=None, unsafe=None, rc12_13=None):
        self.rc12, self.oodh, self.mined, self.unsafe, self.rc12_13 = rc12, oodh, mined, unsafe, rc12_13

    @classmethod
    def from_paths(cls, rc12_dev, trees_gz, mined, rubric):
        from bankpass import human
        dev = rc12_dev_texts(rc12_dev)
        res, _ = oodh_reserved_turns(trees_gz)
        return cls(Ngrams(dev, N_ECHO), Ngrams(res, N_ECHO), Ngrams(mined_texts(mined), N_MINED),
                   human.unsafe_re(safety_terms(rubric)), Ngrams(dev, N_DEC13))

    def describe(self):
        return {"rc12_dev": {"n": N_ECHO, "texts": self.rc12.n_texts if self.rc12 else None},
                "oodh_reserved_user_turns": {"n": N_ECHO, "texts": self.oodh.n_texts if self.oodh else None},
                "mined_examples": {"n": N_MINED, "texts": self.mined.n_texts if self.mined else None},
                "safety_terms": bool(self.unsafe)}

    def text_hits(self, text):
        out = []
        for code, idx in (("RC12DEV_ECHO", self.rc12), ("OODH_ECHO", self.oodh), ("MINED_COPY", self.mined)):
            h = idx.hit(text) if idx else None
            if h:
                out.append((code, h))
        if self.unsafe:
            m = self.unsafe.search(text)
            if m:
                out.append(("SAFETY_LIST", m.group(0)))
        return out

    def n13(self, text):
        return bool(self.rc12_13 and self.rc12_13.hit(text))

    def template_hits(self, template, bank, seed="w4"):
        """hits on the hole-free runs, then on N_FILLS seeded fills; the first fill hit is enough."""
        out = []
        for run in _HOLE.split(template):
            out += [h for h in self.text_hits(run) if h not in out]
        spec = specs.line_specs().get(bank)
        if spec is None or not _HOLE.search(template):
            return out
        rng = P.seeded("bankpass-w4gate", seed, bank, template)
        for _ in range(N_FILLS):
            f = gates.sample_fill(bank, spec, rng)
            txt = B.fill(template, **f)
            new = [h for h in self.text_hits(txt) if h[0] not in {c for c, _ in out}]
            if new:
                out += [(c, "fill " + d) for c, d in new]
                break
        return out


def apply(it, hits):
    """mark a kept item dropped by its first W4 hit; the hits go into its gates block."""
    if not hits:
        return it
    g = dict(it.get("gates") or {})
    g["w4"] = [list(h) for h in hits]
    it.update(status="dropped", drop=hits[0][0], gates=g)
    return it
