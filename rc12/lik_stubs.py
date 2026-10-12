"""Stub tokenizers and stub scorers for the likelihood rows (tests and lik_run.py stub:NAME). No model, no torch.

Tokenizers (lossless: decode(encode(t)) == t; BOS id 1 when special; ids grow per instance):
  StubTok()             one token per word or punctuation mark, leading whitespace attached ("word"): every
                        single-word candidate has 1 token, so every dev row is equal.
  StubTok(chunk=4)      words cut into 4-character pieces ("chunk"): "Quizzards" 3 tokens, "blue" 1, so some rows
                        have unequal candidate token counts.
  StubTok(merge=3)      fixed 3-character pieces that ignore word boundaries ("merge"): prompt + candidate often does
                        not start with the prompt's own tokens, so cand_ids takes the "split" path;
                        canonical=True scores it the E004 canonical way instead.
  StubTok(eos=True)     also appends EOS id 2 when special (the trailing-eos case of E004 cand_ids).
Scorers (logprob(prefix_ids, continuation_ids) -> float):
  length(pre, cont)     -1 per token: a uniform model; prefers the shorter candidate and ties on equal rows.
  Recency(tok)          the end position of the candidate's LAST whole-word, any-case mention in the decoded
                        prefix (-1 if absent): the most recent candidate wins.
  Mentions(tok)         the number of whole-word, any-case mentions of the candidate in the decoded prefix: the
                        most-mentioned candidate wins (the copying heuristic; STEP 11 FIX ROUND: right on every OWN
                        row, whose gold alone is named twice on golden history).
  Ideal(tok, prompts)   prompts: [(prompt text, gold, candidates)]; for a prefix + continuation whose decoded text
                        is some prompt + " " + candidate, +n tokens for the gold, -(n + 1) otherwise. Any other text
                        raises KeyError, so a wrong history or lead-in cannot pass silently."""
import re

import render as RD

PIECE = re.compile(r"\s*(?:\w+|[^\w\s])|\s+")


class StubTok:
    BOS, EOS = 1, 2

    def __init__(self, chunk=None, merge=None, eos=False, canonical=False):
        self.chunk, self.merge, self.add_eos, self.canonical = chunk, merge, eos, canonical
        self.eos = self.EOS if eos else None
        self.vocab, self.inv = {}, {}

    def pieces(self, text):
        if self.merge:
            return [text[i:i + self.merge] for i in range(0, len(text), self.merge)]
        out = []
        for p in PIECE.findall(text):
            body = p.lstrip()
            if not self.chunk or not body:
                out.append(p)
                continue
            ws = p[:len(p) - len(body)]
            parts = [body[i:i + self.chunk] for i in range(0, len(body), self.chunk)]
            out += [ws + parts[0]] + parts[1:]
        return out

    def tid(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 3
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, text, special):
        ids = [self.tid(p) for p in self.pieces(text)]
        return ([self.BOS] if special else []) + ids + ([self.EOS] if special and self.add_eos else [])

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids if i > 2)

    def template(self, messages):
        return RD.template_stub(messages)


def make_tok(name):
    return {"word": StubTok(), "chunk": StubTok(chunk=4), "merge": StubTok(merge=3),
            "canonical": StubTok(merge=3, canonical=True), "eos": StubTok(eos=True)}[name]


def length(pre, cont):
    return -1.0 * len(cont)


class Recency:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, pre, cont):
        text, cand = self.tok.decode(pre), self.tok.decode(cont).strip()
        last = -1
        for m in re.finditer(r"\b%s\b" % re.escape(cand), text, re.I):
            last = m.end()
        return float(last)


class Mentions:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, pre, cont):
        text, cand = self.tok.decode(pre), self.tok.decode(cont).strip()
        return float(len(re.findall(r"\b%s\b" % re.escape(cand), text, re.I)))


class Ideal:
    def __init__(self, tok, prompts):
        self.tok, self.good, self.known = tok, set(), set()
        for text, gold, cands in prompts:
            self.good.add(text + " " + gold)
            self.known.update(text + " " + c for c in cands)

    def __call__(self, pre, cont):
        full = self.tok.decode(list(pre) + list(cont))
        if full not in self.known:
            raise KeyError(f"not a prompt + candidate the oracle was built for: ...{full[-80:]!r}")
        return float(len(cont)) if full in self.good else -float(len(cont)) - 1.0
