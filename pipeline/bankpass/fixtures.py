"""Test fixtures for the bank pass (no model; every text is a FAKE stand-in Claude wrote for tests). A fixture bank
directory is marked "fixture": true in its manifest, so admit refuses it unless a test passes allow_fixture: a fixture
can never be mistaken for a real bank. Its items claim teacher authorship only to exercise the admit checks.

  FakeBankTeacher   a client for gen.run: answers each call with canned lines that satisfy the call's contract
  bank_records      item records for one bank (judged, gated, author thirds), from given templates
  make_bank_dir     a frozen fixture directory with a manifest; tamper hooks for the planted cases"""
import os

import heldout
from bankpass import judge, specs, store

TEACHERS = tuple(sorted(store.TEACHERS))
# FAKE lines in the real banks' shape (the FAKE banks' own lines, re-worded so they differ from banks.py)
FIXTURE_LINES = {
    "key.pet_name.plant": ["Our {o} goes by {v}.", "The {o} we have is named {v}.", "I call my {o} {v}.",
                           "My little {o} answers to {v}.", "We have a {o} called {v}.", "My {o}'s name is {v}."],
    "key.pet_name.query": [("full", "Remind me, what is my {o} called?"), ("head", "What did we name the {o}?"),
                           ("full", "Do you know what I call my {o}?")],
    "open.greet": ["Hello again.", "Hey there!", "Morning, you.", "Hi hi.", "Evening!", "Hey, hello."],
    "close.goodbye": ["Okay, that is everything, bye.", "Right, I am off now. Bye!", "Cheers, talk soon.",
                      "Good, that covers it. Bye.", "Lovely, see you later.", "All done here, goodbye."],
    "marker.fix": ["Hmm, no,", "Let me fix that,", "Rather,", "Ah, I mean,"],
}


class FakeBankTeacher:
    """gen client: n lines (each holding its literal when the call needs one), then END. bad=True returns junk."""
    sampling = {"temperature": 0.8, "top_p": 0.95}

    def __init__(self, model, bad=False, wrong_model=False):
        self.model, self.bad, self.wrong_model, self.calls = model, bad, wrong_model, 0

    def generate(self, prompt, spec, seed):
        self.calls += 1
        if self.bad:
            return {"text": "here you go:\n1. something", "model": self.model}
        import re
        words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        nouns = dict(re.findall(r"^Line (\d+): .*?the ([a-z ]+)$", prompt, re.M))
        lines = []
        for i, lit in enumerate(spec["literals"]):
            noun = nouns.get(str(i + 1), "pet")
            line = f"My {noun} is called {lit}." if lit else f"Line number {words[i % 10]}."
            lines.append(line + (" | rare" if spec["verbalized"] else ""))
        return {"text": "\n".join(lines + ["END"]), "model": "someone-else" if self.wrong_model else self.model}


def class_of(bank):
    """specs.class_of, plus the value banks it does not name (topic T, topicwords W, intent I; else X)."""
    try:
        return specs.class_of(bank)
    except KeyError:
        return {"topic": "T", "topicwords": "W", "intent": "I"}.get(bank, "X")


def _judges(author):
    others = [t for t in TEACHERS if t != author][:2]
    return [{"model": m, "answers": {}, "expected": {}, "fill": {}, "verdict": "keep"} for m in others]


def bank_records(bank, lines, cls=None, author_kind="teacher", models=None):
    """records for one bank; teacher authors rotate (models, default all three) so each has an even share."""
    cls = cls or class_of(bank)
    models = models or TEACHERS
    out = []
    for i, ln in enumerate(lines):
        form, text = ln if isinstance(ln, tuple) else (None, ln)
        model = models[i % len(models)]
        author = store.teacher_author(model) if author_kind == "teacher" else {"kind": author_kind}
        out.append(store.make_item(cls, bank, i, text, author, features={"form": form},
                                   judges=_judges(model) if author_kind == "teacher" else [],
                                   gates={"gate_hash": heldout.gate_hash(), "rc12": heldout.RC12_STATUS,
                                          "hits": []}))
    return out


def make_bank_dir(path, banks=None, author_kind="teacher", fixture=True, status="frozen", tamper=None):
    """write a fixture bank dir; tamper(bank, records) may change records before the files are written, and
    tamper_after(path) (via tamper={"after": fn}) may change files after the manifest."""
    os.makedirs(path, exist_ok=True)
    banks = banks or FIXTURE_LINES
    meta = {}
    for bank, lines in banks.items():
        recs = bank_records(bank, lines, author_kind=author_kind)
        if tamper and tamper.get("records"):
            recs = tamper["records"](bank, recs)
        store.write_jsonl(os.path.join(path, bank + ".jsonl"), recs)
        meta[bank] = {"class": class_of(bank), "target": 30}
    man = store.build_manifest(path, "fixture-v0", heldout.gate_hash(), heldout.RC12_STATUS, meta, status=status,
                               fixture=fixture)
    if tamper and tamper.get("after"):
        tamper["after"](path)
    return man


def judged(rec, verdicts=("keep", "keep")):
    """give rec two non-author judge records with the given verdicts."""
    others = [t for t in TEACHERS if t != rec["author"].get("model")][:2]
    rec["judges"] = [{"model": m, "verdict": v} for m, v in zip(others, verdicts)]
    return judge.vote(rec)
