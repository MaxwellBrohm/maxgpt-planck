"""E004 held-out purity against the training generator (step 3; notes (b) "each family's held-out axis value
occurs in 0 training examples of every seed's stream"). Samples 20,000 training dialogues (seeds 1-5, the first
4,000 of each stream; seed 0's stream is covered by test_train_gen.py) and checks, per example, that no held-out
axis value occurs:
  H1/H2  0 frame echoes (question or answer vs any statement or acknowledgement); no alias (honorific or alias
         surname); no eval marker
  H3     no object corrected 4 or more times (k45)
  H4     d <= 10 (d20)
  H5     at most 2 objects (obj3)
  H6     no held-out object phrase (E001's, the eval objects of the training types, H6's), no sport value, no digit
  H7     every filler turn is a training how-to filler; no H7 filler shares a word 5-gram with any training text
and reports the ranges the training stream actually covers. n-gram overlap lives in ngram_overlap.py.
Usage: python3 -B purity_e004.py > ../logs/purity_e004.txt"""
import re
import sys
from collections import Counter

import heldout_e004 as H
import train_e004 as T
from pools_train import FILLERS_TRAIN
from pools_eval import H7_FILLERS
from text_e004 import normalize, echo_runs, words, values_in

SEEDS, PER_SEED = (1, 2, 3, 4, 5), 4000


def sample():
    out = []
    for s in SEEDS:
        out += T.take(s, PER_SEED)
    return out


def _phrase(text, ph):
    return re.search(r"(?<![a-z])" + re.escape(ph.lower()) + r"s?(?![a-z])", text.lower()) is not None


def violations(ex):
    """{axis: [problem]} for one training example."""
    v = {}
    add = lambda axis, msg: v.setdefault(axis, []).append(msg)
    text = T.prompt(ex) + ex["answer"]
    terms = [t for ph, h in ex["objects"] for t in (ph, h)]
    from pools_train import POOLS
    pool = POOLS[ex["vtype"]]["values"]
    q, a = normalize(ex["question"], terms, pool), normalize(ex["answer"], terms, pool)
    for s in ex["stmts"]:
        for side in ex["turns"][s["turn"]]:
            n = normalize(side, terms, pool)
            if echo_runs(q, n) or echo_runs(a, n):
                add("H1/H2 echo", side)
    if any(h in text for h in H.HONORIFICS) or any(x in text for x in H.ALIAS_SURNAMES):
        add("H1/H2 alias", "honorific or alias surname")
    for m in H.EVAL_MARKERS:
        if re.search(r"(?<![a-z])" + re.escape(m.rstrip(",:")) + r"(?![a-z])", text.lower()):
            add("eval marker", m)
    per_obj = Counter(s["obj"] for s in ex["stmts"] if s["role"] in ("corr", "rev") and s["obj"] is not None)
    if per_obj and max(per_obj.values()) >= 4:
        add("H3 k45", str(dict(per_obj)))
    d_text = len(ex["turns"]) - 1 - max(s["turn"] for s in ex["stmts"])     # from the text, not the annotation
    if ex["d"] > 10 or d_text > 10:
        add("H4 d20", f"{ex['d']} / {d_text}")
    if ex["n_obj"] > 2 or len(ex["objects"]) > 2 or len({s["obj"] for s in ex["stmts"] if s["obj"] is not None}) > 2:
        add("H5 obj3", str(ex["n_obj"]))
    for ph in H.all_heldout_objects():
        if _phrase(text, ph):
            add("H6 object", ph)
    if values_in(text, H.SPORT) or re.search(r"\d", text):
        add("H6 value", "sport or digit")
    st = {s["turn"] for s in ex["stmts"]}
    for i, t in enumerate(ex["turns"]):
        if i not in st and t not in FILLERS_TRAIN:
            add("H7 filler", t[0])
    return v


def ranges(exs):
    out = {}
    out["k (asked)"] = Counter(ex["k"] for ex in exs)
    out["max corrections of any object"] = Counter(
        max(Counter(s["obj"] for s in ex["stmts"] if s["role"] in ("corr", "rev")).values() or [0]) for ex in exs)
    out["d"] = Counter(ex["d"] for ex in exs)
    out["objects"] = Counter(ex["n_obj"] for ex in exs)
    out["value statements"] = Counter(len(ex["stmts"]) for ex in exs)
    return out


def h7_overlap(exs):
    g = lambda t, n=5: {tuple(words(t)[i:i + n]) for i in range(len(words(t)) - n + 1)}
    train = set()
    for ex in exs:
        for u, a in ex["turns"]:
            train |= g(u) | g(a)
        train |= g(ex["question"]) | g(ex["answer"])
    return sorted(set().union(*(g(u) | g(a) for u, a in H7_FILLERS)) & train)


def main():
    exs = sample()
    print(f"E004 purity: {len(exs)} training dialogues (seeds {SEEDS}, first {PER_SEED} of each stream)\n")
    counts, examples = Counter(), {}
    for ex in exs:
        for axis, msgs in violations(ex).items():
            counts[axis] += 1
            examples.setdefault(axis, msgs[0])
    axes = ["H1/H2 echo", "H1/H2 alias", "eval marker", "H3 k45", "H4 d20", "H5 obj3", "H6 object", "H6 value",
            "H7 filler"]
    fail = 0
    for axis in axes:
        n = counts.get(axis, 0)
        fail += n > 0
        print(f"{'PASS' if n == 0 else 'FAIL'}  {axis:12s} training examples with it: {n}"
              + (f"  e.g. {examples[axis]!r}" if n else ""))
    ov = h7_overlap(exs)
    fail += bool(ov)
    print(f"{'PASS' if not ov else 'FAIL'}  H7 genre     H7 filler 5-grams found in training text: {len(ov)} {ov[:5]}")
    print("\nTraining ranges actually covered (held-out values must lie outside):")
    for name, c in ranges(exs).items():
        print(f"  {name:32s} {dict(sorted(c.items()))}")
    print("  held out: k 4-5 (H3), d 20 (H4), 3 objects (H5), value types sport/number and 48 H6 objects (H6),"
          " small-talk/creative fillers (H7), frame echoes and aliases (H1/H2)")
    print("\nRESULT: " + ("ALL PASS" if not fail else f"{fail} FAILED"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
