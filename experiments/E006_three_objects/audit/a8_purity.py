"""A8. Split leak check, pure Python, no tokenizer: word 5-grams shared between BIG + H5L item text (the Q-H5 test
items) and the first 6,600 DRAWN training examples per seed of C (train_e005.stream) and P (train_e006p.stream_p).
Drawn is a superset of kept (kept = the first 6,404 drawn that fit 768 tokens; the most rejected was 129), so 0 here
means 0 in what was trained. G's update part is C's stream (digests equal, A1)."""
import json, itertools, re
from common import *

code_path()
import train_e005, train_e006p

W = re.compile(r"[a-z0-9']+")


def grams(text, n=5):
    w = W.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


BIG = [json.loads(l) for l in open(os.path.join(E6, "big", "big_items.jsonl"))]
EV = set()
for it in BIG:
    for u, a in it["turns"]:
        EV |= grams(u) | grams(a)
    EV |= grams(it["question"]) | grams(it["prefix"] or "")
# Run first with 5-grams over the joined example text: 5-17 hits per seed, all spanning a field boundary (training
# "...end up on? / Friday is the day to keep free." vs BIG "Hold on, Friday is the day of ..."). Per field: below.
print("BIG+H5L 5-grams:", len(EV))
for arm, src in (("C", train_e005.stream), ("P", train_e006p.stream_p)):
    for s in SEEDS:
        hits, ex = 0, []
        for x in itertools.islice(src(s), 6600):
            parts = [z for u, a in x["turns"] for z in (u, a)] + [x["question"], x["answer"]]
            sh = set().union(*(grams(z) for z in parts)) & EV   # per text field: no 5-gram spans two fields
            if sh:
                hits += 1
                ex.append(sorted(sh)[:2])
        print(f"{arm} s{s}: drawn 6600, examples sharing a 5-gram with BIG/H5L: {hits}", ex[:3])
