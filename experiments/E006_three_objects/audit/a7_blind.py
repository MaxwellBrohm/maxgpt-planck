"""A7. The blinded chat sample: each of the 40 turns is checked against the keyed transcript turn, and classified
mechanically with A4's matcher (first sentence a training template / repeated-line loop / neither) before the key
is joined; then the counts per arm."""
import json
from collections import Counter
from common import *
from a4_qchat import tf, loop

R = os.path.join(E6, "logs", "readings", "20261002-181132__")
blind = json.load(open(R + "blind.json"))
cls = {x["k"]: ("loop" if loop(x["reply"]) else "template" if tf(x["reply"]) else "other") for x in blind}
print("blind classes (before the key):", Counter(cls.values()))
key = {x["k"]: x for x in json.load(open(R + "blind_key.json"))}
ok = 0
for x in blind:
    kk = key[x["k"]]
    tr = [json.loads(l) for l in open(os.path.join(E6, "transcripts", f"{PFX}{kk['arm']}{kk['seed']}__greedy.jsonl"))]
    conv = [c for c in tr if c["id"] == kk["conv"]][0]
    t = conv["turns"][kk["turn"]]
    ok += t["user"] == x["user"] and t["assistant"].strip() == x["reply"].strip()
print(f"turns matching the keyed transcript turn: {ok}/40")
print("by arm:", Counter((key[k]["arm"], c) for k, c in cls.items()))
print("arms x seeds:", Counter((v["arm"], v["seed"]) for v in key.values()))
