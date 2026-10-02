"""Writes the two read-only PC jobs used by A5 (broad kbig exposure + reference weight hashes) and the probe overlap
check, with the kbig items, each G seed's used replay thread ids and the probe's user turns embedded, into argv[1]
(a scratch directory; the generated scripts are about 400 KB each, so they are not kept here). Run each with
pcwsl.sh; their outputs are pc_exposure.out and pc_probe.out in this folder (home paths written as ~)."""
import json, os, sys
from common import *

code_path()
import kbig_items as K

out = sys.argv[1]
items = [(x["qid"], x["question"], x["cands"]["gold"].strip()) for x in K.build()]
used = {s: json.load(open(os.path.join(E6, "out", f"{PFX}G{s}__run.json")))["replay_threads"] for s in SEEDS}
convs = [json.loads(l) for l in open(os.path.join(E6, "transcripts", f"{PFX}C1__greedy.jsonl"))]
probe = {c["id"]: [t["user"] for t in c["turns"]] for c in convs}
POOL = 'os.path.expanduser("~/planck/e006_run/replay/replay_pool.jsonl")'

exposure = r'''#!/bin/bash
# Read-only: pool sha256, broad kbig exposure over the pool and each seed's used threads, reference weight hashes.
sha256sum ~/planck/e006_run/replay/replay_pool.jsonl
python3 - <<'PYEOF'
import json, re, os
ITEMS = ''' + json.dumps(items) + r'''
TH = ''' + json.dumps(used) + r'''
pool = {}
for l in open(''' + POOL + r'''):
    t = json.loads(l)
    pool[t["id"]] = "\n".join(x["text"] for x in t["turns"])
print("threads", len(pool))
def wb(w, s):
    return re.search(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", s, re.I) is not None
ent = lambda q: [w for w in re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", q) if w not in ("What", "Who", "Which", "In", "The", "How")]
for name, ids in [("pool", list(pool))] + [("s" + k, v) for k, v in TH.items()]:
    texts = [pool.get(i) for i in ids]
    miss = sum(t is None for t in texts)
    texts = [t for t in texts if t]
    gold_any = [q for q, que, g in ITEMS if any(wb(g, t) for t in texts)]
    co = [q for q, que, g in ITEMS if any(wb(g, t) and any(wb(e, t) for e in ent(que)) for t in texts)]
    print(name, "missing", miss, "gold_any", json.dumps(gold_any), "co", json.dumps(co))
PYEOF
for d in ~/planck/dev/e006_refs_e005/*/ ~/planck/dev/e006_refs_e004/*/; do echo "$(sha256sum $d/model.safetensors | cut -c1-16) ${d#$HOME/}"; done
'''
overlap = r'''#!/bin/bash
# Read-only: word 4-gram overlap between each chat-probe conversation's user turns and the replay threads G used.
python3 - <<'PYEOF'
import json, re, os
PROBE = ''' + json.dumps(probe) + r'''
USED = ''' + json.dumps(used) + r'''
pool = {}
for l in open(''' + POOL + r'''):
    t = json.loads(l)
    pool[t["id"]] = " ".join(x["text"] for x in t["turns"])
W = re.compile(r"[a-z0-9']+")
def g4(s):
    w = W.findall(s.lower())
    return {tuple(w[i:i + 4]) for i in range(len(w) - 3)}
allused = sorted({i for v in USED.values() for i in v})
PG = {i: g4(pool[i]) for i in allused}
for cid, turns in sorted(PROBE.items()):
    q = set().union(*(g4(t) for t in turns))
    if not q:
        print(cid, "no 4-grams"); continue
    best = max(allused, key=lambda i: len(q & PG[i]))
    seeds = [s for s, v in USED.items() if best in v]
    print(cid, len(q), "best share %.2f" % (len(q & PG[best]) / len(q)), best, "seeds", seeds, sorted(q & PG[best])[:4])
PYEOF
'''
for name, src in (("pc_exposure.sh", exposure), ("pc_probe_overlap.sh", overlap)):
    with open(os.path.join(out, name), "w") as f:
        f.write(src)
    print("wrote", os.path.join(out, name), len(src))
