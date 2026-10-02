"""E003 audit A4: split and wording leaks (own code, stdlib; no tokenizer, no model).

  1. dev draw (seed 3003) vs eval items (seed 2026) vs in-training probe (seed 777): shared prompts, shared dialogues
  2. training streams Random(1000*seed+17), seeds 0-3, first 6,600 draws (a superset of the kept examples, which
     the training run filters only by length): exact prompt or dialogue equal to any eval/dev/probe item; eval key
     statements (value replaced by <v>) used as training user turns; eval questions used as training questions
usage: python3 -B a4_leaks.py
"""
import os, random, re, sys

AUD = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(AUD)
sys.path.insert(0, os.path.join(EXP, "code"))
import items as I            # noqa: E402
import items_new as N        # noqa: E402
import eval_extra as X       # noqa: E402
import train_data as TD      # noqa: E402
assert "torch" not in sys.modules

PV = ("same_k1", "same_k2", "same_k3", "twoslot", "noupd")
ev = N.build()
dev = [x for x in N.build(n_scen=32, distances=(10,), seed=3003) if x["var"] in PV]
probe = [x for x in N.build(n_scen=16, distances=(10,), seed=777) if x["var"] in ("same_k1", "twoslot", "noupd")]
cross = X.build_crossed()
VALS = sorted(set(N.DAYS) | set(N.COLORS), key=len, reverse=True)
VRE = re.compile(r"\b(" + "|".join(map(re.escape, VALS)) + r")\b", re.I)


def prompt(x):
    return I.transcript(x["turns"], x["question"], x["prefix"])


def dlg(x):
    return tuple(tuple(t) for t in x["turns"])


def tmpl(s):
    return VRE.sub("<v>", s)


sets = {"eval": ev, "dev": dev, "probe": probe, "cross": cross}
P = {k: {prompt(x) for x in v} for k, v in sets.items()}
D = {k: {dlg(x) for x in v} for k, v in sets.items()}
print(f"items: eval {len(ev)}, dev {len(dev)}, probe {len(probe)}, cross {len(cross)}")
for a, b in (("dev", "eval"), ("dev", "probe"), ("probe", "eval"), ("dev", "cross")):
    print(f"  {a} vs {b}: shared prompts {len(P[a] & P[b])}, shared dialogues {len(D[a] & D[b])}")
# eval key statements: every user turn that holds a value, as a template
key_t = set()
for x in ev + dev + cross:
    for u, a in x["turns"]:
        if VRE.search(u):
            key_t.add(tmpl(u))
eval_q = {x["question"] for x in ev + dev + cross}
print(f"  eval key-statement templates: {len(key_t)}; eval questions: {len(eval_q)}")
allP = set().union(*P.values())
allD = set().union(*D.values())
for seed in (0, 1, 2, 3):
    rng = random.Random(1000 * seed + 17)
    n, hitP, hitD, hitK, hitQ, ex_k = 6600, 0, 0, 0, 0, None
    for _ in range(n):
        ex = TD.gen(rng)
        p = TD.transcript(ex["turns"], ex["question"], ex["prefix"])
        hitP += p in allP
        hitD += tuple(tuple(t) for t in ex["turns"]) in allD
        k = [u for u, a in ex["turns"] if tmpl(u) in key_t]
        if k:
            hitK += 1
            ex_k = ex_k or k[0]
        hitQ += ex["question"] in eval_q
    print(f"  train seed {seed}: {n} draws; exact eval/dev/probe prompt {hitP}, dialogue {hitD}, "
          f"examples with an eval key-statement template {hitK}{' e.g. ' + repr(ex_k) if ex_k else ''}, eval question {hitQ}")
