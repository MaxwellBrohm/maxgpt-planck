"""extra E004 purity lines on the kept examples (my code): eval markers, held-out objects, H6 values,
non-training filler turns; plus: the E004 block is train_e004.stream(s) in order."""
import pickle, re, sys
from collections import Counter
sys.dont_write_bytecode = True
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
import heldout_e004 as H
import train_e004 as T4
import train_e005 as T5
from pools_train import FILLERS_TRAIN
FT = {tuple(f) for f in FILLERS_TRAIN}
EM = [f.split("{s}")[0].strip().lower() for f in H.EVAL_MARKER_FMT.values()]
HO = H.all_heldout_objects()
HO_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(o.lower()) for o in HO) + r")s?(?![a-z])")
SP_RE = re.compile(r"(?<![a-z])(" + "|".join(H.SPORT) + r")(?![a-z])")
c = Counter()
for s in range(1, 6):
    D = pickle.load(open(f"kept_s{s}.pkl", "rb"))
    st = set()
    for i, ex in enumerate(D["kept"]):
        stmt_turns = {x["turn"] for x in ex["stmts"]}
        txt = "\n".join([u + "\n" + a for u, a in ex["turns"]] + [ex["question"], ex["answer"]]).lower()
        f = set()
        if any(u.lower().startswith(m) for u, _ in ex["turns"] for m in EM): f.add("eval marker")
        if HO_RE.search(txt): f.add("held-out object phrase")
        if SP_RE.search(txt): f.add("H6 sport value")
        if any(tuple(t) not in FT for j, t in enumerate(ex["turns"]) if j not in stmt_turns): f.add("non-training filler")
        for x in f: c[x] += 1
    # E004 block in order
    e4 = [x for x in T5.take(s, D["drawn"]) if x["block"] == "e004"]
    ref = T4.take(s, len(e4))
    same = all({k: v for k, v in a.items() if k != "block" and k != "render"} == b for a, b in zip(e4, ref))
    c[f"seed {s} E004 block == train_e004.stream prefix ({len(e4)})"] += same
print(dict(c))
