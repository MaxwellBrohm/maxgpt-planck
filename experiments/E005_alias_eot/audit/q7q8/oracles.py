"""Auditor's own cheap oracles O1-O8 (+ O5 3-gram-run variant) and IDEAL on the rebuilt KEPT examples, per seed
and block, plus the drawn-vs-kept share shift. Definitions from E004 notes.txt CHEAP ORACLES; the code is mine.
Only the stopword / verb word lists (pre-registered definitions) and marker strings are imported."""
import pickle
import re
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
from pools_train import POOLS, MARKER_FMT  # noqa: E402
from heldout_e004 import EVAL_MARKER_FMT  # noqa: E402
from text_e004 import STOPWORDS, OBJECT_FREE_VERBS  # noqa: E402

W = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
MARKS = sorted({f.split("{s}")[0].strip().lower() for f in list(MARKER_FMT.values()) + list(EVAL_MARKER_FMT.values())},
               key=len, reverse=True)


def vre(v):
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", 0 if v[:1].isupper() else re.I)


def vals_in(text, pool):
    hits = []
    for v in pool:
        hits += [(m.start(), v) for m in vre(v).finditer(text)]
    return [v for _, v in sorted(hits)]


def strip_m(t):
    tl = t.lower()
    for m in MARKS:
        if tl.startswith(m):
            return t[len(m):].strip()
    return t


def named(text, term):
    return re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"s?(?![a-z])", text.lower()) is not None


def oracles(ex):
    pool = POOLS[ex["vtype"]]["values"]
    stmts, mentions, tokseq = [], [], []
    for u, a in ex["turns"]:
        vu = vals_in(u, pool)
        if vu:
            stmts.append((u, vu[0]))
        mentions += vu + vals_in(a, pool)
        for t in (u, a):
            for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", t):
                w = m.group(0)
                val = next((v for v in pool if (w == v if v[:1].isupper() else w.lower() == v)), None)
                tokseq.append((w.lower(), val))
    first, last = mentions[0], mentions[-1]
    latest = lambda pred, fb: next((v for t, v in reversed(stmts) if pred(t)), fb)
    asked = ex["objects"][ex["asked"]]
    others = [o for i, o in enumerate(ex["objects"]) if i != ex["asked"]]
    lowpool = {v.lower() for v in pool}
    content = lambda t: {w for w in W.findall(strip_m(t).lower())
                         if w not in STOPWORDS and w not in OBJECT_FREE_VERBS and w not in lowpool}

    def best(score):
        b = max(score(t) for t, _ in stmts)
        return latest(lambda t: score(t) == b, last)

    out = {"O1": first, "O2": last,
           "O3": latest(lambda t: strip_m(t) != t, first)}
    pw = W.findall(ex["answer_pre"].lower())
    toks = [w for w, _ in tokseq]
    o4 = None
    for n in (3, 2):
        if len(pw) < n:
            continue
        pat = pw[-n:]
        occ = [i for i in range(len(toks) - n + 1) if toks[i:i + n] == pat]
        if occ:
            after = [v for _, v in tokseq[occ[-1] + n:] if v is not None]
            if after:
                o4 = after[0]
                break
    out["O4"] = o4 if o4 is not None else last
    qc = content(ex["question"])
    out["O5"] = best(lambda t: len(content(t) & qc))
    g3 = lambda s: {tuple(W.findall(s.lower())[i:i + 3]) for i in range(len(W.findall(s.lower())) - 2)}
    qg = g3(ex["question"]) | g3(ex["answer_pre"])
    out["O5run"] = best(lambda t: len(g3(strip_m(t)) & qg))
    out["O6"] = latest(lambda t: named(t, asked[0]) or named(t, asked[1]), first)
    out["O7"] = latest(lambda t: not any(named(t, ph) or named(t, h) for ph, h in others), first)
    cnt = Counter(mentions)
    top = max(cnt.values())
    lastpos = {v: i for i, v in enumerate(mentions)}
    out["O8"] = max((v for v in cnt if cnt[v] == top), key=lambda v: lastpos[v])
    about = [s for s in ex["stmts"] if s["obj"] == ex["asked"]]
    out["IDEAL"] = about[-1]["value"]
    # text-level sanity for IDEAL: the gold is in the answer and was stated in a user turn
    out["_gold_in_answer"] = bool(vre(ex["gold"]).search(ex["answer"]))
    out["_gold_stated"] = any(v == ex["gold"] for _, v in stmts) or any(
        vre(ex["gold"]).search(u) for u, _ in ex["turns"])
    return out


RULES = ["O1", "O2", "O3", "O4", "O5", "O5run", "O6", "O7", "O8", "IDEAL"]
KEYS = ("kind", "block", "render", "case", "placement")
allgood = True
pooled = defaultdict(lambda: Counter())
pooled_n = Counter()
for seed in range(1, 6):
    D = pickle.load(open(f"kept_s{seed}.pkl", "rb"))
    kept = D["kept"]
    right, per_b, nb = Counter(), defaultdict(Counter), Counter()
    sanity = Counter()
    for ex in kept:
        o = oracles(ex)
        nb[ex["block"]] += 1
        for r in RULES:
            ok = o[r] == ex["gold"]
            right[r] += ok
            per_b[ex["block"]][r] += ok
            pooled[ex["block"]][r] += ok
        pooled_n[ex["block"]] += 1
        sanity["gold_in_answer"] += o["_gold_in_answer"]
        sanity["gold_stated"] += o["_gold_stated"]
    n = len(kept)
    sh = {r: right[r] / n for r in RULES}
    fake = [r for r in RULES if r != "IDEAL"]
    mx = max(fake, key=lambda r: sh[r])
    gate = all(sh[r] <= 0.70 for r in fake) and sh["O1"] <= 0.5 and sh["O2"] <= 0.5 and sh["IDEAL"] == 1.0
    allgood &= gate
    # drawn vs kept shift, drawn = every example drawn up to the last kept one (rejected included)
    worst = (None, None, 0.0)
    for k in KEYS:
        nd, nk = sum(D["cd"][k].values()), sum(D["ck"][k].values())
        for v in D["cd"][k]:
            d = abs(D["cd"][k][v] / nd - D["ck"][k].get(v, 0) / nk)
            if d > worst[2]:
                worst = (k, v, d)
    print(f"seed {seed}: n={n} " + " ".join(f"{r}={sh[r]:.3f}" for r in RULES) + f" | max fake rule {mx} {sh[mx]:.3f}"
          f" | gate {'OK' if gate else 'FAIL'} | largest drawn-vs-kept shift {worst[2]*100:.2f} pts ({worst[0]}={worst[1]})"
          f" | gold in answer {sanity['gold_in_answer']}/{n}, gold stated {sanity['gold_stated']}/{n}")
    print("    per block: " + "; ".join(f"{b} (n={nb[b]}): " + " ".join(f"{r}={per_b[b][r]/nb[b]:.3f}" for r in RULES)
                                     for b in ("e004", "alias", "ind")))
print("pooled per block: " + "; ".join(f"{b}: " + " ".join(f"{r}={pooled[b][r]/pooled_n[b]:.3f}" for r in RULES)
                                      for b in ("e004", "alias", "ind")))
print("ALL SEEDS GATE:", "PASS" if allgood else "FAIL")
