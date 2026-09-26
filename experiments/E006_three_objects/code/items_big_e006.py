"""E006 extra eval items (notes.txt EVAL DRAWS). No model, no tokenizer.
  BIG  items_e004.build(6006, f, 192) for f = H5, C_noupd, C_twoslot: E004's own generator and vocabulary, a new
       draw seed (streams Random("E004:6006:<family>")). Q-H5 reads BIG H5.
  H5L  the gold-last mirror of H5 (critique item 1): 3 objects, E004's H5 planner logic with the after-condition
       inverted (the asked object's latest statement IS the dialogue's last statement), seed 6106, stream
       Random("E004:6106:H5L"), stratified by the asked object's introduction order: 64 items each for first,
       second, third (factors balanced within a stratum; "never corrected" only in the third, where it is the
       only possible order). Items are planned and checked as H5, then labelled family "H5L".
build() makes both; the item checks (checks_eval.ITEM_CHECKS), BIG's shortcut gates G1/G2 (G3 needs the pass
families, not scored here), H5L's IDEAL = 1.00 (its O2 "last mention" is 1.00 by construction and is reported,
not gated), and 0 prompts shared with draws 4004/4104/4204, the AL items or each other.
  python3 -B items_big_e006.py --write   writes ../big/big_items.jsonl + big_items.sha256 ONCE (refuses to rewrite)
  python3 -B items_big_e006.py --check   rebuilds, compares with the file and its sha256 (exit 0 = equal)"""
import hashlib
import json
import os
import sys
from collections import Counter

import items as _I  # noqa: F401  (same import order as e004_sets)
import items_e004 as I
import checks_eval as CE
import shortcuts as SC
from items_plan import FORMS, ev, corr_marker, spec, factors
from items_render import render, seeded, prompt
from heldout_e004 import EVAL_NONCORR_MARKERS
from pools_eval import ALL_EVAL_POOLS, TRAIN_VTYPES

HERE = os.path.dirname(os.path.abspath(__file__))
BIG_DIR = os.path.join(os.path.dirname(HERE), "big")
ITEMS_PATH = os.path.join(BIG_DIR, "big_items.jsonl")
SHA_PATH = os.path.join(BIG_DIR, "big_items.sha256")
BIG_SEED, BIG_N, BIG_FAMS = 6006, 192, ("H5", "C_noupd", "C_twoslot")
H5L_SEED, H5L_PER_STRATUM = 6106, 64
BIG_PROMPT_SHA_PREFIX = "6ad25186"       # the draft's trial build (notes LOG AT WRITING 2)


def intro_order(stmts_or_order, asked):
    seen = []
    for x in stmts_or_order:
        o = x["obj"] if isinstance(x, dict) else x[0]
        if o not in seen:
            seen.append(o)
    return seen.index(asked) + 1


def h5l_one(rng, f):
    nval = len(ALL_EVAL_POOLS[f["pool"]]["values"])
    asked = f["asked"]
    for _ in range(20000):
        ks = [0 if (o == asked and f["ak0"]) else rng.randint(1 if o == asked else 0, 3) for o in range(3)]
        if 3 + sum(ks) > nval:
            continue
        seqs = {o: [(o, "orig")] + [(o, "corr")] * ks[o] for o in range(3)}
        order = []
        while any(seqs.values()):
            o = rng.choice([o for o in seqs if seqs[o]])
            order.append(seqs[o].pop(0))
        if order[-1][0] != asked or intro_order(order, asked) != f["intro"]:
            continue
        events = []
        for i, (o, role) in enumerate(order):
            if role == "orig":
                mk = rng.choice(EVAL_NONCORR_MARKERS) if (i > 0 and rng.random() < 1 / 3) else None
                events.append(ev(o, "orig", "full", marker=mk))
            else:
                adj = order[i - 1][0] == o
                events.append(ev(o, "corr", rng.choice(FORMS if adj else ["full", "head"]), marker=corr_marker(rng)))
        if ks[asked] and (events[-1]["ref"] in ("pron", "ell")) != f["ind"]:
            continue
        return spec("H5", f["pool"], events, 3, asked=asked, q_form=f["q"], asked_k=ks[asked], after_ind=False,
                    intro=f["intro"])
    raise RuntimeError("H5L plan failed")


def build_h5l():
    rng = seeded(H5L_SEED, "H5L")
    specs = []
    for intro in (1, 2, 3):
        ak0 = [True, False, False, False] if intro == 3 else [False]
        F = factors(rng, H5L_PER_STRATUM, pool=TRAIN_VTYPES, asked=[0, 1, 2], ak0=ak0, ind=[True, True, False],
                    q=["full", "head"])
        specs += [h5l_one(rng, dict(f, intro=intro)) for f in F]
    items = []
    for i, sp in enumerate(specs):
        sp["pre"] = rng.randint(0, 3)
        it = render(rng, sp)
        it["idx"], it["seed"] = i, H5L_SEED
        items.append(it)
    return items


def build():
    """-> {"H5": [...], "C_noupd": [...], "C_twoslot": [...], "H5L": [...]} (H5L items labelled family H5L)."""
    out = {f: I.build(BIG_SEED, f, BIG_N) for f in BIG_FAMS}
    h = build_h5l()
    for it in h:
        it["family"] = "H5L"
    out["H5L"] = h
    return out


def as_checked(it):
    return dict(it, family="H5") if it["family"] == "H5L" else it


def item_problems(D):
    bad = Counter()
    for f, its in D.items():
        for it in its:
            for name, fn in CE.ITEM_CHECKS:
                for p in fn(as_checked(it)):
                    bad[(f, name, p.split(" ")[0])] += 1
    return dict(bad)


def checks(D):
    """-> (failures, report lines)."""
    fails, L = [], []
    bad = item_problems(D)
    if bad:
        fails.append(f"item checks: {bad}")
    fr, rows = SC.table({f: D[f] for f in BIG_FAMS})
    g, _ = SC.gates(fr, rows)
    fails += [x for x in g if not x.startswith("G3")]
    L += SC.fmt(fr, rows)
    fr2, rows2 = SC.table({"H5": [as_checked(it) for it in D["H5L"]]})
    L += ["H5L (the mirror; not gated except IDEAL):"] + SC.fmt(fr2, rows2)
    if dict(rows2)["IDEAL"]["H5"] != 1.0:
        fails.append("H5L IDEAL != 1.00")
    old = {}
    for name in I.DRAWS:
        for its in I.draw(name).values():
            for it in its:
                old[prompt(it, True)] = name
    import items_al as AL
    for it in AL.load():
        old[prompt(it, True)] = "AL"
    new = [prompt(it, True) for its in D.values() for it in its]
    shared = sum(p in old for p in new)
    if shared or len(set(new)) != len(new):
        fails.append(f"prompts shared with 4004/4104/4204/AL: {shared}; repeated within BIG+H5L: {len(new) - len(set(new))}")
    h5l_last = sum(it["stmts"][-1]["obj"] == it["asked"] for it in D["H5L"])
    if h5l_last != len(D["H5L"]):
        fails.append(f"H5L: the asked latest is last in {h5l_last}/{len(D['H5L'])}")
    L.append("intro order (first/second/third): " + "; ".join(
        f"{f} {dict(sorted(Counter(intro_order(it['stmts'], it['asked']) for it in D[f]).items()))}"
        for f in ("H5", "H5L")))
    L.append(f"BIG prompt-list sha256 {big_prompt_sha(D)[:16]} (draft trial {BIG_PROMPT_SHA_PREFIX})")
    if not big_prompt_sha(D).startswith(BIG_PROMPT_SHA_PREFIX):
        fails.append("BIG prompt-list hash differs from the draft's trial build")
    return fails, L


def big_prompt_sha(D):
    return hashlib.sha256(json.dumps([prompt(it, True) for f in BIG_FAMS for it in D[f]]).encode()).hexdigest()


def dumps(D):
    return "".join(json.dumps(it, sort_keys=True) + "\n" for f in list(BIG_FAMS) + ["H5L"] for it in D[f])


def load(path=ITEMS_PATH, check_sha=True):
    raw = open(path, "rb").read()
    if check_sha and hashlib.sha256(raw).hexdigest() != open(SHA_PATH).read().split()[0]:
        raise SystemExit(f"refusing: {path} does not match {SHA_PATH}")
    out = {}
    for line in raw.decode().splitlines():
        it = json.loads(line)
        out.setdefault(it["family"], []).append(it)
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    D = build()
    fails, L = checks(D)
    print("\n".join(L))
    text = dumps(D)
    h = hashlib.sha256(text.encode()).hexdigest()
    if "--write" in argv and not fails:
        if os.path.exists(ITEMS_PATH):
            print(f"REFUSED: {ITEMS_PATH} exists (the items are written once)")
            return 1
        os.makedirs(BIG_DIR, exist_ok=True)
        with open(ITEMS_PATH, "x") as f:
            f.write(text)
        with open(SHA_PATH, "x") as f:
            f.write(f"{h}  big_items.jsonl\n")
        print(f"WROTE {ITEMS_PATH}: {sum(map(len, D.values()))} items, sha256 {h}")
    if "--check" in argv:
        same = os.path.exists(ITEMS_PATH) and open(ITEMS_PATH).read() == text and \
            open(SHA_PATH).read().split()[0] == h
        print(f"file equals the rebuild and its sha256: {same}")
        fails += [] if same else ["the items file differs from the rebuild"]
    print("RESULT: " + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"), f"(sha256 {h})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
