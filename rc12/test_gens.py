"""RC-12 DEV generator tests, part 1: structure, counts, L2 (no answer word in any question or later user
turn), G5 filler purity, G7 length, G8 knowledge-free golds, determinism. Part 2 (family layouts, distances,
balance of mention orders) is test_gens_fam.py. Run: python3 -B test_gens.py (exits non-zero on any failure)."""
import json
import os
import re
import subprocess
import sys
from collections import Counter

import build_dev
import common as C
import pools_vals as V
from fam_persist import FIRST_WORDS, CLOSINGS, WORDS

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
EXPECT_FAM = {"T0": 48, "RECALL": 60, "CORR": 64, "BIND": 60, "TWOHOP": 64, "PERSIST": 48, "OWN": 48,
              "TOPIC": 48, "ROLE": 48, "LOOKUP": 48, "LOOP": 40, "K": 64}
EXPECT_CELL = {("RECALL", "abstain"): 12, ("RECALL", "d1-3"): 16, ("RECALL", "d4-7"): 16, ("RECALL", "d8-11"): 16,
               ("CORR", "U-diff"): 16, ("CORR", "U-same"): 16, ("CORR", "C_noupd"): 16, ("CORR", "C_twoslot"): 16,
               ("BIND", "owner"): 24, ("BIND", "perspective"): 20, ("BIND", "third-party"): 16,
               ("TWOHOP", "COMPOSE"): 16, ("PERSIST", "hold"): 32, ("PERSIST", "override"): 16,
               ("OWN", "pick"): 32, ("OWN", "list"): 16, ("ROLE", "named"): 24, ("ROLE", "unnamed"): 24,
               ("K", "followup"): 32, ("K", "control"): 32}
KINDS = set("SLCIOQTDPX")
GRADERS = {"VAL", "ABS", "FMT", "DYN", "ROLEX", "LOOP"}


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def load():
    with open(build_dev.OUT, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9']+", s.lower()))


def golds(p):
    out = [p["gold"]] if p.get("gold") else []
    out += p.get("accepted") or []
    if p.get("gold_fn") and p["gold_fn"].get("options"):
        out += p["gold_fn"]["options"]
    return out


def test_counts(recs):
    fam = Counter(r["family"] for r in recs)
    check(dict(fam) == EXPECT_FAM, f"family counts {dict(fam)}")
    cells = Counter((r["family"], r["cell"]) for r in recs)
    for k, n in EXPECT_CELL.items():
        check(cells[k] == n, f"cell {k}: {cells[k]} != {n}")
    check(len(recs) == 640 and sum(r["n_turns"] == 12 for r in recs) == 608, "640 records / 608 twelve-turn")
    check(len({r["id"] for r in recs}) == len(recs), "duplicate ids")


def test_structure(recs):
    for r in recs:
        n = r["n_turns"]
        check(n == (1 if r["cell"] == "control" else 12), f"{r['id']} n_turns {n}")
        check([t["i"] for t in r["turns"]] == list(range(1, n + 1)), f"{r['id']} turn indices")
        ideals = []
        for t in r["turns"]:
            check(t["kind"] in KINDS, f"{r['id']} u{t['i']} kind {t['kind']}")
            check(t["text"].strip() and t["ideal"].strip(), f"{r['id']} u{t['i']} empty text/ideal")
            lim = 70 if t["kind"] == "T" else 35
            check(C.nwords(t["text"]) <= lim, f"{r['id']} u{t['i']} {C.nwords(t['text'])} words > {lim}")
            check("\u2014" not in t["text"] + t["ideal"], f"{r['id']} u{t['i']} em dash")
            ideals.append(norm(t["ideal"]))
        check(len(set(ideals)) == len(ideals), f"{r['id']} IDEAL replies repeat (loop rule)")
        check(C.token_proxy([t["text"] for t in r["turns"]]) <= C.MAX_TOKENS, f"{r['id']} over the token budget")
        seen = set()
        for p in r["probes"]:
            t = r["turns"][p["turn"] - 1]
            check(p["turn"] not in seen, f"{r['id']} two probes on u{p['turn']}")
            seen.add(p["turn"])
            check(t["text"] == p["question"] and t["kind"] == p["kind"] and p["kind"] in "PX",
                  f"{r['id']} probe u{p['turn']} does not match its turn")
            check(p["grader"] in GRADERS, f"{r['id']} grader {p['grader']}")
            check(t["ideal"] == p["ideal"], f"{r['id']} probe ideal mismatch")
            if p["grader"] == "VAL":
                check(bool(p["gold"]), f"{r['id']} VAL probe without gold")
                if p["candidates"]:
                    check(p["gold"] in p["candidates"], f"{r['id']} gold not among candidates")
                check(V.mentions(p["ideal"], p["gold"]), f"{r['id']} IDEAL reply lacks the gold")
                for c in p["candidates"]:
                    if c != p["gold"]:
                        check(not V.mentions(p["ideal"], c, False), f"{r['id']} IDEAL names another candidate {c}")
            if p["grader"] == "ABS":
                pool = V.GUESS_POOL.get(p["pool"]) or V.POOLS[p["pool"]]
                check(p["gold"] is None and not any(V.mentions(p["ideal"], c, False) for c in pool),
                      f"{r['id']} abstain IDEAL names a value of its pool")
            if p["src"]:
                check(p["d"] == p["turn"] - max(p["src"]), f"{r['id']} d mismatch")
                check(all(r["turns"][s - 1]["kind"] in "SLCITQO" for s in p["src"]), f"{r['id']} src kind")
        check(any(p["kind"] == "P" for p in r["probes"]), f"{r['id']} has no main probe")
        check(r["knowledge"] == (r["family"] == "K"), f"{r['id']} knowledge flag")


def test_l2(recs):
    """L2/G5: no question or prefix contains its gold or any candidate; no user turn after the gold's source
    contains the gold."""
    for r in recs:
        for p in r["probes"]:
            for g in golds(p) + list(p["candidates"]):
                check(not V.mentions(p["question"], g, False), f"{r['id']} question contains {g!r}")
                if p.get("prefix"):
                    check(not V.mentions(p["prefix"], g, False), f"{r['id']} prefix contains {g!r}")
            if p.get("gold") and p["src"]:
                for t in r["turns"][max(p["src"]):]:
                    check(not V.mentions(t["text"], p["gold"], False),
                          f"{r['id']} u{t['i']} repeats the gold {p['gold']!r} after its source")


def test_g8(recs):
    for r in recs:
        if r["knowledge"]:
            continue
        for p in r["probes"]:
            if p["grader"] in ("VAL", "ROLEX") and p.get("gold"):
                check(any(V.mentions(r["turns"][s - 1]["text"], p["gold"]) for s in p["src"]),
                      f"{r['id']} gold {p['gold']!r} not in its source user turn")
            if p["grader"] == "DYN":
                check(p["gold_fn"] and r["turns"][p["gold_fn"]["src_turn"] - 1]["kind"] == "Q",
                      f"{r['id']} DYN probe without a Q source")


def _e_fillers():
    sys.dont_write_bytecode = True
    base = os.path.join(HERE, "..", "experiments")
    sys.path[:0] = [os.path.join(base, "E001_battery_and_probes", "code"),
                    os.path.join(base, "E004_general_updating", "code")]
    import items, items_new, fillers_eval, fillers_train
    del sys.path[:2]
    out = list(items.DISTRACTORS) + list(items_new.EXTRA_DISTRACTORS)
    out += list(fillers_eval.EVAL_FILLERS_NEW) + list(fillers_eval.H7_FILLERS) + list(fillers_train.FILLERS_TRAIN)
    return [x if isinstance(x, str) else " ".join(x) for x in out]


def grams(text, n=5):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def test_fillers(recs):
    check(len(C.FILLERS) >= 80, f"only {len(C.FILLERS)} fillers")
    check(len({q for q, _ in C.FILLERS}) == len(C.FILLERS), "duplicate filler questions")
    old = set()
    for s in _e_fillers():
        old |= grams(s)
    rule_words = FIRST_WORDS + CLOSINGS + WORDS
    holder_words = V.HOLDERS + ["mom", "dad", "friend"]
    for q, a in C.FILLERS:
        both = q + " " + a
        check(not (grams(q) | grams(a)) & old, f"filler shares a 5-gram with E001/E004: {q}")
        check(not re.search(r"\d", both), f"filler has a digit: {q}")
        for v in V.ALL_VALUES + rule_words + holder_words:
            check(not V.mentions(both, v, False), f"filler contains {v!r}: {q}")
        for sent in re.split(r"(?<=[.?!])\s+", both):
            for w in re.findall(r"[A-Za-z][\w'-]*", sent)[1:]:
                check(not w[:1].isupper() or w in ("I", "I'm", "I've"), f"filler has a name-like word {w!r}: {q}")
        check(not re.search(r"[.?!].", a.strip()) and not a.strip().endswith(("?", "!")),
              f"filler IDEAL is not one sentence: {a}")
    for r in recs:
        ds = [t["text"] for t in r["turns"] if t["kind"] == "D"]
        check(len(ds) == len(set(ds)), f"{r['id']} filler used twice")


def test_pools():
    names = list(V.POOLS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(x.lower() for x in V.POOLS[a]) & set(x.lower() for x in V.POOLS[b])
            check(not both, f"pools {a}/{b} share {both}")
    check("May" not in V.MONTH, "May must stay out of MONTH")


def test_determinism(recs):
    out = subprocess.run([sys.executable, "-B", os.path.join(HERE, "build_dev.py"), "--stdout"],
                         capture_output=True, text=True, timeout=110, cwd=HERE)
    with open(build_dev.OUT, encoding="utf-8") as f:
        check(out.stdout == f.read(), "a fresh build differs from dev/rc12_dev.jsonl (not deterministic)")
    check(build_dev.dumps(build_dev.build_all()) == out.stdout, "two builds in two processes differ")


def main():
    recs = load()
    for fn in (test_counts, test_structure, test_l2, test_g8, test_fillers, test_determinism):
        fn(recs)
    test_pools()
    import test_gens_fam
    FAILS.extend(test_gens_fam.run(recs))
    for m in FAILS[:60]:
        print("FAIL", m)
    print(f"{len(FAILS)} failures over {len(recs)} records")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
