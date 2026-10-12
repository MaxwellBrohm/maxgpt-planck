"""A6. My own strict GEN grader, written from E004 notes (c) clauses 1-7 (gen_grade.py is not imported), applied to
every stored GEN record of D4004 (plain and chat) and BIG (plain) for every scored tag; disagreements are listed."""
import json, re
from common import *

code_path()
import items_e004 as I

D4 = I.draw("eval")
BIG = [json.loads(l) for l in open(os.path.join(E6, "big", "big_items.jsonl"))]
BYF = {"e004": D4, "big": {f: [x for x in BIG if x["family"] == f] for f in ("H5", "C_noupd", "C_twoslot")},
       "h5l": {"H5L": [x for x in BIG if x["family"] == "H5L"]}}
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEG = [r"\bnot\b", r"n't\b", r"\bno longer\b", r"\bnever\b", r"\binstead\b", r"\brather than\b", r"\bno\b"]
HEDGE = ["i think", "maybe", "probably", "might", "perhaps", "not sure", "i don't know", "i believe"]


def has(word, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(word.lower()) + r"(?![a-z0-9])", text) is not None


def grade(rec, item, no_comma=False):
    reply = rec["reply"]
    t = reply.strip().lower()
    fails = []
    if not t or rec.get("capped") or rec.get("stop") in ("max", "cap", "length"):
        fails.append(1)
    w = WORD.findall(t)
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    if len(tri) != len(set(tri)) or any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2)):
        fails.append(2)
    if not has(item["gold"], t):
        fails.append(3)
    if any(has(v, t) for v in item["values"] if v.lower() != item["gold"].lower()):
        fails.append(4)
    neg = NEG if not no_comma else NEG[:-1] + [r"\bno,"]
    if any(re.search(p, t) for p in neg):
        fails.append(5)
    if "?" in t or any(has(h, t) for h in HEDGE):
        fails.append(6)
    obj = item["objects"][item["asked"]]
    words = {x for p in obj for x in WORD.findall(p.lower())}
    if item.get("alias"):
        words |= set(WORD.findall(str(item["alias"]).lower()))
    for i, x in enumerate(w):
        if x in words and "my" in w[max(0, i - 3):i]:
            fails.append(7)
            break
    if reply.lstrip().startswith(("User:", "Assistant:")):
        fails.append(7)
    return not fails, sorted(set(fails))


if __name__ == "__main__":
    tags = ["base"] + [f"{a}{s}" for a in ("C", "P", "G", "e005w", "e004w") for s in SEEDS] + ["C1t", "C2t"]
    tot = dis = 0
    by_clause, examples = {}, []
    for t in tags:
        for set_name, renders in (("e004", ("plain", "chat")), ("big", ("plain",)), ("h5l", ("plain",))):
            for render in renders:
                recs = load(t, "gen_" + set_name, render)
                if recs is None:
                    continue
                for r in recs:
                    item = BYF[set_name][r["family"]][r["idx"]]
                    ok, fl = grade(r, item)
                    tot += 1
                    if ok != r["strict"]:
                        dis += 1
                        k = (tuple(fl), tuple(r.get("fails") or []))
                        by_clause[k] = by_clause.get(k, 0) + 1
                        if len(examples) < 8:
                            examples.append((t, set_name, render, r["id"], r["reply"][:90], fl, r.get("fails")))
    print(f"graded {tot} records; disagreements with stored strict: {dis}")
    for k, v in sorted(by_clause.items(), key=lambda x: -x[1])[:12]:
        print("  mine fails", k[0], "stored fails", k[1], ":", v)
    for e in examples:
        print("  e.g.", e)
