import sys, json, collections
sys.dont_write_bytecode = True
import load, mygrade as G
d = load.eval_items()
SW = {"the","a","an","my","your","our","their","his","her","its","this","that","these","those","of","for","with","from","by","run","in","on","at","to","and"}
def objw(it):
    ph, hd = it["objects"][it["asked"]]
    ws = set(G.toks(ph)) | {hd.lower()}
    if it.get("alias") and it.get("alias_obj") == it["asked"]:
        ws |= set(G.toks(it["alias"]))
    return ws - SW
# alias items: H1+H2 whose asked object's latest statement is an alias correction
alias = []
for fam in ("H1", "H2"):
    for it in d[fam]:
        about = [s for s in it["stmts"] if s["obj"] == it["asked"]]
        last = about[-1]
        is_alias = last.get("ref") == "alias"
        assert is_alias == (it["meta"].get("latest_ref") == "alias"), (fam, it["idx"])
        if is_alias:
            prev = about[-2]
            between = [s for s in it["stmts"] if prev["turn"] < s["turn"] < last["turn"]]
            if between: pl = "B_sep"
            elif last["turn"] - prev["turn"] == 1: pl = "adjacent"
            else: pl = "filler"
            assert all(s["obj"] != it["asked"] for s in between)
            alias.append((fam, it["idx"], pl, it))
print("alias items:", collections.Counter(f for f, *_ in alias), "placement:", collections.Counter((f, p) for f, _, p, _ in alias))
# sanity: the alias correction is the gold in every alias item; alias recency oracle
print("gold == alias-correction value on", sum(it["gold"] == [s for s in it["stmts"] if s["obj"]==it["asked"]][-1]["value"] for *_, it in alias), "of", len(alias))
# does the alias correction text contain the alias name? and is it the last statement overall?
lastall = sum(max(it["stmts"], key=lambda s: s["turn"])["obj"] == it["asked"] for *_, it in alias)
print("alias correction is the LAST statement in the dialogue on", lastall, "of", len(alias))
def score(exp, tag):
    L = {(r["family"], r["idx"]): r for r in load.recs(exp, tag, "e004", "plain")}
    Gn = {(r["family"], r["idx"]): r for r in load.recs(exp, tag, "gen_e004", "plain")}
    res = collections.defaultdict(lambda: [0, 0, 0])
    mism = dict(right=0, strict=0, hash=0, gold=0)
    for fam, idx, pl, it in alias:
        r = L[(fam, idx)]; g = Gn[(fam, idx)]
        if r["cand_vals"]["gold"] != it["gold"]: mism["gold"] += 1
        if r["h"] != load.phash(load.plain_prompt(it, True)): mism["hash"] += 1
        lk = load.my_right(r["scores"])
        if lk != r["right"]: mism["right"] += 1
        gs = G.strict(g["reply"], g["stop"], it["gold"], it["values"], objw(it))
        if gs != g["strict"]: mism["strict"] += 1
        for key in ("all", fam, pl):
            res[key][0] += lk; res[key][1] += gs; res[key][2] += 1
    return res, mism
def lab(l, g):
    return "HIGH" if l >= 0.8 and g >= 0.8 else ("LOW" if l <= 0.5 and g <= 0.5 else "MID")
for exp, tags in (("e005", ["base", "s1", "s2", "s3", "s4", "s5"]), ("e004", ["base", "s1", "s2", "s3", "s4", "s5"])):
    hi = lo = 0
    for t in tags:
        res, mism = score(exp, t)
        a = res["all"]; l, g = a[0]/a[2], a[1]/a[2]
        if t != "base":
            hi += lab(l, g) == "HIGH"; lo += lab(l, g) == "LOW"
        parts = " ".join(f"{k} {res[k][0]}/{res[k][1]}/n{res[k][2]}" for k in ("H1", "H2", "adjacent", "filler", "B_sep"))
        print(f"{exp} {t}: LIK {a[0]}/{a[2]}={l:.4f} GEN {a[1]}/{a[2]}={g:.4f} [{lab(l,g)}] | {parts} | mismatches vs stored {mism}")
    maj = "DATA GAP" if hi >= 3 else ("REAL LIMIT" if lo >= 3 else "INCONCLUSIVE")
    print(f"{exp}: HIGH {hi} LOW {lo} -> {maj}")
