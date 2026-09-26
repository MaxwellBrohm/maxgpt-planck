import sys, json, collections
sys.dont_write_bytecode = True
import load, mygrade as G
E5 = load.E5
items = {it["idx"]: it for it in load.al_items()}
ORR = json.load(open("oracle_right.json")); ORR = {k: {int(i): v for i, v in d.items()} for k, d in ORR.items()}
S = json.load(open("sets.json"))
stored = json.load(open(E5 + "/al/al_results.json"))["models"]
SYS = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
def chat_prompt(it):
    s = SYS
    for u, a in it["turns"]:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    s += f"<|im_start|>user\n{it['question']}<|im_end|>\n<|im_start|>assistant\n"
    return s + it["prefix"]
SW = {"the","a","an","my","your","our","their","of","for","with","from","by","run","in","on","at","to","and"}
def objw(it):
    ph, hd = it["objects"][it["asked"]]
    ws = set(G.toks(ph)) | {hd.lower()}
    if it.get("alias") and it.get("alias_obj") == it["asked"]:
        ws |= set(G.toks(it["alias"]))
    return ws - SW
same_title = {i for i, it in items.items() if len([a for a in it["aliases"] if a]) == 2 and
              it["aliases"][0].split()[0] == it["aliases"][1].split()[0]}
groups = {"AL1": lambda it: it["cell"] == "AL1", "AL2": lambda it: it["cell"] == "AL2",
          "AL3": lambda it: it["cell"] == "AL3", "AL4": lambda it: it["cell"] == "AL4",
          "AL124": lambda it: it["cell"] != "AL3", "all": lambda it: True,
          "needs_link14": lambda it: it["idx"] in S["nl"], "same_title15": lambda it: it["idx"] in same_title,
          "T1_wrong": lambda it: not ORR["strict_adjacency(T1)"][it["idx"]],
          "order_wrong": lambda it: not ORR["order_match"][it["idx"]]}
tags = ["base"] + [f"e005_s{i}" for i in range(1, 6)] + [f"e004_s{i}" for i in range(1, 6)]
print("group sizes:", {g: sum(f(it) for it in items.values()) for g, f in groups.items()})
allres = {}
hdr = ["AL1", "AL2", "AL3", "AL4", "AL124", "all", "needs_link14", "same_title15", "T1_wrong", "order_wrong"]
print(f"{'tag':9s} {'measure':10s} " + " ".join(f"{h[:12]:>12s}" for h in hdr))
for tag in tags:
    rr = {}
    for render in ("plain", "chat"):
        L = load.al_recs(tag, "al", render); Gn = load.al_recs(tag, "gen_al", render)
        assert len(L) == 64 and len(Gn) == 64
        mism = collections.Counter()
        lik, gen = {}, {}
        for r in L:
            it = items[r["idx"]]
            if r["cand_vals"]["gold"] != it["gold"] or sorted(r["cand_vals"].values()) != sorted(it["candidates"]): mism["cands"] += 1
            want = load.phash(load.plain_prompt(it, True) if render == "plain" else chat_prompt(it))
            if r["h"] != want: mism["hash"] += 1
            lik[r["idx"]] = load.my_right(r["scores"])
            if lik[r["idx"]] != r["right"]: mism["right"] += 1
        for g in Gn:
            it = items[g["idx"]]
            gen[g["idx"]] = G.strict(g["reply"], g["stop"], it["gold"], it["values"], objw(it))
            if gen[g["idx"]] != g["strict"]: mism["strict"] += 1
        assert set(lik) == set(items) == set(gen)
        rr[render] = (lik, gen)
        for meas, dd in (("LIK", lik), ("GEN", gen)):
            row = []
            for h in hdr:
                ids = [i for i, it in items.items() if groups[h](it)]
                row.append(sum(dd[i] for i in ids) / len(ids))
            # compare to stored where the stored group exists
            st = stored[tag][f"{meas} {render}"]
            cmp = {"AL1": "AL1", "AL2": "AL2", "AL3": "AL3", "AL4": "AL4", "AL124": "AL1+AL2+AL4", "all": "all", "needs_link14": "needs linking"}
            diffs = [h for h in cmp if abs(st[cmp[h]] - row[hdr.index(h)]) > 0.0001]
            print(f"{tag:9s} {meas+' '+render:10s} " + " ".join(f"{x:12.3f}" for x in row) + (f"  STORED DIFFERS {diffs}" if diffs else "") )
        if sum(mism.values()): print("   mismatches vs stored per-record fields:", dict(mism))
    allres[tag] = {r: {"lik": {str(k): v for k, v in rr[r][0].items()}, "gen": {str(k): v for k, v in rr[r][1].items()}} for r in rr}
json.dump(allres, open("al_my_scores.json", "w"))
