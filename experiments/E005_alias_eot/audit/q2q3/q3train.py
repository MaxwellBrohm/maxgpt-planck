import sys, collections, itertools
sys.dont_write_bytecode = True
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
import train_e005 as T5
N = 6600
tot = collections.Counter()
for seed in range(1, 6):
    ex0 = None
    for ex in itertools.islice(T5.stream(seed), N):
        if ex0 is None: ex0 = ex
        st = ex["stmts"]
        n_alias_corr = sum(1 for s in st if s.get("ref") == "alias")
        n_named_objs = len(ex.get("aliases") or {})
        tot["examples"] += 1
        tot[("block", ex.get("block"))] += 1
        tot[("n_alias_corr", n_alias_corr)] += 1
        if ex.get("block") == "alias":
            tot[("case", ex.get("case"))] += 1
            # the asked object's latest statement is an alias correction? a statement after it about the asked object?
            asked = ex["asked"]
            about = [s for s in st if s["obj"] == asked]
            tot[("gold_is_alias_corr", about[-1].get("ref") == "alias")] += 1
            # AL2/AL3-like: two alias corrections in one dialogue
            # AL1-like: other object's alias correction after asked's latest; AL4-like: asked alias corr then a head/full corr of asked
            other_alias_after = any(s.get("ref") == "alias" and s["obj"] != asked and s["turn"] > about[-1]["turn"] for s in st)
            own_alias_then_named = any(s.get("ref") == "alias" and s["obj"] == asked for s in about[:-1]) and about[-1].get("ref") in ("full", "head")
            tot[("AL1_like", other_alias_after)] += 1
            tot[("AL4_like_head_or_full_last", own_alias_then_named)] += 1
            tot[("AL4_like_head_last", own_alias_then_named and about[-1].get("ref") == "head")] += 1
print(sorted(ex0.keys()))
for k, v in sorted(tot.items(), key=str): print(k, v)
