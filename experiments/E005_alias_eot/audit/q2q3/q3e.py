import sys, json, collections, re
sys.dont_write_bytecode = True
import load, alparse as AP, mygrade as G
import q3b_oracles as O
items = {it["idx"]: it for it in load.al_items()}
P = {i: AP.parse(it) for i, it in items.items()}
NAME = AP.NAME
c = collections.Counter()
STOPW = {"the","a","an","my","of","for","with","from","by","run","in","on","at","to","and","new","big","next","our","your"}
for i, it in items.items():
    q, pf = it["question"], it["prefix"]
    c["question_has_any_value"] += any(G.mentions(q, v) for v in it["values"])
    c["prefix_has_any_value"] += any(G.mentions(pf, v) for v in it["values"])
    c["question_or_prefix_has_name"] += bool(NAME.search(q) or NAME.search(pf))
    c["question_or_prefix_has_surname"] += any(a.split()[1] in q + pf for a in it["aliases"] if a)
    # values anywhere outside statement user turns and their acks
    stturns = {p["turn"] for p in P[i]}
    for ti, (u, a) in enumerate(it["turns"]):
        if ti not in stturns:
            c["filler_turn_with_value"] += any(G.mentions(u + " " + a, v) for v in it["values"])
            c["filler_turn_with_name"] += bool(NAME.search(u + " " + a))
    # ack after an alias correction: does it name an object word or a name?
    objwords = {w for ph, hd in it["objects"] for w in G.toks(ph) + [hd.lower()]} - STOPW
    for p in P[i]:
        u, a = it["turns"][p["turn"]]
        if p["obj"] is None:
            c["alias_corr"] += 1
            c["alias_corr_user_text_shares_obj_word"] += bool(set(G.toks(u)) & objwords)
            c["alias_corr_ack_has_obj_word"] += bool(set(G.toks(a)) & objwords)
            c["alias_corr_ack_has_name"] += bool(NAME.search(a))
        c["stmt_ack_mentions_value"] += G.mentions(a, p["value"])
        c["stmt"] += 1
    # mention counts in the whole prompt (user+assistant+question+prefix)
    text = " ".join(u + " " + a for u, a in it["turns"]) + " " + q + " " + pf
    cnt = {v: len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", text, 0 if v == "May" else re.I)) for v in it["candidates"]}
    mx = max(cnt.values())
    c["gold_uniquely_most_mentioned"] += cnt[it["gold"]] == mx and list(cnt.values()).count(mx) == 1
    c["gold_count_" + str(cnt[it["gold"]])] += 1
    c["n_candidates_" + str(len(it["candidates"]))] += 1
    # gold is the last value-bearing turn? the turn just before the question?
    c["gold_in_last_user_turn_before_question"] += G.mentions(it["turns"][-1][0], it["gold"])
    c["gold_is_last_statement"] += P[i][-1]["value"] == it["gold"]
    # gold is the first candidate listed (order of first mention)?
    c["gold_first_mentioned_value"] += it["candidates"][0] == it["gold"]
    c["gold_last_first_mentioned_value"] += it["candidates"][-1] == it["gold"]
    # correction marker oracle: latest statement carrying any marker text in stmts metadata
for k in sorted(c): print(f"  {k}: {c[k]}")
# ntok leak: does 'fewest tokens' pick the gold? (LIK sums log-probs over tokens)
tk = collections.Counter()
for tag in ["e005_s1", "e004_s1"]:
    for r in load.al_recs(tag, "al", "plain"):
        g = r["ntok"]["gold"]; o = [v for k, v in r["ntok"].items() if k != "gold"]
        tk[(tag, "gold_strictly_fewer_tokens")] += g < min(o)
        tk[(tag, "gold_more_tokens")] += g > min(o)
        tk[(tag, "all_single_token")] += all(v == 1 for v in r["ntok"].values())
print(" ", dict(tk))
# marker oracle: latest statement with a correction marker (any), and latest statement WITHOUT a name
def o_marker(it, Pi):
    m = [s for s in it["stmts"] if s["marker"] and s["role"] == "corr"]
    return m[-1]["value"] if m else None
def o_latest_unnamed(it, Pi):
    m = [p for p in Pi if not p["names"]]
    return m[-1]["value"] if m else None
for nm, fn in (("latest_marked_correction", o_marker), ("latest_statement_without_a_name", o_latest_unnamed)):
    per = collections.Counter()
    for i, it in items.items(): per[it["cell"]] += fn(it, P[i]) == it["gold"]
    print(f"  oracle {nm}: " + " ".join(f"{k} {per[k]/16:.2f}" for k in ("AL1", "AL2", "AL3", "AL4")) + f"  all {sum(per.values())/64:.2f}  AL124 {(per['AL1']+per['AL2']+per['AL4'])/48:.2f}")
# hybrids on same-title items: title link when decidable, else fallback X
same = [i for i, it in items.items() if len([a for a in it["aliases"] if a]) == 2 and it["aliases"][0].split()[0] == it["aliases"][1].split()[0]]
S = json.load(open("sets.json"))
print("  same-title items:", len(same), "cells", collections.Counter(items[i]["cell"] for i in same), "of which needs-linking:", len(set(same) & set(S["nl"])))
for nm, fn in O.ORACLES.items():
    if nm in ("title_only_link", "LINK(full name)"): continue
    ok = sum(fn(items[i], P[i]) == items[i]["gold"] for i in same)
    print(f"    on the 15 same-title items, {nm}: {ok}/15")
