"""Aggregate transcripts/*.jsonl into results.json and print summary tables."""
import glob, json, os, re, statistics as st
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "unsloth/gemma-3-270m-it",
         "LiquidAI/LFM2-350M", "LiquidAI/LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct",
         "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"]
SHORT = {"tiiuae/Falcon-H1-Tiny-90M-Instruct": "Falcon-H1-Tiny-90M", "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M",
         "unsloth/gemma-3-270m-it": "Gemma3-270M", "LiquidAI/LFM2-350M": "LFM2-350M", "LiquidAI/LFM2.5-350M": "LFM2.5-350M",
         "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M", "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen/Qwen3-0.6B": "Qwen3-0.6B"}
CATS = ["recall", "followup", "instruction", "correction", "own_answer", "role", "topic_return"]


def toks(t):
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def distinct(ws, n):
    g = ngrams(ws, n)
    return len(set(g)) / len(g) if g else None


def max_repeat(ws, n=4):
    g = Counter(ngrams(ws, n))
    return max(g.values()) if g else 0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


import sys
sys.path.insert(0, HERE)
import battery as B
TESTS = {t["id"]: t for t in B.TESTS}


def regrade(r):
    """Re-run the CURRENT graders on the saved replies, so a grader fix applies to every model."""
    t = TESTS[r["id"]]
    replies = [x["assistant"] for x in r["turns"]]
    checks, lenient = {}, {}
    for c in t["checks"]:
        v = c["fn"](replies[c["turn"]], replies)
        checks[c["name"]] = None if v is None else bool(v)
        if "gold" in c:
            lenient[c["name"]] = B.mentions(c["gold"], replies[c["turn"]])
    for x in r["turns"]:
        x["flags"]["no_memory_claim"] = B.deflects(x["assistant"])
    r["checks"], r["checks_lenient"] = checks, lenient
    return r


def load(mode):
    recs = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, "transcripts", f"*__{mode}.jsonl")):
        for line in open(f):
            r = regrade(json.loads(line))
            recs[r["model"]].append(r)
    return recs


def summarize(recs):
    out = {}
    for m in ORDER:
        R = recs.get(m)
        if not R:
            continue
        seeds = sorted(set(r["seed"] for r in R), key=lambda s: (s is None, s))
        by = defaultdict(list)
        for r in R:
            by[r["id"]].append(r)
        res = {"params_m": R[0]["params_m"], "n_conversations": len(R), "seeds": seeds}

        # ability scores: mean over all gradable checks in category (strict and lenient)
        abil, abil_len, abil_n = {}, {}, {}
        for c in CATS + ["control"]:
            vals, lvals = [], []
            for r in R:
                if r["cat"] != c:
                    continue
                vals += [v for v in r["checks"].values() if v is not None]
                lvals += [v for v in r.get("checks_lenient", {}).values()]
            abil[c] = mean([float(v) for v in vals])
            abil_len[c] = mean([float(v) for v in lvals])
            abil_n[c] = len(vals)
        res["ability_strict"] = abil
        res["ability_lenient_gold_mentioned"] = abil_len
        res["ability_n_checks"] = abil_n
        multi_vals = [float(v) for r in R if r["cat"] in CATS for v in r["checks"].values() if v is not None]
        res["overall_multiturn_strict"] = mean(multi_vals)
        res["overall_multiturn_macro"] = mean([abil[c] for c in CATS])

        # recall vs distance (recall tests + their distance-0 controls)
        dist = defaultdict(lambda: {"strict": [], "lenient": [], "lp_gold": [], "rank1": [], "gold_beats_foil": [], "margin": []})
        for r in R:
            if r["cat"] == "recall" or (r["cat"] == "control" and (r["control_for"] or "").startswith("R_")):
                d = r["distance"]
                for k, v in r["checks"].items():
                    dist[d]["strict"].append(float(v))
                for k, v in r.get("checks_lenient", {}).items():
                    dist[d]["lenient"].append(float(v))
                for f in r["forced"]:
                    if f["lp_gold"] is not None:
                        dist[d]["lp_gold"].append(f["lp_gold"])
                        dist[d]["rank1"].append(float(f["greedy_gold"]))
                        dist[d]["gold_beats_foil"].append(float((f["margin"] or 0) > 0))
                        dist[d]["margin"].append(f["margin"])
        res["recall_by_distance"] = {str(d): {k: mean(v) for k, v in dist[d].items()} | {"n": len(dist[d]["strict"])}
                                     for d in sorted(dist)}

        # forced-choice margin on correction / binding tests (gold = corrected / right value)
        fc = {}
        for r in R:
            if r["cat"] in ("correction", "role") or (r["cat"] == "control" and (r["control_for"] or "")[:2] in ("K_", "RI")):
                for f in r["forced"]:
                    fc.setdefault(r["id"], []).append({"margin": f["margin"], "gold_rank1": f["greedy_gold"],
                                                       "gold_beats_foil": (f["margin"] or 0) > 0})
        res["forced_choice_corrections_binding"] = {k: {"mean_margin": mean([x["margin"] for x in v]),
                                                        "gold_beats_foil_rate": mean([float(x["gold_beats_foil"]) for x in v])}
                                                    for k, v in fc.items()}

        # context vs capability: multi-turn test vs its single-turn control, per shared check, per seed
        cls = Counter()
        detail = {}
        for r in R:
            if r["cat"] != "control":
                continue
            base = r["control_for"]
            twins = [x for x in by[base] if x["seed"] == r["seed"]]
            if not twins:
                continue
            mt = twins[0]
            for k, cv in r["checks"].items():
                mv = mt["checks"].get(k)
                if cv is None or mv is None:
                    continue
                key = ("both_pass" if cv and mv else "context_failure" if cv and not mv
                       else "capability_failure" if not cv and not mv else "multi_only_pass")
                cls[key] += 1
                detail.setdefault(f"{base}:{k}", []).append(key)
        res["context_vs_capability"] = dict(cls)
        res["context_vs_capability_detail"] = detail

        # instruction persistence by turn position (t1 = first question after the instruction)
        pers = defaultdict(list)
        for r in R:
            if r["id"] in ("I_one_sentence", "I_end_question", "I_caps"):
                for k, v in r["checks"].items():
                    pers[k[-2:]].append(float(v))
            if r["id"] == "I_persona":
                for k, v in r["checks"].items():
                    if k.startswith("arr"):
                        pers["persona_" + k[-2:]].append(float(v))
        res["instruction_persistence_by_turn"] = {k: mean(v) for k, v in sorted(pers.items())}

        # per-turn flags over every assistant turn
        flags = Counter()
        nturns = 0
        defl_graded = []
        long_replies = 0
        degenerate = 0
        for r in R:
            for t in r["turns"]:
                nturns += 1
                for k, v in t["flags"].items():
                    flags[k] += int(v)
                ws = toks(t["assistant"])
                if len(ws) >= 20 and max_repeat(ws, 4) >= 3:
                    degenerate += 1
            if r["cat"] in ("recall", "correction", "role", "topic_return") or (r["cat"] == "control" and (r["control_for"] or "")[:2] in ("R_", "K_", "RI", "T_")):
                defl_graded.append(float(r["turns"][-1]["flags"]["no_memory_claim"]))
        res["turn_flags_rate"] = {k: round(v / nturns, 4) for k, v in flags.items()}
        res["turn_flags_count"] = dict(flags)
        res["n_assistant_turns"] = nturns
        res["replies_with_4gram_repeated_3x"] = round(degenerate / nturns, 4)
        res["deflection_rate_on_final_recall_turns"] = mean(defl_graded)

        # self-copy: a reply that re-uses most of an EARLIER reply of the same conversation
        # (>= 50% of its 8-grams already appeared in a previous assistant turn)
        sc, sc_n = 0, 0
        for r in R:
            seen = set()
            for t in r["turns"]:
                ws = toks(t["assistant"])
                g8 = set(ngrams(ws, 8))
                if g8 and seen:
                    sc_n += 1
                    if len(g8 & seen) / len(g8) >= 0.5:
                        sc += 1
                seen |= g8
        res["self_copy_rate"] = round(sc / sc_n, 4) if sc_n else None

        # role capture: in the reply to a user's self-description, the model speaks AS the user
        CAPTURE = re.compile(r"\b(my (cat|dog|parrot|grandmother|apartment|sister|brother|meeting|dentist|favorite color|dinner party)|"
                             r"i'm (priya|marcus|elena|oscar|jordan)|my name is (priya|marcus|elena|oscar|jordan|lena)|"
                             r"i (live|work) (in|at|as)|i'm (excited|hosting|writing)|welcome to 417|i have a (cat|dog|parrot))\b", re.I)
        cap, cap_n = 0, 0
        for r in R:
            if r["cat"] == "control":
                continue
            t0 = r["turns"][0]
            if any(w in t0["user"].lower() for w in ("my name", "i'm", "i live", "i have a", "my dog", "my sister", "my brother", "my grandmother", "my meeting", "my favorite", "i just moved", "i'm hosting", "i'm writing", "remember that i")):
                cap_n += 1
                cap += int(bool(CAPTURE.search(t0["assistant"])))
        res["role_capture_on_self_description"] = {"rate": round(cap / cap_n, 3) if cap_n else None, "n": cap_n}

        # loop metrics on the free-form chats
        loops = {}
        for tid in ("L_tell_more", "L_chitchat"):
            d3, over, mr, lens = [], [], [], []
            for r in by.get(tid, []):
                seen = set()
                for i, t in enumerate(r["turns"]):
                    ws = toks(t["assistant"])
                    lens.append(len(ws))
                    d3.append(distinct(ws, 3))
                    mr.append(max_repeat(ws, 4))
                    g4 = set(ngrams(ws, 4))
                    if i > 0 and g4:
                        over.append(len(g4 & seen) / len(g4))
                    seen |= g4
            loops[tid] = {"distinct3_mean": mean(d3), "cross_turn_4gram_overlap_mean": mean(over),
                          "max_within_reply_4gram_repeat": max(mr) if mr else None, "mean_words": mean(lens)}
        res["loops"] = loops

        # final prompt lengths for recall tests (context actually spanned)
        res["recall_prompt_tokens"] = {r["id"]: r["turns"][-1]["n_prompt_tokens"] for r in R if r["cat"] == "recall" and r["seed"] in (None, 0)}

        # per-test pass table
        res["per_test"] = {tid: {k: mean([float(x["checks"][k]) for x in v if x["checks"].get(k) is not None])
                                 for k in v[0]["checks"]} for tid, v in by.items()}
        out[m] = res
    return out


def table(res, key, cols):
    lines = ["| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for m in ORDER:
        if m not in res:
            continue
        row = res[m][key]
        cells = []
        for c in cols:
            v = row.get(c) if isinstance(row, dict) else None
            if isinstance(v, dict):
                v = v.get("strict")
            cells.append("-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {SHORT[m]} ({res[m]['params_m']}M) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fixed_history():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "exp", "fixed_history__*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        GOLD = {"name": r"\bpriya\b", "number": B.num_pat(417), "pet": r"\bbiscuit", "city": r"\btucson\b"}
        for r in rows:  # regrade with the current graders
            g = GOLD[r["fact"]]
            r["lenient"] = B.mentions(g, r["answer"])
            r["deflect"] = B.deflects(r["answer"])
            r["capture"] = B.captures(g, r["answer"])
            r["strict"] = B.mentions(g, B.strip_vocative(g, r["answer"])) and not r["deflect"] and not r["capture"]
        key = rows[0]["model"] + ("  [plain transcript]" if rows[0].get("format") == "plain" else "") + ("  [memory system prompt]" if rows[0].get("memsys") else "")
        agg = {}
        for style in ("short", "long"):
            for n in sorted({r["n_distractors"] for r in rows}):
                rs = [r for r in rows if r["style"] == style and r["n_distractors"] == n]
                if not rs:
                    continue
                agg[f"{style}_n{n}"] = {"strict": mean([float(r["strict"]) for r in rs]),
                                        "lenient": mean([float(r["lenient"]) for r in rs]),
                                        "deflect": mean([float(r["deflect"]) for r in rs]),
                                        "capture": mean([float(r["capture"]) for r in rs]),
                                        "greedy_gold_from_prefix": mean([float(r["greedy_gold"]) for r in rs if r["greedy_gold"] is not None]),
                                        "gold_beats_foil": mean([float((r["margin"] or 0) > 0) for r in rs]),
                                        "lp_gold": mean([r["lp_gold"] for r in rs]),
                                        "prompt_tokens": mean([r["prompt_tokens"] for r in rs]), "n": len(rs)}
        out[key] = agg
    return out


if __name__ == "__main__":
    results = {"greedy": summarize(load("greedy")), "sampled": summarize(load("sampled")), "fixed_history": fixed_history()}
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    for mode in ("greedy", "sampled"):
        res = results[mode]
        if not res:
            continue
        print(f"\n## {mode}: ability (strict)")
        print(table(res, "ability_strict", CATS + ["control"]))
        print(f"\n## {mode}: recall strict by distance")
        ds = sorted({d for m in res for d in res[m]["recall_by_distance"]}, key=int)
        print(table({m: {**res[m], "rbd": {d: res[m]["recall_by_distance"][d]["strict"] for d in res[m]["recall_by_distance"]}} for m in res}, "rbd", ds))
        print(f"\n## {mode}: recall lenient (gold mentioned) by distance")
        print(table({m: {**res[m], "rbd": {d: res[m]["recall_by_distance"][d]["lenient"] for d in res[m]["recall_by_distance"]}} for m in res}, "rbd", ds))
        print(f"\n## {mode}: forced-prefix greedy-from-prefix emits gold (rate) by distance")
        print(table({m: {**res[m], "rbd": {d: res[m]["recall_by_distance"][d]["rank1"] for d in res[m]["recall_by_distance"]}} for m in res}, "rbd", ds))
        print(f"\n## {mode}: forced-prefix mean logprob(gold) by distance")
        print(table({m: {**res[m], "rbd": {d: res[m]["recall_by_distance"][d]["lp_gold"] for d in res[m]["recall_by_distance"]}} for m in res}, "rbd", ds))
        print(f"\n## {mode}: context vs capability")
        print(table(res, "context_vs_capability", ["both_pass", "context_failure", "capability_failure", "multi_only_pass"]))
        print(f"\n## {mode}: instruction persistence by turn")
        print(table(res, "instruction_persistence_by_turn", ["t1", "t2", "t3", "persona_t0", "persona_t1", "persona_t2", "persona_t3"]))
        print(f"\n## {mode}: turn flags")
        for m in ORDER:
            if m in res:
                r = res[m]
                print(SHORT[m], r["turn_flags_count"], "turns", r["n_assistant_turns"], "deflect_final", r["deflection_rate_on_final_recall_turns"],
                      "degenerate", r["replies_with_4gram_repeated_3x"], "selfcopy", r["self_copy_rate"], "capture", r["role_capture_on_self_description"], "overall", r["overall_multiturn_strict"], "macro", r["overall_multiturn_macro"])
        print(f"\n## {mode}: loops")
        for m in ORDER:
            if m in res:
                print(SHORT[m], json.dumps(res[m]["loops"]))

    fh = results["fixed_history"]
    if fh:
        print("\n## fixed-history sweep: strict free-answer recall | greedy-from-prefix emits gold  (per n distractors)")
        for m, agg in fh.items():
            for style in ("short", "long"):
                ks = [k for k in agg if k.startswith(style)]
                print(f"{m:<48} {style:<5} " + "  ".join(f"n{k.split('_n')[1]}:{agg[k]['strict']:.2f}|{agg[k]['greedy_gold_from_prefix']:.2f}|tok{agg[k]['prompt_tokens']:.0f}" for k in ks))


def e2_pooled():
    """Pooled fixed-history numbers: n>=1 distractors, per style, plus n=0 and n=12."""
    out = {}
    GOLD = {"name": r"\bpriya\b", "number": B.num_pat(417), "pet": r"\bbiscuit", "city": r"\btucson\b"}
    for f in sorted(glob.glob(os.path.join(HERE, "exp", "fixed_history__*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        if not rows:
            continue
        key = os.path.basename(f)[len("fixed_history__"):-len(".jsonl")]
        for r in rows:
            g = GOLD[r["fact"]]
            r["deflect"] = B.deflects(r["answer"]); r["capture"] = B.captures(g, r["answer"])
            r["strict"] = B.mentions(g, B.strip_vocative(g, r["answer"])) and not r["deflect"] and not r["capture"]
        d = {}
        for style in ("short", "long"):
            rs = [r for r in rows if r["style"] == style and r["n_distractors"] >= 1]
            if not rs:
                continue
            d[style] = {"strict": mean([float(r["strict"]) for r in rs]), "deflect": mean([float(r["deflect"]) for r in rs]),
                        "capture": mean([float(r["capture"]) for r in rs]),
                        "gold_beats_foil": mean([float((r["margin"] or 0) > 0) for r in rs]),
                        "greedy_gold": mean([float(r["greedy_gold"]) for r in rs if r["greedy_gold"] is not None]),
                        "lp_gold": mean([r["lp_gold"] for r in rs]), "n": len(rs)}
            r12 = [r for r in rows if r["style"] == style and r["n_distractors"] == 12]
            d[style]["n12_strict"] = mean([float(r["strict"]) for r in r12])
            d[style]["n12_gold_beats_foil"] = mean([float((r["margin"] or 0) > 0) for r in r12])
            d[style]["n12_tokens"] = mean([r["prompt_tokens"] for r in r12])
        r0 = [r for r in rows if r["n_distractors"] == 0 and r["style"] == "short"]
        d["n0_strict"] = mean([float(r["strict"]) for r in r0])
        out[key] = d
    return out


if __name__ == "__main__":
    pooled = e2_pooled()
    results["fixed_history_pooled"] = pooled
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\n## fixed-history pooled over n>=1 (strict | deflect | capture | gold>foil | greedy_gold | n12 strict | n12 gold>foil @tokens)")
    for k, d in pooled.items():
        for style in ("short", "long"):
            if style in d:
                x = d[style]
                print(f"{k:<58} {style:<5} {x['strict']:.2f} | {x['deflect']:.2f} | {x['capture']:.2f} | {x['gold_beats_foil']:.2f} | {x['greedy_gold']:.2f} | {x['n12_strict']:.2f} | {x['n12_gold_beats_foil']:.2f} @{x['n12_tokens']:.0f}  (n={x['n']})")


USERFACT_CHECKS = {"recall", "name", "city", "pet", "corrected", "binding", "coref", "user_job", "fear", "party_math", "topic_recall"}
STALE = {"K_day": r"\bmonday\b", "K_time": r"(?<![\d:])3(?![\d])\s*(pm|p\.m|o'clock|:00)|\bthree\b", "K_color": r"\bblue\b"}


def taxonomy(mode="greedy", include_controls=False):
    """Classify every failed user-fact check (first matching label wins)."""
    out = {}
    for m, R in load(mode).items():
        c = Counter()
        for r in R:
            if (r["cat"] == "control") != include_controls or r["cat"] in ("loop",) and r["id"] != "L_chitchat":
                continue
            t = TESTS[r["id"]]
            replies = [x["assistant"] for x in r["turns"]]
            for ch in t["checks"]:
                if ch["name"] not in USERFACT_CHECKS or r["checks"].get(ch["name"]) is not False:
                    continue
                a = replies[ch["turn"]]
                g8 = set(ngrams(toks(a), 8))
                prev = set().union(*[set(ngrams(toks(x), 8)) for x in replies[:ch["turn"]]]) if ch["turn"] else set()
                if g8 and prev and len(g8 & prev) / len(g8) >= 0.5:
                    lab = "copy_of_own_earlier_reply"
                elif B.deflects(a):
                    lab = "deflection"
                elif ("gold" in ch and B.mentions(ch["gold"], a) and B.captures(ch["gold"], a)) or \
                        (re.search(r"\b(i|i'm|my)\b", B.unquote(a), re.I) and not re.search(r"\b(you|your)\b", a, re.I)):
                    lab = "perspective_error (answers as the user or about itself)"
                elif r["id"] in STALE and B.asserted(STALE[r["id"]], a):
                    lab = "stale_value"
                elif "gold" in ch and B.mentions(ch["gold"], a):
                    lab = "gold_only_in_greeting_or_other"
                else:
                    lab = "wrong_or_ignored"
                c[lab] += 1
        out[m] = dict(c)
    return out


if __name__ == "__main__":
    tx = taxonomy("greedy")
    results["failure_taxonomy_greedy_multiturn"] = tx
    txs = taxonomy("sampled")
    results["failure_taxonomy_sampled_multiturn"] = txs
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("\n## failure taxonomy (greedy, multi-turn user-fact checks)")
    for m in ORDER:
        if m in tx:
            print(f"{SHORT[m]:<20} {tx[m]}")
