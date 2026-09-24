"""Print the markdown tables used in lanes/probe.md, straight from results.json (run analyze.py first)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
ORDER = ["tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "unsloth/gemma-3-270m-it",
         "LiquidAI/LFM2-350M", "LiquidAI/LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct",
         "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"]
SHORT = {"tiiuae/Falcon-H1-Tiny-90M-Instruct": "Falcon-H1-Tiny-90M", "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M",
         "unsloth/gemma-3-270m-it": "Gemma-3-270M", "LiquidAI/LFM2-350M": "LFM2-350M", "LiquidAI/LFM2.5-350M": "LFM2.5-350M",
         "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M", "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen/Qwen3-0.6B": "Qwen3-0.6B"}
CATS = ["recall", "followup", "instruction", "correction", "own_answer", "role", "topic_return"]


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def ability(mode):
    g = R[mode]
    print(f"\n### ability scores, {mode} (strict, fraction of checks passed)\n")
    print("| model | params | " + " | ".join(CATS) + " | macro | single-turn controls |")
    print("|" + "---|" * (len(CATS) + 4))
    for m in ORDER:
        if m not in g:
            continue
        r = g[m]
        print(f"| {SHORT[m]} | {r['params_m']}M | " + " | ".join(f(r['ability_strict'][c]) for c in CATS)
              + f" | **{f(r['overall_multiturn_macro'])}** | {f(r['ability_strict']['control'])} |")


def recall_distance(modes):
    print(f"\n### main-battery recall by distance ({'+'.join(modes)}; strict free answer | forced-prefix gold>foil)\n")
    ds = ["0", "2", "4", "5", "6", "10"]
    print("| model | " + " | ".join(f"d={d}" for d in ds) + " |")
    print("|" + "---|" * (len(ds) + 1))
    for m in ORDER:
        cells = []
        for d in ds:
            vals, n, gb = [], 0, []
            for mode in modes:
                x = R[mode].get(m, {}).get("recall_by_distance", {}).get(d)
                if x:
                    vals.append((x["strict"], x["n"]))
                    if x.get("gold_beats_foil") is not None:
                        gb.append(x["gold_beats_foil"])
            if not vals:
                cells.append("-")
                continue
            tot = sum(v * k for v, k in vals) / sum(k for _, k in vals)
            n = sum(k for _, k in vals)
            cells.append(f"{tot:.2f} (n={n})" + (f" / {sum(gb)/len(gb):.2f}" if gb else ""))
        if any(c != "-" for c in cells):
            print(f"| {SHORT[m]} | " + " | ".join(cells) + " |")


def e2():
    P = R.get("fixed_history_pooled", {})
    print("\n### fixed-history sweep, pooled over 1-12 distractor turns (n=20 per cell)\n")
    print("| model / format | replies | strict recall | deflect | role capture | greedy-from-prefix = gold | gold > foil | strict @12 turns | tokens @12 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, d in P.items():
        for style in ("short", "long"):
            if style in d:
                x = d[style]
                print(f"| {k} | {style} | {f(x['strict'])} | {f(x['deflect'])} | {f(x['capture'])} | {f(x['greedy_gold'])} | "
                      f"{f(x['gold_beats_foil'])} | {f(x['n12_strict'])} | {f(x['n12_tokens'], 0)} |")


def misc(mode):
    g = R[mode]
    print(f"\n### per-turn behavior, {mode}\n")
    print("| model | deflection on final user-fact turns | self-copy rate | replies with a 4-gram x3 | 'tell me more' cross-turn 4-gram overlap | instruction t1/t2/t3 | persona 'Arr' t0..t3 |")
    print("|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in g:
            continue
        r = g[m]
        p = r["instruction_persistence_by_turn"]
        print(f"| {SHORT[m]} | {f(r['deflection_rate_on_final_recall_turns'])} | {f(r['self_copy_rate'])} | {f(r['replies_with_4gram_repeated_3x'])} | "
              f"{f(r['loops']['L_tell_more']['cross_turn_4gram_overlap_mean'])} | {f(p.get('t1'))}/{f(p.get('t2'))}/{f(p.get('t3'))} | "
              f"{f(p.get('persona_t0'))}/{f(p.get('persona_t1'))}/{f(p.get('persona_t2'))}/{f(p.get('persona_t3'))} |")


def corrections(mode):
    g = R[mode]
    print(f"\n### forced-choice margin, corrected minus stale value (log-prob), {mode}\n")
    ids = ["K_day", "K_time", "K_color"]
    print("| model | " + " | ".join(f"{i} multi / single" for i in ids) + " | RI_binding multi / single (Oscar minus Lena) |")
    print("|" + "---|" * (len(ids) + 2))
    for m in ORDER:
        if m not in g:
            continue
        fc = g[m]["forced_choice_corrections_binding"]
        cells = [f"{f(fc.get(i, {}).get('mean_margin'),1)} / {f(fc.get(i + '__ctrl', {}).get('mean_margin'),1)}" for i in ids + ["RI_binding"]]
        print(f"| {SHORT[m]} | " + " | ".join(cells) + " |")


def ctxcap(mode):
    g = R[mode]
    print(f"\n### context vs capability ({mode}; each multi-turn check paired with its single-turn control)\n")
    print("| model | both pass | context failure (control passes, multi fails) | capability failure (both fail) | multi-only pass |")
    print("|---|---|---|---|---|")
    for m in ORDER:
        if m not in g:
            continue
        c = g[m]["context_vs_capability"]
        print(f"| {SHORT[m]} | {c.get('both_pass', 0)} | {c.get('context_failure', 0)} | {c.get('capability_failure', 0)} | {c.get('multi_only_pass', 0)} |")


def taxonomy(key):
    T = R.get(key, {})
    labels = ["deflection", "perspective_error (answers as the user or about itself)", "stale_value", "copy_of_own_earlier_reply",
              "gold_only_in_greeting_or_other", "wrong_or_ignored"]
    print(f"\n### failure taxonomy ({key})\n")
    print("| model | " + " | ".join(l.split(" (")[0] for l in labels) + " | total |")
    print("|" + "---|" * (len(labels) + 2))
    tot = {l: 0 for l in labels}
    for m in ORDER:
        if m not in T:
            continue
        row = T[m]
        for l in labels:
            tot[l] += row.get(l, 0)
        print(f"| {SHORT[m]} | " + " | ".join(str(row.get(l, 0)) for l in labels) + f" | {sum(row.values())} |")
    print("| all | " + " | ".join(str(tot[l]) for l in labels) + f" | {sum(tot.values())} |")


if __name__ == "__main__":
    ability("greedy"); ability("sampled")
    recall_distance(["greedy"]); recall_distance(["greedy", "sampled"])
    e2(); misc("greedy"); misc("sampled"); corrections("greedy"); ctxcap("greedy"); ctxcap("sampled")
    taxonomy("failure_taxonomy_greedy_multiturn"); taxonomy("failure_taxonomy_sampled_multiturn")


def e2_curves():
    F = R.get("fixed_history", {})
    print("\n### fixed-history sweep by number of distractor turns (4 facts per cell): strict free recall / greedy-from-prefix emits gold\n")
    ns = ["0", "1", "2", "4", "8", "12"]
    print("| model | replies | " + " | ".join(f"n={n}" for n in ns) + " |")
    print("|" + "---|" * (len(ns) + 2))
    for k, agg in F.items():
        for style in ("short", "long"):
            cells = []
            for n in ns:
                x = agg.get(f"{style}_n{n}")
                cells.append("-" if not x else f"{x['strict']:.2f} / {x['greedy_gold_from_prefix']:.2f}")
            if any(c != "-" for c in cells):
                tok = agg.get(f"{style}_n12", {}).get("prompt_tokens")
                print(f"| {k} | {style} (n=12: {tok:.0f} tok) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    e2_curves()
