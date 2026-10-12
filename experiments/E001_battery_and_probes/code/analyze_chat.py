"""Regrade chat-probe transcripts with the E001 hardened graders (and, for comparison, the
original graders), and write ../chat_results.json.

Sources: research/probe/transcripts/*.jsonl (the 8 models probed on 2026-09-23, read only)
and ../transcripts/*.jsonl (E001 runs). An E001 transcript for the same model and mode
replaces the old one in the tables; both are kept in the JSON.
Scores: ability = mean over GRADABLE checks in the category (truncated fails excluded);
macro = mean of the 7 multi-turn categories. "old" = original graders on the same replies.
"""
import glob, importlib.util, json, os, sys
from collections import defaultdict, Counter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RESEARCH = "/Users/brohm/Documents/Projects/maxgpt-planck/research/probe"
sys.path.insert(0, HERE)
import battery as B

spec = importlib.util.spec_from_file_location("battery_orig", os.path.join(RESEARCH, "battery.py"))
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)

CATS = ["recall", "followup", "instruction", "correction", "own_answer", "role", "topic_return"]
ORDER = ["tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "LiquidAI/LFM2.5-230M",
         "unsloth/gemma-3-270m-it", "LiquidAI/LFM2-350M", "LiquidAI/LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct",
         "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"]
SHORT = {"tiiuae/Falcon-H1-Tiny-90M-Instruct": "Falcon-90M", "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M",
         "LiquidAI/LFM2.5-230M": "LFM2.5-230M", "unsloth/gemma-3-270m-it": "Gemma3-270M", "LiquidAI/LFM2-350M": "LFM2-350M",
         "LiquidAI/LFM2.5-350M": "LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M",
         "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B", "Qwen/Qwen3-0.6B": "Qwen3-0.6B"}
NEW_T = {t["id"]: t for t in B.TESTS}
OLD_T = {t["id"]: t for t in OLD.TESTS}
USERFACT = {"recall", "name", "city", "pet", "corrected", "binding", "coref", "user_job", "fear", "topic_recall"}


def regrade(r):
    replies = [x["assistant"] for x in r["turns"]]
    new, old, trunc = {}, {}, []
    for c in NEW_T[r["id"]]["checks"]:
        v, tr = B.grade(c, replies, hit_max=r["turns"][c["turn"]]["flags"]["hit_max"])
        new[c["name"]] = v
        if tr:
            trunc.append((c["name"], v))
    for c in OLD_T[r["id"]]["checks"]:
        v = c["fn"](replies[c["turn"]], replies)
        old[c["name"]] = None if v is None else bool(v)
    return new, old, trunc


def load(source):
    recs = defaultdict(lambda: defaultdict(list))  # model -> mode -> records
    for f in sorted(glob.glob(os.path.join(source, "*__greedy.jsonl")) + glob.glob(os.path.join(source, "*__sampled.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[r["model"]][r["mode"]].append(r)
    return recs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(R):
    abil_new, abil_old = defaultdict(list), defaultdict(list)
    flips, n_ungraded, n_capped_fail, n_capped_replies, n_replies = [], 0, 0, 0, 0
    defl = []
    forced = defaultdict(list)
    for r in R:
        new, old, trunc = regrade(r)
        n_ungraded += sum(v is None for _, v in trunc)
        n_capped_fail += sum(v is False for _, v in trunc)
        for x in r["turns"]:
            n_replies += 1
            n_capped_replies += int(x["flags"]["hit_max"])
        for k, v in new.items():
            abil_new[r["cat"]].append(None if v is None else float(v))
            ov = old.get(k)
            abil_old[r["cat"]].append(None if ov is None else float(ov))
            if ov is not None and v is not None and bool(ov) != bool(v):
                c = [c for c in NEW_T[r["id"]]["checks"] if c["name"] == k][0]
                flips.append(dict(id=r["id"], seed=r["seed"], check=k, old=bool(ov), new=bool(v),
                                  reply=r["turns"][c["turn"]]["assistant"][:300]))
            elif ov is not None and v is None:
                c = [c for c in NEW_T[r["id"]]["checks"] if c["name"] == k][0]
                flips.append(dict(id=r["id"], seed=r["seed"], check=k, old=bool(ov), new=None,
                                  reply=r["turns"][c["turn"]]["assistant"][-200:]))
        if r["cat"] in ("recall", "correction", "role"):
            for fc in r.get("forced", []):
                if fc.get("greedy_gold") is not None:
                    forced[r["cat"]].append((float(fc["greedy_gold"]), float((fc["margin"] or 0) > 0)))
        if r["cat"] in ("recall", "correction", "role", "topic_return") or (
                r["cat"] == "control" and (r["control_for"] or "")[:2] in ("R_", "K_", "RI", "T_")):
            defl.append(float(B.deflects(r["turns"][-1]["assistant"])))
    out = {"ability_new": {c: mean(abil_new[c]) for c in CATS + ["control"]},
           "ability_old": {c: mean(abil_old[c]) for c in CATS + ["control"]},
           "n_gradable_new": {c: sum(v is not None for v in abil_new[c]) for c in CATS + ["control"]},
           "n_checks": {c: len(abil_new[c]) for c in CATS + ["control"]}}
    out["macro_new"] = mean([out["ability_new"][c] for c in CATS])
    out["macro_old"] = mean([out["ability_old"][c] for c in CATS])
    out["ungraded_truncated_checks"] = n_ungraded
    out["capped_content_fails"] = n_capped_fail
    out["capped_reply_rate"] = round(n_capped_replies / max(n_replies, 1), 3)
    out["deflection_rate_final_userfact"] = mean(defl)
    out["forced_prefix"] = {c: {"greedy_gold": mean([a for a, b in v]), "gold_beats_foil": mean([b for a, b in v]), "n": len(v)}
                            for c, v in forced.items()}
    out["flips"] = flips
    out["flip_counts"] = dict(Counter(("pass->fail" if f["old"] and f["new"] is False else
                                       "fail->pass" if (not f["old"]) and f["new"] else
                                       "pass->ungraded" if f["old"] and f["new"] is None else "fail->ungraded") for f in flips))
    return out


def main():
    old_src, new_src = load(os.path.join(RESEARCH, "transcripts")), load(os.path.join(EXP, "transcripts"))
    res = {"research_transcripts": {}, "e001_transcripts": {}}
    for tag, src in (("research_transcripts", old_src), ("e001_transcripts", new_src)):
        for m, modes in src.items():
            res[tag][m] = {mode: summarize(R) for mode, R in modes.items()}
    # table: prefer E001 transcripts for a model/mode when they exist
    best = {}
    for m in ORDER:
        for mode in ("greedy", "sampled"):
            if m in res["e001_transcripts"] and mode in res["e001_transcripts"][m]:
                best[(m, mode)] = ("E001", res["e001_transcripts"][m][mode])
            elif m in res["research_transcripts"] and mode in res["research_transcripts"][m]:
                best[(m, mode)] = ("research", res["research_transcripts"][m][mode])
    res["table_source"] = {f"{m}|{mode}": v[0] for (m, mode), v in best.items()}
    json.dump(res, open(os.path.join(EXP, "chat_results.json"), "w"), indent=1)
    for mode in ("greedy", "sampled"):
        print(f"\n## chat probe, {mode}, hardened graders (old graders' macro in brackets)")
        print("| model | src | recall | follow-up | instruction | correction | own answer | role | topic return | macro new [old] | controls | end_q ungraded (capped) | fails on capped replies | deflect |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in ORDER:
            if (m, mode) not in best:
                continue
            src, s = best[(m, mode)]
            a = s["ability_new"]
            f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"| {SHORT[m]} | {src} | " + " | ".join(f(a[c]) for c in CATS) +
                  f" | {f(s['macro_new'])} [{f(s['macro_old'])}] | {f(a['control'])} | {s['ungraded_truncated_checks']} | {s['capped_content_fails']} | {f(s['deflection_rate_final_userfact'])} |")
        print("forced answer prefix on multi-turn items (greedy emits gold / gold beats foil):")
        for m in ORDER:
            if (m, mode) in best:
                fp = best[(m, mode)][1]["forced_prefix"]
                print("  " + SHORT[m] + ": " + "; ".join(f"{c} {f(v['greedy_gold'])} / {f(v['gold_beats_foil'])} (n={v['n']})" for c, v in sorted(fp.items())))
        print("flip counts (old grader -> new grader):")
        for m in ORDER:
            if (m, mode) in best:
                print(" ", SHORT[m], best[(m, mode)][1]["flip_counts"])


if __name__ == "__main__":
    main()
