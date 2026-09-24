"""Summarize E001 likelihood runs (../out/*.jsonl) into ../battery_results.json and print tables.

New control items: accuracy (gold strictly above every other candidate) per variant, render
and distance, paired metrics, what the model picks, and mean margins.
Old items (for models first probed in E001): the REPORT section 3.3 columns, plus khard.
The 9 models of the original battery are read from research/capacity_probe/results*.json
(read only) so the old-item table covers everything.
"""
import glob, json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import metrics as M

RESEARCH_CAP = "/Users/brohm/Documents/Projects/maxgpt-planck/research/capacity_probe"
ORDER = ["tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct",
         "LiquidAI/LFM2.5-230M", "maxgpt3:final", "maxgpt3:final_sft", "unsloth/gemma-3-270m-it", "LiquidAI/LFM2-350M",
         "LiquidAI/LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B",
         "LiquidAI/LFM2-700M", "LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-2.6B"]
SHORT = {"tiiuae/Falcon-H1-Tiny-90M-Instruct": "Falcon-90M-I", "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M-base",
         "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M-I", "LiquidAI/LFM2.5-230M": "LFM2.5-230M",
         "maxgpt3:final": "MaxGPT-3 base", "maxgpt3:final_sft": "MaxGPT-3 SFT", "unsloth/gemma-3-270m-it": "Gemma3-270M-it",
         "LiquidAI/LFM2-350M": "LFM2-350M", "LiquidAI/LFM2.5-350M": "LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M-I",
         "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-I", "Qwen/Qwen3-0.6B": "Qwen3-0.6B", "LiquidAI/LFM2-700M": "LFM2-700M",
         "LiquidAI/LFM2-1.2B": "LFM2-1.2B", "LiquidAI/LFM2-2.6B": "LFM2-2.6B"}


def load_out():
    runs = defaultdict(dict)  # model -> (set, render) -> {"meta":..., "recs": [...]}
    for f in sorted(glob.glob(os.path.join(EXP, "out", "*.jsonl"))):
        meta, recs, skipped = None, [], 0
        for line in open(f):
            r = json.loads(line)
            if r.get("meta"):
                meta = r
            elif "skipped" in r:
                skipped += 1
            else:
                recs.append(r)
        if meta is None or meta.get("dry"):
            continue
        runs[meta["model"]][(meta["set"], meta["render"])] = {"meta": meta, "recs": recs, "skipped": skipped, "file": f}
    return runs


def acc_of(s, key):
    v = s.get(key)
    return None if v is None else v["acc"]


def f2(v):
    return "-" if v is None else f"{v:.2f}"


def new_tables(runs, res):
    VARS = ["same_k1", "same_k2", "same_k3", "pos", "keyorig", "keycorr", "neutral", "noupd", "twoslot"]
    PAIRS = ["PAIR_upd_bind", "PAIR_key_pair", "PAIR_all5"]
    models = [m for m in ORDER if any(k[0] == "new" for k in runs.get(m, {}))]
    summ = {}
    for m in models:
        recs = []
        for (st, rd), v in runs[m].items():
            if st == "new":
                recs += v["recs"]
        summ[m] = M.summarize_new(recs)
        res.setdefault(m, {})["new_items"] = summ[m]
        res[m]["new_items_n_records"] = len(recs)
    for render in ("plain", "chat"):
        for dkey, dlabel in (("d4-10", "d4 and d10 pooled"), ("0", "d0")):
            print(f"\n### New control items, {render} render, {dlabel}, both families (n = {'128' if dkey == 'd4-10' else '64'} per cell)")
            print("| model | " + " | ".join(VARS) + " | upd+noupd pair | key pair | all 5 |")
            print("|---|" + "---|" * (len(VARS) + 3))
            for m in models:
                s = summ[m]
                if f"{render}|same_k1|all|0" not in s:
                    continue
                row = [f2(acc_of(s, f"{render}|{v}|all|{dkey}")) for v in VARS + PAIRS]
                print(f"| {SHORT.get(m, m)} | " + " | ".join(row) + " |")
    # attribution indices (d4-10, both families)
    print("\n### Attribution, d4-d10 pooled (plain unless noted). Differences are accuracy points.")
    print("| model | same_k1 plain | same_k1 chat | chat - plain (same_k1) | chat - plain (all 5) | keycorr - keyorig | pos - same_k1 | neutral - keyorig | noupd | twoslot picks: gold / orig (first) / other (latest) | twoslot picks at d0 | same_k1 wrong picks orig | mean margin same_k1 (nats) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    attrib = {}
    for m in models:
        s = summ[m]
        g = lambda k: acc_of(s, k)
        def diff(a, b):
            return None if a is None or b is None else round(a - b, 3)
        pk = s.get("picks|plain|twoslot|all|4", {}), s.get("picks|plain|twoslot|all|10", {})
        tot = sum(sum(p.values()) for p in pk) or 1
        tw = {k: round(sum(p.get(k, 0) for p in pk) / tot, 2) for k in ("gold", "orig", "other")}
        p0 = s.get("picks|plain|twoslot|all|0", {})
        t0 = sum(p0.values()) or 1
        tw0 = {k: round(p0.get(k, 0) / t0, 2) for k in ("gold", "orig", "other")}
        ps = s.get("picks|plain|same_k1|all|4", {}), s.get("picks|plain|same_k1|all|10", {})
        wrong = sum(p.get("orig", 0) + p.get("tie", 0) for p in ps)
        worig = sum(p.get("orig", 0) for p in ps)
        mm = [s.get(f"margin|plain|same_k1|all|{d}") for d in (4, 10)]
        mm = None if None in mm else round(sum(mm) / 2, 2)
        a = dict(same_plain=g("plain|same_k1|all|d4-10"), same_chat=g("chat|same_k1|all|d4-10"),
                 format_same=diff(g("chat|same_k1|all|d4-10"), g("plain|same_k1|all|d4-10")),
                 format_all5=diff(g("chat|PAIR_all5|all|d4-10"), g("plain|PAIR_all5|all|d4-10")),
                 key=diff(g("plain|keycorr|all|d4-10"), g("plain|keyorig|all|d4-10")),
                 position=diff(g("plain|pos|all|d4-10"), g("plain|same_k1|all|d4-10")),
                 prefix=diff(g("plain|neutral|all|d4-10"), g("plain|keyorig|all|d4-10")),
                 noupd=g("plain|noupd|all|d4-10"), twoslot_picks=tw, twoslot_picks_d0=tw0,
                 same_wrong_orig_share=(round(worig / wrong, 2) if wrong else None), same_margin=mm)
        attrib[m] = a
        res[m]["attribution"] = a
        print(f"| {SHORT.get(m, m)} | {f2(a['same_plain'])} | {f2(a['same_chat'])} | {f2(a['format_same'])} | {f2(a['format_all5'])} | "
              f"{f2(a['key'])} | {f2(a['position'])} | {f2(a['prefix'])} | {f2(a['noupd'])} | {tw['gold']:.2f} / {tw['orig']:.2f} / {tw['other']:.2f} | {tw0['gold']:.2f} / {tw0['orig']:.2f} / {tw0['other']:.2f} | "
              f"{f2(a['same_wrong_orig_share'])} | {'-' if mm is None else f'{mm:+.2f}'} |")
    # per family, plain, d4-10
    print("\n### By item family, plain, d4-d10 (day = dentist weekday, color = car colour; n = 64 per cell)")
    print("| model | fam | same_k1 | pos | keyorig | keycorr | noupd | twoslot | all 5 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        for fam in ("day", "color"):
            s = summ[m]
            row = [f2(acc_of(s, f"plain|{v}|{fam}|d4-10")) for v in ("same_k1", "pos", "keyorig", "keycorr", "noupd", "twoslot", "PAIR_all5")]
            print(f"| {SHORT.get(m, m)} | {fam} | " + " | ".join(row) + " |")
    # U_same by k and distance (plain), the REPORT's latest-wins grid on the clean item
    print("\n### U_same (identical wording) latest-wins, plain: k = 1 / 2 / 3 corrections")
    print("| model | d0 | d4 | d10 |")
    print("|---|---|---|---|")
    for m in models:
        s = summ[m]
        cells = []
        for d in (0, 4, 10):
            cells.append(" / ".join(f2(acc_of(s, f"plain|same_k{k}|all|{d}")) for k in (1, 2, 3)))
        print(f"| {SHORT.get(m, m)} | " + " | ".join(cells) + " |")
    return attrib


def old_tables(runs, res):
    """REPORT 3.3 columns for every model: research results for the 9 old models, E001 for new ones."""
    old_res = json.load(open(os.path.join(RESEARCH_CAP, "results.json")))
    old_kh = json.load(open(os.path.join(RESEARCH_CAP, "results_khard.json")))
    rows = {}
    for m in ORDER:
        src = None
        if m in runs and ("old", "plain") in runs[m]:
            s = M.summarize_old(runs[m][("old", "plain")]["recs"])
            src = "E001"
            kh = runs[m].get(("khard", "plain"))
            khd = None
            if kh:
                khd = {t: M.acc([x["scores"]["gold"] > x["scores"]["foil"] for x in kh["recs"] if x["task"] == t])
                       for t in ("K4_closed", "K4_open")}
            res.setdefault(m, {})["old_items"] = s
            res[m]["khard"] = khd
            if ("old", "chat") in runs[m]:
                res[m]["old_items_own_format"] = M.summarize_old(runs[m][("old", "chat")]["recs"])
                kc = runs[m].get(("khard", "chat"))
                if kc:
                    res[m]["khard_own_format"] = {t: M.acc([x["scores"]["gold"] > x["scores"]["foil"] for x in kc["recs"] if x["task"] == t])
                                                  for t in ("K4_closed", "K4_open")}
        elif m in old_res:
            s = old_res[m]
            src = "research"
            khd = old_kh.get(m)
        else:
            continue
        g = lambda k: (s.get(k) or {}).get("acc") if isinstance(s.get(k), dict) else None
        two = [g(f"R2_twohop|{d}|all") for d in (0, 4, 10)]
        two = None if None in two else round(sum(two) / 3, 3)
        lw = []
        for k in (1, 2, 3):
            for d in (4, 10):
                c = s.get(f"U_k{k}|{d}|all")
                if isinstance(c, dict):
                    lw.append((c["acc"], c["n"]))
        lwv = round(sum(a * n for a, n in lw) / sum(n for _, n in lw), 3) if lw else None
        rows[m] = dict(src=src, owner_pair10=g("R1_both|10|all"), onehop_pair10=g("R2one_both|10|all"), twohop_all=two,
                       persp_pair10=g("P_both|10|all"), latest_wins_4_10=lwv, K=g("K_closedbook|0|all"),
                       khard_closed=(khd or {}).get("K4_closed", {}).get("acc") if khd else None,
                       khard_open=(khd or {}).get("K4_open", {}).get("acc") if khd else None,
                       U_d0=" / ".join(f2(g(f"U_k{k}|0|all")) for k in (1, 2, 3)))
    print("\n### Old battery items (REPORT 3.3 columns); src = which run the numbers come from")
    print("| model | src | owner pair @10 | one-hop pair @10 | two-hop, all d | perspective pair @10 | latest wins @4-10 (old U) | old U d0, k=1/2/3 | K closed-book | long-tail closed / open |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, r in rows.items():
        print(f"| {SHORT.get(m, m)} | {r['src']} | {f2(r['owner_pair10'])} | {f2(r['onehop_pair10'])} | {f2(r['twohop_all'])} | "
              f"{f2(r['persp_pair10'])} | {f2(r['latest_wins_4_10'])} | {r['U_d0']} | {f2(r['K'])} | {f2(r['khard_closed'])} / {f2(r['khard_open'])} |")
    res["_old_item_table"] = rows
    for m in ("maxgpt3:final", "maxgpt3:final_sft"):
        if m in res and "old_items_own_format" in res[m]:
            s = res[m]["old_items_own_format"]
            g = lambda k: (s.get(k) or {}).get("acc") if isinstance(s.get(k), dict) else None
            print(f"{SHORT[m]} in its own USER:/ASSISTANT: format: owner pair @10 {f2(g('R1_both|10|all'))}, one-hop pair @10 "
                  f"{f2(g('R2one_both|10|all'))}, perspective pair @10 {f2(g('P_both|10|all'))}, K {f2(g('K_closedbook|0|all'))}, "
                  f"khard {res[m].get('khard_own_format')}")


def main():
    runs = load_out()
    res = {}
    for m, v in runs.items():
        res.setdefault(m, {})["runs"] = {f"{a}|{b}": dict(n=len(x["recs"]), skipped=x["skipped"], file=os.path.basename(x["file"]),
                                                          dtype=x["meta"].get("dtype"), device=x["meta"].get("device"),
                                                          params=x["meta"].get("params"), emb=x["meta"].get("emb"))
                                         for (a, b), x in v.items()}
    new_tables(runs, res)
    old_tables(runs, res)
    json.dump(res, open(os.path.join(EXP, "battery_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
