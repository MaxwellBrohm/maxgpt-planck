"""Summarize out/*.jsonl into a table per model x task x distance, write results.json."""
import json, glob, os, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["tiiuae/Falcon-H1-Tiny-90M-Instruct", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct",
         "unsloth/gemma-3-270m-it", "LiquidAI/LFM2-350M", "LiquidAI/LFM2.5-350M", "HuggingFaceTB/SmolLM2-360M-Instruct",
         "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B", "LiquidAI/LFM2-2.6B",
         "HuggingFaceTB/SmolLM2-135M-Instruct+FT", "HuggingFaceTB/SmolLM2-360M-Instruct+FT"]
SHORT = {"tiiuae/Falcon-H1-Tiny-90M-Instruct": "Falcon-90M-I", "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M-base",
         "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M-I", "unsloth/gemma-3-270m-it": "Gemma3-270M-it",
         "LiquidAI/LFM2-350M": "LFM2-350M", "LiquidAI/LFM2.5-350M": "LFM2.5-350M",
         "HuggingFaceTB/SmolLM2-360M-Instruct": "SmolLM2-360M-I", "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-I",
         "Qwen/Qwen3-0.6B": "Qwen3-0.6B", "LiquidAI/LFM2-2.6B": "LFM2-2.6B",
         "HuggingFaceTB/SmolLM2-135M-Instruct+FT": "SmolLM2-135M-I +FT", "HuggingFaceTB/SmolLM2-360M-Instruct+FT": "SmolLM2-360M-I +FT"}


def load(path):
    meta, recs = None, []
    for line in open(path):
        r = json.loads(line)
        if r.get("meta"):
            meta = r
        else:
            recs.append(r)
    return meta, recs


def acc(xs):
    n = len(xs)
    if n == 0:
        return None
    p = sum(xs) / n
    return {"acc": round(p, 3), "n": n, "se": round(math.sqrt(max(p * (1 - p), 1e-9) / n), 3)}


def summarize(recs):
    cells = defaultdict(list)
    for r in recs:
        s, t, d = r["scores"], r["task"], r["d"]
        g = s["gold"]
        others = {k: v for k, v in s.items() if k != "gold"}
        cells[(t, d, "all")].append(g > max(others.values()))
        for k, v in others.items():
            cells[(t, d, "vs_" + k)].append(g > v)
        if t == "K_closedbook":
            cells[(t, r["cond"], "all")].append(g > max(others.values()))
        margin = g - max(others.values())
        cells[(t, d, "margin")].append(margin)
    # paired "both right" metrics: a model that always picks the first-mentioned (or the
    # most recent) value gets ~0 here, a coin-flipper ~0.25
    for a_, b_ in zip(recs, recs[1:]):
        ok = lambda r: r["scores"]["gold"] > max(v for k, v in r["scores"].items() if k != "gold")
        if a_["task"] == "R1_owner" and a_["cond"] == "mine" and b_["task"] == "R1_owner" and b_["cond"] == "sister":
            cells[("R1_both", a_["d"], "all")].append(ok(a_) and ok(b_))
        if a_["task"] == "P_myname" and b_["task"] == "P_yourname":
            cells[("P_both", a_["d"], "all")].append(ok(a_) and ok(b_))
    for i in range(len(recs) - 3):
        w = recs[i:i + 4]
        if [x["task"] for x in w] == ["R2_twohop", "R2_onehop", "R2_twohop", "R2_onehop"]:
            ok = lambda r: r["scores"]["gold"] > r["scores"]["foil"]
            cells[("R2two_both", w[0]["d"], "all")].append(ok(w[0]) and ok(w[2]))
            cells[("R2one_both", w[0]["d"], "all")].append(ok(w[1]) and ok(w[3]))
    out = {}
    for (t, d, m), xs in cells.items():
        key = f"{t}|{d}|{m}"
        if m == "margin":
            out[key] = round(sum(xs) / len(xs), 3)
        else:
            out[key] = acc([bool(x) for x in xs])
    return out


def main():
    res = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "*.jsonl"))):
        if os.path.basename(path).startswith("khard__"):
            continue
        meta, recs = load(path)
        if meta is None or not recs:
            continue
        s = summarize(recs)
        s["_params_M"] = round(meta["params"] / 1e6, 1)
        s["_nonemb_M"] = round((meta["params"] - meta["emb"]) / 1e6, 1)
        s["_n_records"] = len(recs)
        s["_split_tokenization"] = sum(r["split"] for r in recs)
        s["_max_prompt_tokens"] = max(r["prompt_tokens"] for r in recs)
        res[meta["model"]] = s
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)

    models = [m for m in ORDER if m in res]
    cols = [("K", "K_closedbook", 0, "all"),
            ("R1 d0", "R1_owner", 0, "all"), ("R1 d10", "R1_owner", 10, "all"),
            ("1hop d10", "R2_onehop", 10, "all"),
            ("2hop d0", "R2_twohop", 0, "all"), ("2hop d4", "R2_twohop", 4, "all"), ("2hop d10", "R2_twohop", 10, "all"),
            ("U1 d10", "U_k1", 10, "all"), ("U2 d10", "U_k2", 10, "all"), ("U3 d0", "U_k3", 0, "all"), ("U3 d10", "U_k3", 10, "all"),
            ("U3 vs orig d10", "U_k3", 10, "vs_orig"), ("U3 vs prev d10", "U_k3", 10, "vs_prev"),
            ("myname d10", "P_myname", 10, "all"), ("myname vs asst d10", "P_myname", 10, "vs_assistant"),
            ("myname vs sis d10", "P_myname", 10, "vs_sister"), ("yourname d10", "P_yourname", 10, "all")]
    hdr = "| model | total M | non-emb M | " + " | ".join(c[0] for c in cols) + " |"
    print(hdr); print("|" + "---|" * (3 + len(cols)))
    for m in models:
        s = res[m]
        row = [SHORT.get(m, m), str(s["_params_M"]), str(s["_nonemb_M"])]
        for _, t, d, k in cols:
            v = s.get(f"{t}|{d}|{k}")
            row.append("" if v is None else f"{v['acc']:.2f}")
        print("| " + " | ".join(row) + " |")
    print()
    print("knowledge by tier:")
    for m in models:
        s = res[m]
        print(SHORT.get(m, m), [s.get(f"K_closedbook|tier{t}|all", {}).get("acc") for t in (1, 2, 3)],
              "split-tok:", s["_split_tokenization"], "max prompt tok:", s["_max_prompt_tokens"])
    print()
    print("all tasks x distance (acc all-foils):")
    tasks = ["R1_owner", "R1_both", "R2_onehop", "R2one_both", "R2_twohop", "R2two_both", "U_k1", "U_k2", "U_k3", "P_myname", "P_yourname", "P_both"]
    for m in models:
        s = res[m]
        parts = []
        for t in tasks:
            parts.append(t + ":" + "/".join(f"{s[f'{t}|{d}|all']['acc']:.2f}" if f"{t}|{d}|all" in s else "-" for d in (0, 4, 10)))
        print(SHORT.get(m, m), " ".join(parts))


def khard():
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "khard__*.jsonl")) + glob.glob(os.path.join(HERE, "out", "ftk__*.jsonl"))):
        b = os.path.basename(path)
        m = b.split("__", 1)[1][:-len(".jsonl")].replace("__", "/") + ("+FT" if b.startswith("ftk__") else "")
        recs = [json.loads(l) for l in open(path)]
        r = {}
        for t in ("K4_closed", "K4_open"):
            xs = [x["scores"]["gold"] > x["scores"]["foil"] for x in recs if x["task"] == t]
            r[t] = acc(xs)
        rows[m] = r
    if rows:
        print()
        print("| model | long-tail closed book | same facts, open book |")
        print("|---|---|---|")
        for m in ORDER:
            if m in rows:
                print(f"| {SHORT.get(m, m)} | {rows[m]['K4_closed']['acc']:.2f} | {rows[m]['K4_open']['acc']:.2f} |")
        json.dump(rows, open(os.path.join(HERE, "results_khard.json"), "w"), indent=1)


def uprobe():
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "uprobe__*.jsonl")) + glob.glob(os.path.join(HERE, "out", "ftu__*.jsonl"))):
        b = os.path.basename(path)
        m = b.split("__", 1)[1][:-len(".jsonl")].replace("__", "/") + ("+FT" if b.startswith("ftu__") else "")
        cells = defaultdict(list)
        for l in open(path):
            x = json.loads(l)
            s = x["scores"]; g = s["gold"]
            cells[(x["task"], x["d"])].append(g > max(v for k, v in s.items() if k != "gold"))
        rows[m] = {f"{t}|{d}": acc(v) for (t, d), v in cells.items()}
    if rows:
        print()
        cols = [(f"{v}_k{k}", d) for v in ("Usame", "Uneutral") for k in (1, 3) for d in (0, 10)]
        print("| model | " + " | ".join(f"{t} d{d}" for t, d in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for m in ORDER:
            if m in rows:
                print(f"| {SHORT.get(m, m)} | " + " | ".join((f"{rows[m][f'{t}|{d}']['acc']:.2f}" if f'{t}|{d}' in rows[m] else "-") for t, d in cols) + " |")
        json.dump(rows, open(os.path.join(HERE, "results_uprobe.json"), "w"), indent=1)


def copysplit():
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "out", "copysplit__*.json"))):
        m = os.path.basename(path)[len("copysplit__"):-len(".json")].replace("__", "/")
        rows[m] = json.load(open(path))
    if rows:
        print()
        keys = ["all", "hist_bigram/content", "hist_bigram", "hist_token", "self_only", "novel/content", "novel"]
        print("| model | " + " | ".join(keys) + " |")
        print("|---|" + "---|" * len(keys))
        for m in ORDER:
            if m in rows:
                print(f"| {SHORT.get(m, m)} | " + " | ".join(
                    f"{rows[m][k]['nll']:.3f} ({rows[m][k]['n']})" if k in rows[m] else "" for k in keys) + " |")


if __name__ == "__main__":
    main()
    khard()
    uprobe()
    copysplit()
