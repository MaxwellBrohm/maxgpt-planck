import sys, os, json, hashlib, pickle, math
sys.dont_write_bytecode = True
E5 = "REPO/experiments/E005_alias_eot"
E4 = "REPO/experiments/E004_general_updating"
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = HERE + "/items_eval_e005code.pkl"

def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]

def eval_items():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    sys.path.insert(0, E5 + "/code")
    import items_e004 as E
    d = E.draw("eval")
    pickle.dump(d, open(CACHE, "wb"))
    return d

def plain_prompt(item, with_prefix=True):
    lines = []
    for u, a in item["turns"]:
        lines += [f"User: {u}", f"Assistant: {a}"]
    lines += [f"User: {item['question']}", "Assistant:" + (" " + item["prefix"] if with_prefix else "")]
    return "\n".join(lines)

def recs(exp, tag, kind="e004", render="plain"):
    base = E5 if exp == "e005" else E4
    p = f"{base}/out/{SLUG}__{tag}__{kind}__{render}.jsonl"
    return [json.loads(l) for l in open(p)]

def al_items():
    return [json.loads(l) for l in open(E5 + "/al/al_items.jsonl")]

def al_recs(tag, kind, render):
    return [json.loads(l) for l in open(f"{E5}/al/out/{tag}__{kind}__{render}.jsonl")]

def my_right(scores):
    vals = list(scores.values())
    if len(vals) < 2 or any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in vals):
        return False
    return all(scores["gold"] > v for k, v in scores.items() if k != "gold")

def lik_pick(r):
    lab = max(r["scores"], key=lambda k: r["scores"][k])
    return r["cand_vals"][lab]
