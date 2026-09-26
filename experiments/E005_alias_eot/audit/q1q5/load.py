import sys, os, json, hashlib, pickle, math
sys.dont_write_bytecode = True
E5 = "REPO/experiments/E005_alias_eot"
E4 = "REPO/experiments/E004_general_updating"
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"
TAGS = ["base", "s1", "s2", "s3", "s4", "s5"]
PASSF = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "C_noupd", "C_twoslot"]
FAMS = PASSF + ["ID"]
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = HERE + "/items_eval.pkl"

def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]

def items():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    sys.path.insert(0, E5 + "/code")
    import items as I
    import items_e004 as E
    d = E.draw("eval")
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    pickle.dump(d, open(CACHE, "wb"))
    return d

def recs(tag, kind="e004", render="plain", exp=E5):
    p = f"{exp}/out/{SLUG}__{tag}__{kind}__{render}.jsonl"
    return [json.loads(l) for l in open(p)]

def plain_prompt(it, with_prefix):
    lines = []
    for u, a in it["turns"]:
        lines += [f"User: {u}", f"Assistant: {a}"]
    lines += [f"User: {it['question']}", "Assistant:" + (" " + it["prefix"] if with_prefix else "")]
    return "\n".join(lines)

def my_gold(it):
    about = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    return about[-1]["value"]

def my_cands(it):
    out = []
    for s in it["stmts"]:
        if s["value"] not in out:
            out.append(s["value"])
    return out

def lik_right(scores):
    vals = list(scores.values())
    if len(vals) < 2 or any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in vals):
        return False
    g = scores["gold"]
    return all(g > v for k, v in scores.items() if k != "gold")

def lik_pick(r):
    lab = max(r["scores"], key=lambda k: r["scores"][k])
    return r["cand_vals"][lab]
