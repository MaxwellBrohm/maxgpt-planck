"""POST-HOC read-outs (not part of the pre-registered pass rule):
  1. free generation on the E001 update items + crossed items (gen_probe.py outputs)
  2. the 53-conversation chat probe (E001 hardened graders, greedy) on the untouched and fine-tuned models,
     scored exactly as E001 analyze_chat.py does (ability = mean over gradable checks; macro = mean of the
     7 multi-turn categories; truncated end-with-question checks excluded)
  3. seed-0 rerun reproducibility (s0 vs s0r on every shared eval set)
Writes ../posthoc_results.json and ../logs/tables_posthoc.txt.
"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
OUT = os.path.join(EXP, "out")
TR = os.path.join(EXP, "transcripts")
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import battery as B
import metrics_ft as MF

CATS = ["recall", "followup", "instruction", "correction", "own_answer", "role", "topic_return"]
MODELS = {"HuggingFaceTB__SmolLM2-135M-Instruct": "135M", "HuggingFaceTB__SmolLM2-360M-Instruct": "360M"}
NEW_T = {t["id"]: t for t in B.TESTS}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def gen_summary(recs):
    out = {}
    for render in ("plain", "chat"):
        R = [r for r in recs if r["render"] == render]
        for var in ("same_k1", "twoslot", "noupd", "cross_A", "cross_B"):
            xs = [r for r in R if r["var"] == var]
            if xs:
                out[f"{render}|{var}|strict"] = round(sum(r["strict"] for r in xs) / len(xs), 3)
                out[f"{render}|{var}|lenient"] = round(sum(r["lenient"] for r in xs) / len(xs), 3)
        by = defaultdict(list)
        for r in R:
            if r["var"] in ("cross_A", "cross_B"):
                by[r["sid"]].append(r["strict"])
        if by:
            out[f"{render}|cross_pair|strict"] = round(sum(all(v) and len(v) == 2 for v in by.values()) / len(by), 3)
        # the pass rule's three cells, generated instead of scored
        ok = [out.get(f"{render}|{v}|strict") for v in ("same_k1", "twoslot", "noupd")]
        out[f"{render}|all3>=0.8"] = all(x is not None and x >= 0.8 for x in ok)
    return out


def chat_summary(recs):
    abil = defaultdict(list)
    n_trunc = 0
    defl = []
    for r in recs:
        replies = [x["assistant"] for x in r["turns"]]
        for c in NEW_T[r["id"]]["checks"]:
            v, tr = B.grade(c, replies, hit_max=r["turns"][c["turn"]]["flags"]["hit_max"])
            abil[r["cat"]].append(None if v is None else float(v))
            n_trunc += v is None
        if r["cat"] in ("recall", "correction", "role", "topic_return") or (
                r["cat"] == "control" and (r["control_for"] or "")[:2] in ("R_", "K_", "RI", "T_")):
            defl.append(float(B.deflects(r["turns"][-1]["assistant"])))
    out = {"ability": {c: mean(abil[c]) for c in CATS + ["control"]}}
    out["macro"] = mean([out["ability"][c] for c in CATS])
    out["ungraded_truncated"] = n_trunc
    out["deflection_rate"] = mean(defl)
    out["correction_items"] = {r["id"]: {"checks": r["checks"], "final_reply": r["turns"][-1]["assistant"][:240]}
                               for r in recs if r["cat"] == "correction"}
    return out


def repro(slug):
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT, f"{slug}__s0__*.jsonl"))):
        name = os.path.basename(p).split("__s0__")[1]
        q = os.path.join(OUT, f"{slug}__s0r__{name}")
        if not os.path.exists(q):
            continue
        a, b = load(p), load(q)
        if len(a) != len(b):
            out[name] = {"error": f"length {len(a)} vs {len(b)}"}
            continue
        mx = max(abs(x["scores"][k] - y["scores"][k]) for x, y in zip(a, b) for k in x["scores"])
        flips = sum(MF.right(x["scores"]) != MF.right(y["scores"]) for x, y in zip(a, b))
        out[name] = {"n": len(a), "max_abs_diff": round(mx, 5), "decision_flips": flips}
    return out


def main():
    res = {"gen": {}, "chat": {}, "repro_s0_vs_s0r": {}}
    lines = []
    for slug, short in MODELS.items():
        # generation
        files = sorted(glob.glob(os.path.join(OUT, f"{slug}__*__gen__plain.jsonl")))
        tags = [os.path.basename(f)[len(slug) + 2:].split("__gen__")[0] for f in files]
        tags = [t for t in tags if not t.startswith("dry")]
        if tags:
            lines.append(f"== {short}: POST-HOC free generation, greedy, d10, strict grader (lenient in brackets)")
            lines.append("tag   | render | same_k1     twoslot     noupd       cross_A     cross_B     cross_pair | all3>=0.8")
        for t in tags:
            recs = load(os.path.join(OUT, f"{slug}__{t}__gen__plain.jsonl"))
            cp = os.path.join(OUT, f"{slug}__{t}__gen__chat.jsonl")
            if os.path.exists(cp):
                recs += load(cp)
            g = gen_summary(recs)
            res["gen"][f"{short}|{t}"] = g
            for render in ("plain", "chat"):
                if f"{render}|same_k1|strict" not in g:
                    continue
                cells = " ".join(f"{g[f'{render}|{v}|strict']:.2f} [{g[f'{render}|{v}|lenient']:.2f}]"
                                 for v in ("same_k1", "twoslot", "noupd", "cross_A", "cross_B"))
                lines.append(f"{t:5s} | {render:6s} | {cells} {g.get(f'{render}|cross_pair|strict', float('nan')):.2f}       | {g[f'{render}|all3>=0.8']}")
        # chat probe
        cfiles = sorted(glob.glob(os.path.join(TR, f"{slug}__*__greedy.jsonl")))
        ctags = [os.path.basename(f)[len(slug) + 2:].split("__greedy")[0] for f in cfiles]
        if ctags:
            lines.append("")
            lines.append(f"== {short}: POST-HOC chat probe (53 conversations, own chat template, greedy, E001 hardened graders)")
            lines.append("tag   | " + " ".join(f"{c[:10]:>10s}" for c in CATS) + " |  macro | controls | deflect | correction items (K_day K_time K_color)")
        for t, f in zip(ctags, cfiles):
            recs = load(f)
            if len(recs) < 53:
                lines.append(f"{t:5s} | incomplete ({len(recs)} conversations)")
                continue
            c = chat_summary(recs)
            res["chat"][f"{short}|{t}"] = c
            fm = lambda v: "   -" if v is None else f"{v:.2f}"
            corr = " ".join(str(all(v["checks"].values())) for k, v in sorted(c["correction_items"].items()))
            lines.append(f"{t:5s} | " + " ".join(f"{fm(c['ability'][k]):>10s}" for k in CATS) +
                         f" | {fm(c['macro']):>6s} | {fm(c['ability']['control']):>8s} | {fm(c['deflection_rate']):>7s} | {corr}")
        rp = repro(slug)
        if rp:
            res["repro_s0_vs_s0r"][short] = rp
            lines.append("")
            lines.append(f"== {short}: seed 0 trained twice (s0 vs s0r), same code path except the post-hoc sets")
            for k, v in rp.items():
                lines.append(f"  {k}: {v}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    json.dump(res, open(os.path.join(EXP, "posthoc_results.json"), "w"), indent=1)
    open(os.path.join(EXP, "logs", "tables_posthoc.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
