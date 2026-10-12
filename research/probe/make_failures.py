"""Build FAILURES.md from a curated list. All model text is pulled verbatim from
transcripts/*.jsonl (never retyped); long replies are cut at MAXC characters and
the cut is marked explicitly. Distractor turns are collapsed to one line."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B
from analyze import regrade

HERE = os.path.dirname(os.path.abspath(__file__))
MAXC = 420
SLUG = {"Falcon-H1-Tiny-90M": "tiiuae__Falcon-H1-Tiny-90M-Instruct", "SmolLM2-135M": "HuggingFaceTB__SmolLM2-135M-Instruct",
        "Gemma3-270M": "unsloth__gemma-3-270m-it", "LFM2-350M": "LiquidAI__LFM2-350M", "LFM2.5-350M": "LiquidAI__LFM2.5-350M",
        "SmolLM2-360M": "HuggingFaceTB__SmolLM2-360M-Instruct", "Qwen2.5-0.5B": "Qwen__Qwen2.5-0.5B-Instruct", "Qwen3-0.6B": "Qwen__Qwen3-0.6B"}


def get(model, tid, mode="greedy", seed=None):
    for line in open(os.path.join(HERE, "transcripts", f"{SLUG[model]}__{mode}.jsonl")):
        r = json.loads(line)
        if r["id"] == tid and r["seed"] == seed:
            return regrade(r)
    raise KeyError((model, tid, mode, seed))


def cut(t):
    t = t.replace("\n", " ⏎ ")
    return t if len(t) <= MAXC else t[:MAXC] + f" [... cut, {len(t) - MAXC} more chars]"


def render(model, tid, show=None, mode="greedy", seed=None):
    r = get(model, tid, mode, seed)
    out = []
    if r["system"]:
        out.append(f"> **system:** {r['system']}")
    for i, t in enumerate(r["turns"]):
        if show is not None and i not in show:
            if t["user"] in B.D:
                out.append(f"> *(turn {i}: distractor \"{t['user']}\", reply omitted)*")
            else:
                out.append(f"> *(turn {i} omitted)*")
            continue
        out.append(f"> **user {i}:** {t['user']}")
        out.append(f">")
        out.append(f"> **{model}:** {cut(t['assistant'])}")
        out.append(">")
    fz = ""
    if r["forced"]:
        f = r["forced"][0]
        fz = (f"\nForced-prefix probe on the same context: after \"{f['prefix']}\", gold \"{f['gold'].strip()}\" token ranks "
              f"{f['ranks_gold']} (1 = the model's top choice), log-prob {f['lp_gold']:.2f}; foil \"{f['foil'].strip()}\" log-prob {f['lp_foil']:.2f}.")
    grade = ", ".join(f"{k}={'PASS' if v else ('n/a' if v is None else 'FAIL')}" for k, v in r["checks"].items())
    return "\n".join(out).rstrip(">\n") + f"\n\nGrade: {grade}.{fz}\n"


if __name__ == "__main__":
    from failures_list import ENTRIES, HEADER
    parts = [HEADER]
    for sec in ENTRIES:
        parts.append(f"\n## {sec['section']}\n\n{sec.get('intro', '')}\n")
        for e in sec["items"]:
            parts.append(f"\n### {e['title']}\n\n*{e['model']}, `{e['tid']}`, {e.get('mode', 'greedy')}*\n\n")
            parts.append(render(e["model"], e["tid"], e.get("show"), e.get("mode", "greedy"), e.get("seed")))
            parts.append(f"\n**Why it matters:** {e['why']}\n")
    open(os.path.join(HERE, "FAILURES.md"), "w").write("".join(parts))
    print("wrote FAILURES.md")
