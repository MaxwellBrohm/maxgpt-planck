"""RC-12 STEP 10b side check (prereg s12 baseline clause; not s11): does any existing model under 0.5B already meet the
Level R CI part (paired 95% CI lower bound >= -3 vs Qwen2.5-0.5B-Instruct) on DEV, template, seeds 1-3? With the
composite (9 families since LOOKUP left it; Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md
(item 1)) and, beside it, the old 10-family list with LOOKUP. The baselines have 1 training seed and T0 far below
0.90, so none meets Level R in full; this only reads the CI part, which is how s12's baseline clause reads (item 7).
The 10-family variant rebinds S.COMPOSITE in this process only. No model, no GPU.
  python -B dev_baseline_ci.py <a copy of the PC's ~/planck/runs/rc12_dev> 10000 [out.json]"""
import json
import os
import sys

import score as S
import score_stats as ST

ROOT, N = sys.argv[1], int(sys.argv[2])
MODELS = ["Falcon-H1-Tiny-90M-Instruct", "SmolLM2-135M-Instruct", "Doge-160M-Instruct", "LFM2.5-230M",
          "gemma-3-270m-it", "LFM2.5-350M", "SmolLM2-360M-Instruct"]


def rows_of(m):
    out = []
    for s in ("1", "2", "3"):
        for x in (s, s + "_owncf"):
            out += [json.loads(line) for line in open(os.path.join(ROOT, m, "template", x, "transcripts.jsonl"))]
    return out


comp = rows_of("Qwen2.5-0.5B-Instruct")
full = list(S.COMPOSITE)
out = {}
for m in MODELS:
    rows = rows_of(m)
    for label, fams in (("9fam", full), ("10fam_withLOOKUP", full[:-1] + S.REPORTED + full[-1:])):
        S.COMPOSITE[:] = fams
        ci = ST.bootstrap_diff(rows, comp, n=N, seed=0)
        ra, rb = S.summarize(rows)["R"], S.summarize(comp)["R"]
        out[f"{m}|{label}"] = dict(R=ra, R_qwen=rb, D=ra - rb, ci=ci, ci_part_met=ST.level_r(ci))
        print(m, label, f"R {ra:.2f} vs {rb:.2f} D {ra - rb:+.2f} CI {ci['lo']:+.2f} to {ci['hi']:+.2f}",
              "CI part met" if ST.level_r(ci) else "CI part not met", flush=True)
    S.COMPOSITE[:] = full
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                                         "dev_panel", "baseline_ci.json")
json.dump(out, open(OUT, "w"), indent=1)
