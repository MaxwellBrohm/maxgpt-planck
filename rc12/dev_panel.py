"""RC-12 STEP 10a: score the dev-baseline panel (prereg draft s9-s12) with the committed scorer. No model, no GPU.
  python -B dev_panel.py --root <a copy of the PC's ~/planck/runs/rc12_dev> --out runs/dev_panel
Per model, render and seed: score.summarize(the dev rows + that seed's --own-cf OWN rows) (OD1 b gate). Per model and
render, the mean of the 3 sampling seeds: score.summarize(the rows of seeds 1-3 and their twins), i.e. the scorer's
own nested means (s9), with the comparator Qwen2.5-0.5B-Instruct's summary of the same render for the Level A loop
sub-criterion (s10). Template render, mean of 3 seeds: score_stats.headroom over the s12 core panel (s11),
persist_base_rates / persist_drops over the core panel (s11), sensitivity vs the comparator (s9), and LFM2-2.6B's
Level A keys (the re-anchor input, s10). It computes and writes; it decides nothing (rebuild / drop / re-anchor are
listed for Max in notes STEP 10a). Also checks every run's stored scores.jsonl against score.score_lines on its rows
and that the pooled mean of 3 equals the arithmetic mean of the 3 per-seed values.
Since 2026-10-02 (Max: took all recommendations in rc12/DECISIONS_FOR_MAX.md, items 1 and 12): R has 9 families,
LOOKUP's family score is a reported column (F_LOOKUP, from the keys) with its headroom row beside (decides nothing),
and the near-duplicate rate sits beside the loop rate. Since 2026-10-04 (Max: LOOKUP is its own pre-registered
headline claim, prereg draft s10b): its P and X probe rates and score.lookup_claim's verdict are columns, and
rules.json lookup_claim holds every panel model's LOOKUP rate with score_stats.lookup_ci (template, seeds 1-3,
10,000 resamples), the comparison s10b reports (a dev reading of public models, never a claim). Since 2026-10-04 (Max:
took all recommendations in rc12/DECISIONS_LEVEL_R_FOR_MAX.md, decision 1 D): S7, S7_ungated and the s12 clause's
verdict (S7 point >= 40 and PERSIST >= 0.40) are columns, and rules.json level_r holds every panel model's S7 with
score_stats.s7_ci and score.level_r's verdict (template, seeds 1-3, 10,000 resamples; one training seed, so never
claimable: a dev reading, never a claim)."""
import argparse
import json
import os

import runner as RN
import score as S
import score_stats as ST

COMPARATOR = "Qwen2.5-0.5B-Instruct"
CORE = ["Qwen2.5-0.5B-Instruct", "Qwen3.5-0.8B", "LFM2.5-230M", "LFM2.5-350M", "Falcon-H1-Tiny-90M-Instruct",
        "SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct", "Doge-160M-Instruct", "LFM2-2.6B"]       # s12
EXTRAS = ["Qwen3-0.6B", "gemma-3-270m-it", "LFM2-700M", "LFM2-1.2B"]
RENDERS = ["template", "plain"]
SEEDS = ["greedy", "1", "2", "3"]
LA = list(S.LEVEL_A)
COLS = (["R", "R_ungated", "S7", "S7_ungated", "s12_clause"] + [f"F_{f}" for f in S.COMPOSITE + S.REPORTED] +
        ["LOOKUP_P", "LOOKUP_X"] +
        ["OWN_ungated", "OWN_CF"] + LA +
        ["T0", "loop", "near_duplicate", "ack_repeat", "ack_repeat_of_statements", "ack_repeat_of_answers", "RUNAWAY", "EMPTY", "LEAK",
         "cf_unswapped", "replies", "level_a_met", "lookup_claim_met"])


def load(path):
    return [json.loads(line) for line in open(path)]


def run_rows(root, model, render, seed):
    d = os.path.join(root, model, render)
    for s in (seed, seed + "_owncf"):
        assert os.path.exists(os.path.join(d, s, "DONE")), f"{model} {render} {s}: no DONE"
    return load(os.path.join(d, seed, "transcripts.jsonl")), load(os.path.join(d, seed + "_owncf", "transcripts.jsonl"))


def flat(summ):
    """the panel columns of one score.summarize result."""
    ks, deg = summ["keys"], summ["degenerate_rates"]
    out = dict(R=summ["R"], R_ungated=summ["R_ungated"], S7=summ["S7"], S7_ungated=summ["S7_ungated"],
               s12_clause=summ["s12_clause"]["passes"])
    out.update({f"F_{f}": summ["families"][f] for f in S.COMPOSITE})
    out.update({f"F_{f}": ks.get(f) for f in S.REPORTED})
    out.update(LOOKUP_P=summ["lookup_claim"]["P"], LOOKUP_X=summ["lookup_claim"]["X"],
               lookup_claim_met=summ["lookup_claim"]["met"])
    out.update(OWN_ungated=summ["own_gate"]["OWN"], OWN_CF=summ["own_gate"]["OWN_CF"])
    out.update({k: ks.get(k) for k in LA})
    out.update(T0=summ["t0"]["score"], loop=summ["loop_rate"], near_duplicate=summ["near_duplicate"],
               ack_repeat=summ["ack_repeat"],
               ack_repeat_of_statements=summ["ack_repeat_of_statements"],
               ack_repeat_of_answers=summ["ack_repeat_of_answers"], RUNAWAY=deg.get("RUNAWAY"),
               EMPTY=deg.get("EMPTY"), LEAK=deg.get("LEAK"), cf_unswapped=summ["own_gate"]["cf_unswapped"],
               replies=summ["replies"], level_a_met=summ["level_a_met"])
    return out


def stored_scores_ok(root, model, render, seed, rows):
    """the run's scores.jsonl (written by dev_batch at run time) == score.score_lines(rows) under this tree."""
    p = os.path.join(root, model, render, seed, "scores.jsonl")
    want = [json.loads(json.dumps(x)) for x in S.score_lines(rows)]
    return load(p) == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=RN.DEV)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = [m for m in CORE + EXTRAS if os.path.isdir(os.path.join(args.root, m))]
    assert models == CORE + EXTRAS, f"missing models: {sorted(set(CORE + EXTRAS) - set(models))}"
    per_seed, mean3, summ3, ids, checks = [], {}, {}, {}, []
    pooled_rows = {}
    for render in RENDERS:
        for model in models:
            rows3 = []
            vals = []
            for seed in SEEDS:
                dev, cf = run_rows(args.root, model, render, seed)
                ids[model] = dev[0]["responder"]
                ok_dev = stored_scores_ok(args.root, model, render, seed, dev)
                ok_cf = stored_scores_ok(args.root, model, render, seed + "_owncf", cf)
                checks.append(dict(model=model, render=render, seed=seed, scores_jsonl_dev=ok_dev, scores_jsonl_cf=ok_cf,
                                   dev_rows=len(dev), cf_rows=len(cf)))
                s = S.summarize(dev + cf)
                row = dict(model=model, render=render, seed=seed, **flat(s))
                per_seed.append(row)
                if seed != "greedy":
                    rows3 += dev + cf
                    vals.append(row)
            pooled_rows[(model, render)] = rows3
            summ3[(model, render)] = S.summarize(rows3)
            m = dict(model=model, render=render, seed="mean3", **flat(summ3[(model, render)]))
            for c in COLS:      # pooled (nested) mean vs the arithmetic mean of the 3 per-seed values
                xs = [v[c] for v in vals]
                if isinstance(m[c], float) and all(isinstance(x, float) for x in xs):
                    d = abs(m[c] - sum(xs) / 3)
                    if d > 1e-9:
                        checks.append(dict(model=model, render=render, seed="mean3", column=c, pooled=m[c],
                                           arithmetic=sum(xs) / 3))
            mean3[(model, render)] = m
        comp = summ3[(COMPARATOR, render)]
        for model in models:          # Level A with the comparator's loop rate (s10); a dev reading, never a claim
            s = S.summarize(pooled_rows[(model, render)], comparator=comp)
            mean3[(model, render)].update(level_a_met=s["level_a_met"],
                                          loop_vs_comparator=s["level_a"]["LOOP_vs_comparator"]["met"])
    # s11 headroom, template, mean of 3, core panel only
    panel = {m: summ3[(m, "template")] for m in CORE}
    head = ST.headroom(panel)
    head_reported = ST.headroom(panel, S.REPORTED)          # LOOKUP's row, out of R since 2026-10-02 (item 1)
    # s11 PERSIST base rates over every reply of every non-PERSIST conversation (template, seeds 1-3, core panel)
    recs = RN.load(args.data)
    rules = ST.persist_rules(recs)
    prates = {m: ST.persist_base_rates(pooled_rows[(m, "template")], rules) for m in CORE}
    prates_x = {m: ST.persist_base_rates(pooled_rows[(m, "template")], rules) for m in EXTRAS}
    drops = ST.persist_drops(prates)
    # s9 sensitivity row vs the comparator (template, mean of 3)
    sens = {m: ST.sensitivity(summ3[(m, "template")], summ3[(COMPARATOR, "template")]) for m in models}
    eq_only = {f"{m}|{r}": summ3[(m, r)]["equality_only"] for m in models for r in RENDERS}
    # the LOOKUP claim's comparison (s10b): every panel model's LOOKUP rate and 95% CI, template, seeds 1-3
    lookup = {m: dict(ST.lookup_ci(pooled_rows[(m, "template")], n=10000), P=summ3[(m, "template")]["lookup_claim"]["P"],
                      X=summ3[(m, "template")]["lookup_claim"]["X"], met=summ3[(m, "template")]["lookup_claim"]["met"])
              for m in models}
    # Level R since 2026-10-04: S7, its one-model CI and score.level_r's verdict per model (template, seeds 1-3)
    level = {}
    for m in models:
        st, ci = summ3[(m, "template")], ST.s7_ci(pooled_rows[(m, "template")], n=10000)
        level[m] = dict(S7=st["S7"], ci=ci, PERSIST=st["families"]["PERSIST"], T0=st["t0"]["score"],
                        s12_clause=st["s12_clause"]["passes"],
                        verdict=S.level_r(ci["lo"], st["families"]["PERSIST"], st["t0"]["score"], st["n_train_seeds"]))
    extra = {f"{m}|{r}": dict(k=summ3[(m, r)]["k"], t0_failures=summ3[(m, r)]["t0"]["failures"],
                              own_source_fail=summ3[(m, r)]["own_source_fail"],
                              capture_rate=summ3[(m, r)]["capture_rate"], lenient=summ3[(m, r)]["lenient"])
             for m in models for r in RENDERS}
    with open(os.path.join(args.out, "panel.json"), "w") as f:
        json.dump(dict(model_ids=ids, core=CORE, extras=EXTRAS, columns=COLS, per_seed=per_seed,
                       mean3=list(mean3.values())), f, indent=0)
    with open(os.path.join(args.out, "panel.tsv"), "w") as f:
        cols = ["model", "render", "seed"] + COLS + ["loop_vs_comparator"]
        f.write("\t".join(cols) + "\n")
        for row in per_seed + list(mean3.values()):
            f.write("\t".join("" if row.get(c) is None else f"{row[c]:.4f}" if isinstance(row.get(c), float)
                              else str(row.get(c)) for c in cols) + "\n")
    with open(os.path.join(args.out, "rules.json"), "w") as f:
        json.dump(dict(headroom=head, headroom_reported=head_reported, persist_rules=[f"{a}:{b}" for a, b in rules], persist_rates_core=prates,
                       persist_rates_extras=prates_x, persist_drops_core=drops, sensitivity=sens,
                       reanchor_input={k: mean3[("LFM2-2.6B", "template")][k] for k in LA},
                       equality_only=eq_only, reported_beside=extra, lookup_claim=lookup, level_r=level), f, indent=0)
    with open(os.path.join(args.out, "checks.json"), "w") as f:
        json.dump(checks, f, indent=0)
    bad = [c for c in checks if c.get("scores_jsonl_dev") is False or c.get("scores_jsonl_cf") is False
           or "column" in c]
    print(f"runs {len(per_seed)} x 2 (dev + owncf), check failures {len(bad)}")
    for c in bad[:20]:
        print("CHECK", c)
    print("headroom fails:", {k: v["reason"] for k, v in head.items() if v["fail"]})
    print("persist drops (core, > 0.30):", drops)
    print("LOOKUP claim (template, mean of 3, 95% CI):", {m: f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
                                                         for m, v in lookup.items()})
    print("Level R (template, mean of 3; S7 [95% CI], PERSIST, T0, s12 clause, met):",
          {m: f"{v['S7']:.2f} [{v['ci']['lo']:.2f}, {v['ci']['hi']:.2f}] {v['PERSIST']:.3f} {v['T0']:.3f} "
              f"{v['s12_clause']} {v['verdict']['met']}" for m, v in level.items()})
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
