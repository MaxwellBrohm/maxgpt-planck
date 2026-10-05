"""RC-12 item 14 extras (Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 14); notes STEP 11):
score the sub-30M public chat models' dev runs (and cRia-75M) as REPORTED extras (never headroom-panel members) with
the scorer of this tree, plus the Vertex-0.6-15M card row (its card's sampling, engines_card.json; NOVELTY_2026-10
s6), reported beside its plain-decoding row. No model, no GPU.
  python -B dev_extras14.py --root <copy of the PC's rc12_dev holding the extras> --panel-root <regraded STEP 10a copy>
      [--card-root <copy of rc12_dev_card>] --out runs/dev_panel/extras14 [--boot 10000]
Per model, render and seed: score.summarize(dev rows + that seed's --own-cf twin rows) (OD1 b); per model and render
the mean of the 3 sampling seeds (the scorer's nested means over seeds 1-3 and their twins), Level A with
Qwen2.5-0.5B-Instruct's summary of the same render (s10). Every stored row is regraded with this tree's graders
(runner.make_row on the stored turns) and the rows whose grades differ from the stored ones are counted, so a grader
change made after the runs (e.g. item 3) is applied, and the count says whether it touched these runs. Template, seeds
1-3: score_stats.bootstrap_diff against Qwen2.5 (the s12 clause as item 7 reads it: the CI part only, at every size),
the s9 sensitivity row (families where Qwen2.5 > 0.05), the PERSIST base rates (s11; reported, the extras never drop
a rule), and the LOOKUP claim (score.lookup_claim, prereg draft s10b; Max, 2026-10-04): its value, P and X probe
rates, cells and score_stats.lookup_ci. Level R since 2026-10-04 (Max: took all recommendations in
rc12/DECISIONS_LEVEL_R_FOR_MAX.md, decision 1 D): S7 with score_stats.s7_ci, score.level_r's verdict and the s12
clause (S7 point >= 40 and PERSIST >= 0.40); the CI vs Qwen2.5 above is the -3 non-inferiority test, reported beside
(score_stats.noninferior). The card row has seeds 1-3 only (the card asks for sampling).
Writes extras14.json and extras14.tsv."""
import argparse
import json
import os

import dev_panel as DP
import runner as RN
import score as S
import score_stats as ST

EXTRAS14 = ["Veyra2-Blueberry-5M-Instruct", "BananaMind-2-Nano-Chat", "Veyra2-Blueberry-10M-Instruct",
            "Vertex-0.6-15M-Instruct", "Veyra2-Mango-15M-Instruct", "Loom-Spark-3.2", "BananaMind-2-Mini-Chat",
            "Swen-28M", "Veyra2-Mango-30M-Instruct", "cRia-LM-75M-Instruct"]          # by size (rc12/pins.json)
CARD = {"Vertex-0.6-15M-Instruct": "Vertex-0.6-15M-Instruct@card"}           # model dir -> its card row's label
SEEDS = ["greedy", "1", "2", "3"]
LCELLS = ["menu", "rota", "tour"]


def regrade(rows, recs):
    """the rows regraded by this tree's graders; and how many differ from the stored grade (probes, unit, flags)."""
    out, diff = [], 0
    for r in rows:
        n = RN.make_row(recs[r["id"]], r["turns"], r["render"], r["seed"], r["responder"], r["train_seed"], r["own_cf"])
        diff += any(n[k] != r[k] for k in ("probes", "unit", "flags", "leaks", "ack_repeat", "ack_of_answer"))
        out.append(n)
    return out, diff


def complete(root, model, render, seeds):
    d = os.path.join(root, model, render)
    return all(os.path.exists(os.path.join(d, s, "DONE")) and os.path.exists(os.path.join(d, s + "_owncf", "DONE"))
               for s in seeds)


def lookup_cols(summ):
    """the LOOKUP claim's cells (dev_panel.flat already has its family score, P and X rates and met)."""
    return {f"LOOKUP:{c}": summ["lookup_claim"]["cells"].get(c) for c in LCELLS}


def score_one(root, model, label, render, seeds, recs, comp, comp_rows, boot, out):
    """per seed and mean of the sampled seeds for one model dir (or card row); fills out's lists and dicts."""
    rows3, ndiff, nrows, meta = [], 0, 0, None
    for seed in seeds:
        dev, cf = DP.run_rows(root, model, render, seed)
        out["ids"][label] = dev[0]["responder"]
        meta = meta or json.load(open(os.path.join(root, model, render, seed, "meta.json")))
        ok = DP.stored_scores_ok(root, model, render, seed, dev)
        dev, d1 = regrade(dev, recs)
        cf, d2 = regrade(cf, recs)
        ndiff, nrows = ndiff + d1 + d2, nrows + len(dev) + len(cf)
        s = S.summarize(dev + cf)
        out["per_seed"].append(dict(model=label, render=render, seed=seed, scores_jsonl_ok=ok, **DP.flat(s),
                                    **lookup_cols(s)))
        if seed != "greedy":
            rows3 += dev + cf
    s3 = S.summarize(rows3, comparator=comp[render])
    out["mean3"].append(dict(model=label, render=render, seed="mean3", **DP.flat(s3), **lookup_cols(s3),
                             loop_vs_comparator=s3["level_a"]["LOOP_vs_comparator"]["met"],
                             t0_failures=len(s3["t0"]["failures"]), regraded_rows_differing=ndiff, rows=nrows,
                             decode=meta.get("decode"), engine=meta.get("engine"), dtype=meta.get("dtype")))
    out["status"][f"{label}|{render}"] = f"scored; {ndiff} of {nrows} stored rows regrade differently"
    if render == "template":
        c = ST.bootstrap_diff(rows3, comp_rows["template"], n=boot, seed=0)
        out["ci"][label] = dict(R=s3["R"], R_qwen=comp["template"]["R"], D=s3["R"] - comp["template"]["R"], ci=c,
                                ci_part_met=ST.noninferior(c), sensitivity=ST.sensitivity(s3, comp["template"]),
                                sensitivity_qwen=ST.sensitivity(comp["template"], comp["template"]),
                                lookup=dict(s3["lookup_claim"], ci=ST.lookup_ci(rows3, n=boot, seed=0)))
        c7 = ST.s7_ci(rows3, n=boot, seed=0)
        out["ci"][label].update(S7=s3["S7"], s7_ci=c7, s12_clause=s3["s12_clause"],
                                level_r=S.level_r(c7["lo"], s3["families"]["PERSIST"], s3["t0"]["score"],
                                                  s3["n_train_seeds"]))
        out["persist"][label] = ST.persist_base_rates(rows3, out["rules"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel-root", required=True)
    ap.add_argument("--card-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=RN.DEV)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    recs = {r["id"]: r for r in RN.load(args.data)}
    comp, comp_rows = {}, {}
    for render in DP.RENDERS:
        rows = []
        for s in ("1", "2", "3"):
            dev, cf = DP.run_rows(args.panel_root, DP.COMPARATOR, render, s)
            rows += dev + cf
        comp_rows[render], _ = regrade(rows, recs)
        comp[render] = S.summarize(comp_rows[render])
    out = dict(per_seed=[], mean3=[], ci={}, persist={}, status={}, ids={}, rules=ST.persist_rules(list(recs.values())))
    jobs = [(args.root, m, m, r, SEEDS) for m in EXTRAS14 for r in DP.RENDERS]
    if args.card_root:
        jobs += [(args.card_root, m, lab, "template", ["1", "2", "3"]) for m, lab in CARD.items()]
    for root, model, label, render, seeds in jobs:
        if not complete(root, model, render, seeds):
            out["status"][f"{label}|{render}"] = "incomplete: not scored"
            continue
        score_one(root, model, label, render, seeds, recs, comp, comp_rows, args.boot, out)
    cols = ["model", "render", "seed"] + DP.COLS + [f"LOOKUP:{c}" for c in LCELLS] + ["loop_vs_comparator"]
    with open(os.path.join(args.out, "extras14.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in out["per_seed"] + out["mean3"]:
            f.write("\t".join("" if row.get(c) is None else f"{row[c]:.4f}" if isinstance(row.get(c), float)
                              else str(row.get(c)) for c in cols) + "\n")
    json.dump(dict(model_ids=out["ids"], composite=list(S.COMPOSITE), comparator=DP.COMPARATOR,
                   comparator_R={r: comp[r]["R"] for r in comp}, status=out["status"], per_seed=out["per_seed"],
                   mean3=out["mean3"], ci_vs_qwen=out["ci"], persist_base_rates=out["persist"]),
              open(os.path.join(args.out, "extras14.json"), "w"), indent=0)
    for k, v in out["status"].items():
        print(k, v)
    for m, c in out["ci"].items():
        sv, sq, lk = c["sensitivity"], c["sensitivity_qwen"], c["lookup"]
        print(f"{m}: R {c['R']:.2f} vs Qwen2.5 {c['R_qwen']:.2f}, D {c['D']:+.2f}, CI {c['ci']['lo']:+.2f} to "
              f"{c['ci']['hi']:+.2f}, CI part {'met' if c['ci_part_met'] else 'not met'}; sensitivity row "
              f"({'+'.join(sv['kept'] or [])}) {sv['R']:.2f} vs {sq['R']:.2f}; LOOKUP {lk['value']:.3f} "
              f"(CI {lk['ci']['lo']:.3f} to {lk['ci']['hi']:.3f}, P {lk['P']:.3f}, X {lk['X']:.3f}); S7 {c['S7']:.2f} "
              f"(CI {c['s7_ci']['lo']:.2f} to {c['s7_ci']['hi']:.2f}), PERSIST {c['s12_clause']['PERSIST']:.3f}, "
              f"s12 clause {'passes' if c['s12_clause']['passes'] else 'fails'}, Level R met {c['level_r']['met']}")
    bad = [r for r in out["per_seed"] if r["scores_jsonl_ok"] is False]
    print("stored scores.jsonl == score_lines(rows) under this tree: "
          f"{len(out['per_seed']) - len(bad)} of {len(out['per_seed'])} (differences are expected only if the scorer "
          "changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
