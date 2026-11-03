"""RC-12 frontier calibration report: Level R numbers for the Claude runs (api_run.py) beside a panel model, all
regraded from the stored transcripts (CPU only, nothing replayed), under three rule sets (RULES):

  before_waiver  graders.py with the answer-first waiver switched off (grade_text.OFF "first"): the graders until
                 2026-11-03, which fail "The banjo. Your nephew is the one with the trombone." on v3_other.
  current        graders.py as it stands (the waiver adopted 2026-11-03, rc12/FRONTIER.md).
  no_v3          v3_other and v3_shotgun dropped from G-VAL: a probe of the grader, never a rule, since a list of
                 every candidate passes it (the SHOTGUN fake scores 92 on it).

  python -B frontier_report.py --root runs/frontier --ref <dir with LFM2-2.6B/template/<seed>[_owncf]/> --out <json>
Per model: seeds 1-3 where present (Claude Sonnet 5.5 has one sample: a single seed's CI is wider), each with its
_owncf twin, pooled exactly as dev_panel.py pools them; S7 CI by score_stats.s7_ci; verdict by score.level_r."""
import argparse
import json
import os

import grade_text as T
import graders as G
import runner as RN
import score as S
import score_stats as ST

RECS = {r["id"]: r for r in RN.load()}
ABLATE = {"v3_other", "v3_shotgun"}
_g_val = G.g_val


def g_val_no_v3(*a, **k):
    out = _g_val(*a, **k)
    out["fails"] = [f for f in out["fails"] if f not in ABLATE]
    out["ok"] = not out["fails"]
    return out


# (rule set, G-VAL, grade_text.OFF): the graders before the answer-first waiver (OFF "first"), as they stand, and
# with both clauses dropped (a probe of the grader, never a rule: a shotgun list passes it)
RULES = [("before_waiver", _g_val, {"first"}), ("current", _g_val, set()), ("no_v3", g_val_no_v3, {"first"})]


def runs(base, model):
    d = os.path.join(base, model, "template")
    seeds = [s for s in ("1", "2", "3") if os.path.isfile(os.path.join(d, s, "transcripts.jsonl"))]
    rows = []
    for s in seeds:
        for sub in (s, s + "_owncf"):
            rows += [json.loads(line) for line in open(os.path.join(d, sub, "transcripts.jsonl"))]
    return seeds, rows


def regrade(rows):
    return [RN.make_row(RECS[r["id"]], r["turns"], r["render"], r["seed"], r["responder"], r["train_seed"],
                        r["own_cf"]) for r in rows]


def level(rows, seeds):
    s = S.summarize(rows)
    ci = ST.s7_ci(rows, n=10000)
    dev = [r for r in rows if not r.get("own_cf")]
    replies = [t for r in dev for t in r["turns"]]
    return dict(seeds=seeds, conversations=len(dev), S7=s["S7"], ci=[ci["lo"], ci["hi"]],
                PERSIST=s["families"]["PERSIST"], T0=s["t0"]["score"],
                verdict=S.level_r(ci["lo"], s["families"]["PERSIST"], s["t0"]["score"], s["n_train_seeds"]),
                families=s["families"], cap_rate=sum(t["stop"] == "cap" for t in replies) / len(replies))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RN.HERE, "runs", "frontier"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--models", default="claude-sonnet-5-5,claude-haiku-5-5")
    ap.add_argument("--panel", default="LFM2-2.6B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    todo = [(m, args.root) for m in args.models.split(",") if os.path.isdir(os.path.join(args.root, m))]
    todo.append((args.panel, args.ref))
    report = {}
    for model, base in todo:
        seeds, rows = runs(base, model)
        report[model] = {}
        for rules, fn, off in RULES:
            G.g_val = fn
            T.OFF.clear()
            T.OFF.update(off)
            report[model][rules] = level(regrade(rows), seeds)
            r = report[model][rules]
            print(f"{model:20s} {rules:10s} seeds {','.join(seeds)}  S7 {r['S7']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
                  f"  PERSIST {r['PERSIST']:.3f}  T0 {r['T0']:.3f}  capped {r['cap_rate']:.0%}  {r['verdict']}", flush=True)
    G.g_val = _g_val
    T.OFF.clear()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
