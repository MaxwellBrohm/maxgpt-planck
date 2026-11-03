"""RC-12 frontier calibration report: Level R numbers for the Claude runs (api_run.py) beside a panel model, under
the working-tree graders and under one ablation, all regraded from the stored transcripts (CPU only, nothing replayed).

  rules "current":   graders.py as it stands.
  rules "attributed": the same, with v3_other and v3_shotgun dropped from G-VAL. Those two fail a reply that names any
                     other in-context value, which is how a frontier model answers ("The banjo. Your nephew is the one
                     with the trombone."; "Saturday. It started on Wednesday, then moved to Friday ..."). The ablation
                     is a probe of the grader, not a proposed rule: it also lets a shotgun list through, so it is read
                     only as the gap it opens between models, never as a pass.

  python -B frontier_report.py --root runs/frontier --ref <dir with LFM2-2.6B/template/<seed>[_owncf]/> --out <json>
Per model: seeds 1-3 where present (Claude Sonnet 5.5 has one sample: a single seed's CI is wider), each with its
_owncf twin, pooled exactly as dev_panel.py pools them; S7 CI by score_stats.s7_ci; verdict by score.level_r."""
import argparse
import json
import os
import re

import grade_text as T
import graders as G
import runner as RN
import score as S
import score_stats as ST

RECS = {r["id"]: r for r in RN.load()}
ABLATE = {"v3_other", "v3_shotgun"}
_g_val = G.g_val


def g_val_attributed(*a, **k):
    out = _g_val(*a, **k)
    out["fails"] = [f for f in out["fails"] if f not in ABLATE]
    out["ok"] = not out["fails"]
    return out


FIRST = re.compile(r"[.!?\n]")


def g_val_answer_first(reply, stop, prior, rec, probe, gold=None):
    """v3_other and v3_shotgun are waived when the reply's first sentence asserts a gold and names no other value of
    the probe, so "The banjo. Your nephew is the one with the trombone." passes and a list of every candidate does
    not (the shortcut ceiling under these rules is in the report: a cheater has to know the gold to lead with it)."""
    out = _g_val(reply, stop, prior, rec, probe, gold)
    if not ABLATE & set(out["fails"]):
        return out
    text = T.norm(reply)
    first = FIRST.split(text.strip(), maxsplit=1)[0]
    gs = G.golds(probe, gold)
    pool = G.pool_values(probe)
    rest = [v for v in dict.fromkeys(list(probe.get("candidates") or []) + G.in_context(rec, probe, pool)) if v not in gs]
    if any(T.asserted_hits(first, g, pool, gold=True) for g in gs) and not any(T.mentioned(first, v) for v in rest):
        out["fails"] = [f for f in out["fails"] if f not in ABLATE]
        out["ok"] = not out["fails"]
    return out


RULES = {"current": _g_val, "attributed": g_val_attributed, "answer_first": g_val_answer_first}


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
        for rules, fn in RULES.items():
            G.g_val = fn
            report[model][rules] = level(regrade(rows), seeds)
            r = report[model][rules]
            print(f"{model:20s} {rules:10s} seeds {','.join(seeds)}  S7 {r['S7']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
                  f"  PERSIST {r['PERSIST']:.3f}  T0 {r['T0']:.3f}  capped {r['cap_rate']:.0%}  {r['verdict']}", flush=True)
    G.g_val = _g_val
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
