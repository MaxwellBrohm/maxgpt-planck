"""RC-12 scoring (SPEC s6): units, family and cell scores, the Level R composite, the Level A criteria, the Tier 0
gate, the K report and the diagnostics. Input: graded rows from runner.py (one per conversation x seed).
Bootstrap, headroom, PERSIST base rates and the sensitivity row: score_stats.py.

Unit: a conversation's unit score (runner/graders: mean of its probes, or all-right for RECALL abstain, ROLE,
  LOOKUP), except BIND, whose unit is the PAIR (both twins right; twins joined by pair_id within one run).
Run: one (train_seed, seed) combination. Family score = mean over its units in a run, then over sampling seeds,
  then over training seeds (nested means). Seeds: sampling seeds when any are present, else greedy (--seeds).
Level R composite R = 100 x unweighted mean of the 10 COMPOSITE family scores (T0 is a gate, K is reported
  separately, TWOHOP:COMPOSE is diagnostic and never enters TWOHOP).
Level A (proposal thresholds, SPEC s6): CORR:U (U-diff + U-same units pooled) >= 0.80, CORR:C_noupd >= 0.80,
  CORR:C_twoslot >= 0.80, BIND pair rate >= 0.80, LOOP rate over every reply of every conversation (all
  families, LOOP flag only) <= 2% and no higher than the comparator's. Claimable only with >= 3 training seeds
  and T0 >= 0.90. Without a comparator the last sub-criterion is unknown and Level A is not met."""
import argparse
import json
from collections import defaultdict

COMPOSITE = ["RECALL", "CORR", "BIND", "TWOHOP", "PERSIST", "OWN", "TOPIC", "ROLE", "LOOKUP", "LOOP"]
DIAG = {("TWOHOP", "COMPOSE")}
LEVEL_A = {"CORR:U": ("CORR", ("U-diff", "U-same")), "CORR:C_noupd": ("CORR", ("C_noupd",)),
           "CORR:C_twoslot": ("CORR", ("C_twoslot",)), "BIND": ("BIND", None)}
BARS = dict(level_a=0.80, loop=0.02, t0=0.90, min_train_seeds=3)
DEGEN = ("LOOP", "RUNAWAY", "EMPTY", "LEAK")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def select(rows, seeds=None):
    """rows of the chosen seeds: sampling seeds if present, else greedy (seed None)."""
    if seeds is None:
        sampled = {r["seed"] for r in rows if r["seed"] is not None}
        seeds = sampled or {None}
    return [r for r in rows if r["seed"] in set(seeds)]


def runs(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r.get("train_seed", 0), r["seed"])].append(r)
    return out


def units(run_rows):
    """units of one run: dicts family, cell, uid, score, knowledge."""
    out, pairs = [], defaultdict(list)
    for r in run_rows:
        if r["family"] == "BIND":
            pairs[r["pair_id"]].append(r)
        else:
            out.append(dict(family=r["family"], cell=r["cell"], uid=r["id"], score=r["unit"]))
    for pid, twins in sorted(pairs.items()):
        assert len(twins) == 2, f"BIND pair {pid} has {len(twins)} twins in one run"
        out.append(dict(family="BIND", cell=twins[0]["cell"], uid=pid,
                        score=float(all(t["unit"] == 1.0 for t in twins))))
    return out


def unit_keys(u):
    """every score key a unit counts toward."""
    f, c = u["family"], u["cell"]
    keys = [f"{f}:{c}"]
    if (f, c) not in DIAG:
        keys.append(f)
    for k, (fam, cells) in LEVEL_A.items():
        if f == fam and (cells is None or c in cells) and k not in keys:
            keys.append(k)
    return keys


def nested(per_run):
    """per_run {(train_seed, seed): value} -> mean over sampling seeds, then over training seeds."""
    by_train = defaultdict(list)
    for (tr, _), v in per_run.items():
        if v is not None:
            by_train[tr].append(v)
    return mean(mean(v) for v in by_train.values())


def key_scores(rows):
    """{key: nested score} over every family, family:cell and Level A key."""
    per = defaultdict(dict)
    for rk, rr in runs(rows).items():
        acc = defaultdict(list)
        for u in units(rr):
            for k in unit_keys(u):
                acc[k].append(u["score"])
        for k, v in acc.items():
            per[k][rk] = mean(v)
    return {k: nested(v) for k, v in per.items()}


def loop_stats(rows):
    n = sum(len(r["flags"]) for r in rows)
    rates = {d: sum(d in f for r in rows for f in r["flags"]) / n for d in DEGEN} if n else {}
    return n, rates


def composite(fam):
    missing = [f for f in COMPOSITE if fam.get(f) is None]
    if missing:
        return None
    return 100 * mean(fam[f] for f in COMPOSITE)


def level_a(ks, loop_rate, comparator_loop=None, n_train=1, t0=None):
    crit = {k: dict(value=ks.get(k), bar=BARS["level_a"], met=ks.get(k) is not None and ks[k] >= BARS["level_a"])
            for k in LEVEL_A}
    crit["LOOP"] = dict(value=loop_rate, bar=BARS["loop"], met=loop_rate is not None and loop_rate <= BARS["loop"])
    crit["LOOP_vs_comparator"] = dict(value=loop_rate, bar=comparator_loop,
                                      met=comparator_loop is not None and loop_rate <= comparator_loop)
    met = all(c["met"] for c in crit.values())
    claimable = met and n_train >= BARS["min_train_seeds"] and t0 is not None and t0 >= BARS["t0"]
    return crit, met, claimable


def summarize(rows, comparator=None, seeds=None):
    """comparator: a summary dict of the comparator model (for the loop sub-criterion), or None."""
    rows = select(rows, seeds)
    ks = key_scores(rows)
    fam = {f: ks.get(f) for f in COMPOSITE}
    n_rep, degen = loop_stats(rows)
    t0 = ks.get("T0")
    n_train = len({r.get("train_seed", 0) for r in rows})
    crit, met, claimable = level_a(ks, degen.get("LOOP"), comparator and comparator["loop_rate"], n_train, t0)
    own = [p for r in rows if r["family"] == "OWN" for p in r["probes"] if "source_fail" in p]
    lenient = defaultdict(list)
    for r in rows:
        for p in r["probes"]:
            lenient[r["family"]].append(bool(p.get("lenient")))
    k_follow, k_control = ks.get("K:followup"), ks.get("K:control")
    return dict(
        R=composite(fam), families=fam, keys=ks, loop_rate=degen.get("LOOP"), degenerate_rates=degen,
        replies=n_rep, level_a=crit, level_a_met=met, level_a_claimable=claimable, n_train_seeds=n_train,
        seeds=sorted({str(r["seed"]) for r in rows}),
        t0=dict(score=t0, met=t0 is not None and t0 >= BARS["t0"],
                failures=sorted({r["id"] for r in rows if r["family"] == "T0" and r["unit"] < 1})),
        k=dict(followup=k_follow, control=k_control,
               gap=None if k_follow is None or k_control is None else k_control - k_follow),
        own_source_fail=mean(p["source_fail"] for p in own) if own else None,
        capture_rate=mean(bool(r["leaks"]) for r in rows) if rows else None,
        lenient={f: mean(v) for f, v in lenient.items()})


def score_lines(rows):
    """per-run family and cell score lines (scores.jsonl)."""
    out = []
    for (tr, sd), rr in sorted(runs(rows).items(), key=lambda x: (x[0][0], str(x[0][1]))):
        acc = defaultdict(list)
        for u in units(rr):
            for k in unit_keys(u):
                acc[k].append(u["score"])
        for k in sorted(acc):
            out.append(dict(train_seed=tr, seed=sd, key=k, score=mean(acc[k]), n_units=len(acc[k])))
    return out


def load_rows(path):
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", help="transcripts.jsonl files of ONE model (training seeds)")
    ap.add_argument("--comparator", default=None, help="the comparator's transcripts.jsonl")
    args = ap.parse_args()
    rows = [r for p in args.transcripts for r in load_rows(p)]
    comp = summarize(load_rows(args.comparator)) if args.comparator else None
    s = summarize(rows, comp)
    s.pop("keys")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
