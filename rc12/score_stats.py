"""RC-12 statistics on top of score.py (SPEC s3, s6, s7): the paired bootstrap for Level R, the headroom rule,
the PERSIST base-rate drop rule and the sensitivity row. Pure Python.

bootstrap_diff(rows_a, rows_b, n)  D = R(a) - R(b). Each resample: units resampled within each family (stratified,
    the SAME draw for both models: paired), training seeds resampled per model, sampling seeds resampled inside
    each drawn training seed (nested); percentile 95% CI. Level R holds when the CI's lower bound is >= -3 points
    (level_r). The OWN slot holds the OD1 b gated units (score.py), so both models need their --own-cf OWN rows;
    without them it refuses.
lookup_ci(rows, n)  the LOOKUP claim's 95% CI (prereg draft s10b; Max, 2026-10-04): one model, LOOKUP units
    resampled, training and sampling seeds resampled as above; reported beside the claim, never part of R.
OW = oodh_wording.py, the OOD-H wording rule of s1: OW.paired_diff (OOD-H paired difference, paired bootstrap over
    threads and seeds), OW.decide (D worse than -5 points: "non-inferior on RC-12's format" with the OOD-H result
    beside it, the unqualified wording refused; provisional until OOD-H Part 2 is scored), OW.refusals.
headroom(panel, keys)  panel {model: score.summarize(...)} (template render, mean of 3 seeds, core panel only). A
    family or Level A key fails if EVERY model is <= 0.05 (floor) or EVERY model is >= 0.95 (ceiling). A composite
    family is judged on the score that enters R (summary families: OWN = OWN_GATED, OD1 b). keys: HEADROOM_KEYS (the
    composite families and the Level A keys); a reported family (score.REPORTED, LOOKUP since 2026-10-02) can be
    passed to see its row, which decides nothing.
persist_base_rates(rows, rules)  compliance share of each (rule, arg) over every reply of every NON-PERSIST
    conversation; rules above 0.30 for any core baseline are dropped (persist_drops). The rows are the TEMPLATE
    render's (Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 6); prereg draft s11):
    PERSIST is scored on the template render and the plain render is diagnostic.
sensitivity(summary, comparator)  the composite without the families where the comparator is <= 0.05 (R None when
    a composite family is missing, e.g. OWN without the --own-cf rows)."""
import random

import grade_fmt_dyn as FD
import grade_text as T
import oodh_wording as OW  # noqa: F401  (the s1 OOD-H wording rule, used as ST.OW)
import score as S

HEADROOM_KEYS = S.COMPOSITE + [k for k in S.LEVEL_A if k not in S.COMPOSITE]
HEADROOM = dict(floor=0.05, ceiling=0.95)
LEVEL_R_MARGIN = -3.0
PERSIST_DROP = 0.30


def unit_table(rows):
    """{train_seed: {seed: {family: {uid: score}}}} over the composite families (BIND = pairs; the OWN slot holds
    the OWN_GATED units, never the ungated OWN ones: OD1 b)."""
    out = {}
    for (tr, sd), rr in S.runs(S.select(rows)).items():
        fam = {f: {} for f in S.COMPOSITE}
        for u in S.units(rr):
            f = S.SLOT.get(u["family"], u["family"])
            if f in fam and u["family"] not in S.GATE and (u["family"], u["cell"]) not in S.DIAG:
                fam[f][u["uid"]] = u["score"]
        out.setdefault(tr, {})[sd] = fam
    return out


def _arrays(table, uids):
    return {tr: {sd: {f: [fam[f][u] for u in uids[f]] for f in S.COMPOSITE} for sd, fam in by_seed.items()}
            for tr, by_seed in table.items()}


def _r(arrs, draw, rng):
    trains = list(arrs)
    picks = []
    for tr in [rng.choice(trains) for _ in trains]:
        seeds = list(arrs[tr])
        picks += [arrs[tr][rng.choice(seeds)] for _ in seeds]
    tot = 0.0
    for f in S.COMPOSITE:
        idx = draw[f]
        tot += sum(sum(a[f][i] for i in idx) / len(idx) for a in picks) / len(picks)
    return 100 * tot / len(S.COMPOSITE)


def bootstrap_diff(rows_a, rows_b, n=10000, seed=0):
    ta, tb = unit_table(rows_a), unit_table(rows_b)
    first = next(iter(next(iter(ta.values())).values()))
    uids = {f: sorted(first[f]) for f in S.COMPOSITE}
    assert all(uids.values()), f"no units for {[f for f in uids if not uids[f]]} (OWN needs the --own-cf rows, OD1 b)"
    for t in (ta, tb):
        for by_seed in t.values():
            for fam in by_seed.values():
                for f in S.COMPOSITE:
                    assert sorted(fam[f]) == uids[f], f"unit sets differ in {f}: paired bootstrap impossible"
    aa, ab = _arrays(ta, uids), _arrays(tb, uids)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        draw = {f: [rng.randrange(len(uids[f])) for _ in uids[f]] for f in S.COMPOSITE}
        diffs.append(_r(aa, draw, rng) - _r(ab, draw, rng))
    diffs.sort()
    point = S.summarize(rows_a)["R"] - S.summarize(rows_b)["R"]
    return dict(D=point, lo=diffs[int(0.025 * n)], hi=diffs[min(n - 1, int(0.975 * n))], n=n)


def level_r(ci):
    return ci["lo"] >= LEVEL_R_MARGIN


def lookup_ci(rows, n=10000, seed=0):
    """95% percentile CI of the LOOKUP claim's score (score.lookup_claim; prereg draft s10b; Max, 2026-10-04): per
    resample one draw of LOOKUP units (shared by every run of the model), training seeds resampled, sampling seeds
    resampled inside each drawn training seed (nested), as bootstrap_diff does for R. One model, not paired."""
    table = {}
    for (tr, sd), rr in S.runs(S.select(rows)).items():
        table.setdefault(tr, {})[sd] = {u["uid"]: u["score"] for u in S.units(rr) if u["family"] == S.LOOKUP_CLAIM}
    uids = sorted(next(iter(next(iter(table.values())).values())))
    assert uids and all(sorted(us) == uids for by in table.values() for us in by.values()), "LOOKUP unit sets differ"
    arrs = {tr: [[us[u] for u in uids] for us in by.values()] for tr, by in table.items()}
    trs, rng, vals = list(arrs), random.Random(seed), []
    for _ in range(n):
        draw = [rng.randrange(len(uids)) for _ in uids]
        picks = [a for tr in [rng.choice(trs) for _ in trs] for a in [rng.choice(arrs[tr]) for _ in arrs[tr]]]
        vals.append(sum(sum(a[i] for i in draw) / len(draw) for a in picks) / len(picks))
    vals.sort()
    return dict(value=S.summarize(rows)["lookup_claim"]["value"], lo=vals[int(0.025 * n)],
                hi=vals[min(n - 1, int(0.975 * n))], n=n, units=len(uids))


def headroom(panel, keys=None):
    out = {}
    for k in HEADROOM_KEYS if keys is None else keys:
        vals = {m: s["families"][k] if k in s["families"] else s["keys"].get(k) for m, s in panel.items()}
        known = [v for v in vals.values() if v is not None]
        floor = bool(known) and len(known) == len(vals) and all(v <= HEADROOM["floor"] for v in known)
        ceil = bool(known) and len(known) == len(vals) and all(v >= HEADROOM["ceiling"] for v in known)
        out[k] = dict(scores=vals, fail=floor or ceil, reason="floor" if floor else "ceiling" if ceil else None)
    return out


def persist_rules(recs):
    return sorted({(rule, arg) for r in recs if r["family"] == "PERSIST" for rule, arg in r["meta"]["rules"]},
                  key=lambda x: (x[0], str(x[1])))


def persist_base_rates(rows, rules):
    replies = [T.norm(t["reply"]) for r in S.select(rows) if r["family"] != "PERSIST" and not r.get("own_cf")
               for t in r["turns"]]
    return {f"{rule}:{arg}": sum(FD.rule_ok(rule, arg, x) for x in replies) / len(replies) for rule, arg in rules}


def persist_drops(panel_rates):
    """panel_rates {model: persist_base_rates(...)} -> rules whose base rate exceeds 0.30 for ANY model."""
    keys = sorted({k for rates in panel_rates.values() for k in rates})
    return [k for k in keys if any(rates.get(k, 0) > PERSIST_DROP for rates in panel_rates.values())]


def sensitivity(summary, comparator):
    if any(s["families"][f] is None for s in (summary, comparator) for f in S.COMPOSITE):
        return dict(kept=None, dropped=None, R=None)
    keep = [f for f in S.COMPOSITE if comparator["families"][f] > HEADROOM["floor"]]
    r = 100 * sum(summary["families"][f] for f in keep) / len(keep) if keep else None
    return dict(kept=keep, dropped=[f for f in S.COMPOSITE if f not in keep], R=r)
