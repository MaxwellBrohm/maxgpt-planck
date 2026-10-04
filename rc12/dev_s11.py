"""RC-12 STEP 10b: prereg draft s11 applied to the dev baselines (headroom, PERSIST base rates). No model or GPU.
  python -B dev_s11.py --root <a copy of the PC's ~/planck/runs/rc12_dev> [--out runs/dev_panel/s11.json]
Core panel of s12, template render, mean of the 3 sampling seeds = score.summarize(seeds 1-3 + their --own-cf twins),
the scorer's own nested means (OWN is the gated OWN, OD1 b). Headroom: score_stats.headroom over HEADROOM_KEYS (the
composite families, 9 since 2026-10-02, + the Level A keys CORR:U, C_noupd, C_twoslot; the Level A BIND key is the
BIND family, checked equal). Also per seed and greedy, how many core models clear each bar, and for LOOKUP: probe-level strict and lenient
rates, failing clauses, and what the abstain probe (X: a key the named table lacks, the other table holds) says.
PERSIST: score_stats.persist_base_rates (every reply of every non-PERSIST conversation, cf twins out) per core model,
pooled over seeds 1-3 and per seed; persist_drops (> 0.30 for any core model). Cross-checked against STEP 10a's
rules.json; the two rule functions are re-run on edited copies of the measured inputs (boundary checks). It computes;
it decides nothing (rebuild or drop is Max's, notes STEP 10b).
Since 2026-10-02 (Max: took all recommendations in rc12/DECISIONS_FOR_MAX.md, item 1): LOOKUP is out of the composite;
the headroom rule runs over the 9 composite families and the Level A keys, and LOOKUP's row is written beside it
(headroom_reported, the reason in s11; it decides nothing). --panel names the dev_panel.py output to cross-check
(the regraded panel after item 3, and after STEP 11b). Since 2026-10-04 LOOKUP is its own headline claim (Max; prereg
draft s10b): an absolute bar, so the headroom row stays a report; the claim's dev comparison is dev_panel.py's
rules.json lookup_claim."""
import argparse
import copy
import json
import os
from collections import Counter, defaultdict

import grade_text as T
import runner as RN
import score as S
import score_stats as ST

CORE = ["Qwen2.5-0.5B-Instruct", "Qwen3.5-0.8B", "LFM2.5-230M", "LFM2.5-350M", "Falcon-H1-Tiny-90M-Instruct",
        "SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct", "Doge-160M-Instruct", "LFM2-2.6B"]          # s12
EXTRAS = ["Qwen3-0.6B", "gemma-3-270m-it", "LFM2-700M", "LFM2-1.2B"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return [json.loads(x) for x in open(p)]


def rows_of(root, model, seeds):
    out = []
    for s in seeds:
        d = os.path.join(root, model, "template")
        for x in (s, s + "_owncf"):
            assert os.path.exists(os.path.join(d, x, "DONE")), f"{model} {x}: no DONE"
            out += load(os.path.join(d, x, "transcripts.jsonl"))
    return out


def key_values(summ):
    return {k: summ["families"][k] if k in summ["families"] else summ["keys"].get(k)
            for k in ST.HEADROOM_KEYS + S.REPORTED}


def as_panel(vals):
    """{model: {key: value}} -> the summary shape score_stats.headroom reads."""
    return {m: dict(families={k: v[k] for k in S.COMPOSITE}, keys={k: v[k] for k in list(S.LEVEL_A) + S.REPORTED})
            for m, v in vals.items()}


def lookup_evidence(rows, recs):
    rr = [r for r in S.select(rows) if r["family"] == "LOOKUP" and not r.get("own_cf")]
    by, clauses, x = defaultdict(list), defaultdict(Counter), Counter()
    for r in rr:
        for p in r["probes"]:
            by[p["grader"]].append(p)
            clauses[p["grader"]].update(p["fails"] or ["(ok)"])
        xp = [p for p in recs[r["id"]]["probes"] if p["grader"] == "ABS"][0]
        gp = [p for p in r["probes"] if p["grader"] == "ABS"][0]
        txt = T.norm([t for t in r["turns"] if t["i"] == xp["turn"]][0]["reply"])
        anyv = any(T.mentioned(txt, v) for v in xp["candidates"])
        cue = "a2_cue" not in gp["fails"]
        x.update(other_table_value=T.mentioned(txt, xp["other_value"]), any_value=anyv, cue=cue,
                 cue_no_value=cue and not anyv, degenerate="a1_degen" in gp["fails"])
    n = len(rr)
    return dict(units=n, unit=sum(r["unit"] for r in rr) / n,
                unit_lenient=sum(all(p["lenient"] for p in r["probes"]) for r in rr) / n,
                probe_ok={g: sum(p["ok"] for p in ps) / len(ps) for g, ps in by.items()},
                probe_lenient={g: sum(bool(p["lenient"]) for p in ps) / len(ps) for g, ps in by.items()},
                clauses={g: dict(c.most_common()) for g, c in clauses.items()},
                abstain_probe={k: v / n for k, v in x.items()})


def boundary_checks(vals, rates):
    """the two rules on edited copies of the measured inputs; True where each reacts as s11's text says."""
    def fails(over):
        v = copy.deepcopy(vals)
        for (m, k), x in over.items():
            v[m][k] = x
        return {k: h["reason"] for k, h in ST.headroom(as_panel(v), ST.HEADROOM_KEYS + S.REPORTED).items()
                if h["fail"]}
    top = max(CORE, key=lambda m: vals[m]["LOOKUP"])
    rule = "one_sentence:None"
    hi = max(CORE, key=lambda m: rates[m][rule])
    r1, r2 = copy.deepcopy(rates), copy.deepcopy(rates)
    r1[hi][rule], r2[hi][rule] = 0.30, 0.3001
    ceil = {(m, "CORR:U"): 0.95 for m in CORE}
    return {"LOOKUP top model at 0.050 still floor": fails({(top, "LOOKUP"): 0.05}).get("LOOKUP") == "floor",
            "LOOKUP top model at 0.051 passes": "LOOKUP" not in fails({(top, "LOOKUP"): 0.051}),
            "OWN: LFM2-2.6B at 0.05 makes OWN fail": fails({("LFM2-2.6B", "OWN"): 0.05}).get("OWN") == "floor",
            "every CORR:U 0.95 -> ceiling": fails(ceil).get("CORR:U") == "ceiling",
            "one CORR:U 0.949 -> passes": "CORR:U" not in fails({**ceil, (CORE[0], "CORR:U"): 0.949}),
            "base rate 0.30 not dropped": ST.persist_drops(r1) == [],
            "base rate 0.3001 dropped": ST.persist_drops(r2) == [rule]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "dev_panel", "s11.json"))
    ap.add_argument("--panel", default=os.path.join(HERE, "runs", "dev_panel"))
    args = ap.parse_args()
    recs = RN.load(RN.DEV)
    lk_recs = {r["id"]: r for r in recs if r["family"] == "LOOKUP"}
    res = dict(core=CORE, headroom_keys=ST.HEADROOM_KEYS, thresholds=ST.HEADROOM, persist_drop=ST.PERSIST_DROP)
    rows, vals = {}, {}
    for m in CORE + EXTRAS:
        rows[m] = rows_of(args.root, m, ("1", "2", "3"))
        summ = S.summarize(rows[m])
        vals[m] = key_values(summ)
        if m in CORE:
            assert summ["keys"].get("BIND") == summ["families"]["BIND"], m
    core_vals = {m: vals[m] for m in CORE}
    head = ST.headroom(as_panel(core_vals))
    res["headroom"] = head
    res["headroom_reported"] = ST.headroom(as_panel(core_vals), S.REPORTED)
    res["clear_floor"] = {k: sum(v > ST.HEADROOM["floor"] for v in h["scores"].values()) for k, h in head.items()}
    res["under_ceiling"] = {k: sum(v < ST.HEADROOM["ceiling"] for v in h["scores"].values()) for k, h in head.items()}
    per_seed = {m: {s: key_values(S.summarize(rows_of(args.root, m, (s,)))) for s in ("1", "2", "3", "greedy")}
                for m in CORE}
    res["per_seed"] = per_seed
    res["headroom_fails_per_seed"] = {s: {k: h["reason"] for k, h in ST.headroom(as_panel(
        {m: per_seed[m][s] for m in CORE})).items() if h["fail"]} for s in ("1", "2", "3", "greedy")}
    res["lookup"] = {m: lookup_evidence(rows[m], lk_recs) for m in CORE + EXTRAS}
    rules = ST.persist_rules(recs)
    res["persist_rules"] = [f"{a}:{b}" for a, b in rules]
    rates = {m: ST.persist_base_rates(rows[m], rules) for m in CORE}
    res["persist_rates_core"] = rates
    res["persist_rates_extras"] = {m: ST.persist_base_rates(rows[m], rules) for m in EXTRAS}
    res["persist_replies"] = {m: sum(len(r["turns"]) for r in S.select(rows[m])
                                     if r["family"] != "PERSIST" and not r.get("own_cf")) for m in CORE}
    res["persist_drops_core"] = ST.persist_drops(rates)
    seed_rates = {(m, s): ST.persist_base_rates([r for r in rows[m] if r["seed"] == s], rules)
                  for m in CORE for s in (1, 2, 3)}
    res["persist_max_core"] = {k: max((rates[m][k], m) for m in CORE) for k in res["persist_rules"]}
    res["persist_max_core_one_seed"] = {k: max((v[k], m, s) for (m, s), v in seed_rates.items())
                                        for k in res["persist_rules"]}
    res["boundary_checks"] = boundary_checks(core_vals, rates)
    old = os.path.join(args.panel, "rules.json")
    if os.path.exists(old):
        o = json.load(open(old))
        res["equals_panel_rules_json"] = dict(
            headroom=all(abs(o["headroom"][k]["scores"][m] - head[k]["scores"][m]) < 1e-12
                         and o["headroom"][k]["fail"] == head[k]["fail"] for k in head for m in CORE),
            persist=all(abs(o["persist_rates_core"][m][k] - rates[m][k]) < 1e-12 for m in CORE for k in rates[m]),
            drops=o["persist_drops_core"] == res["persist_drops_core"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("headroom fails:", {k: h["reason"] for k, h in head.items() if h["fail"]})
    print("reported (out of R):", {k: (h["reason"], max(h["scores"].values())) for k, h in res["headroom_reported"].items()})
    print("per seed:", res["headroom_fails_per_seed"])
    print("PERSIST drops:", res["persist_drops_core"])
    print("boundary checks:", res["boundary_checks"])
    print("equals the panel rules.json:", res.get("equals_panel_rules_json"))
    return 0 if all(res["boundary_checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
