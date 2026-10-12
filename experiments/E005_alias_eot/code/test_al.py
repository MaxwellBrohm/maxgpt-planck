"""No-model mutation test of the AL validation (checks_al.py, purity_al.py) and the AL tables (analyze_al.py).
System python, no torch. Every mutant must be flagged by the checker it targets; the baselines must be clean.
  checks  M1 a lure that belongs to the asked object (AL1's alias correction made A's) -> G2 LN gate
          M2 one wrong gold -> G1;  M3 a value in a question -> G5;  M4 one question form flipped -> G4
          M5 a training surname in the AL surname list -> G6;  M6 an alias correction naming its object -> G4
  purity  P1 a training surname in an AL turn;  P2 a training alias template as an alias correction;
          P3/P5 a training filler as an AL filler turn;  P4 a training sentence frame as a statement
  tables  scripted models on scratch outputs: an IDEAL model, an LN model (picks "latest statement with a name"),
          a T model (adjacency tracker); exact expected shares; the ADDITION sentence fires only with DATA GAP and
          >= 3 of 5 E005 seeds below 0.5
usage: python3 -B test_al.py <scratch dir> > ../logs/test_al.txt"""
import copy
import json
import os
import shutil
import sys

import items_al as A
import checks_al as CA
import purity_al as PU
import analyze_al as AN
import pools_alias_train as PA
import train_e005 as T5

RES = []


def expect(name, ok):
    RES.append((name, ok))
    print(f"{'ok  ' if ok else 'FAIL'} {name}")


def gates(items):
    return CA.rule_gates(CA.rule_table(items)) + CA.structure(items)


def mutant_items():
    base = A.load()
    expect("baseline items: rule + structure gates clean", gates(base) == [])
    orig = A.PLANS["AL1"]

    def al1_own(rng, place, f):     # the late alias correction belongs to the ASKED object: alias recency wins
        ev, asked, aliased = orig(rng, place, f)
        ev[-1]["obj"] = 0
        return ev, asked, sorted(set(aliased) | {0})
    A.PLANS["AL1"] = al1_own
    try:
        m1 = A.build()
    finally:
        A.PLANS["AL1"] = orig
    f = gates(m1)
    expect(f"M1 own-object lure flagged by G2 ({[x for x in f if x.startswith('G2')][:1]})",
           any(x.startswith("G2 LN") for x in f))
    m2 = copy.deepcopy(base)
    m2[5]["gold"] = [c for c in m2[5]["candidates"] if c != m2[5]["gold"]][0]
    expect("M2 wrong gold flagged by G1", any(x.startswith("G1 IDEAL") for x in gates(m2)))
    m3 = copy.deepcopy(base)
    m3[9]["question"] = m3[9]["question"].rstrip("?") + f" on {m3[9]['gold']}?"
    expect("M3 value in a question flagged by G5", any("value in question" in x for x in CA.structure(m3)))
    m4 = copy.deepcopy(base)
    m4[0]["q_form"] = "head" if m4[0]["q_form"] == "full" else "full"
    expect("M4 unbalanced question form flagged by G4", any("question form" in x for x in CA.structure(m4)))
    keep = list(A.AL_SURNAMES)
    A.AL_SURNAMES.append(PA.SURNAMES[0])
    try:
        expect("M5 training surname in the AL list flagged by G6",
               any("training pool" in x for x in CA.names_check(base)))
    finally:
        A.AL_SURNAMES[:] = keep
    m6 = copy.deepcopy(base)
    it = m6[20]
    s = [x for x in it["stmts"] if x["ref"] == "alias"][0]
    u, a = it["turns"][s["turn"]]
    it["turns"][s["turn"]] = [u.rstrip(".") + f" for the {it['objects'][s['obj']][1]}.", a]
    expect("M6 alias correction naming its object flagged by G4", any("names an object" in x for x in CA.structure(m6)))
    return base


def purity_mutants(base):
    small = dict(seeds=(1,), per_seed=7000)
    expect("baseline purity (seed 1) clean", PU.purity(base, **small) == [])
    p1 = copy.deepcopy(base)
    u, a = p1[3]["turns"][0]
    p1[3]["turns"][0] = [u + f" Mr. {PA.SURNAMES[3]} agrees.", a]
    expect("P1 training surname in AL text flagged", any(x.startswith("P1 training name") for x in PU.purity(p1, **small)))
    p2 = copy.deepcopy(base)
    it = p2[21]
    s = [x for x in it["stmts"] if x["ref"] == "alias"][0]
    tpl = PA.ALIAS_CORR[it["vtype"]][0]
    it["turns"][s["turn"]] = [tpl.format(a=it["aliases"][s["obj"]], v=s["value"]), it["turns"][s["turn"]][1]]
    expect("P2 training alias template flagged", any(x.startswith("P2") for x in PU.purity(p2, **small)))
    p3 = copy.deepcopy(base)
    it = p3[40]
    st = {x["turn"] for x in it["stmts"]}
    j = [i for i in range(len(it["turns"])) if i not in st][-1]
    from fillers_train import FILLERS_TRAIN
    it["turns"][j] = list(FILLERS_TRAIN[0])
    f = PU.purity(p3, **small)
    expect("P3 and P5 training filler flagged", any(x.startswith("P3") for x in f) and any(x.startswith("P5") for x in f))
    from pools_train import POOLS
    p4 = copy.deepcopy(base)
    it, s = [(y, x) for y in p4 for x in y["stmts"] if x["role"] == "orig" and not y["aliases"][x["obj"]]][0]
    tpl = POOLS[it["vtype"]]["orig"][0]         # a training template filled with the AL object and value
    it["turns"][s["turn"]] = [tpl.format(o=it["objects"][s["obj"]][0], v=s["value"]), it["turns"][s["turn"]][1]]
    expect(f"P4 training sentence frame flagged ({tpl!r})", any(x.startswith("P4") for x in PU.purity(p4, **small)))


def fake_model(tag, items, pick_of, out):
    for render in AN.RENDERS:
        with open(os.path.join(out, f"{tag}__al__{render}.jsonl"), "w") as f:
            for it in items:
                vals = {"gold": it["gold"], **{f"c{i}": v for i, v in enumerate(
                    [c for c in it["candidates"] if c != it["gold"]], 1)}}
                p = pick_of(it)
                sc = {k: (0.0 if v == p else -1.0) for k, v in vals.items()}
                f.write(json.dumps(dict(idx=it["idx"], cell=it["cell"], scores=sc, cand_vals=vals,
                                        right=p == it["gold"])) + "\n")
        with open(os.path.join(out, f"{tag}__gen_al__{render}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(dict(idx=it["idx"], strict=pick_of(it) == it["gold"])) + "\n")
    sha = open(A.SHA_PATH).read().split()[0]
    json.dump(dict(model=tag, items_sha256=sha, selftest_fail=[]), open(os.path.join(out, f"{tag}__run.json"), "w"))


def tables(base, scratch):
    out = os.path.join(scratch, "al_out")
    shutil.rmtree(out, ignore_errors=True)       # a stale scratch dir would carry models from an earlier run
    os.makedirs(out)
    AN.OUT, AN.TABLES, AN.RESULTS = out, os.path.join(scratch, "t.txt"), os.path.join(scratch, "r.json")
    AN.E005_RESULTS = os.path.join(scratch, "results.json")
    rv = AN.rule_values(base)
    fake_model("base", base, lambda it: it["gold"], out)
    fake_model("e005_s1", base, lambda it: rv[it["idx"]]["LN"], out)
    fake_model("e005_s2", base, lambda it: rv[it["idx"]]["LN"], out)
    fake_model("e005_s3", base, lambda it: rv[it["idx"]]["T"], out)
    json.dump({"e005": {"reading": {"alias": {"label": "DATA GAP"}}}}, open(AN.E005_RESULTS, "w"))
    AN.main()
    r = json.load(open(AN.RESULTS))["models"]
    lp = lambda t, k: r[t]["LIK plain"][k]
    expect("IDEAL model 1.00 everywhere", all(v == 1.0 for v in r["base"]["LIK plain"].values()))
    expect("LN model: all .25, AL3 1.00, AL1+AL2+AL4 0, needs linking 0",
           (lp("e005_s1", "all"), lp("e005_s1", "AL3"), lp("e005_s1", "AL1+AL2+AL4"), lp("e005_s1", "needs linking"))
           == (0.25, 1.0, 0.0, 0.0))
    expect("LN model picks LN's value on all 48 items where LN is wrong",
           r["e005_s1"]["LIK plain picks LN's value where LN is wrong"] == [48, 48])
    expect("T model: all .75, needs linking 0", (lp("e005_s3", "all"), lp("e005_s3", "needs linking")) == (0.75, 0.0))
    ad = json.load(open(AN.RESULTS))["addition"]
    expect("ADDITION sentence off with 2 low seeds", ad["resolved_by_alias_recency"] is False)
    fake_model("e005_s4", base, lambda it: rv[it["idx"]]["LN"], out)
    AN.main()
    ad = json.load(open(AN.RESULTS))["addition"]
    expect("ADDITION sentence on with 3 low seeds and DATA GAP", ad["resolved_by_alias_recency"] is True)
    json.dump({"e005": {"reading": {"alias": {"label": "INCONCLUSIVE"}}}}, open(AN.E005_RESULTS, "w"))
    AN.main()
    expect("ADDITION sentence off without DATA GAP", json.load(open(AN.RESULTS))["addition"]["resolved_by_alias_recency"]
           is False)


def main():
    scratch = sys.argv[1]
    base = mutant_items()
    purity_mutants(base)
    tables(base, scratch)
    bad = [n for n, ok in RES if not ok]
    print(f"\nRESULT: {'ALL PASS' if not bad else f'{len(bad)} FAILED: {bad}'} ({len(RES)} checks)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
