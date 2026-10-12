"""K2 mask-fit defaults sweep on the planted-set fixture (tests/plantfix.py; review R-2). CPU, tiny model only.
Each run is one masks.fit at the longest step count; the hardened top-b set after S updates is read from the live
gates at the start of step S (fit draws its gate noise in the same order, so a run of S steps ends there too;
check_equivalence confirms it on one case). One JSON line per (case, floors, init, lam, seed, S). K3 round 3: with a
single step count the record is fit's own result (restarts included, and per restart whether it found the set);
reading several step counts inside one run needs --restarts 1.
  python tests/mask_sweep.py OUT.jsonl [--init 0 1] [--lam 1 3] [--steps 300 1000] [--seeds 0 1 2]
      [--floors k2 default] [--cases SF1 ...] [--out-gain 2] [--restarts 2] [--procs 4]  (defaults: masks.fit's)
  python tests/mask_sweep.py --summary OUT.jsonl [--tau 0.5]
floors k2: K2's normalization (fact floor = CE with the planted fact units ablated, skill floor = log V);
default: fit's (None, None), CE0 + 1 on each side. --out-gain g gives every plant the output gain g (default: the
fixture's mix of graded and saturated plants). The 2026-09-27 runs behind the defaults: K2 notes K2-e.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import kcommon as K                                                   # noqa: E402

if K.HARNESS not in sys.path:
    sys.path.insert(1, K.HARNESS)
import plantfix as PF                                                 # noqa: E402

CASES = PF.CASES


def one(case, floors, init, lam, seed, steps, out=None, per_budget=False, restarts=1):
    """-> list of records, one per checkpoint in steps (sorted). out: a scalar output gain for every plant (else
    plantfix.OUT). restarts: masks.fit's restarts (K3 round 3; 1 before it)."""
    import torch
    torch.set_num_threads(1)
    import masks as M
    import plantfix as PF
    want_out = PF.OUT_DEFAULT if out is None else {k: float(out) for k in PF.OUT_DEFAULT}
    if PF.OUT != want_out:
        PF.OUT = want_out
        PF.build.cache_clear()
    m, *_, units = PF.build()
    obj, facts, b, want = CASES[case]
    fc, sc, fl = PF.objectives(facts)
    want_u = {units[k] for k in want}
    if len(steps) == 1:                         # fit's own result, restarts included
        t0 = time.time()
        r = M.fit(m, fc, sc, obj, budget=b, steps=steps[0], lam=lam, init=init, seed=seed,
                  floors=fl if floors == "k2" else (None, None), per_budget=bool(per_budget), restarts=restarts)
        drop = {(layer, j): v for layer, ps in r["p_drop"].items() for j, v in enumerate(ps)}
        top = sorted(PF.as_set(r["units"]))
        return [{"case": case, "out": out, "floors": floors, "init": init, "lam": lam, "seed": seed,
                 "steps": steps[0], "per_budget": bool(per_budget), "restarts": restarts,
                 "found": set(top) == want_u, "top": [list(u) for u in top],
                 "runs_found": [PF.as_set(u) == want_u for _, u in r["runs"]], "hard": r["hard"],
                 "fp_max": max(v for u, v in drop.items() if u not in want_u),
                 "tp_min": min(drop[u] for u in want_u), "e_drop": sum(drop.values()),
                 "sec": round(time.time() - t0, 1)}]
    assert restarts == 1, "several step counts are read inside one run: restarts 1"
    spy, marks, at = [], sorted(steps), {}
    calls = [0]

    class Spy(M.GatedModel):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            spy.append(self)

    def fact(model):
        calls[0] += 1
        s = calls[0] - 2                         # updates applied before this call (call 1 is the no-grad f0)
        if s in marks[:-1] and spy:
            at[s] = snapshot(spy[-1], b)
        return fc(model)
    orig, M.GatedModel = M.GatedModel, Spy
    try:
        t0 = time.time()
        r = M.fit(m, fact, sc, obj, budget=b, steps=marks[-1], lam=lam, init=init, seed=seed,
                  floors=fl if floors == "k2" else (None, None), per_budget=bool(per_budget), restarts=1)
        at[marks[-1]] = snapshot(spy[-1], b)
        dt = time.time() - t0
    finally:
        M.GatedModel = orig
    assert at[marks[-1]]["top"] == sorted(PF.as_set(r["units"]))
    recs = []
    for s in marks:
        a = at[s]
        drop = a["drop"]
        recs.append({"case": case, "out": out, "floors": floors, "init": init, "lam": lam, "seed": seed, "steps": s,
                     "per_budget": bool(per_budget), "restarts": 1,
                    "found": set(map(tuple, a["top"])) == want_u, "top": [list(u) for u in a["top"]],
                    "fp_max": max(v for u, v in drop.items() if u not in want_u),
                    "tp_min": min(drop[u] for u in want_u), "e_drop": sum(drop.values()), "sec": round(dt, 1)})
    return recs


def snapshot(gm, b):
    drop = {(l, j): float(1 - g.p_open()[j].detach()) for l, g in gm.gates.items() for j in range(len(g.log_alpha))}
    return {"top": sorted(gm.ranking()[:b]), "drop": drop}


def check_equivalence(seed=1, steps=120):
    """A fit of `steps` updates ends on the set this sweep reads at `steps` inside a longer run, both at masks.fit's
    defaults (K3 round 2, REVIEW 3b D-5: since R-2 this compared the defaults with init 1.0, lam 1 per unit).
    -> (same set, |E[dropped] difference|, the set)."""
    import inspect
    import masks as M
    import plantfix as PF
    d = {k: v.default for k, v in inspect.signature(M.fit).parameters.items()}
    fc, sc, fl = PF.objectives(PF.ALLF)
    short = M.fit(PF.build()[0], fc, sc, "SF", budget=4, steps=steps, seed=seed, floors=fl, restarts=1)
    rec = one("MX-SF", "k2", d["init"], d["lam"], seed, (steps, steps + 80), per_budget=d["per_budget"],
              restarts=1)[0]
    same_set = rec["top"] == sorted(map(list, PF.as_set(short["units"])))
    return same_set, abs(rec["e_drop"] - short["history"][-1]["drop"]), rec["top"]


def _job(args):
    return one(*args)


def summary(path, tau):
    rows = [json.loads(x) for x in open(path)]
    key = lambda r: (str(r.get("out")), r["floors"], r["init"], r["lam"], r["steps"],      # noqa: E731
                     int(r.get("per_budget", False)), r.get("warm", 0), r.get("restarts", 1))   # warm: K3 r3 trial
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    cases = list(CASES)
    print(f"out floors init lam steps pb warm rs | found (of seeds) per case: {' '.join(cases)} | all | fp>{tau} | "
          "max fp | min tp | runs found")
    for k in sorted(groups):
        g = groups[k]
        per = {c: [r for r in g if r["case"] == c] for c in cases}
        cells = " ".join(f"{sum(r['found'] for r in per[c])}/{len(per[c])}" for c in cases)
        ok = all(r["found"] for r in g)
        nfp = sum(r["fp_max"] > tau for r in g)
        runs = [x for r in g for x in r.get("runs_found", [r["found"]])]
        print(f"{k[0]:4s} {k[1]:7s} {k[2]:4} {k[3]:4} {k[4]:5} {k[5]} {k[6]:4} {k[7]} | {cells} | "
              f"{'ALL' if ok else '-':3s} | "
              f"{nfp:3d} | {max(r['fp_max'] for r in g):.3f} | {min(r['tp_min'] for r in g):.3f} | "
              f"{sum(runs)}/{len(runs)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    import inspect
    import masks as M
    dflt = {k: v.default for k, v in inspect.signature(M.fit).parameters.items()}
    ap.add_argument("--init", type=float, nargs="*", default=[dflt["init"]])      # masks.fit's defaults
    ap.add_argument("--lam", type=float, nargs="*", default=[dflt["lam"]])
    ap.add_argument("--steps", type=int, nargs="*", default=[dflt["steps"]])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--floors", nargs="*", default=["k2"])
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out-gain", type=float, nargs="*", default=[None])
    ap.add_argument("--per-budget", type=int, nargs="*", default=[int(dflt["per_budget"])])
    ap.add_argument("--restarts", type=int, default=dflt["restarts"])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.summary:
        return summary(a.out, a.tau)
    jobs = [(c, f, i, lam, s, tuple(a.steps), o, pb, a.restarts) for c, f, i, lam, s, o, pb in
            itertools.product(a.cases, a.floors, a.init, a.lam, a.seeds, a.out_gain, a.per_budget)]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool, open(a.out, "a") as fh:
        for n, recs in enumerate(pool.imap_unordered(_job, jobs), 1):
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    summary(a.out, a.tau)


if __name__ == "__main__":
    main()
