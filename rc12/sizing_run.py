"""RC-12 STEP 10d: run the s9 sealed-size rule on the dev baselines (sizing.py). No model, no GPU.
  python -B sizing_run.py --root <copy of ~/planck/runs/rc12_dev> --out runs/sizing [--sims 1000 --procs 8]
Trio (prereg s9 "the two baselines nearest to it"): PRIMARY = nearest on the dev composite R (template, mean of 3
seeds, runs/dev_panel/panel.json, STEP 10a); SIZE = nearest in parameter count (safetensors headers, tied
embeddings once: Qwen2.5-0.5B-Instruct 494.0M, Qwen3-0.6B 596.0M, SmolLM2-360M-Instruct 361.8M, LFM2.5-350M
354.5M; every other panel model is farther). Writes estimates.json, power.json, power.tsv; prints the table.
Configs (each a full curve n = 600..1500 step 100, 10,000 resamples per split, seeds fixed):
  primary_x / primary_s   PRIMARY trio, Planck's training seeds as different as two models (rt = rx) / as alike
                          as two sampling seeds (rt = rs); the rule's n is the larger of the two
  size_x / size_s         the SIZE trio
  base608                 units scaled per 608 twelve-turn conversations
  seedsd_<s>              a training-seed main effect: sd s points of a Planck seed's expected R (sensitivity)
The families are score.COMPOSITE: 9 since 2026-10-02, LOOKUP out (Max, 2026-10-02: took all recommendations in
rc12/DECISIONS_FOR_MAX.md (item 1)), so the old "nolookup" config is the primary one and is gone."""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool  # noqa: E402

import dev_panel as DP  # noqa: E402
import score as S  # noqa: E402
import score_stats as ST  # noqa: E402
import sizing as SZ  # noqa: E402

COMP = DP.COMPARATOR
SIZE_TRIO = ["Qwen3-0.6B", "SmolLM2-360M-Instruct"]
NS = list(range(600, 1501, 100))
SEED = 20261002
SEED_SDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_model(root, model):
    rows = []
    for sd in ("1", "2", "3"):
        dev, cf = DP.run_rows(root, model, "template", sd)
        rows += dev + cf
    table = ST.unit_table(rows)
    assert list(table) == [0] and sorted(table[0]) == [1, 2, 3], (model, list(table))
    ks = {}
    for r in rows:
        if r["family"] in SZ.MEAN_UNITS and not r.get("own_cf"):
            k = len(r["probes"])
            assert abs(sum(p["ok"] for p in r["probes"]) / k - r["unit"]) < 1e-12, r["id"]
            assert ks.setdefault(r["family"], {}).setdefault(r["id"], k) == k, r["id"]
    for f in S.COMPOSITE:
        assert len(table[0][1][f]) == SZ.DEV_UNITS[f], (model, f, len(table[0][1][f]))
        if f not in SZ.MEAN_UNITS:
            assert all(x in (0.0, 1.0) for s in table[0].values() for x in s[f].values()), (model, f)
    r_mean = sum(100 * sum(sum(v.values()) / len(v) for v in table[0][s].values()) / len(S.COMPOSITE)
                 for s in (1, 2, 3)) / 3
    return table[0], ks, r_mean


def job(a):
    name, est, fams, n, sims, B, rt, sd, base = a
    out = SZ.power(est, fams, n, sims, B, SEED, rt, sd, base)
    out["config"] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "runs/dev_panel/panel.json"))
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--sens-sims", type=int, default=500)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    panel = {r["model"]: r["R"] for r in json.load(open(args.panel))["mean3"] if r["render"] == "template"}
    by_r = sorted((abs(r - panel[COMP]), m) for m, r in panel.items() if m != COMP)
    r_trio = [m for _, m in by_r[:2]]
    models = sorted({COMP, *r_trio, *SIZE_TRIO})
    units, ks = {}, {}
    for m in models:
        units[m], k, r_mean = load_model(args.root, m)
        assert abs(r_mean - panel[m]) < 1e-9, (m, r_mean, panel[m])      # same units as the committed scorer
        for f, d in k.items():
            assert ks.setdefault(f, d) == d, f"k differs across models in {f}"
    fams = list(S.COMPOSITE)
    est = {"PRIMARY": SZ.estimate(units, ks, COMP, r_trio, fams), "SIZE": SZ.estimate(units, ks, COMP, SIZE_TRIO, fams)}
    cfg = [("primary_x", "PRIMARY", fams, args.sims, "x", 0.0, 640),
           ("primary_s", "PRIMARY", fams, args.sims, "s", 0.0, 640),
           ("size_x", "SIZE", fams, args.sens_sims, "x", 0.0, 640),
           ("size_s", "SIZE", fams, args.sens_sims, "s", 0.0, 640)]
    jobs = [(c, est[t], fs, n, sims, args.B, rt, sd, base) for c, t, fs, sims, rt, sd, base in cfg for n in NS]
    with Pool(args.procs) as pool:
        res = pool.map(job, jobs, chunksize=1)
    pw = {(r["config"], r["n"]): r["power"] for r in res}
    worst = min(("x", "s"), key=lambda rt: sum(pw[(f"primary_{rt}", n)] for n in NS))
    more = [("base608", "PRIMARY", fams, args.sens_sims, worst, 0.0, 608)]
    more += [(f"seedsd_{sd}", "PRIMARY", fams, args.sens_sims, worst, sd, 640) for sd in SEED_SDS]
    jobs = [(c, est[t], fs, n, sims, args.B, rt, sd, base) for c, t, fs, sims, rt, sd, base in more for n in NS]
    with Pool(args.procs) as pool:
        res += pool.map(job, jobs, chunksize=1)
    curves = {}
    for r in res:
        curves.setdefault(r["config"], []).append(r)
    rule = {}
    for c, rs in curves.items():
        ok = [r["n"] for r in sorted(rs, key=lambda r: r["n"]) if r["power"] >= 0.80]
        rule[c] = dict(n=ok[0] if ok else None, cap_power=max(rs, key=lambda r: r["n"])["power"])
    n_rule = max(rule["primary_x"]["n"] or 10 ** 9, rule["primary_s"]["n"] or 10 ** 9)
    summary = dict(comparator=COMP, primary_trio=r_trio, size_trio=SIZE_TRIO, dev_R={m: panel[m] for m in models},
                   conservative_rt=worst, rule=rule,
                   n=None if n_rule == 10 ** 9 else n_rule, B=args.B, seed=SEED)
    with open(os.path.join(args.out, "estimates.json"), "w") as f:
        json.dump(est, f, indent=1)
    with open(os.path.join(args.out, "power.json"), "w") as f:
        json.dump(dict(summary=summary, results=res), f, indent=1)
    cols = ["config", "n", "power", "mean_lo", "mean_D", "sd_D", "mean_boot_se", "tau", "sims", "B"]
    with open(os.path.join(args.out, "power.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(res, key=lambda r: (r["config"], r["n"])):
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(json.dumps(summary, indent=1))
    for c, rs in curves.items():
        print(f"{c:14s}", " ".join(f"{r['n']}:{r['power']:.3f}" for r in sorted(rs, key=lambda r: r["n"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
