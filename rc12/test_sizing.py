"""RC-12 tests of sizing.py (the s9 sealed-size simulation), no model, fixed seeds.
1 phi2 / rho_for: the bivariate-normal orthant against exact endpoints and a fixed-seed Monte Carlo; round trip.
2 analysis = score_stats.bootstrap_diff: synthetic rows (comparator 1 x 3 seeds, Planck 3 x 3, OWN with --own-cf
  twins, BIND as twin pairs) through the committed bootstrap_diff, and sizing.boot on the SAME random draws (a
  replica of bootstrap_diff's random.Random call order): D, lo and hi equal to 1e-9.
3 sizing.draws (numpy) vs the replica draws, 20,000 resamples each, on data where only the comparator's
  sampling seeds differ and on data where only Planck's training and sampling seeds differ: sd and mean of D agree.
4 estimator round trip: data generated from KNOWN p, rw, rs, rx (Planck's training seeds 0 and 1 as the two
  baselines, rt = rx) are estimated back to those values.
5 known answer: every seed identical (rs = rw = rt = 1) and 0/1 units with paired disagreement delta, so the analysis
  is the plain paired stratified bootstrap and power = Phi(3 / sigma - 1.96), sigma^2 = (100 / F)^2 sum_f delta / n_f
  over the F composite families (F = 9 since LOOKUP left the composite, 2026-10-02; the old text had 100 = (100/10)^2);
  delta set for power 0.50 and 0.80; also delta 0 -> D = lo = 0 in every split, power 1.
6 tau_for: the training-seed main effect gives the asked sd of a seed's expected R. 7 with that effect the mean D
  over 1,000 generated splits is 0 (Planck's threshold c sqrt(1 + tau^2) keeps its rate at p).
Run: ~/.venvs/planck/bin/python -B test_sizing.py   (exit 1 on any failure)"""
import math
import random
import sys

import numpy as np

import score as S
import score_stats as ST
import sizing as SZ

FAMS = list(S.COMPOSITE)
fails = []


def check(name, ok, info=""):
    print(("ok  " if ok else "FAIL") + f" {name} {info}")
    if not ok:
        fails.append(name)


def est_const(p, rx, rs, rw, ks_mean=(2, 3, 5, 6), ks_loop=(12,)):
    return {f: dict(p=p, rx=rx, rs=rs, rw=rw if f in SZ.MEAN_UNITS else rs,
                    ks=list(ks_mean if f == "PERSIST" else ks_loop if f == "LOOP" else (1,))) for f in FAMS}


def rows_of(data, model):
    rows, n_train = [], 3 if model == "P" else 1
    for t in range(n_train):
        for s in range(3):
            for f in FAMS:
                mat = data[f][1] if model == "P" else data[f][0]
                for i in range(mat.shape[0]):
                    base = dict(family=f, cell="c", knowledge=False, seed=s + 1,
                                train_seed=t + 1 if model == "P" else 0,
                                responder=model, render="template", own_cf=False, cf_unswapped=False, flags=[],
                                probes=[], leaks=[], pair_id=None, unit=float(mat[i, 3 * t + s]))
                    if f == "BIND":
                        for tw in "ab":
                            rows.append(dict(base, id=f"BIND-{i:04d}-{tw}", pair_id=f"BIND-{i:04d}"))
                        continue
                    rows.append(dict(base, id=f"{f}-{i:04d}"))
                    if f == "OWN":
                        rows.append(dict(base, id=f"{f}-{i:04d}", own_cf=True, unit=1.0))
    return rows


def replica_draws(nfs, B, seed):
    """bootstrap_diff's random.Random calls, in its order: the unit draw (S.COMPOSITE order), then Planck's picks
    (rows_a), then the comparator's (rows_b), as count / share arrays for sizing.boot."""
    rng = random.Random(seed)
    counts = [np.zeros((B, nf)) for nf in nfs]
    wq, wp = np.zeros((B, 3)), np.zeros((B, 9))
    for b in range(B):
        for j, nf in enumerate(nfs):
            for _ in range(nf):
                counts[j][b, rng.randrange(nf)] += 1
        for t in [rng.choice([0, 1, 2]) for _ in range(3)]:
            for s in [rng.choice([0, 1, 2]) for _ in range(3)]:
                wp[b, 3 * t + s] += 1 / 9
        for _t in [rng.choice([0]) for _ in range(1)]:
            for s in [rng.choice([0, 1, 2]) for _ in range(3)]:
                wq[b, s] += 1 / 3
    return counts, wq, wp


def main():
    # 1
    for p, rho in ((0.3, 0.0), (0.3, 1.0), (0.05, 1.0), (0.5, 0.0)):
        c = SZ.ND.inv_cdf(p)
        check(f"phi2 endpoint p={p} rho={rho}", abs(SZ.phi2(c, rho) - (p if rho == 1 else p * p)) < 1e-9)
    rng = np.random.default_rng(1)
    z1, e = rng.standard_normal(2_000_000), rng.standard_normal(2_000_000)
    for p, rho in ((0.2, 0.4), (0.05, 0.7), (0.5, 0.25)):
        c = SZ.ND.inv_cdf(p)
        mc = float(np.mean((z1 < c) & (rho * z1 + math.sqrt(1 - rho * rho) * e < c)))
        check(f"phi2 vs Monte Carlo p={p} rho={rho}", abs(SZ.phi2(c, rho) - mc) < 0.0015,
              f"{SZ.phi2(c, rho):.5f} {mc:.5f}")
        cov = SZ.phi2(c, rho) - p * p
        check(f"rho_for round trip p={p} rho={rho}", abs(SZ.rho_for(p, cov) - rho) < 1e-6)
    # 2
    est = est_const(0.3, 0.2, 0.4, 0.5)
    data = SZ.generate(est, FAMS, 128, np.random.default_rng(2), rt="x", eta=(np.array([-0.4, 0.0, 0.5]), 0.3))
    nfs = [data[f][0].shape[0] for f in FAMS]
    ref = ST.bootstrap_diff(rows_of(data, "P"), rows_of(data, "Q"), n=400, seed=11)
    got = SZ.ci(*SZ.boot(data, FAMS, *replica_draws(nfs, 400, 11)))
    check("boot == score_stats.bootstrap_diff (same draws)",
          all(abs(got[k] - ref[k]) < 1e-9 for k in ("D", "lo", "hi")), f"ref {ref} got {got}")
    # 3 one data set where only the comparator's sampling seeds differ, one where only Planck's seeds differ
    g3 = np.random.default_rng(3)
    for name, rq, rp in (("comparator seeds", (0.1, 0.5, 0.9), [0.5] * 9),
                         ("Planck seeds", (0.5,) * 3, [r + d for r in (0.2, 0.5, 0.8) for d in (-0.1, 0.0, 0.1)])):
        d3 = {f: ((g3.random((n, 3)) < np.array(rq)).astype(float), (g3.random((n, 9)) < np.array(rp)).astype(float))
              for f, n in zip(FAMS, nfs)}
        _, a = SZ.boot(d3, FAMS, *SZ.draws(nfs, 20000, g3))
        _, b = SZ.boot(d3, FAMS, *replica_draws(nfs, 20000, 12))
        check(f"numpy draws ~ replica draws, {name} (sd, mean of D)",
              abs(a.std() / b.std() - 1) < 0.04 and abs(a.mean() - b.mean()) < 0.1 * b.std(),
              f"sd {a.std():.4f} {b.std():.4f} mean {a.mean():.4f} {b.mean():.4f}")
    # 4
    known = est_const(0.25, 0.3, 0.5, 0.6)
    big = SZ.generate(known, FAMS, 640 * 100, np.random.default_rng(4), rt="x")
    units = {"Q": {s: {f: {i: big[f][0][i, s] for i in range(big[f][0].shape[0])} for f in FAMS} for s in range(3)}}
    for t in (0, 1):
        units[f"B{t}"] = {s: {f: {i: big[f][1][i, 3 * t + s] for i in range(big[f][1].shape[0])} for f in FAMS}
                          for s in range(3)}
    ks = {f: {i: sorted(known[f]["ks"])[int(i * len(known[f]["ks"]) / big[f][0].shape[0])]
              for i in range(big[f][0].shape[0])} for f in SZ.MEAN_UNITS}
    back = SZ.estimate(units, ks, "Q", ["B0", "B1"], fams=["PERSIST", "LOOP", "CORR"])
    for f, e in back.items():
        dev = {k: round(e[k] - known[f][k], 4) for k in ("p", "rx", "rs", "rw")}
        check(f"estimate round trip {f}", all(abs(v) < 0.03 for v in dev.values()), str(dev))
    # 5
    p = 0.3
    sum_inv = sum(1 / SZ.n_units(f, 640) for f in FAMS)
    for target in (0.50, 0.80):
        sigma = 3 / (1.96 + ND_inv(target))
        delta = sigma * sigma / ((100 / len(FAMS)) ** 2 * sum_inv)     # F families (9 since 2026-10-02, item 1)
        rx = SZ.rho_for(p, p * (1 - p) - delta / 2)
        pw = SZ.power(est_const(p, rx, 1.0, 1.0, (1,), (1,)), FAMS, 640, 1000, 2000, seed=5, rt="s")
        check(f"known-answer power {target}", abs(pw["power"] - target) < 0.05,
              f"delta {delta:.4f} sim {pw['power']:.3f} sd_D {pw['sd_D']:.3f} vs sigma {sigma:.3f}")
    pw = SZ.power(est_const(p, 1.0, 1.0, 1.0, (1,), (1,)), FAMS, 600, 50, 500, seed=6, rt="s")
    check("delta 0: D = lo = 0 (to float noise), power 1",
          pw["power"] == 1.0 and abs(pw["mean_lo"]) < 1e-9 and pw["sd_D"] < 1e-9, f"{pw['mean_lo']:.2e}")
    # 6
    e6 = est_const(0.2, 0.2, 0.4, 0.5)
    tau = SZ.tau_for(e6, FAMS, 1.5)
    etas = np.random.default_rng(7).normal(0, tau, 200000)
    c = SZ.ND.inv_cdf(0.2) * math.sqrt(1 + tau * tau)
    r = 100 * np.array([SZ.ND.cdf(c + x) for x in etas[:20000]])
    check("tau_for gives the asked seed sd (1.5 points) and keeps the rate", abs(r.std() - 1.5) < 0.05
          and abs(r.mean() - 20.0) < 0.05, f"tau {tau:.4f} sd {r.std():.3f} mean {r.mean():.3f}")
    # 7 the training-seed main effect keeps the true difference at 0 (Planck's rate averaged over seeds = p)
    e7 = est_const(0.1, 0.2, 0.4, 0.5)
    tau = SZ.tau_for(e7, FAMS, 5.0)
    g = np.random.default_rng(8)
    ds = []
    for _ in range(1000):
        d = SZ.generate(e7, FAMS, 640, g, rt="x", eta=(g.normal(0, tau, 3), tau))
        ds.append(10 * sum(d[f][1].mean() - d[f][0].mean() for f in FAMS))
    se = np.std(ds) / math.sqrt(len(ds))
    check("seed main effect: mean D ~ 0", abs(np.mean(ds)) < 3 * se, f"mean {np.mean(ds):.3f} se {se:.3f}")
    print("ALL SIZING CHECKS PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
    return 1 if fails else 0


def ND_inv(x):
    return SZ.ND.inv_cdf(x)


if __name__ == "__main__":
    sys.exit(main())
