"""RC-12 sealed-size rule (prereg draft s9 "Sealed size", PLAN sizing rule). No model, no GPU; numpy only.
estimate(units, ks, comparator, baselines): from dev runs (template, sampling seeds 1-3, the analysis's own units:
  score_stats.unit_table, OWN gated), per composite family: p = the mean rate of the two symmetrized pairs
  (comparator, baseline); the within-conversation correlation phi_w (ICC of the graded turns of one conversation,
  for PERSIST and LOOP, whose unit is a share of 2 to 12 turns; every other composite unit is one 0/1 outcome);
  msd_x, the paired disagreement (mean squared paired unit difference over all seed pairings, = the disagreement
  rate for 0/1 units) of the comparator vs each baseline; msd_s, the same model across sampling seeds. Each is
  mapped to a latent (probit) correlation at p: rw (turns of one conversation), rs (sampling seeds), rx (models).
  A latent correlation is never below 0, so where the observed disagreement exceeds that of two independent units
  at p, msd_*_model (what the simulation reproduces) is below msd_* and the shortfall is reported.
generate(est, fams, n, rng, rt, eta): a split of n conversations (family f has round(DEV_UNITS[f] n / 640) units),
  a comparator (1 training seed x 3 sampling seeds) and a Planck stand-in (3 x 3), both at rate p (true difference
  0). Turn = 1[Z < c], Z = sqrt(rx) A_u + sqrt(rt-rx) B_mu + sqrt(rs-rt) C_mtu + sqrt(rw-rs) E_mtsu + sqrt(1-rw) e.
  rt (Planck's training seeds) is an assumption in [rx, rs]. eta (optional): Planck seed t's threshold c' + eta_t,
  c' = c sqrt(1 + tau^2), so the rate stays p (a training-seed main effect, sensitivity only).
draws(nfs, B, rng) + boot(...): score_stats.bootstrap_diff vectorised (one unit draw per family shared by both
  models, training seeds resampled per model, sampling seeds inside each drawn training seed, percentile CI with
  lo = sorted[int(0.025 B)]); test_sizing.py checks it against bootstrap_diff on the same random draws.
power(...): share of simulated splits whose CI passes score_stats.noninferior (lower bound >= -3), PLAN's original
  Level R test. Since 2026-10-04 Level R reads S7's one-model CI lower bound >= 40 instead (Max, 2026-10-04: took all
  recommendations in rc12/DECISIONS_LEVEL_R_FOR_MAX.md (decision 1, D)) and the -3 test is reported beside; the
  sealed size this rule gave, 600, stands (that bar barely depends on n; prereg draft s9)."""
import math
from statistics import NormalDist

import numpy as np

import score_stats as ST

ND = NormalDist()
DEV_RECORDS = 640      # prereg s5: 640 records = 608 twelve-turn conversations + 32 one-turn K controls
DEV_UNITS = dict(RECALL=60, CORR=64, BIND=30, TWOHOP=48, PERSIST=48, OWN=48, TOPIC=48, ROLE=48, LOOKUP=48, LOOP=40)
MEAN_UNITS = ("PERSIST", "LOOP")
_GL = np.polynomial.legendre.leggauss(64)


def phi2(c, rho):
    """P(Z1 < c, Z2 < c) for a standard bivariate normal with correlation rho in [0, 1] (Plackett; r = sin t)."""
    p = ND.cdf(c)
    top = math.asin(min(max(rho, 0.0), 1.0))
    if top == 0.0:
        return p * p
    t = 0.5 * top * (_GL[0] + 1)
    return p * p + 0.5 * top * float(np.sum(_GL[1] * np.exp(-c * c / (1 + np.sin(t))))) / (2 * math.pi)


def rho_for(p, cov):
    """the latent correlation whose two 0/1 outcomes at rate p have covariance cov (clamped to [0, 1])."""
    if p <= 0 or p >= 1 or cov <= 0:
        return 0.0
    c = ND.inv_cdf(p)
    if p * p + cov >= p:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi2(c, mid) - p * p < cov else (lo, mid)
    return (lo + hi) / 2


def unit_var(p, rw, ks):
    """variance of a unit that is the share of k turns at rate p with latent turn correlation rw (mean over ks)."""
    if p <= 0 or p >= 1:
        return 0.0
    cw = phi2(ND.inv_cdf(p), rw) - p * p
    return sum((p * (1 - p) + (k - 1) * cw) / k for k in ks) / len(ks)


def _msd(xs, ys):
    return float(np.mean((np.asarray(xs) - np.asarray(ys)) ** 2))


def estimate(units, ks, comparator, baselines, fams=None):
    """units {model: {seed: {family: {uid: score}}}} (one training seed); ks {family: {uid: k}} for MEAN_UNITS."""
    fams = fams or list(DEV_UNITS)
    trio = [comparator] + list(baselines)
    out = {}
    for f in fams:
        uids = sorted(units[comparator][next(iter(units[comparator]))][f])
        arr = {m: {s: [units[m][s][f][u] for u in uids] for s in units[m]} for m in trio}
        rate = {m: float(np.mean([x for v in arr[m].values() for x in v])) for m in trio}
        p = float(np.mean([(rate[comparator] + rate[b]) / 2 for b in baselines]))
        kl = [ks[f][u] for u in uids] if f in MEAN_UNITS else [1] * len(uids)
        phi_w, rw = None, None
        if f in MEAN_UNITS:
            einv = float(np.mean([1 / k for k in kl]))
            num = den = 0.0
            for m in trio:
                for v in arr[m].values():
                    pr = float(np.mean(v))
                    num += float(np.var(v, ddof=1)) - pr * (1 - pr) * einv
                    den += pr * (1 - pr) * (1 - einv)
            phi_w = min(max(num / den, 0.0), 1.0) if den > 0 else 0.0
            rw = rho_for(p, phi_w * p * (1 - p))
        var = unit_var(p, rw or 0.0, kl)
        msd_x = float(np.mean([_msd(arr[comparator][a], arr[b][s]) for b in baselines
                               for a in arr[comparator] for s in arr[b]]))
        msd_s = float(np.mean([_msd(arr[m][a], arr[m][s]) for m in trio for a in arr[m] for s in arr[m] if a < s]))
        msd_bb = float(np.mean([_msd(arr[b1][a], arr[b2][s]) for i, b1 in enumerate(baselines)
                                for b2 in baselines[i + 1:] for a in arr[b1] for s in arr[b2]]))
        rs_raw, rx_raw = rho_for(p, var - msd_s / 2), rho_for(p, var - msd_x / 2)
        rs = min(rs_raw, rw) if rw is not None else rs_raw
        rx = min(rx_raw, rs)
        cov = {k: phi2(ND.inv_cdf(p), r) - p * p if 0 < p < 1 else 0.0 for k, r in (("x", rx), ("s", rs))}
        model = {k: 2 * (var - c) for k, c in cov.items()}
        out[f] = dict(p=p, rates=rate, n_dev=len(uids), ks=kl, phi_w=phi_w, rw=rw if rw is not None else rs, rs=rs,
                      rx=rx, msd_x=msd_x, msd_s=msd_s, msd_bb=msd_bb, unit_var=var, msd_x_model=model["x"],
                      msd_s_model=model["s"], clamped=dict(rs=rs < rs_raw, rx=rx < rx_raw))
    return out


def n_units(f, n, base=DEV_RECORDS):
    return max(1, round(DEV_UNITS[f] * n / base))


def generate(est, fams, n, rng, rt="x", eta=None, base=DEV_RECORDS):
    """{family: (Q (nf x 3), P (nf x 9, column 3 t + s))} unit scores; eta: Planck's per-seed threshold shifts."""
    tau = 0.0 if eta is None else float(eta[1])
    shift = np.zeros(3) if eta is None else np.asarray(eta[0], dtype=float)
    out = {}
    for f in fams:
        e, nf = est[f], n_units(f, n, base)
        p = e["p"]
        if p <= 0 or p >= 1:
            out[f] = (np.full((nf, 3), float(p >= 1)), np.full((nf, 9), float(p >= 1)))
            continue
        kd = sorted(e["ks"])
        k = np.array([kd[int(u * len(kd) / nf)] for u in range(nf)])
        km = int(k.max())
        mask = np.arange(km)[None, :] < k[:, None]
        rx, rs, rw = e["rx"], e["rs"], e["rw"]
        r_t = rx if rt == "x" else rs if rt == "s" else float(rt)
        r_t = min(max(r_t, rx), rs)
        w = [math.sqrt(max(v, 0.0)) for v in (rx, r_t - rx, rs - r_t, rw - rs, 1 - rw)]
        a = rng.standard_normal(nf)[:, None, None]
        c = ND.inv_cdf(p)
        zq = (w[0] * a + w[1] * rng.standard_normal(nf)[:, None, None] + w[2] * rng.standard_normal(nf)[:, None, None]
              + w[3] * rng.standard_normal((nf, 3))[:, :, None] + w[4] * rng.standard_normal((nf, 3, km)))
        zp = (w[0] * a[:, :, :, None] + w[1] * rng.standard_normal(nf)[:, None, None, None]
              + w[2] * rng.standard_normal((nf, 3))[:, :, None, None]
              + w[3] * rng.standard_normal((nf, 3, 3))[:, :, :, None] + w[4] * rng.standard_normal((nf, 3, 3, km)))
        thr_p = c * math.sqrt(1 + tau * tau) + shift[None, :, None, None]
        q = ((zq < c) & mask[:, None, :]).sum(-1) / k[:, None]
        pp = ((zp < thr_p) & mask[:, None, None, :]).sum(-1) / k[:, None, None]
        out[f] = (q.astype(float), pp.reshape(nf, 9).astype(float))
    return out


def draws(nfs, B, rng):
    """numpy draws of the bootstrap: per family a (B x nf) count matrix (units drawn with replacement), the
    comparator's (B x 3) sampling-seed pick shares, Planck's (B x 9) (training seed, sampling seed) pick shares."""
    counts = []
    for nf in nfs:
        idx = rng.integers(0, nf, (B, nf)) + nf * np.arange(B)[:, None]
        counts.append(np.bincount(idx.ravel(), minlength=B * nf).reshape(B, nf).astype(float))
    sq = rng.integers(0, 3, (B, 3))
    wq = np.stack([(sq == j).sum(1) for j in range(3)], 1) / 3.0
    t = rng.integers(0, 3, (B, 3))
    s = rng.integers(0, 3, (B, 3, 3))
    flat = (t[:, :, None] * 3 + s).reshape(B, 9) + 9 * np.arange(B)[:, None]
    wp = np.bincount(flat.ravel(), minlength=B * 9).reshape(B, 9) / 9.0
    return counts, wq, wp


def boot(data, fams, counts, wq, wp):
    """(D point, sorted resampled D) for D = R(Planck) - R(comparator), R = 100 x mean over fams of F_f."""
    scale = 100.0 / len(fams)
    rq = np.zeros(len(wq))
    rp = np.zeros(len(wp))
    d = 0.0
    for f, cnt in zip(fams, counts):
        q, p = data[f]
        nf = q.shape[0]
        rq += ((cnt @ q) * wq).sum(1) / nf
        rp += ((cnt @ p) * wp).sum(1) / nf
        d += p.mean() - q.mean()
    return scale * d, np.sort(scale * (rp - rq))


def ci(point, diffs):
    B = len(diffs)
    return dict(D=point, lo=float(diffs[int(0.025 * B)]), hi=float(diffs[min(B - 1, int(0.975 * B))]), n=B)


def tau_for(est, fams, sigma_r):
    """tau whose common per-seed threshold shift gives sd sigma_r (R points) of a training seed's expected R."""
    if sigma_r <= 0:
        return 0.0
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / w.sum()

    def sd(tau):
        g = np.zeros_like(x)
        for f in fams:
            p = est[f]["p"]
            if 0 < p < 1:
                c = ND.inv_cdf(p) * math.sqrt(1 + tau * tau)
                g += np.array([ND.cdf(c + tau * v) for v in x])
        g *= 100.0 / len(fams)
        return math.sqrt(max(float((w * g * g).sum() - (w * g).sum() ** 2), 0.0))
    lo, hi = 0.0, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sd(mid) < sigma_r else (lo, mid)
    return (lo + hi) / 2


def power(est, fams, n, sims, B, seed, rt="x", sigma_r=0.0, base=DEV_RECORDS):
    """simulate the s9 analysis on `sims` splits of n conversations; power = share with noninferior(ci) true."""
    rng = np.random.default_rng([seed, n, int(round(1000 * sigma_r)), {"x": 1, "s": 2}.get(rt, 3), base])
    tau = tau_for(est, fams, sigma_r)
    nfs = [n_units(f, n, base) for f in fams]
    los, ds, ses = [], [], []
    for _ in range(sims):
        eta = (rng.normal(0.0, tau, 3), tau) if tau > 0 else None
        data = generate(est, fams, n, rng, rt, eta, base)
        point, diffs = boot(data, fams, *draws(nfs, B, rng))
        c = ci(point, diffs)
        los.append(c["lo"])
        ds.append(point)
        ses.append(float(np.std(diffs)))
    passed = [ST.noninferior(dict(lo=x)) for x in los]
    return dict(n=n, units=dict(zip(fams, nfs)), sims=sims, B=B, seed=seed, rt=rt, sigma_r=sigma_r, tau=tau,
                base=base, power=sum(passed) / sims, mean_lo=float(np.mean(los)), sd_D=float(np.std(ds)),
                mean_D=float(np.mean(ds)), mean_boot_se=float(np.mean(ses)))
