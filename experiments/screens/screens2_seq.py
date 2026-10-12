"""SCREENS2 B4 (5) SEQUENTIAL RULE (experiments/SCREENS2.txt, entry "SEQUENTIAL RULE", 2026-10-12): C7 read as a
two-look group-sequential design. Look 1 at k 5 (seeds 101-105); a contrast INCONCLUSIVE there (after B4 (3)) runs
seeds 106-110 for the arm AND its reference (base2, an own BASE, base7 or S019's canon_ac) and is read once more at
k 10 (look 2), which is final. Stdlib only (scipy is not in Planck's venv); no file is read here.

  spend      Lan-DeMets Pocock-type alpha spending, alpha(t) = 0.025 ln(1 + (e - 1) t), one-sided per direction
             (C7's t(0.975)); t = k1 / 10 at look 1, the rest (0.025 - alpha(t1)) at look 2
  bounds     (c1, c2) in t units for a reference SD with df_ref: c1 = t_ppf(1 - alpha(t1), df); c2 solves
             P0(T1 <= c1, T2 > c2) = 0.025 - alpha(t1), where T_j = Z_j / W share ONE W = SD_ref / sigma
             (sqrt(chi2_df / df), the reference is reused at both looks) and corr(Z1, Z2) = sqrt(k1 / k2)
  reading    one metric at one look (as analyze_lib.metric_reading): thr = c x SD_ref / sqrt(k); BETTER needs
             -dbar > thr and, at look 1, every d_s < 0 (C7); at look 2 at least sign_min(k2) of the d_s < 0 (9 of 10:
             a one-sided sign test no weaker than C7's every-seed clause at k 5, p <= 1/32); WORSE mirrored;
             EQUAL |dbar| + thr < delta; p is the stagewise-ordering p (look 1: C7's two-sided t p)
The error control does not depend on the sign clauses, the EQUAL stop or B4 (3): each only removes crossings, so
P0(BETTER) <= 0.025 (and WORSE, and EQUAL at |mu| = delta, by the same boundaries) under C7's t model.
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analyze_lib as A  # noqa: E402
from analyze_lib import chi2_ppf, t_cdf, t_ppf  # noqa: E402

ALPHA = 0.025                                      # one-sided, per direction: C7's t(0.975, df)
K1, K2 = 5, 10                                     # look 1, look 2 (planned)
SEEDS1, SEEDS2 = (101, 102, 103, 104, 105), (106, 107, 108, 109, 110)
SIGN_P = 1 / 32                                    # C7's every-seed clause at k 5, read as a one-sided sign test
N_GL = 64


def spend(t: float) -> float:
    """Pocock-type spending: alpha(t) = ALPHA ln(1 + (e - 1) t), t clipped to [0, 1]."""
    return ALPHA * math.log(1 + (math.e - 1) * min(max(t, 0.0), 1.0))


@lru_cache(maxsize=None)
def _gl(n: int = N_GL) -> tuple[tuple[float, float], ...]:
    """Gauss-Legendre (node, weight) pairs on [-1, 1] (Newton on P_n)."""
    out = []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        out.append((x, 2 / ((1 - x * x) * dp * dp)))
    return tuple(out)


def _phi_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _low_high(a: float, b: float, rho: float) -> float:
    """P(Z1 <= a, Z2 > b) for standard normals with correlation rho."""
    lo = -9.0
    if a <= lo:
        return 0.0
    s, h, m, tot = math.sqrt(1 - rho * rho), (a - lo) / 2, (a + lo) / 2, 0.0
    for x, w in _gl():
        z = m + h * x
        tot += w * math.exp(-z * z / 2) * (1 - _phi_cdf((b - rho * z) / s))
    return tot * h / math.sqrt(2 * math.pi)


@lru_cache(maxsize=None)
def _w_nodes(df: int | None) -> tuple[tuple[float, float], ...]:
    """(W, weight) for E_W[.], W = sqrt(chi2_df / df), on the quantile scale u in (0, 1); df None: W = 1."""
    if df is None:
        return ((1.0, 1.0),)
    return tuple((math.sqrt(chi2_ppf((x + 1) / 2, df) / df), w / 2) for x, w in _gl())


def cont_cross(c1: float, c2: float, rho: float, df: int | None) -> float:
    """P0(T1 <= c1, T2 > c2) with T_j = Z_j / W (one shared W)."""
    return sum(w * _low_high(c1 * v, c2 * v, rho) for v, w in _w_nodes(df))


def _q(p: float, df: int | None) -> float:
    return NormalDist().inv_cdf(p) if df is None else t_ppf(p, df)


@lru_cache(maxsize=None)
def bounds(df: int | None, k1: int = K1, k2: int = K2, kmax: int = K2) -> dict:
    """Look boundaries in t units for one reference df; k1, k2 the seeds actually read (C9 drops a diverged BASE seed)."""
    if not 2 <= k1 < k2:
        raise ValueError(f"need 2 <= k1 < k2, got {k1}, {k2}")
    a1 = spend(k1 / kmax)
    c1, rho, rest = _q(1 - a1, df), math.sqrt(k1 / k2), ALPHA - a1
    lo, hi = 0.0, 60.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cont_cross(c1, mid, rho, df) > rest:
            lo = mid
        else:
            hi = mid
    return {"df": df, "k1": k1, "k2": k2, "a1": a1, "a2": rest, "c1": c1, "c2": (lo + hi) / 2, "rho": rho}


def sign_min(k: int) -> int:
    """Fewest d_s on the arm's side for BETTER/WORSE at look 2: the smallest n <= k whose one-sided sign-test tail
    P(Bin(k, 1/2) >= n) is at most 1/32 (C7's clause at k 5); every d_s if no n qualifies."""
    for n in range(k + 1):
        if sum(math.comb(k, j) for j in range(n, k + 1)) / 2 ** k <= SIGN_P:
            return n
    return k


def p_stagewise(t_abs: float, look: int, df: int | None, k1: int = K1, k2: int = K2) -> float:
    """Two-sided stagewise-ordering p: look 1 C7's t p; look 2 P0(|T1| >= c1) + P0(|T1| < c1, |T2| >= t_abs)."""
    if look == 1:
        return 2 * (1 - (_phi_cdf(t_abs) if df is None else t_cdf(t_abs, df)))
    b = bounds(df, k1, k2)
    c1 = b["c1"]
    inner = cont_cross(c1, t_abs, b["rho"], df) - cont_cross(-c1, t_abs, b["rho"], df)
    return min(1.0, 2 * b["a1"] + 2 * inner)


def reading(d: list[float], sd: float, df: int, base_mean: float, look: int, k1: int | None = None) -> dict:
    """One metric at one look (bpb; +inf for a diverged arm seed). Look 2 needs k1, the seeds read at look 1."""
    k = len(d)
    if look == 1:
        b, need = bounds(df, k, K2), k
        c = b["c1"]
    elif look == 2:
        b, need = bounds(df, k1, k), sign_min(k)
        c = b["c2"]
    else:
        raise ValueError(f"look {look}")
    dbar = A.INF if any(x == A.INF for x in d) else sum(d) / k
    se = sd / math.sqrt(k)
    thr, delta = c * se, A.DELTA * base_mean
    better = -dbar > thr and sum(x < 0 for x in d) >= need
    worse = dbar > thr and sum(x > 0 for x in d) >= need
    equal = abs(dbar) + thr < delta
    label = "BETTER" if better else "WORSE" if worse else "EQUAL" if equal else "UNRESOLVED"
    p = 0.0 if not math.isfinite(dbar) else p_stagewise(abs(dbar) / se, look, df, b["k1"], b["k2"])
    return {"label": label, "d": d, "k": k, "dbar": dbar, "thr": thr, "c": c, "need": need, "delta": delta,
            "sd_ref": sd, "df": df, "look": look, "p_two_sided": p}


def extends(verdict: str) -> bool:
    """Look 1 -> look 2 only on INCONCLUSIVE (after B4 (3)); every other look-1 verdict is final."""
    return verdict == "INCONCLUSIVE"
