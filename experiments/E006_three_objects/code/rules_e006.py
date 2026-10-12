"""E006 decision rules (notes.txt PRE-REGISTERED RULES AND READINGS), pure functions: no files, no model, no torch.
Bootstraps use numpy.random.default_rng(6006) and 10,000 resamples; a 95% CI is the 2.5th and 97.5th percentile.
  paired   right_X(s, i) - right_Y(s, i), paired by seed number and item; each resample draws k seed numbers and n
           items with replacement and averages the difference over the drawn (seed, item) pairs
  chat     the same two-way resampling over seeds and conversations; a turn-level measure is the drawn turns' mean,
           CHECKS is the mean over drawn seeds of the checks passed summed over the drawn conversations
Labels (first match wins; the tests exercise every branch): q_h5, q_chat, knowledge, cost_mark."""
import numpy as np

B, BOOT_SEED = 10000, 6006
GAIN, NTC, DROP, DEV_TOL, TRADE = 0.06, 0.06, 0.05, 0.03, -0.05
K_BIG, LOOP_MARGIN, CHECKS_MARGIN, HALF = 0.03, 0.10, -3.0, 0.5


def _ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def paired(X, Y, seeds=None, items=None, n_boot=B, seed=BOOT_SEED):
    """X, Y: {seed: {item: 0/1}} -> (d, lo, hi, k seeds, n items); None if no complete common seed."""
    seeds = sorted(set(X) & set(Y)) if seeds is None else list(seeds)
    if not seeds or any(s not in X or s not in Y for s in seeds):
        return None
    items = sorted(set.intersection(*(set(X[s]) & set(Y[s]) for s in seeds))) if items is None else list(items)
    if not items:
        return None
    D = np.array([[float(X[s][i]) - float(Y[s][i]) for i in items] for s in seeds])
    k, n = D.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, k, size=(n_boot, k))
    ii = rng.integers(0, n, size=(n_boot, n))
    sc = np.zeros((n_boot, k))
    np.add.at(sc, (np.arange(n_boot)[:, None], si), 1.0)
    ic = np.zeros((n_boot, n))
    np.add.at(ic, (np.arange(n_boot)[:, None], ii), 1.0)
    bm = np.einsum("bk,kn,bn->b", sc, D, ic) / (k * n)
    lo, hi = _ci(bm)
    return float(D.mean()), lo, hi, k, n


def chat_boot(Xs, Ys, key, n_boot=B, seed=BOOT_SEED):
    """Xs, Ys: {seed: {conv: [turn rows]}} (rows hold key, 0/1) -> (d, lo, hi): drawn turns' mean of X - Y."""
    seeds = sorted(set(Xs) & set(Ys))
    convs = sorted(set.intersection(*(set(Xs[s]) & set(Ys[s]) for s in seeds)))
    S = np.array([[sum(float(r[key]) for r in Xs[s][c]) - sum(float(r[key]) for r in Ys[s][c]) for c in convs]
                  for s in seeds])
    N = np.array([[len(Xs[s][c]) for c in convs] for s in seeds], dtype=float)
    return _two_way(S, N, n_boot, seed)


def checks_boot(Xc, Yc, n_boot=B, seed=BOOT_SEED):
    """Xc, Yc: {seed: {conv: checks passed}} -> (d, lo, hi) in checks per model (of 73)."""
    seeds = sorted(set(Xc) & set(Yc))
    convs = sorted(set.intersection(*(set(Xc[s]) & set(Yc[s]) for s in seeds)))
    S = np.array([[float(Xc[s][c] - Yc[s][c]) for c in convs] for s in seeds])
    return _two_way(S, None, n_boot, seed, per_model=True)


def _two_way(S, N, n_boot, seed, per_model=False):
    k, n = S.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, k, size=(n_boot, k))
    ci_ = rng.integers(0, n, size=(n_boot, n))
    sc = np.zeros((n_boot, k))
    np.add.at(sc, (np.arange(n_boot)[:, None], si), 1.0)
    cc = np.zeros((n_boot, n))
    np.add.at(cc, (np.arange(n_boot)[:, None], ci_), 1.0)
    num = np.einsum("bk,kn,bn->b", sc, S, cc)
    if per_model:
        bm = num / k
        d = float(S.sum() / k)
    else:
        bm = num / np.einsum("bk,kn,bn->b", sc, N, cc)
        d = float(S.sum() / N.sum())
    lo, hi = _ci(bm)
    return d, lo, hi


def q_h5(lik, gen, p_l4, c_l4_gap_lik, dev_shift_lik, h5l):
    """lik, gen: (d, lo, hi) of P - C on BIG H5; p_l4: {"LIK": (d, lo, hi), "GEN": ...} of P - L4 (e004w);
    c_l4_gap_lik: L4 - C, pooled BIG H5 LIK (None when e004w is incomplete: flag "L4 NOT READ"); dev_shift_lik: C - e005w, pooled BIG H5 LIK; h5l: {"LIK": (d, lo, hi),
    "GEN": ...} of P - C on H5L. -> dict(label, flags, suffix)."""
    if lik is None or gen is None:
        return {"label": "NOT READ", "flags": [], "suffix": None}
    both = lambda f: f(lik) and f(gen)
    if c_l4_gap_lik is None:  # e004w incomplete: no drop is measured, so neither flag state is claimed
        flags = ["L4 NOT READ"]
    else:
        flags = [] if c_l4_gap_lik >= DROP else ["NO DROP TO RESTORE ON CUDA"]
    suffix = None
    if both(lambda x: x[0] >= GAIN and x[1] > 0):
        if dev_shift_lik is None or abs(dev_shift_lik) >= DEV_TOL:
            label = "GAIN (RESTORED VS PARTLY NOT READ: training-device shift >= 0.03)"
        elif p_l4 and all(p_l4.get(m) and p_l4[m][2] > 0 for m in ("LIK", "GEN")):
            label = "RESTORED"
        else:
            label = "PARTLY RESTORED"
        traded = [m for m in ("LIK", "GEN") if h5l and h5l.get(m) and h5l[m][0] <= TRADE and h5l[m][2] < 0]
        suffix = f"recency traded ({', '.join(traded)})" if traded else "no loss where the gold is last"
    elif both(lambda x: x[2] < 0):
        label = "WORSE"
    elif both(lambda x: x[2] < NTC):
        label = "NOT THE CAUSE"
    else:
        label = "INCONCLUSIVE"
    return {"label": label, "flags": flags, "suffix": suffix}


def q_chat(tf, tf2, tf_c, tf_g, tf2_c, tf2_g, loop, checks, stops):
    """tf, tf2, loop, checks: (d, lo, hi) of G - C; tf_c.. pooled shares; stops: G's stopping yardstick (bool).
    -> dict(label, failed): PROTECTS / PARTLY PROTECTS / SMALL EFFECT / WORSE / NO EFFECT."""
    if None in (tf, tf2, loop, checks):
        return {"label": "NOT READ", "failed": []}
    a_ci = tf[2] < 0 and tf2[2] < 0
    a = a_ci and tf_g <= HALF * tf_c and tf2_g <= HALF * tf2_c
    failed = []
    if not loop[2] < LOOP_MARGIN:
        failed.append("(b) loops")
    if not checks[1] > CHECKS_MARGIN:
        failed.append("(c) probe checks")
    if not stops:
        failed.append("(d) stopping")
    if a and not failed:
        return {"label": "PROTECTS", "failed": []}
    if a:
        return {"label": "PARTLY PROTECTS", "failed": failed}
    if a_ci:
        return {"label": "SMALL EFFECT", "failed": failed}
    if tf[1] > 0 or tf2[1] > 0:
        return {"label": "WORSE", "failed": failed}
    return {"label": "NO EFFECT", "failed": failed}


def knowledge(k):
    """k: (K, lo, hi) of G - C on the unexposed kbig items (items x seeds bootstrap)."""
    if k is None:
        return "NOT READ"
    K, lo, hi = k[:3]
    if K >= K_BIG and lo > 0:
        return "REDUCES THE COST"
    if lo > 0:
        return "SMALL REDUCTION"
    if hi < 0:
        return "INCREASES THE COST"
    return "NO EFFECT"


def cost_mark(x):
    if x is None:
        return "NOT READ"
    d, lo, hi = x[:3]
    if d <= -DROP and hi < 0:
        return "COST"
    if hi < 0:
        return "LOWER"
    return "-"


def sentence(name, x):
    return f"{name}: " + ("not read" if x is None else f"{x[0]:+.3f}, CI {x[1]:+.3f} to {x[2]:+.3f} (k {x[3]}, n {x[4]})"
                          if len(x) > 3 else f"{x[0]:+.3f}, CI {x[1]:+.3f} to {x[2]:+.3f}")
