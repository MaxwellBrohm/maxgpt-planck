"""A2. Pass rule (E004 (c) as E005 used it), written from the notes' text: cells from raw scores (LIK) and stored
strict (GEN), D4004 plain; PASS / PARTIAL-G / PARTIAL-X / FAIL; the any-seed variant of the never-partial clause;
leave-one-seed-out; fewest item flips (up to 3 cells); alias reading; stopping yardstick."""
import itertools
from common import *

U_X = ["H3", "H4", "H5", "H6", "H7"]
NEVER = ["H1", "H2", "C_noupd", "C_twoslot"]


def cells(arm, seeds=SEEDS):
    L = matrix(arm, "e004", "plain", "LIK", PASS9, seeds)
    G = matrix(arm, "e004", "plain", "GEN", PASS9, seeds)
    return {(s, kd, f): int(M[f][j].sum()) for kd, M in (("LIK", L), ("GEN", G)) for f in PASS9
            for j, s in enumerate(seeds)}


def seed_ok(c, s, fams, kinds=("LIK", "GEN")):
    return all(c[(s, kd, f)] >= 52 for f in fams for kd in kinds)   # >= 0.8 of 64 means 52 or more


def reading(c, seeds=SEEDS, planned=5, any_seed=False):
    need = planned // 2 + 1
    if sum(seed_ok(c, s, PASS9) for s in seeds) >= need:
        return "PASS"
    if any_seed and any(c[(s, "LIK", f)] < 52 for s in seeds for f in NEVER):
        return "FAIL"
    if sum(seed_ok(c, s, PASS9, ("LIK",)) for s in seeds) >= need:
        return "PARTIAL-G"
    best = None
    for r in (1, 2):
        for X in itertools.combinations(U_X, r):
            n = sum(seed_ok(c, s, [f for f in PASS9 if f not in X]) for s in seeds)
            if n >= need and (best is None or (len(X), -n) < (len(best[0]), -best[1])):
                best = (X, n)
        if best:
            return f"PARTIAL-X {'+'.join(best[0])} ({best[1]} seeds)"
    return "FAIL"


def fewest_flips(c, label):
    keys = list(c)
    dist = {k: (c[k] - 51) if c[k] >= 52 else (52 - c[k]) for k in keys}
    flip = lambda k: (51 if c[k] >= 52 else 52)
    best = None
    for r in (1, 2, 3):
        for combo in itertools.combinations(sorted(keys, key=lambda k: dist[k])[:40], r):
            tot = sum(dist[k] for k in combo)
            if best and tot >= best[0]:
                continue
            c2 = dict(c)
            for k in combo:
                c2[k] = flip(k)
            if reading(c2).split(" (")[0] != label.split(" (")[0]:
                best = (tot, combo)
    return best


def alias_and_stop(arm):
    hi = lo = 0
    rows, stops = [], []
    for s in SEEDS:
        t = tag_of(arm, s)
        lik = [lik_right(r) for r in load(t, "e004", "plain", root_of(arm)) if r["family"] in ("H1", "H2")
               and r.get("latest_ref") == "alias"]
        gen = [r["strict"] for r in load(t, "gen_e004", "plain", root_of(arm)) if r["family"] in ("H1", "H2")
               and r.get("latest_ref") == "alias"]
        a, b = sum(lik) / len(lik), sum(gen) / len(gen)
        hi += a >= 0.8 and b >= 0.8
        lo += a <= 0.5 and b <= 0.5
        rows.append(f"s{s} {sum(lik)}/{len(lik)} {sum(gen)}/{len(gen)}")
        P = matrix(arm, "e004", "plain", "GEN", PASS9, [s])
        C = matrix(arm, "e004", "chat", "GEN", PASS9, [s])
        gaps = [C[f].mean() - P[f].mean() for f in PASS9]
        stops.append((bool(max(abs(g) for g in gaps) <= 0.10 + 1e-12), float(min(gaps)), float(max(gaps))))
    label = "DATA GAP" if hi >= 3 else "REAL LIMIT" if lo >= 3 else "INCONCLUSIVE"
    return label, hi, lo, rows, stops


if __name__ == "__main__":
    for arm in ("C", "P", "G", "e005w", "e004w", "E005"):
        c = cells(arm)
        lab = reading(c)
        anyv = reading(c, any_seed=True)
        loo = {f"-s{d}": reading({k: v for k, v in c.items() if k[0] != d}, [s for s in SEEDS if s != d])
               for d in SEEDS}
        ff = fewest_flips(c, lab)
        print(f"== {arm}: {lab} | any-seed {anyv}")
        print("   LOO", {k: v.split(' (')[0] for k, v in loo.items()})
        print("   fewest flips", ff[0] if ff else None, [k for k in ff[1]] if ff else "")
        print("   seeds passing all but H5:", [s for s in SEEDS if seed_ok(c, s, [f for f in PASS9 if f != "H5"])])
        for s in SEEDS:
            bad = [f"{kd} {f} {c[(s, kd, f)]}" for f in PASS9 for kd in ("LIK", "GEN") if c[(s, kd, f)] < 52]
            thin = [f"{kd} {f} {c[(s, kd, f)]}" for f in PASS9 for kd in ("LIK", "GEN")
                    if 52 <= c[(s, kd, f)] <= 53 and f != "H5"]
            print(f"   s{s} failing: {bad}; thin (52-53): {thin}")
        lab2, hi, lo, rows, stops = alias_and_stop(arm)
        print(f"   alias {lab2} (HIGH {hi}, LOW {lo}): {rows}")
        print(f"   stopping learned: {sum(x[0] for x in stops) >= 3} per seed {[x[0] for x in stops]}"
              f" gaps {[(round(x[1], 3), round(x[2], 3)) for x in stops]}")
