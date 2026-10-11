"""Fake complete T001 output directories for testing results.py and rederive.py (never results). Each run file has
train.py's JSON shape: raw counts per (split, family, n) with the rates train.score derives from them, a loss curve,
params and wall time. Arm skills are drawn per fake seed, so different seeds give different R1/R2/R3 outcomes; runplan
names every run, so a fake dir is complete exactly when runplan says DONE."""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runplan as R  # noqa: E402

ARMS = R.PICK_ARMS + R.DIAG_ARMS


def _rec(rnd, skill, ovs, n, fam, force=None):
    q = 50 * n
    ok = int(q * min(1, skill * (1.1 - n / 200) + rnd.uniform(-.03, .03)))
    uq = q // 3
    uok = int(uq * min(1, ovs + .1 + rnd.uniform(-.05, .05)))
    oq = q // 12
    ook = int(oq * min(1, ovs * (1.05 - n / 300) + rnd.uniform(-.05, .05)))
    tq = q // 5
    tok = int(tq * rnd.uniform(.5, 1))
    kq = q // 4 if fam == "track" else 0
    kok = int(kq * rnd.uniform(.3, .9))
    if rnd.random() < 0.02:
        oq = ook = 0                                       # an n without outvoted queries (counts 0 in LWO)
    if force is not None:                  # fixed accuracies on all and on outvoted queries: a rate or (all, outvoted)
        fa, fo = force if isinstance(force, tuple) else (force, force)
        oq = max(oq, 1)
        ok, ook = int(fa * q), int(round(fo * oq))
    stale = int((uq - uok) * rnd.uniform(.5, 1))
    first = int(stale * rnd.uniform(0, 1))
    d = dict(q=q, ok=ok, upd_q=uq, upd_ok=uok, stale=stale, first=first, trk_q=kq, trk_ok=kok, ov_q=oq, ov_ok=ook,
             tie_q=tq, tie_ok=tok)
    d.update(acc=ok / q, lw=uok / uq, trk_acc=kok / kq if kq else None, acc_ov=ook / oq if oq else None,
             acc_tie=tok / tq)
    return d


def build(out: str, seed: int, p_div: float = 0.02, fire_a: bool = False, fire_b=False, ceiling=(),
          ar32: dict | None = None, lr_tie: tuple = ()) -> str:
    """Writes a complete fake out dir (every run runplan asks for) and returns out. Knobs that reach rule branches:
    fire_a / fire_b make base's C0 run trip C0 rule (a) / (b) (fire_b "acc_only": dev acc 0.95 but outvoted acc 0.5
    at every n, which must not trip it); ceiling, a tuple of arms or a dict arm -> top n, makes every verdict run of
    the arm score 0.95 on all and on outvoted test queries up to top n (all n for a tuple: a censored cap) and 0.5
    above; ar32 maps an arm to a fixed plain-recall accuracy at n = 32 in its verdict runs; every seed-0 LR run of an
    arm in lr_tie gets the same dev accuracies (a tie between LRs)."""
    ceiling = ceiling if isinstance(ceiling, dict) else {a: 10 ** 6 for a in ceiling}
    ar32 = ar32 or {}
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(seed)
    skill = {a: rnd.uniform(0.6, 1.0) for a in ARMS}
    ovs = {a: rnd.uniform(0.2, 1.0) for a in ARMS}
    params = {a: 400_000 + 1_000 * i for i, a in enumerate(ARMS)}
    for _ in range(600):
        sp, phase = R.plan(out)
        if sp is None:
            return out
        a, g = sp["arm"], sp["grid"]
        div = phase == "VERDICT" and rnd.random() < p_div
        sk = skill[a] + rnd.uniform(-.04, .04)              # seed-to-seed spread

        def recs(fam, force=None):
            return {str(n): _rec(rnd, sk, ovs[a], n, fam, force(n) if force else None) for n in g}
        dev = test = None
        if phase == "C0" and fire_a and sp["steps"] == 6000:
            dev = lambda n: 0.3 if n == 8 else 0.95                     # noqa: E731
        elif phase == "C0" and fire_b and sp["n_hi"] == 96:
            dev = (lambda n: (0.95, 0.5)) if fire_b == "acc_only" else (lambda n: 0.95)     # noqa: E731
        elif phase == "LR" and a in lr_tie:
            dev = lambda n: 0.5                                          # noqa: E731
        vd = phase == "VERDICT" and sp["family"] == "mqrar"
        if vd and a in ceiling:
            test = lambda n: 0.95 if n <= ceiling[a] else 0.5            # noqa: E731
        arf = (lambda n: ar32[a] if n == 32 else None) if vd and a in ar32 else None      # noqa: E731
        if sp["family"] == "mqrar":
            ev = {"test": {"mqrar": recs("mqrar", test), "ar": recs("ar", arf)}, "dev": {"mqrar": recs("mqrar", dev)}}
        else:
            ev = {"test": {"track": recs("track"), "mqrar": recs("mqrar")}}
        curve = [[0, 5.0], [sp["steps"] - 1, None if div else round(rnd.uniform(0.1, 2.0), 4)]]
        js = dict(arm=a, seed=sp["seed"], lr=sp["lr"], steps=sp["steps"], n_hi=sp["n_hi"], grid=g,
                  family=sp["family"], diverged=div, eval=None if div else ev, params=params[a],
                  wall_s=round(rnd.uniform(100, 300), 2), train_s=90.0, loss_curve=curve)
        with open(os.path.join(out, sp["name"] + ".json"), "w") as f:
            json.dump(js, f)
    raise RuntimeError("fake plan did not finish")


if __name__ == "__main__":
    print(build(sys.argv[1], int(sys.argv[2])))
