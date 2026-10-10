"""T001 run order (PREREG.txt CALIBRATION C0, LEARNING RATE, SEEDS AND PAIRING, GPU PLAN). Stateless: the next run
is derived from the JSON files already in the output directory, so the PC driver can stop and resume anywhere.

  python runplan.py next --out DIR      -> "DONE", or the train.py arguments of the next run
  python runplan.py status --out DIR    -> counts per phase
  python runplan.py verdict --out DIR   -> R1, R2, R3 and the per-arm table (only once every verdict run exists)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
from model import DIAG_ARMS, PICK_ARMS  # noqa: E402

GRID0 = (8, 16, 24, 32, 48, 64, 96)
GRID_EXT = (128, 160)
LRS = (3e-4, 1e-3, 3e-3)
EXT = {3e-4: 1e-4, 3e-3: 1e-2}
TRACK_ARMS, TRACK_SEEDS = ("base", "gdn", "canon"), (1, 2, 3)
C0_A_STEPS, C0_A_ACC8, C0_B_ACC = 12000, 0.5, 0.9


def name(arm, family, seed, lr, st) -> str:
    return f"{arm}_{family}_s{seed}_lr{lr:g}_st{st['steps']}_n{st['n_hi']}"


def load(out, nm):
    p = os.path.join(out, nm + ".json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def dev_acc(js) -> dict:
    """{n: {"acc": dev mqrar accuracy, "acc_ov": dev outvoted-query accuracy}}; {} for a diverged run."""
    if js.get("diverged") or not js.get("eval"):
        return {}
    return {int(n): dict(acc=r["acc"], acc_ov=r["acc_ov"]) for n, r in js["eval"]["dev"]["mqrar"].items()}


def _get(d: dict, n: int, key: str) -> float:
    return (d.get(n) or {}).get(key) or 0.0


def spec(arm, family, seed, lr, st) -> dict:
    return dict(arm=arm, family=family, seed=seed, lr=lr, steps=st["steps"], n_hi=st["n_hi"], grid=st["grid"],
                name=name(arm, family, seed, lr, st))


def settings(out):
    """(settings, None) once C0 is final, else (settings so far, the C0 run still needed). Each rule fires once."""
    st = dict(steps=6000, n_hi=96, grid=list(GRID0), a=False, b=False)
    while True:
        js = load(out, name("base", "mqrar", 0, 1e-3, st))
        if js is None:
            return st, spec("base", "mqrar", 0, 1e-3, st)
        d = dev_acc(js)
        if not st["a"] and _get(d, 8, "acc") < C0_A_ACC8:
            st.update(steps=C0_A_STEPS, a=True)
            continue
        if not st["b"] and all(_get(d, n, "acc") >= C0_B_ACC and _get(d, n, "acc_ov") >= C0_B_ACC for n in GRID0):
            st.update(n_hi=max(GRID_EXT), grid=list(GRID0) + list(GRID_EXT), b=True)
            continue
        return st, None


def picked_lr(out, arm, st):
    """(lr, None) once the arm's seed-0 runs are complete, else (None, the run still needed)."""
    dev = {}
    for lr in LRS:
        js = load(out, name(arm, "mqrar", 0, lr, st))
        if js is None:
            return None, spec(arm, "mqrar", 0, lr, st)
        dev[lr] = dev_acc(js)
    first = metrics.lr_pick(dev, st["grid"])
    if first in EXT:
        lr = EXT[first]
        js = load(out, name(arm, "mqrar", 0, lr, st))
        if js is None:
            return None, spec(arm, "mqrar", 0, lr, st)
        dev[lr] = dev_acc(js)
    return metrics.lr_pick(dev, st["grid"]), None


def plan(out):
    st, need = settings(out)
    if need:
        return need, "C0"
    lrs = {}
    for arm in PICK_ARMS + DIAG_ARMS:
        lrs[arm], need = picked_lr(out, arm, st)
        if need:
            return need, "LR"
    for seed in metrics.SEEDS:
        for arm in PICK_ARMS + DIAG_ARMS:
            if load(out, name(arm, "mqrar", seed, lrs[arm], st)) is None:
                return spec(arm, "mqrar", seed, lrs[arm], st), "VERDICT"
    for seed in TRACK_SEEDS:
        for arm in TRACK_ARMS:
            if load(out, name(arm, "track", seed, lrs[arm], st)) is None:
                return spec(arm, "track", seed, lrs[arm], st), "TRACK"
    return None, "DONE"


def train_args(sp, out) -> str:
    return (f"--arm {sp['arm']} --family {sp['family']} --seed {sp['seed']} --lr {sp['lr']:g} --steps {sp['steps']} "
            f"--n-hi {sp['n_hi']} --grid {','.join(str(g) for g in sp['grid'])} "
            f"--out {os.path.join(out, sp['name'] + '.json')}")


def table(out):
    st, need = settings(out)
    if need:
        raise SystemExit("C0 not final")
    tab, lrs = {}, {}
    for arm in PICK_ARMS + DIAG_ARMS:
        lrs[arm], need = picked_lr(out, arm, st)
        if need:
            raise SystemExit(f"LR pick of {arm} incomplete")
        tab[arm] = {}
        for s in metrics.SEEDS:
            js = load(out, name(arm, "mqrar", s, lrs[arm], st))
            if js is None:
                raise SystemExit(f"missing verdict run {arm} seed {s}")
            tab[arm][s] = metrics.run_metrics(js)
    return st, lrs, tab


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("next", "status", "verdict"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "next":
        sp, phase = plan(a.out)
        print("DONE" if sp is None else train_args(sp, a.out))
    elif a.cmd == "status":
        sp, phase = plan(a.out)
        n = len([f for f in os.listdir(a.out) if f.endswith(".json")]) if os.path.isdir(a.out) else 0
        print(f"phase {phase}; {n} run files; next {sp['name'] if sp else '-'}")
    else:
        st, lrs, tab = table(a.out)
        print(json.dumps(dict(settings=st, lr=lrs, R1=metrics.r1(tab), R2=metrics.r2(tab), R3=metrics.r3(tab),
                              table={k: {str(s): v for s, v in r.items()} for k, r in tab.items()}), indent=1))


if __name__ == "__main__":
    main()
