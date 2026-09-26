"""Write E3 run configs and queue plans from E2's recorded picks (stdlib; Mac or PC, no torch).

  python e3plan.py --a5 ETA,R --b5 ETA,R [--a20 ETA,R --b20 ETA,R] [--part3] [--out-dir DIR]

ETA = optim.lr, R = embed_lr / lr = scalar_lr / lr, exactly as E2's result lines give A, B, A20 and B20 (E3
notes, ARMS). Nothing is run. Configs go to configs/ (or --out-dir, for a dry look), plans to plans/:
  part1.txt  waits for E2's "E2 5M DONE"; A1 B1 A2 B2 ... A8 B8 (whole pairs first), then the replays
             e3_5m_A_s1_R2 and _R3 (config text identical to A1's except name and out_dir); marks "E3 PART 1 DONE"
  part2.txt  (with --a20/--b20) waits for "E2 20M DONE"; 20M A1 B1 A2 B2 A3 B3; marks "E3 PART 2 DONE"
  part3.txt  (with --part3) 5M arm A: init seeds 2..5 at data_seed 1 and data seeds 2..5 at seed 1, interleaved
             init2 data2 init3 ... so a stopped queue leaves balanced families; marks "E3 PART 3 DONE". It needs
             the data_seed harness change (e2e3/data_seed.patch) committed and tested: preflight.py refuses a
             data_seed the harness does not read, so the queue stops there instead of running a wrong Part 3.
Schedules (E3 notes): full WSD, warmup 76, 20% linear decay; 5M 7,630 steps (decay from 6,104), 20M 15,259
steps (decay from 12,207); ckpt_every = round(2% of total) = 153 / 305, keep_last 2 (base), plus final.
The joint order with E2 comes from the marks: E2's 20M plan starts with "wait_mark E3 E3 PART 1 DONE".
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "../../../planck_root"
BASE = {"5m": "../../E2_lr_transfer/configs/base5m.yaml", "20m": "../../E2_lr_transfer/configs/base20m.yaml"}
TOTAL = {"5m": 7630, "20m": 15259}


def num(x: float) -> str:
    return f"{x:.6g}"


def lr_pair(s: str) -> tuple[float, float]:
    eta, r = (float(v) for v in s.split(","))
    assert eta > 0 and r > 0, s
    return eta, r


def config(name: str, size: str, arm: str, eta: float, r: float, seed: int, data_seed: int | None = None,
           note: str = "") -> str:
    tot = TOTAL[size]
    txt = (f"# E3 {size.upper()} arm {arm} (eta {num(eta)}, r {num(r)}), seed {seed}{note} (e3plan.py)\n"
           f"extends: {BASE[size]}\n"
           f"name: {name}\n"
           f"out_dir: {ROOT}/runs/E3/{name}\n"
           f"seed: {seed}\n")
    if data_seed is not None:
        txt += f"data_seed: {data_seed}\n"
    return txt + (f"optim: {{lr: {num(eta)}, embed_lr: {num(r * eta)}, scalar_lr: {num(r * eta)}}}\n"
                  f"schedule: {{mode: full, decay_frac: 0.2}}\n"
                  f"train: {{total_steps: {tot}, ckpt_every: {max(1, round(0.02 * tot))}}}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5", required=True, help="E2's final 5M pick: ETA,R")
    ap.add_argument("--b5", required=True, help="E2's 5M runner-up: ETA,R")
    ap.add_argument("--a20", default=None)
    ap.add_argument("--b20", default=None)
    ap.add_argument("--part3", action="store_true")
    ap.add_argument("--out-dir", default=None, help="write configs and plans here instead (dry look)")
    a = ap.parse_args(argv)
    conf = a.out_dir or os.path.join(HERE, "configs")
    plans = a.out_dir or os.path.join(HERE, "plans")
    os.makedirs(conf, exist_ok=True)
    os.makedirs(plans, exist_ok=True)
    arms = {"A": lr_pair(a.a5), "B": lr_pair(a.b5)}
    assert arms["A"] != arms["B"], "A and B must differ"
    files: dict[str, str] = {}
    p1 = ["# E3 Part 1 (e3plan.py): 5M pairs, then two replays of A1", "wait_mark E2 E2 5M DONE"]
    for s in range(1, 9):
        for arm in "AB":
            n = f"e3_5m_{arm}_s{s}"
            files[n] = config(n, "5m", arm, *arms[arm], s)
            p1.append(f"train {n}")
    for k in (2, 3):
        n = f"e3_5m_A_s1_R{k}"
        files[n] = config(n, "5m", "A", *arms["A"], 1)
        p1.append(f"train {n}")
    p1.append("mark E3 PART 1 DONE")
    out_plans = {"part1": p1}
    if a.a20 or a.b20:
        a20, b20 = lr_pair(a.a20), lr_pair(a.b20)
        assert a20 != b20, "A20 and B20 must differ"
        p2 = ["# E3 Part 2 (e3plan.py): 20M pairs", "wait_mark E2 E2 20M DONE"]
        for s in range(1, 4):
            for arm, lr in (("A", a20), ("B", b20)):
                n = f"e3_20m_{arm}_s{s}"
                files[n] = config(n, "20m", arm, *lr, s)
                p2.append(f"train {n}")
        out_plans["part2"] = p2 + ["mark E3 PART 2 DONE"]
    if a.part3:
        p3 = ["# E3 Part 3 (e3plan.py): 5M arm A, init and data families (needs harness data_seed)",
              "wait_mark E3 E3 PART 2 DONE"]
        for k in range(2, 6):
            for fam, seed, dseed in (("init", k, 1), ("data", 1, k)):
                n = f"e3_5m_A_{fam}{k}"
                files[n] = config(n, "5m", "A", *arms["A"], seed, dseed, f", data_seed {dseed}")
                p3.append(f"train {n}")
        out_plans["part3"] = p3 + ["mark E3 PART 3 DONE"]
    for n, txt in files.items():
        p = os.path.join(conf, n + ".yaml")
        if os.path.exists(p) and open(p).read() != txt:
            sys.exit(f"refusing to overwrite {p} with different content")
        with open(p, "w") as f:
            f.write(txt)
    for n, lines in out_plans.items():
        with open(os.path.join(plans, n + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{len(files)} configs in {conf}; plans {sorted(out_plans)} in {plans}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
