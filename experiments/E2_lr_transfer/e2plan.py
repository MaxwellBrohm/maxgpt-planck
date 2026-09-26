"""Write E2 run configs and queue plans (stdlib + PyYAML-free: configs are written as text). Mac or PC, no torch.

  python e2plan.py stage-a                         the 5 stage A arms (20 configs) + plans/stageA.txt
  python e2plan.py arm --size 5m --eta 6e-3 --r 2  one arm: its trunk and branch configs (later stages,
                                                   written by the analyzer or a person from recorded results)
  python e2plan.py plan NAME ARM [ARM ...] [--mark TEXT] [--wait "EXP TEXT"]
                                                   plans/NAME.txt: each arm's trunk then its branches
  python e2plan.py check                           recompute every schedule number below from harness/schedule.py
                                                   rules (pure arithmetic) and list configs with their sha256

An arm is one LR point: eta = optim.lr (NorMuon matrices), r = embed_lr / lr = scalar_lr / lr. Its name is
<size>_e<eta x 1e3>_r<r>, e.g. 5m_e3_r1, so a point shared by two stages (stage B reuses stage A's pick at r 1)
has one set of runs. Schedules (E2 notes, SCHEDULE, batch 32,768 slots, warmup 76 in every config):
  5M   trunk 6,105 steps, stable checkpoints 1,526 / 3,052 / 6,104; branches decay s/4 steps:
       b62M 1,908 steps, b125M 3,815, b250M 7,630 (ckpt_every 38 / 76 / 153 = 2% of each run)
  20M  trunk 12,208 steps, stable checkpoints 6,104 / 12,207; b250M 7,630, b500M 15,259 (ckpt_every 153 / 305)
Trunk ckpt_every is 2% of the trunk (rolling saves only serve a crash resume). Every config extends
base5m.yaml or base20m.yaml (and so engine.yaml) and writes only name, out_dir, LRs, schedule, cadence.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONF, PLANS = os.path.join(HERE, "configs"), os.path.join(HERE, "plans")
WARMUP, BATCH = 76, 32768
SIZES = {
    "5m": {"base": "base5m.yaml", "trunk": 6105, "branches": [("b62M", 1526), ("b125M", 3052), ("b250M", 6104)]},
    "20m": {"base": "base20m.yaml", "trunk": 12208, "branches": [("b250M", 6104), ("b500M", 12207)]},
}
STAGE_A_ETAS = [0.75e-3, 1.5e-3, 3e-3, 6e-3, 12e-3]
ROOT = "../../../planck_root"


def num(x: float) -> str:
    return f"{x:.6g}"


def arm_name(size: str, eta: float, r: float) -> str:
    return f"{size}_e{num(eta * 1e3)}_r{num(r)}"


def decay_steps(at: int) -> int:
    """s * 0.2 / 0.8 = s / 4 rounded half up, as the notes list (1,526 -> 382). Written explicitly in every branch
    config, so harness schedule.branch's float round (381 for 1,526) never applies."""
    return max(1, (at + 2) // 4)


def cadence(total: int) -> int:
    return max(1, round(0.02 * total))


def runs_for(size: str, eta: float, r: float, exp: str = "E2") -> list[tuple[str, str]]:
    """-> [(run name, config text)]: the trunk, then each branch."""
    s, arm = SIZES[size], arm_name(size, eta, r)
    lrs = f"optim: {{lr: {num(eta)}, embed_lr: {num(r * eta)}, scalar_lr: {num(r * eta)}}}\n"
    head = (f"# E2 {size.upper()} arm eta {num(eta)}, r {num(r)} (written by e2plan.py; E2 notes, SCHEDULE)\n"
            f"extends: {s['base']}\n")
    trunk = f"{arm}_trunk"
    pts = [at for _, at in s["branches"]]
    assert max(pts) == s["trunk"] - 1, "the trunk must be one step longer than its last branch point"
    out = [(trunk, head + f"name: {trunk}\nout_dir: {ROOT}/runs/{exp}/{trunk}\n" + lrs +
            f"schedule: {{mode: trunk, branch_points: [{', '.join(map(str, pts))}]}}\n"
            f"train: {{total_steps: {s['trunk']}, ckpt_every: {cadence(s['trunk'])}}}\n")]
    for tag, at in s["branches"]:
        name, d = f"{arm}_{tag}", decay_steps(at)
        out.append((name, head + f"name: {name}\nout_dir: {ROOT}/runs/{exp}/{name}\n" + lrs +
                    f"schedule: {{mode: branch, init_from: {ROOT}/runs/{exp}/{trunk}/stable_{at:08d}.pt, "
                    f"decay_steps: {d}}}\n"
                    f"train: {{total_steps: {at + d}, ckpt_every: {cadence(at + d)}}}\n"))
    return out


def write_arm(size: str, eta: float, r: float) -> list[str]:
    names = []
    for name, txt in runs_for(size, eta, r):
        p = os.path.join(CONF, name + ".yaml")
        if os.path.exists(p) and open(p).read() != txt:
            sys.exit(f"refusing to overwrite {p} with different content")
        with open(p, "w") as f:
            f.write(txt)
        names.append(name)
    return names


def write_plan(plan: str, arms: list[str], header: str = "", mark: str | None = None,
               wait: str | None = None) -> str:
    lines = [f"# {header or plan} (e2plan.py). train = preflight, train, then score if it is a branch"]
    if wait:
        lines.append(f"wait_mark {wait}")
    for arm in arms:
        size = arm.split("_")[0]
        lines.append(f"train {arm}_trunk")
        lines += [f"train {arm}_{tag}" for tag, _ in SIZES[size]["branches"]]
    if mark:
        lines.append(f"mark {mark}")
    p = os.path.join(PLANS, plan + ".txt")
    os.makedirs(PLANS, exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p


def check() -> int:
    ok = True
    for size, s in SIZES.items():
        for tag, at in s["branches"]:
            tot = at + decay_steps(at)
            full_start = max(WARMUP, tot - round(tot * 0.2))    # schedule.full(tot, warmup 76, 0.2)
            same = full_start == at
            ok &= same
            print(f"{size} {tag}: branch at {at}, decay {decay_steps(at)}, total {tot} = {tot * BATCH / 1e6:.1f}M "
                  f"slots, ckpt_every {cadence(tot)}; schedule.full({tot}) decays from {full_start}: "
                  f"{'same' if same else 'DIFFERENT'}")
        print(f"{size} trunk {s['trunk']} steps, ckpt_every {cadence(s['trunk'])}")
    for p in sorted(os.listdir(CONF)):
        if p.endswith(".yaml"):
            print(hashlib.sha256(open(os.path.join(CONF, p), "rb").read()).hexdigest(), p)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage-a")
    a1 = sub.add_parser("arm")
    a1.add_argument("--size", choices=sorted(SIZES), required=True)
    a1.add_argument("--eta", type=float, required=True)
    a1.add_argument("--r", type=float, default=1.0)
    a2 = sub.add_parser("plan")
    a2.add_argument("name")
    a2.add_argument("arms", nargs="+")
    a2.add_argument("--mark", default=None, help='e.g. "E2 5M DONE" (the last 5M plan)')
    a2.add_argument("--wait", default=None, help='e.g. "E3 E3 PART 1 DONE" (the 20M plan: joint order)')
    sub.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "stage-a":
        arms = []
        for eta in STAGE_A_ETAS:
            write_arm("5m", eta, 1.0)
            arms.append(arm_name("5m", eta, 1.0))
        print(write_plan("stageA", arms, "E2 stage A: eta 0.75..12 x 1e-3, r 1", "E2 STAGE A DONE"))
    elif a.cmd == "arm":
        print("\n".join(write_arm(a.size, a.eta, a.r)))
    elif a.cmd == "plan":
        print(write_plan(a.name, a.arms, mark=a.mark, wait=a.wait))
    else:
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
