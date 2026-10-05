"""SCREENS config generator and queue plans (experiments/SCREENS.txt C1-C4, C1-b, ORDER). Mac or PC, no model.

  python screens.py build                 BASE runs (seeds from E3), own BASEs, every C3 g-check (seed 1,
                                          g 0.5 / 1 / 2), S003 stage A; plans stage1_select, stage2_select
  python screens.py config SID.ARM --g G [--seed S]   one arm config (a C3 extension: g 0.25 / 4 / ...)
  python screens.py s003-arm --eta E --r R            S003 search point: trunk + 3 branches (stages B, C)
  python screens.py seeds --stage N --pick SID.ARM=G ... [--s003 ETA,R]
                                          the seed sets after the picks, plans/stage<N>_seeds.txt (ORDER)
  python screens.py plan NAME RUN ... [--mark TEXT] [--wait "EXP TEXT"]
  python screens.py check [RUN ...]       C2 + RC-12 GUARD + C1-b + C6 hook + 2% on every config (or named runs)
  python screens.py hours [--measured-file TSV] [--s003-rb R]
                                          GPU HOURS estimate and ORDER's 25 h cap check (screens_hours.report):
                                          measured runs count once, queued estimates only for runs not yet run
Every config is written, then checked (screens_lib.check); a refused config is deleted and the command fails.
An existing config is never overwritten with different content. Plans: queue_screens.sh plan lines.
"""
from __future__ import annotations

import argparse
import os
import sys

import screens_lib as L

PLANS = os.path.join(L.HERE, "plans")


def write(sid: str | None, name: str, txt: str, p: dict) -> str:
    path = os.path.join(L.config_dir(sid), name + ".yaml")
    if os.path.exists(path):
        if open(path).read() != txt:
            sys.exit(f"refusing to overwrite {path} with different content")
    else:
        with open(path, "w") as f:
            f.write(txt)
    bad = L.check(path, p)
    if bad:
        os.unlink(path)
        sys.exit("refused (config deleted):\n  " + "\n  ".join(bad))
    return name


def extends(sid: str | None) -> str:
    return "screens_base.yaml" if sid is None else "../../screens/configs/screens_base.yaml"


def base(seed: int, p: dict, sid: str | None = None) -> str:
    name = f"{sid.lower()}_base_s{seed}" if sid else f"base_s{seed}"
    head = f"SCREENS {sid} own BASE (ENGINE keys)" if sid else "SCREENS BASE (C1: E3 arm A, C1-b eager)"
    return write(sid, name, L.config_text(name, f"{head}, LR5, seed {seed}", extends(sid), seed,
                                          {**L.lrs(*p["lr5"]), **L.engine(sid)}), p)


def arm(sid: str, a: str, g: float, seed: int, p: dict, note: str = "") -> str:
    name = f"{sid.lower()}_{a}_g{L.num(g)}_s{seed}"
    keys = {**L.lrs(*p["lr5"], g), **L.engine(sid), **L.SCREENS[sid]["arms"][a]}
    head = f"SCREENS {sid} arm {a}, g {L.num(g)} x LR5, seed {seed}{note}"
    return write(sid, name, L.config_text(name, head, extends(sid), seed, keys), p)


def s003_arm(eta: float, r: float, p: dict) -> list[str]:
    """E2's trunk + branch schedule for one LR point (e2plan's constants), optim.kind adamw, eager (C1-b)."""
    import e2plan
    s, stem = e2plan.SIZES["5m"], "s003_adamw" + e2plan.arm_name("5m", eta, r)[2:]
    keys = {**L.lrs(eta, r), **L.engine("S003"), **L.SCREENS["S003"]["arms"]["adamw"]}
    runs = [(f"{stem}_trunk", {"schedule.mode": "trunk", "schedule.branch_points": [at for _, at in s["branches"]],
                               "train.total_steps": s["trunk"], "train.ckpt_every": e2plan.cadence(s["trunk"])})]
    for tag, at in s["branches"]:
        d = e2plan.decay_steps(at)
        runs.append((f"{stem}_{tag}", {"schedule.mode": "branch", "schedule.decay_steps": d,
                                       "schedule.init_from": f"{L.ROOTREL}/runs/SCREENS/{stem}_trunk/stable_{at:08d}.pt",
                                       "train.total_steps": at + d, "train.ckpt_every": e2plan.cadence(at + d)}))
    head = f"SCREENS S003 AdamW search point eta {L.num(eta)}, r {L.num(r)} (E2 SCHEDULE, seed 1)"
    return [write("S003", n, L.config_text(n, head, extends("S003"), 1, {**keys, **sch}), p) for n, sch in runs]


def s003_seed(eta: float, r: float, seed: int, p: dict) -> str:
    name = f"s003_adamw_e{L.num(eta * 1e3)}_r{L.num(r)}_s{seed}"
    keys = {**L.lrs(eta, r), **L.engine("S003"), **L.SCREENS["S003"]["arms"]["adamw"]}
    return write("S003", name, L.config_text(name, f"SCREENS S003 AdamW at LR5_adamw, seed {seed}", extends("S003"),
                                             seed, keys), p)


def plan(name: str, runs: list[str], header: str, mark: str | None = None, wait: str | None = None) -> str:
    for r in runs:
        if len(L.find(r)) != 1:
            sys.exit(f"plan {name}: run {r} has {len(L.find(r))} configs (want exactly 1)")
    lines = [f"# {header} (screens.py). train = screens check, preflight, train, score"]
    lines += [f"wait_mark {wait}"] if wait else []
    lines += [f"train {r}" for r in runs] + ([f"mark {mark}"] if mark else [])
    os.makedirs(PLANS, exist_ok=True)
    path = os.path.join(PLANS, name + ".txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def gcheck(sid: str, p: dict) -> list[str]:
    return [arm(sid, a, g, 1, p, ": C3 g-check") for a in L.SCREENS[sid]["arms"] for g in (0.5, 1.0, 2.0)]


def build(p: dict) -> None:
    seeds = sorted({s for v in p["seeds"].values() for s in v})
    for s in seeds:
        base(s, p)
        for sid in L.SCREENS:
            if L.engine(sid):
                base(s, p, sid)
    s1 = gcheck("S001", p) + gcheck("S002", p)
    for eta in L.S003_STAGE_A:
        s1 += s003_arm(eta, 1.0, p)
    print(plan("stage1_select", s1, "SCREENS stage 1 selection at seed 1 (ORDER 1): S001 g-check, S002 g-checks, "
               "S003 stage A", "SCREENS STAGE 1 SELECTION A DONE", "SCREENS SCREENS PRECONDITIONS OK"))
    s2 = [r for sid in L.ORDER[2] for r in gcheck(sid, p)]
    print(plan("stage2_select", s2, "SCREENS stage 2 selection at seed 1 (ORDER 2, priority S005 S004 S007 S006)",
               "SCREENS STAGE 2 SELECTION DONE", "SCREENS SCREENS STAGE 2 SMOKES RECORDED"))


def seeds(stage: int, picks: dict, s003, p: dict) -> None:
    runs, top = [], max(len(p["seeds"][L.SCREENS[s]["cls"]]) for s in L.ORDER[stage])
    for i in range(top):
        s = 101 + i
        if any(not L.engine(sid) for sid in L.ORDER[stage]):
            runs.append(base(s, p))
        for sid in L.ORDER[stage]:
            if s not in p["seeds"][L.SCREENS[sid]["cls"]]:
                continue
            if L.engine(sid):
                runs.append(base(s, p, sid))
            for a in L.SCREENS[sid]["arms"]:
                if sid == "S003":
                    runs.append(s003_seed(*s003, s, p))
                    continue
                g = picks[f"{sid}.{a}"]
                runs.append(arm(sid, a, g, s, p, ": screen seed at the g pick"))
                if sid in ("S006", "S007") and g != 1.0:      # C6 MATCHED LR: IND-only run at g = 1
                    runs.append(arm(sid, a, 1.0, s, p, ": IND-only matched-LR run (C6), never in a bpb verdict"))
    print(plan(f"stage{stage}_seeds", runs, f"SCREENS stage {stage} seed sets, seed-major (ORDER {stage})",
               f"SCREENS STAGE {stage} SEEDS DONE"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("config")
    c.add_argument("arm")
    c.add_argument("--g", type=float, required=True)
    c.add_argument("--seed", type=int, default=1)
    s3 = sub.add_parser("s003-arm")
    s3.add_argument("--eta", type=float, required=True)
    s3.add_argument("--r", type=float, required=True)
    sd = sub.add_parser("seeds")
    sd.add_argument("--stage", type=int, choices=(1, 2), required=True)
    sd.add_argument("--pick", action="append", default=[], help="SID.ARM=G, e.g. S002.novres=0.5")
    sd.add_argument("--s003", default=None, help="LR5_adamw as ETA,R (stage 1)")
    pl = sub.add_parser("plan")
    pl.add_argument("name")
    pl.add_argument("runs", nargs="+")
    pl.add_argument("--mark", default=None)
    pl.add_argument("--wait", default=None)
    ck = sub.add_parser("check")
    ck.add_argument("runs", nargs="*")
    hr = sub.add_parser("hours")
    hr.add_argument("--measured-file", default=None, help="measured_hours.tsv (default: experiments/screens/)")
    hr.add_argument("--s003-rb", type=float, default=None, help="S003 stage B pick r_B (r_B 1: stage C adds no run)")
    a = ap.parse_args(argv)
    p = L.params()
    if a.cmd == "build":
        build(p)
    elif a.cmd == "config":
        sid, arm_ = a.arm.split(".")
        print(arm(sid, arm_, a.g, a.seed, p, ": C3 g-check extension" if a.seed == 1 else ""))
    elif a.cmd == "s003-arm":
        print("\n".join(s003_arm(a.eta, a.r, p)))
    elif a.cmd == "seeds":
        picks = {k: float(v) for k, v in (x.split("=") for x in a.pick)}
        need = {f"{sid}.{arm_}" for sid in L.ORDER[a.stage] if sid != "S003" for arm_ in L.SCREENS[sid]["arms"]}
        if set(picks) != need or (a.stage == 1) != (a.s003 is not None):
            sys.exit(f"stage {a.stage} needs --pick for exactly {sorted(need)}" + (" and --s003" if a.stage == 1 else ""))
        seeds(a.stage, picks, tuple(float(x) for x in a.s003.split(",")) if a.s003 else None, p)
    elif a.cmd == "plan":
        print(plan(a.name, a.runs, a.name, a.mark, a.wait))
    elif a.cmd == "check":
        paths = [q for r in a.runs for q in L.find(r)] if a.runs else L.all_configs()
        if a.runs and len(paths) != len(a.runs):
            print(f"refused: {len(paths)} configs found for {len(a.runs)} names")
            return 1
        bad = {q: L.check(q, p) for q in paths}
        for q, b in bad.items():
            print(("ok      " if not b else "REFUSED ") + os.path.relpath(q, L.ROOT) + "".join(f"\n  {x}" for x in b))
        return 1 if any(bad.values()) else 0
    else:
        import screens_hours as H
        H.report(p, a.measured_file or H.TSV, a.s003_rb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
