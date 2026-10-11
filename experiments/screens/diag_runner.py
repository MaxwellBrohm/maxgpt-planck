"""C6 diagnostics runner (SCREENS C6 HD, SINK, S001's 200-window GATE, final IND; S004 / S005 / S007 per-module
readings; E3 Part 4d's IND and HD, experiments/E3_seed_noise/notes.txt PART 4d ADDENDUM). Report-only: no verdict.
Reads the kept final checkpoint of each registered run and writes one JSON per (run, job) into
<runs root>/<EXP>/<run>/c6/<job>.json. It opens nothing else in a run directory (never bpb.jsonl, bpb/, diag.jsonl or
log.jsonl: those are results), lists it only to find final_*.pt and the stop markers.

  python diag_runner.py list
  python diag_runner.py run --runs-root R --evalset E [--groups G,..] [--only RUN,..] [--device cuda]
                            [--budget-s 1500] [--cpu-only]
Jobs (diag_lib.py): modules (CPU, S004 / S005 / S007 arms), ind (every run), hd (every run), attn (SCREENS runs),
halflife (S005's arm). A job whose JSON exists is skipped (a cut run resumes); --cpu-only runs the modules jobs alone.
Exit: 0 every selected job done; 3 the time budget ran out, or a replay waits for a later process (call again); 1 a job
raised (the rest ran; nothing written for it); 2 refused (arguments, a pin, a run or checkpoint check). 2 > 1 > 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(ROOT, "data_prep"), os.path.join(ROOT, "harness"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import diag_lib as DL  # noqa: E402  (the pins and the readings)

CODE = ["experiments/screens/diag_runner.py", "experiments/screens/diag_lib.py", "data_prep/bpb.py",
        "data_prep/bpb_hd.py", "data_prep/evalwin.py", "harness/attn_diag.py", "harness/induction.py",
        "harness/model.py", "harness/blocks.py"]

SEEDS = (101, 102)
STAGE1 = ["base_s101", "base_s102"] + [f"{a}_s{s}" for a in ("s001_nogate_g1", "s002_novres_g1", "s002_noqknorm_g1",
                                                            "s002_nonormscale_g1", "s003_adamw_e3_r2") for s in SEEDS]
STAGE2 = [f"{a}_s{s}" for a in ("s005_base", "s005_forget_g1", "s004_canonac_g1", "s007_smear_g1", "s006_mtp_g2")
          for s in SEEDS]
BASEDIAG = ["base_s103", "base_s104", "base_s105"]
E3P1 = [f"e3_5m_{a}_s{s}" for s in range(1, 9) for a in "AB"] + ["e3_5m_A_s1_R2", "e3_5m_A_s1_R3"]
E3P2 = [f"e3_20m_{a}_s{s}" for s in (1, 2, 3) for a in "AB"]
SCREEN_JOBS = ("modules", "ind", "hd", "attn", "halflife")
# group: (experiment directory, runs, jobs, file suffix), in queue order (C6 SDs first, then the screens)
GROUPS = {"e3p1": ("E3", E3P1, ("ind", "hd"), ""), "stage2": ("SCREENS", STAGE2, SCREEN_JOBS, ""),
          "basediag": ("SCREENS", BASEDIAG, SCREEN_JOBS, ""), "stage1": ("SCREENS", STAGE1, SCREEN_JOBS, ""),
          "replay": ("E3", ["e3_5m_A_s1"], ("ind", "hd"), ".replay"), "e3p2": ("E3", E3P2, ("ind", "hd"), "")}
EST = {"load": 5.0, "modules": 2.0, "ind": 5.0, "hd": 60.0, "attn": 30.0, "halflife": 15.0}   # s at 5M; x4 at 20M
STOPS = ("DIVERGED", "GAP", "rc12_eval.jsonl")
OFF = {"canon": "", "forget_gate": False, "smear_key": False, "attn_gate": True, "value_residual": True,
       "qk_norm": True, "norm_scaling": True}
FLAGS = {"s004_canonac": {"canon": "AC", "canon_kernel": 4}, "s005_forget": {"forget_gate": True},
         "s007_smear": {"smear_key": True}, "s001_nogate": {"attn_gate": False},
         "s002_novres": {"value_residual": False}, "s002_noqknorm": {"qk_norm": False},
         "s002_nonormscale": {"norm_scaling": False}}


class Refused(Exception):
    pass


def worst(a: int, b: int) -> int:
    """Exit codes by precedence: 2 (refused) over 1 (a job raised) over 3 (budget) over 0."""
    rank = {0: 0, 3: 1, 1: 2, 2: 3}
    return a if rank[a] >= rank[b] else b


def applies(job: str, run: str) -> bool:
    if job == "modules":
        return run.startswith(("s004_canonac", "s005_forget", "s007_smear"))
    if job == "halflife":
        return run.startswith("s005_forget")
    return True


def expected(run: str) -> tuple[int, int, dict]:
    """-> (final step, d_model, the model_cfg values this run's checkpoint must carry)."""
    want = dict(OFF)
    for pre, kv in FLAGS.items():
        if run.startswith(pre):
            want.update(kv)
    return (15259, 320, want) if run.startswith("e3_20m") else (7630, 192, want)


def jobs(groups=None, only=None) -> list[tuple]:
    """-> [(group, exp, run, job, out file name)] in queue order."""
    out = []
    for g, (exp, runs, js, suf) in GROUPS.items():
        if groups and g not in groups:
            continue
        for r in runs:
            if only and r not in only:
                continue
            out += [(g, exp, r, j, f"{j}{suf}.json") for j in js if applies(j, r)]
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def final_ckpt(rdir: str, run: str) -> str:
    if not os.path.isdir(rdir):
        raise Refused(f"{run}: no run directory")
    names = os.listdir(rdir)
    bad = [s for s in STOPS if s in names]
    if bad:
        raise Refused(f"{run}: {bad[0]} in its run directory")
    fin = sorted(n for n in names if n.startswith("final_") and n.endswith(".pt"))
    if len(fin) != 1:
        raise Refused(f"{run}: {len(fin)} final_*.pt (want exactly 1)")
    step = expected(run)[0]
    if fin[0] != f"final_{step:08d}.pt":
        raise Refused(f"{run}: {fin[0]}, want final_{step:08d}.pt")
    return os.path.join(rdir, fin[0])


def check_ck(ck: dict, run: str) -> None:
    step, d, want = expected(run)
    mc = ck["model_cfg"]
    got = {k: mc.get(k, OFF.get(k, 4 if k == "canon_kernel" else None)) for k in want}
    if ck.get("step") != step or mc.get("d_model") != d or got != want:
        raise Refused(f"{run}: checkpoint step {ck.get('step')}, d_model {mc.get('d_model')}, flags {got}; "
                      f"want step {step}, d_model {d}, {want}")


def run(a) -> int:
    import torch
    t0, est, status, done_any, cache = time.time(), dict(EST), 0, False, {}
    if sha256(DL.TOK_FILE) != DL.TOK_SHA:
        raise Refused(f"tokenizer {DL.TOK_FILE} is not {DL.TOK_SHA[:16]}")
    if not a.cpu_only:
        man = json.load(open(os.path.join(a.evalset, "manifest.json")))
        if man.get("evalset_sha256") != DL.EVALSET_SHA:
            raise Refused(f"eval set {a.evalset} is {man.get('evalset_sha256')}, not {DL.EVALSET_SHA[:16]}")
    code = {p: sha256(os.path.join(ROOT, p)) for p in CODE}
    todo: dict = {}
    for g, exp, r, j, fn in jobs(a.groups, a.only):
        if a.cpu_only and j != "modules":
            continue
        out = os.path.join(a.runs_root, exp, r, "c6", fn)
        if not os.path.exists(out):
            todo.setdefault((g, exp, r), []).append((j, out))
    seen = set()
    for (g, exp, r), js in todo.items():
        if g == "replay" and (exp, r) in seen:      # a replay is scored by a later process, never this one
            print(f"replay of {r} left for a later process", flush=True)
            return worst(status, 3)
        seen.add((exp, r))
        scale = 4.0 if r.startswith("e3_20m") else 1.0
        need = scale * (est["load"] + sum(est[j] for j, _ in js))
        if done_any and time.time() - t0 + need > a.budget_s:
            print(f"budget: {time.time() - t0:.0f} s spent, {r} needs about {need:.0f} s: stopping", flush=True)
            return worst(status, 3)
        try:
            ckp = final_ckpt(os.path.join(a.runs_root, exp, r), r)
            ts = time.time()
            ck = torch.load(ckp, map_location="cpu", weights_only=False)
            check_ck(ck, r)
            ck_sha = sha256(ckp)
            model = cfg = None
            if not a.cpu_only:
                import bpb
                model, cfg, _ = bpb.load_model(ckp, a.device)
            est["load"] = max(est["load"], (time.time() - ts) / scale)
        except Refused as e:
            print(f"REFUSED {e}", flush=True)
            status = worst(status, 2)
            continue
        for j, out in js:
            ts = time.time()
            try:
                res = DL.compute(j, model, cfg, ck["model"], a.device, a.evalset, cache)
            except Exception as e:      # one failed job does not stop the others; it writes nothing
                print(f"FAILED {r} {j}: {type(e).__name__}: {e}", flush=True)
                status = worst(status, 1)
                continue
            sec = time.time() - ts
            est[j] = max(est[j], sec / scale)
            DL.write_json(out, {"run": r, "exp": exp, "group": g, "job": j, "checkpoint": os.path.basename(ckp),
                             "checkpoint_sha256": ck_sha, "step": ck.get("step"), "device": a.device if j != "modules"
                             else "cpu", "torch": torch.__version__, "code_sha256": code, "seconds": round(sec, 2),
                             "evalset_sha256": DL.EVALSET_SHA, "tokenizer_sha256": DL.TOK_SHA, "result": res})
            print(f"done {exp}/{r} {j} {sec:.1f} s", flush=True)
            done_any = True
        del model, ck
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    for k in ("--runs-root", "--evalset", "--groups", "--only"):
        ap.add_argument(k, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget-s", type=float, default=1500.0)
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args(argv)
    a.groups = a.groups.split(",") if a.groups else None
    a.only = a.only.split(",") if a.only else None
    if a.groups and set(a.groups) - set(GROUPS):
        print(f"unknown groups {sorted(set(a.groups) - set(GROUPS))}; known {list(GROUPS)}")
        return 2
    if a.cmd == "list":
        for g, exp, r, j, fn in jobs(a.groups, a.only):
            print(f"{g}\t{exp}/{r}\t{j}\t{fn}")
        return 0
    if not a.runs_root or (not a.evalset and not a.cpu_only):
        print("run needs --runs-root and --evalset (or --cpu-only)")
        return 2
    if a.cpu_only:
        a.device = "cpu"
    try:
        return run(a)
    except Refused as e:
        print(f"REFUSED {e}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
