"""SCREENS config library (experiments/SCREENS.txt C1-C4, C1-b; each S00x/notes.txt). Mac or PC, no model built.

  BASE      configs/screens_base.yaml: E3 arm A key for key (eager, AMENDMENT C1-b: train.compile false as E3 ran),
            plus engine_fixed and the C6 eval.induction hook (HOOK keys: logging only). Run configs add name,
            out_dir, seed and the LRs. One BASE run per seed serves every screen but S005: configs/base_s<seed>.yaml.
  own BASE  a screen with ENGINE keys (S005: doc_attn mask, micro 8 x accum 2) pairs with <sid>_base_s<seed>.
            Registered switches go in ENGINE by a logged edit, then that screen's configs are rebuilt: S005's
            micro 4 x accum 4 fallback, S006's micro 8 x accum 2 (if mem_fit fails at 16).
  arm       its BASE + the keys its notes.txt registers (ARMS) at g x LR5 (C3; r_5 kept), seeds from E3 (C4).
  S003      AdamW search: E2's trunk + branch configs (e2plan.runs_for) with optim.kind adamw; seed runs full WSD.
check(path) is C2 (only allowed keys differ from the reference, every registered key at its value, LRs right),
the RC-12 GUARD (no eval.rc12 key, BASE included), C1-b's engine values (every run eager) and the 2% rule.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXPD = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPD)
E2D, E3D = os.path.join(EXPD, "E2_lr_transfer"), os.path.join(EXPD, "E3_seed_noise")
for _p in (os.path.join(ROOT, "harness"), E2D):
    if _p not in sys.path:
        sys.path.insert(0, _p)
BASE_YAML = os.path.join(HERE, "configs", "screens_base.yaml")
E2_SHA = "b08062c8b8778796f6bdaf086684fd254e4f751f814df26fcfa152c995d509ea"   # E2 results.json (FIXED REFILL, q2 key)
E3_SHA = "1f017ecb1e6d14dd4f676c90ca703c20e9395b217fc841f83fffcb93e57994a2"   # E3 results.json (FIXED log)
ENGINE_FIXED = "2026-10-03 f314eb7 SCREENS C1-b"
COMPILE = False      # AMENDMENT C1-b: every screen run eager (train.compile false), withdrawing C1-a
STEPS, SLOTS, BASE_PARAMS = 7630, 32768, 5_010_133
ROOTREL = "../../../planck_root"
LRK = ("optim.lr", "optim.embed_lr", "optim.scalar_lr")
SCREENS = {   # dir, C4 class, stage, arms {arm: registered keys}, ENGINE keys (both arms + own BASE); hours: screens_hours
    "S001": {"dir": "S001_attn_gate", "cls": "new init", "stage": 1, "arms": {"nogate": {"model.attn_gate": False}}},
    "S002": {"dir": "S002_block_ablations", "cls": "SIA", "stage": 1,
             "arms": {"novres": {"model.value_residual": False}, "noqknorm": {"model.qk_norm": False},
                      "nonormscale": {"model.norm_scaling": False}}},
    "S003": {"dir": "S003_adamw", "cls": "same init", "stage": 1, "arms": {"adamw": {"optim.kind": "adamw"}}},
    "S004": {"dir": "S004_canon", "cls": "SIA", "stage": 2,
             "arms": {"canonac": {"model.canon": "AC", "model.canon_kernel": 4}}},
    "S005": {"dir": "S005_forget_gate", "cls": "SIA", "stage": 2, "arms": {"forget": {"model.forget_gate": True}},
             "engine": {"train.doc_attn": "mask", "train.micro_batch": 8, "train.grad_accum": 2}},
    "S006": {"dir": "S006_mtp_aux", "cls": "SIA", "stage": 2,
             "arms": {"mtp": {"train.mtp": 1, "train.mtp_weight": 1.0}}},
    "S007": {"dir": "S007_smeared_key", "cls": "SIA", "stage": 2, "arms": {"smear": {"model.smear_key": True}}},
}
ORDER = {1: ["S001", "S002", "S003"], 2: ["S005", "S004", "S007", "S006"]}   # SCREENS ORDER, must-run first
G_ALLOWED = [2.0 ** n for n in range(-3, 4)]     # C3 grid {0.5, 1, 2} plus at most 2 factor-2 extensions each way
S003_STAGE_A = [0.75e-3, 1.5e-3, 3e-3, 6e-3, 12e-3]
NAME_RE = re.compile(r"^(?:(?P<sid>s00\d)_)?(?P<arm>[a-z]+)(?:_g(?P<g>[0-9.]+))?(?:_e(?P<e>[0-9.]+)_r(?P<r>[0-9.]+))?"
                     r"(?:_s(?P<seed>\d+)|_(?P<tag>trunk|b62M|b125M|b250M))$")


def num(x: float) -> str:
    return f"{x:.6g}"


def yv(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):     # PyYAML reads "1e-05" as a string: every float gets a "."
        s = num(v)
        return s if "." in s else s.replace("e", ".0e") if "e" in s else s + ".0"
    return str(v)


def sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def params() -> dict:
    """LR5, k and the class reference SDs from E2 and E3 results.json (C1, C4), pinned by sha256."""
    e2p, e3p = os.path.join(E2D, "results.json"), os.path.join(E3D, "results.json")
    assert sha256(e2p) == E2_SHA and sha256(e3p) == E3_SHA, "E2/E3 results.json changed: refill the FIXED log"
    e2, e3 = json.load(open(e2p)), json.load(open(e3p))
    pick = tuple(e2["stage_C"]["lrs"]["pick"])
    rc = e3["e2_recheck"]
    lr5 = tuple(rc["lr5_for_screens"])
    assert rc["verdict"].startswith("CONFIRM") and lr5 == pick, (rc["verdict"], lr5, pick)
    q, k = e3["quantiles"], {}
    for cls in ("same init", "SIA", "new init"):
        need = 1
        for m in ("F(CHAT)", "F(PROSE)"):
            n = e3["noise_5M_250M"][m]
            if cls == "same init":
                kn = n["k_needed_paired"]["1.000%"]
            elif cls == "new init":     # sqrt(2) sigma_seed: E3 Part 3 (sigma_init) not run
                assert any(x.startswith("Part 3") for x in e3["not_run"])
                kn = n["k_needed_unpaired"]["1.000%"]
            else:                       # SD_ref = u x SD_d, u = E3's 80% upper bound at df 7 (C4)
                kn = next(k_ for k_ in range(1, 99) if q["factor_df7"] * q["ub80_df7"] * n["SD_d_rel"] / math.sqrt(k_) <= 0.01)
            need = max(need, kn)
        k[cls] = min(5, max(2, need))
    return {"lr5": lr5, "k": k, "u": q["ub80_df7"], "seeds": {c: [100 + i for i in range(1, v + 1)] for c, v in k.items()}}


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for key, v in d.items():
        out.update(flat(v, f"{pre}{key}.") if isinstance(v, dict) else {f"{pre}{key}": v})
    return out


def resolve(path: str) -> dict:
    import runio
    return runio.load_yaml(path)


def resolve_text(txt: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(txt)
    try:
        return resolve(f.name)
    finally:
        os.unlink(f.name)


def config_text(name: str, header: str, extends: str, seed: int, keys: dict) -> str:
    """Run config text: name, out_dir, seed, then one flow mapping per section of the dotted keys."""
    secs: dict = {}
    for dk, v in keys.items():
        sec, k = dk.split(".", 1)
        secs.setdefault(sec, []).append(f"{k}: {yv(v)}")
    body = "".join(f"{s}: {{{', '.join(kv)}}}\n" for s, kv in secs.items())
    return (f"# {header} (screens.py)\nextends: {extends}\nname: {name}\nout_dir: {ROOTREL}/runs/SCREENS/{name}\n"
            f"seed: {seed}\n{body}")


def lrs(eta: float, r: float, g: float = 1.0) -> dict:
    return {"optim.lr": g * eta, "optim.embed_lr": g * r * eta, "optim.scalar_lr": g * r * eta}

def hook_keys() -> dict:
    """BASE's eval.induction keys (C6 IND and GATE logging): every SCREENS run carries them unchanged."""
    return {k: v for k, v in flat(resolve(BASE_YAML)).items() if k.startswith("eval.induction.")}


def engine(sid: str | None) -> dict:
    return dict(SCREENS[sid].get("engine", {})) if sid else {}


def reference(sid: str | None, lr5) -> dict:
    """Resolved BASE at LR5 (the C2 reference), with the screen's ENGINE keys applied."""
    ref = flat(resolve(BASE_YAML))
    ref.update(lrs(*lr5))
    ref.update(engine(sid))
    return ref


def diff(a: dict, b: dict) -> set:
    return {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>") or (k in a) != (k in b)}

def s003_e2_reference(eta: float, r: float, tag: str) -> dict:
    """E2's own stage config for this LR point (e2plan.runs_for), resolved against E2's base5m.yaml."""
    import e2plan
    txt = dict(e2plan.runs_for("5m", eta, r))[f"{e2plan.arm_name('5m', eta, r)}_{tag}"]
    return flat(resolve_text(txt.replace("extends: base5m.yaml", f"extends: {os.path.join(E2D, 'configs', 'base5m.yaml')}")))


def check(path: str, p: dict | None = None) -> list[str]:
    """C2 + RC-12 GUARD + C1-b + the C6 hook + 2% rule for one run config. [] = ok."""
    from budget import count_analytic
    from config import PlanckConfig
    p = p or params()
    cfg = resolve(path)
    f, name, bad = flat(cfg), cfg.get("name", ""), []
    if "rc12" in (cfg.get("eval") or {}):
        bad.append("RC-12 GUARD: eval.rc12 is set (no RC-12 scoring before the lock)")
    m = NAME_RE.match(name)
    if not m or os.path.basename(path) != f"{name}.yaml" or not str(cfg.get("out_dir", "")).endswith(f"runs/SCREENS/{name}"):
        return bad + [f"name {name!r}: not a SCREENS run name, or file name / out_dir do not match it"]
    sid = m["sid"].upper() if m["sid"] else None
    arm, seed = m["arm"], int(m["seed"]) if m["seed"] else None
    home = SCREENS.get(sid, {}).get("dir") if sid else "screens"     # shared BASE runs: experiments/screens
    if os.path.basename(os.path.dirname(os.path.dirname(path))) != home:
        bad.append(f"{name}: not in experiments/{home}/configs")
    lr5, regd, allowed = p["lr5"], {}, {"name", "out_dir", "seed"}
    if m["tag"]:                                       # S003 search run: E2's stage config + optim.kind adamw
        e, r = float(m["e"]) * 1e-3, float(m["r"])
        ref = s003_e2_reference(e, r, m["tag"])
        allowed |= {"optim.kind", "train.compile", "engine_fixed", "schedule.init_from"} | set(hook_keys())
        regd = {"optim.kind": "adamw", "train.compile": COMPILE, "engine_fixed": ENGINE_FIXED, **hook_keys()}
        init, e2i = f.get("schedule.init_from"), ref.get("schedule.init_from")   # E2's own branch point
        want_i = e2i.replace("/runs/E2/5m_", "/runs/SCREENS/s003_adamw_", 1) if e2i else None
        if init != want_i:          # the same trunk step E2 branches from, in this arm's SCREENS trunk (review 10-03)
            bad.append(f"{name}: init_from {init} is not this arm's SCREENS trunk at E2's branch point ({want_i})")
    elif arm == "base":
        if sid and not engine(sid):
            return bad + [f"{name}: {sid} has no ENGINE keys, so no own BASE (it pairs with base_s<seed>)"]
        ref, regd = reference(None, lr5), engine(sid)
        allowed |= set(regd)
    else:
        if sid not in SCREENS or arm not in SCREENS[sid]["arms"]:
            return bad + [f"{name}: {sid} has no arm {arm!r}"]
        ref, regd = reference(sid, lr5), dict(SCREENS[sid]["arms"][arm])
        allowed |= set(regd) | set(LRK)
        if m["e"]:                                     # S003 seed run at its own searched LRs
            want = lrs(float(m["e"]) * 1e-3, float(m["r"]))
        else:
            g = float(m["g"] or "nan")
            if g not in G_ALLOWED:
                bad.append(f"{name}: g {m['g']} is not a power of 2 in [1/8, 8] (C3)")
            want = lrs(*lr5, g if g == g else 1.0)
        for k, v in want.items():
            if not math.isclose(float(f.get(k, -1)), v, rel_tol=1e-9):
                bad.append(f"{name}: {k} = {f.get(k)} but C3 gives {v:.6g}")
    for k in sorted(diff(f, ref) - allowed):
        bad.append(f"C2: {name} differs from its BASE in {k}: {ref.get(k, '<absent>')!r} -> {f.get(k, '<absent>')!r}")
    for k, v in regd.items():
        if f.get(k, "<absent>") != v:
            bad.append(f"C2: {name} must set {k} = {v!r} (registered), has {f.get(k, '<absent>')!r}")
    if (f.get("train.compile"), f.get("train.compile_dynamic"), f.get("train.ce_chunk_rows")) != (COMPILE, None, 0):
        bad.append(f"C1-b: {name} engine compile/compile_dynamic/ce_chunk_rows = {f.get('train.compile')!r}/"
                   f"{f.get('train.compile_dynamic')!r}/{f.get('train.ce_chunk_rows')!r}, want {COMPILE!r}/None/0")
    if f.get("engine_fixed") != ENGINE_FIXED:
        bad.append(f"C1-b: {name} engine_fixed {f.get('engine_fixed')!r} != {ENGINE_FIXED!r}")
    hk = hook_keys()
    if not hk or any(f.get(k, "<absent>") != v for k, v in hk.items()):
        bad.append(f"C6: {name} does not carry BASE's eval.induction hook unchanged")
    if seed is not None and seed != 1 and seed not in range(101, 106):
        bad.append(f"C4: {name} seed {seed} is neither the g-check seed 1 nor a screen seed 101..105")
    if cfg.get("seed") != (seed if seed is not None else 1):
        bad.append(f"{name}: seed key {cfg.get('seed')} does not match the name")
    n = count_analytic(PlanckConfig.from_dict(cfg["model"]))["total"]
    if abs(n - BASE_PARAMS) > 0.02 * BASE_PARAMS:
        bad.append(f"C2: {name} has {n:,} params, outside 2% of BASE {BASE_PARAMS:,}")
    return bad


def config_dir(sid: str | None) -> str:
    return os.path.join(EXPD, SCREENS[sid]["dir"] if sid else "screens", "configs")


def find(name: str) -> list[str]:
    """Every config file of this run name in the screens' config dirs (the queue wants exactly one)."""
    dirs = [config_dir(None)] + [config_dir(s) for s in SCREENS]
    return [os.path.join(d, name + ".yaml") for d in dirs if os.path.isfile(os.path.join(d, name + ".yaml"))]


def all_configs() -> list[str]:
    dirs = [config_dir(None)] + [config_dir(s) for s in SCREENS]
    return sorted(os.path.join(d, x) for d in dirs for x in os.listdir(d)
                  if x.endswith(".yaml") and x not in ("prereg.yaml", "screens_base.yaml"))
