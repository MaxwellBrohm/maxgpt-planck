"""Track K run configs and queue plans for STAGE 0 and K1 (SPEC 7, 11, 13; K1 notes STAGE 0, ARMS). Mac or PC, no
model. Built 2026-11-21 (K3 round 16, SPEC 14 BUILD's "a run-config writer (NOT BUILT)").

  python kconfig.py base                          ../configs/k_base.yaml
  python kconfig.py k0 --evals EVALS_DIR          K0-smoke (probe on, probe off, and a probe-off replay: CUDA eager is
                                                  not bitwise repeatable, so the replay measures the noise floor) and
                                                  K0-g: g 0.5, 1, 2 at seeds 199 and 200 [G-2]; plans/k0.txt, where
                                                  the grid waits for k0smoke.py's pass mark (SPEC 13's precondition)
  python kconfig.py k0x G --evals EVALS_DIR       K0-g's one extension (G 0.25 or 4) at both seeds; plans/k0x.txt
  python kconfig.py k1 --evals EVALS_DIR          K0-cap (seed 200) and K1 (F0, FL, FH at 201-203) at g_K, read from
                                                  the final K0-g pick in results/readings.jsonl (never typed in);
                                                  plans/k1a.txt (K0-cap, then FH seed 201: K1-b's interim stop) and
                                                  plans/k1b.txt (the other 8, seed-major)
  python kconfig.py check [NAME ...]              every config (or the named ones) against REG's rules

EVALS_DIR holds gen.py evals' manifest.json; each config pins K-PROBE's sha256 from it, so a run whose data root holds
other eval files is refused at its start (kprobe.py). Every config is written, then checked; a refused config is
deleted and the command fails; an existing config is never overwritten with different content. Configs sit in
experiments/K/configs, three levels below the code root (gen.data_block's paths), next to prereg.yaml (train.py's
gate). Every K run starts through ktrain.py (code/notes.txt SPEC 14 BUILD).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "..", "configs")
PLANS = os.path.join(HERE, "..", "plans")
READINGS = os.path.join(HERE, "..", "results", "readings.jsonl")
ROOTREL = "../../../planck_root"
STEPS, CKPT, WARMUP, DECAY = 7_630, 153, 76, 0.2        # SPEC 7: SCREENS C1 key for key, B = 250.0M slots
SMOKE_STEPS = 2 * CKPT                                  # K0-smoke: two probe readings and a checkpoint, never read
G_GRID, G_EXT, SEL, PAIRED = (0.5, 1.0, 2.0), (0.25, 4.0), (199, 200), (201, 202, 203)
SMOKE_MARK = "K0-smoke passed"                         # queue_k.sh wait_mark text; k0smoke.py writes it on a pass
K1_ARMS = ("F0", "FL", "FH")
NAME_RE = re.compile(r"^(?:(?P<smoke>k0_smoke(?P<off>_off|_rep)?)|k0g_g(?P<g>[0-9p]+)|(?P<cap>k0cap)|"
                     r"k1_(?P<arm>f0|fl|fh))_s(?P<seed>\d+)$")
BASE_TEXT = f"""# Track K base (SPEC 7; K1 notes STAGE 0; written by kconfig.py). Every K run config extends this file and sets
# only name, out_dir, seed, its LRs (LR5 x g), its data sources (gen.py data_block of its arm) and its k: block
# (ktrain.py). The model, data ids, optimizer and engine are E2's base5m.yaml (E2 5M shape, 5,010,133 params);
# schedule and length are SCREENS C1's ({STEPS:,} steps x 32,768 slots, warmup 76, linear decay over the last 20%,
# ckpt_every {CKPT}); every K run is eager (code/notes.txt K COMPILE DECISION: SCREENS C1-b's engine values).
extends: ../../E2_lr_transfer/configs/base5m.yaml
engine_fixed: "2026-10-03 f314eb7 SCREENS C1-b (K COMPILE DECISION, 2026-10-11)"
schedule: {{mode: full, decay_frac: {DECAY}}}
train: {{total_steps: {STEPS}, ckpt_every: {CKPT}, compile: false}}
"""


def gname(g: float) -> str:
    return f"g{g:g}".replace(".", "p")


def g_of(tok: str) -> float:
    return float(tok.replace("p", "."))


def lr5() -> tuple[float, float]:
    """LR5 = (eta_5, r_5): E2 results.json stage C's pick, as E3's recheck confirmed it (SPEC 7; SCREENS C1)."""
    ex = os.path.join(K.ROOT, "experiments")
    e2 = json.load(open(os.path.join(ex, "E2_lr_transfer", "results.json")))
    e3 = json.load(open(os.path.join(ex, "E3_seed_noise", "results.json")))
    pick, rc = tuple(e2["stage_C"]["lrs"]["pick"]), e3["e2_recheck"]
    if not (rc["verdict"].startswith("CONFIRM") and tuple(rc["lr5_for_screens"]) == pick):
        raise ValueError(f"LR5: E2's pick {pick} is not E3's confirmed {rc['lr5_for_screens']}")
    return float(pick[0]), float(pick[1])


def lrs(g: float) -> dict:
    eta, r = lr5()
    return {"lr": g * eta, "embed_lr": g * r * eta, "scalar_lr": g * r * eta}


def probe_block(evals_dir: str) -> dict:
    man = json.load(open(os.path.join(evals_dir, "manifest.json")))
    return {"every": CKPT, "files": [{"file": f"{ROOTREL}/data/k/evals/k_probe.jsonl",
                                      "sha256": man["sets"]["k_probe"]["sha256"]}]}


def spec(name: str) -> dict:
    """REG: what a registered name must hold. -> {arm, g (None: g_K), seeds, steps, probe}."""
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"{name}: not a registered K run name")
    if m["smoke"]:
        return {"arm": "FH", "g": 1.0, "seeds": (200,), "steps": SMOKE_STEPS, "probe": not m["off"], "kind": "smoke"}
    if m["g"]:
        g = g_of(m["g"])
        if g not in G_GRID + G_EXT:
            raise ValueError(f"{name}: g {g} is neither K0-g's grid {G_GRID} nor an extension {G_EXT}")
        return {"arm": "FH", "g": g, "seeds": SEL, "steps": STEPS, "probe": True, "kind": "k0g"}
    if m["cap"]:
        return {"arm": "CAP", "g": None, "seeds": (200,), "steps": STEPS, "probe": True, "kind": "k0cap"}
    return {"arm": m["arm"].upper(), "g": None, "seeds": PAIRED, "steps": STEPS, "probe": True, "kind": "k1"}


def text(name: str, g: float, probe: dict | None, header: str) -> str:
    import gen
    s, sp = spec(name), NAME_RE.match(name)
    seed = int(sp["seed"])
    src = gen.data_block(s["arm"])["sources"]
    lines = [f"# {header} (kconfig.py)", "extends: k_base.yaml", f"name: {name}",
             f"out_dir: {ROOTREL}/runs/K/{name}", f"seed: {seed}",
             "optim: " + json.dumps(lrs(g))]
    if s["steps"] != STEPS:
        lines.append(f"train: {{total_steps: {s['steps']}}}")
    lines += ["data:", "  sources:"] + [f"    - {json.dumps(x)}" for x in src]
    if probe is not None:
        lines += ["k:", "  probe: " + json.dumps(probe)]
    return "\n".join(lines) + "\n"


def check(path: str, g_k: float | None = None) -> list[str]:
    """K's registry rules for one config (independent of text()): [] = ok."""
    import runio
    import ktrain
    import gen
    from budget import count_analytic
    from config import PlanckConfig
    bad, cfg = [], runio.load_yaml(path)
    name = cfg.get("name", "")
    try:
        s = spec(name)
    except ValueError as e:
        return [str(e)]
    if os.path.basename(path) != f"{name}.yaml" or cfg.get("out_dir") != f"{ROOTREL}/runs/K/{name}":
        bad.append(f"{name}: file name or out_dir does not match the name")
    if not os.path.isfile(os.path.join(os.path.dirname(path), "prereg.yaml")):
        bad.append(f"{name}: no prereg.yaml next to it")
    seed = int(NAME_RE.match(name)["seed"])
    if cfg.get("seed") != seed or seed not in s["seeds"]:
        bad.append(f"{name}: seed {cfg.get('seed')} is not its name's, or not among {s['seeds']}")
    g = s["g"] if s["g"] is not None else g_k
    oc = cfg.get("optim", {})
    if g is None:
        bad.append(f"{name}: g_K unknown (no final K0-g pick)")
    elif any(abs(float(oc.get(k, -1)) - v) > 1e-12 for k, v in lrs(g).items()):
        bad.append(f"{name}: LRs {[oc.get(k) for k in lrs(g)]} are not LR5 x {g}")
    if cfg.get("data", {}).get("sources") != gen.data_block(s["arm"])["sources"]:
        bad.append(f"{name}: data sources are not gen.data_block({s['arm']!r})'s")
    tc, sc = cfg.get("train", {}), cfg.get("schedule", {})
    want = (s["steps"], CKPT, False, "full", WARMUP, DECAY)
    have = (tc.get("total_steps"), tc.get("ckpt_every"), tc.get("compile"), sc.get("mode"), sc.get("warmup_steps"),
            sc.get("decay_frac"))
    if have != want:
        bad.append(f"{name}: (total_steps, ckpt_every, compile, mode, warmup, decay_frac) {have}, not {want}")
    if (tc.get("micro_batch"), tc.get("grad_accum"), cfg["model"].get("seq_len")) != (16, 1, 2048):
        bad.append(f"{name}: not 16 x 2048 slots per step")
    if count_analytic(PlanckConfig.from_dict(cfg["model"]))["total"] != 5_010_133:
        bad.append(f"{name}: not SPEC 7's 5,010,133-parameter model")
    kc = cfg.get("k") or {}
    pr = kc.get("probe")
    if s["probe"] != (pr is not None) or set(kc) - {"probe"}:
        bad.append(f"{name}: k block {sorted(kc)}; registered: {'probe only' if s['probe'] else 'none'}")
    if pr is not None and (pr.get("every") != CKPT or [f.get("file") for f in pr.get("files", [])] !=
                           [f"{ROOTREL}/data/k/evals/k_probe.jsonl"] or "sets" in pr["files"][0]):
        bad.append(f"{name}: k.probe is not K-PROBE every {CKPT}")
    try:
        ktrain.check(cfg)
    except ValueError as e:
        bad.append(f"{name}: ktrain refuses it: {e}")
    return bad


def final_pick(path: str | None = None) -> float | None:
    """g_K: the last K0-g reading in the readings file (READINGS unless given), if it is final (k0read.py writes it)."""
    path, last = path or READINGS, None
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("reading") == "K0-g":
                last = r
    return float(last["g"]) if last and last.get("final") and last.get("g") is not None else None


def write(name: str, txt: str, g_k: float | None = None) -> str:
    path = os.path.join(CONF, name + ".yaml")
    if os.path.exists(path):
        if open(path).read() != txt:
            raise SystemExit(f"refusing to overwrite {path} with different content")
    else:
        with open(path, "w") as f:
            f.write(txt)
    bad = check(path, g_k)
    if bad:
        os.unlink(path)
        raise SystemExit("refused (config deleted):\n  " + "\n  ".join(bad))
    return name


def plan(fname: str, lines: list[str]) -> None:
    os.makedirs(PLANS, exist_ok=True)
    with open(os.path.join(PLANS, fname), "w") as f:
        f.write("".join(x + "\n" for x in lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "k0", "k0x", "k1", "check"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--evals")
    a = ap.parse_args(argv)
    os.makedirs(CONF, exist_ok=True)
    if a.cmd == "base":
        p = os.path.join(CONF, "k_base.yaml")
        if os.path.exists(p) and open(p).read() != BASE_TEXT:
            raise SystemExit(f"refusing to overwrite {p} with different content")
        open(p, "w").write(BASE_TEXT)
        return 0
    if a.cmd == "check":
        names = a.rest or sorted(f[:-5] for f in os.listdir(CONF) if NAME_RE.match(f[:-5]))
        bad = [b for n in names for b in check(os.path.join(CONF, n + ".yaml"), final_pick())]
        print("\n".join(bad) or f"ok: {len(names)} configs")
        return 1 if bad else 0
    pb = probe_block(a.evals)
    if a.cmd == "k0":
        runs = [write("k0_smoke_s200", text("k0_smoke_s200", 1.0, pb, "K0-smoke, FH at g 1, probe on")),
                write("k0_smoke_off_s200", text("k0_smoke_off_s200", 1.0, None, "K0-smoke, FH at g 1, probe off")),
                write("k0_smoke_rep_s200", text("k0_smoke_rep_s200", 1.0, None, "K0-smoke, probe-off replay"))]
        runs += [write(n, text(n, g, pb, f"K0-g, FH at g {g:g}, selection seed {s}")) for s in SEL for g in G_GRID
                 for n in [f"k0g_{gname(g)}_s{s}"]]
        # 2026-11-22 (REVIEW 17b): SPEC 13 makes "the K smoke run passed" a precondition of every K run, so the
        # grid waits for k0smoke.py's pass mark (written only on a pass) after the three smoke runs
        plan("k0.txt", [f"train {n}" for n in runs[:3]] + [f"wait_mark K {SMOKE_MARK}"] +
             [f"train {n}" for n in runs[3:]] + ["mark K0-g grid done: read k0read.py, then k0x or k1"])
    elif a.cmd == "k0x":
        g = float(a.rest[0])
        if g not in G_EXT:
            raise SystemExit(f"k0x: g {g:g} is not one of K0-g's extensions {G_EXT}")
        runs = [write(n, text(n, g, pb, f"K0-g extension, FH at g {g:g}, selection seed {s}")) for s in SEL
                for n in [f"k0g_{gname(g)}_s{s}"]]
        plan("k0x.txt", [f"train {n}" for n in runs] + ["mark K0-g extension done: read k0read.py"])
    else:
        g = final_pick()
        if g is None:
            raise SystemExit(f"no final K0-g pick in {READINGS}: run k0read.py first")
        cap = write("k0cap_s200", text("k0cap_s200", g, pb, f"K0-cap, FACT share 1.0, g_K {g:g}"), g)
        k1 = [write(n, text(n, g, pb, f"K1 {arm}, g_K {g:g}, seed {s}"), g) for s in PAIRED for arm in K1_ARMS
              for n in [f"k1_{arm.lower()}_s{s}"]]
        first = "k1_fh_s201"
        plan("k1a.txt", [f"train {cap}", f"train {first}", "mark K1-b: read LOADED(FH) at seed 201", "stop"])
        plan("k1b.txt", [f"train {n}" for n in k1 if n != first] + ["mark K1 done"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
