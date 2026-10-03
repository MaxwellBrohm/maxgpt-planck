"""C1-a gate for one screen arm (experiments/SCREENS.txt AMENDMENT C1-a). PC, CUDA, under gpu.lock (one hold).

  python flag_parity.py TARGET --out OUT.json [compile_parity.py args: --steps 200 --arms ...]
  TARGET = SID (a one-arm screen), SID:ARM (S002:novres), or SID:base (a screen's own BASE: its ENGINE keys only)

Before a code screen's arms start: (1) a compiled smoke of its flag arm (gate_smoke.py), graph count and graph
breaks logged, no recompile-limit message; (2) a 200-step parity of that arm, compiled vs eager, by
compile_parity.py's rule. The harness's compile_parity.main runs unchanged; it builds budget.py's 5M shape on
train.py's defaults and has no option for a screen flag, so this wrapper adds, to every arm of the parity (eager
and compiled alike), the arm's registered keys and its screen's ENGINE train keys (screens_lib: S004 canon AC,
S005 forget_gate + doc_attn mask at micro 8 x accum 2, S006 mtp, S007 smear_key; S001 and S002 model flags off).
Each run record in the JSON gains "dynamo" (unique graphs, graph breaks and their reasons, every torch._dynamo log
line that mentions a recompile or cache limit) and "ran" (the arm's start record fields and the model flags read
back from the config it trained on). Exit 0 = parity pass AND no such line in a compiled arm; the screen then runs
compiled. Anything else: that screen runs ALL its arms eager (screens_lib ENGINE "train.compile": False, configs
rebuilt), logged in its notes.txt first.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys

import yaml

import screens_lib as L    # puts harness/ on sys.path

LIMIT_WORDS = ("recompile", "cache_size_limit", "cache size limit")
START_KEYS = ("n_params", "doc_attn", "compile", "compile_dynamic", "optim_batched", "mtp", "batch_tokens", "precision")


class Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.msgs: list[str] = []

    def emit(self, rec: logging.LogRecord) -> None:
        m = rec.getMessage()
        if any(w in m.lower() for w in LIMIT_WORDS):
            self.msgs.append(m[:400])


def arm_keys(target: str) -> dict:
    """TARGET -> every key the parity arms must carry (ENGINE keys, then the arm's registered keys)."""
    sid, _, arm = target.partition(":")
    arms = L.SCREENS[sid]["arms"]
    if arm == "base":
        assert L.engine(sid), f"{sid} has no ENGINE keys: it pairs with the default-engine BASE, no own BASE"
        return L.engine(sid)
    if not arm:
        assert len(arms) == 1, f"{sid} has arms {sorted(arms)}: name one as {sid}:ARM"
        (arm,) = arms
    assert arm in arms, f"{sid} has no arm {arm!r}"
    return {**L.engine(sid), **arms[arm]}


def flag_keys(target: str) -> tuple[dict, dict, dict]:
    """-> (model keys, train keys for every parity arm, micro/accum) of TARGET."""
    keys = arm_keys(target)
    assert keys.get("train.compile", "default") != False, f"{target} is already eager: no gate to run"  # noqa: E712
    model = {k[6:]: v for k, v in keys.items() if k.startswith("model.")}
    train = {k[6:]: v for k, v in keys.items() if k.startswith("train.") and k not in ("train.micro_batch", "train.grad_accum")}
    micro = {k[6:]: v for k, v in keys.items() if k in ("train.micro_batch", "train.grad_accum")}
    assert model or train, target
    return model, train, micro


def ran(work: str, i: int, model: dict) -> dict:
    """What arm i really ran: its start record (runs.jsonl) and the model flags in the config it trained on."""
    starts = [json.loads(x) for x in open(os.path.join(work, "runs.jsonl")) if '"event": "start"' in x]
    st = [s for s in starts if s.get("run") == f"arm{i}"][-1]
    cfg = yaml.safe_load(open(os.path.join(work, f"arm{i}", "config.yaml")))
    return {**{k: st[k] for k in START_KEYS if k in st}, "model_flags": {k: cfg["model"].get(k) for k in model}}


def carried_ok(r: dict, model: dict, train: dict) -> bool:
    """The arm trained with the flag, the engine and its compile mode (start records name doc_attn only when it is
    not mask, and compile only when on)."""
    got = r["ran"]
    return (got["model_flags"] == model
            and ("doc_attn" not in train or got.get("doc_attn", "mask") == train["doc_attn"])
            and (not train.get("mtp") or (got.get("mtp") or {}).get("heads") == train["mtp"])
            and ("compile" in got) == (not r["arm"].startswith("eager")))


def install(target: str):
    """Patch compile_parity so every arm carries the flag. Returns (compile_parity module, Grab)."""
    import compile_parity as CP
    from torch._dynamo.utils import counters
    model, train, _ = flag_keys(target)
    solve0, write0, run0 = CP.budget.solve, CP.write_run, CP.run_arm

    def solve(target, k=None, top=1):
        s = solve0(target, k, top)
        return dataclasses.replace(s, cfg=s.cfg.replace(**model))

    def write_run(d, data_dir, **over):
        over["train"] = {**over.get("train", {}), **train}
        return write0(d, data_dir, **over)

    grab = Grab()
    logging.getLogger("torch._dynamo").addHandler(grab)

    def run_arm(arm, i, a, model_cfg, data_dir, work):
        grab.msgs.clear()
        counters.clear()
        r = run0(arm, i, a, model_cfg, data_dir, work)
        r["dynamo"] = {"unique_graphs": counters["stats"].get("unique_graphs", 0),
                       "graph_breaks": sum(counters["graph_break"].values()),
                       "break_reasons": {k.splitlines()[0][:160]: v for k, v in counters["graph_break"].items()},
                       "limit_msgs": list(grab.msgs)}
        r["ran"] = ran(work, i, model)
        print(f"[c1a] arm {i}:{arm} {r['dynamo']['unique_graphs']} graphs, {r['dynamo']['graph_breaks']} breaks, "
              f"{len(grab.msgs)} limit lines, ran {r['ran']}", flush=True)
        return r

    CP.budget.solve, CP.write_run, CP.run_arm = solve, write_run, run_arm
    return CP, grab


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    target = argv.pop(0)
    model, train, micro = flag_keys(target)
    for k, flag in (("micro_batch", "--micro"), ("grad_accum", "--accum")):
        if k in micro and flag not in argv:
            argv += [flag, str(micro[k])]
    CP, _ = install(target)
    rc = CP.main(argv)
    out = argv[argv.index("--out") + 1]
    res = json.load(open(out))
    comp = [r for r in res["runs"] if not r["arm"].startswith("eager")]
    msgs = [m for r in comp for m in r["dynamo"]["limit_msgs"]]
    carried = all(carried_ok(r, model, train) for r in res["runs"])
    res["c1a_gate"] = {"screen": target, "flag": [model, train], "parity_pass": res["pass"],
                       "compiled_graphs": sorted({r["dynamo"]["unique_graphs"] for r in comp}),
                       "compiled_breaks": sorted({r["dynamo"]["graph_breaks"] for r in comp}),
                       "limit_msgs": msgs, "flag_carried": carried,
                       "pass": bool(res["pass"]) and not msgs and rc == 0 and carried}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[c1a] {target} {res['c1a_gate']}")
    return 0 if res["c1a_gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
