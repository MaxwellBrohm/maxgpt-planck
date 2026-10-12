"""RC-12 lockstep runner test (notes STEP 9; no model). lockstep.rows must give EXACTLY runner.run's rows (every
field, same order) for a deterministic responder:
  every_fake  every fake (validate_e2e.NAMES + fakes_family.TESTS) on every dev record, plain render
  template    every fake, template render, on one record per (family, cell) plus every OWN record
  seeds       SAMPLER (FIRST when sampled) with seeds greedy, 1, 2 in ONE lockstep pass
  trunc       FORGETFUL and HISTCHECK with ctx 556 (history dropped), plain and template; HISTCHECK saw exactly
              the user turns plus its own stripped replies in lockstep too (no violations in any instance)
  owncf       every fake on the OWN records with --own-cf (the rewrite happens per conversation between batches)
  shape       turn t of every live conversation in one reply_batch call: the first call holds every conversation,
              each call's requests are all at the same turn position, sizes never grow, calls = the longest length
  serial      an HFResponder subclass (no torch, deterministic) through lockstep.Serial: rows equal
  cli         runner.py --lockstep writes byte-identical transcripts.jsonl and scores.jsonl to a sequential run
  refuse      lockstep.engine refuses a bare fake (it must come wrapped in PerConv)
Run: python3 -B test_lockstep.py   (writes logs/test_lockstep.txt; exit 1 on any failure)"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import fakes_family as FF
import hf_responder as HR
import lockstep as LS
import render as RD
import runner as R
import validate_e2e as VE

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = list(dict.fromkeys(VE.NAMES + FF.TESTS))


def dumps(rows):
    return [json.dumps(r, sort_keys=True) for r in rows]


def same(tag, seq, lock):
    a, b = dumps(seq), dumps(lock)
    if a == b:
        return []
    k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return [f"{tag}: {len(a)} vs {len(b)} rows, first differing row {k}"]


def guarded(fn):
    """a check that raises is a failure line, not a crash (mutation_step9.py counts it as a kill)."""
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            return [f"{fn.__name__}: raised {type(e).__name__}: {str(e)[:120]}"]
    return run


@guarded
def compare(recs, name, render="plain", seeds=(None,), ctx=None, own_cf=False, tag=None):
    seq = R.run(recs, FF.make(name), render, list(seeds), ctx, None, name, 0, own_cf)
    lock = R.run(recs, LS.PerConv(lambda: FF.make(name), render), render, list(seeds), ctx, None, name, 0, own_cf,
                 lockstep=True)
    return same(tag or f"{name} {render} seeds={list(seeds)} ctx={ctx} own_cf={own_cf}", seq, lock)


def sample(recs):
    seen, out = set(), []
    for r in recs:
        if (r["family"], r["cell"]) not in seen or r["family"] == "OWN":
            seen.add((r["family"], r["cell"]))
            out.append(r)
    return out


class Recorder(LS.PerConv):
    def __init__(self, factory, render):
        super().__init__(factory, render)
        self.made, self.calls = [], []
        make = self.factory
        self.factory = lambda: self.made.append(make()) or self.made[-1]

    def reply_batch(self, reqs):
        self.calls.append([(k, i) for k, _, _, _, i in reqs])
        return super().reply_batch(reqs)


@guarded
def c_shape(recs):
    fails = []
    eng = Recorder(lambda: FF.make("HISTCHECK"), "plain")
    R.run(recs, eng, "plain", [None], 556, None, "HISTCHECK", lockstep=True)
    order = [sorted(t["i"] for t in r["turns"]) for r in recs]
    if not eng.calls or len(eng.calls[0]) != len(recs):
        fails.append("shape: the first reply_batch call does not hold every conversation")
    for step, call in enumerate(eng.calls):
        if any(order[k][step] != i for k, i in call):
            fails.append(f"shape: call {step} mixes turn positions")
        if step and len(call) > len(eng.calls[step - 1]):
            fails.append(f"shape: call {step} grew")
    if len(eng.calls) != max(len(o) for o in order):
        fails.append(f"shape: {len(eng.calls)} calls for a longest conversation of {max(len(o) for o in order)}")
    bad = [v for f in eng.made for v in getattr(f, "violations", [])]
    if bad:
        fails.append(f"shape: HISTCHECK history violations in lockstep: {bad[:3]}")
    if len(eng.made) != len(recs):
        fails.append(f"shape: {len(eng.made)} per-conversation instances for {len(recs)} conversations")
    return fails


class StubHF(HR.HFResponder):
    """an HFResponder whose start is the real one (it only sets the id and seed) and whose reply is deterministic."""
    def __init__(self, render):
        self.render, self.ctx = render, None

    def count(self, messages, render=None):
        return RD.proxy_tokens(messages)

    def reply(self, history, i):
        last = history[-1]["content"].split()
        return f"Reply {self.rid} {self.seed} {i} {len(history)} {' '.join(last[:3])}.", "eos"


@guarded
def c_serial(recs):
    seq = R.run(recs, StubHF("template"), "template", [None, 7], 900, None, "stub")
    lock = R.run(recs, StubHF("template"), "template", [None, 7], 900, None, "stub", lockstep=True)
    return same("serial: StubHF through lockstep.Serial", seq, lock)


def c_cli():
    fails = []
    with tempfile.TemporaryDirectory() as d:
        for tag, extra in (("seq", []), ("lock", ["--lockstep"])):
            cmd = [sys.executable, "-B", os.path.join(HERE, "runner.py"), "--responder", "fake:IDEAL_ALT", "--render",
                   "template", "--seeds", "greedy,3", "--limit", "80", "--out", os.path.join(d, tag)] + extra
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=100)
            if p.returncode:
                fails.append(f"cli: {tag} exit {p.returncode}: {p.stderr[-200:]}")
        for f in ("transcripts.jsonl", "scores.jsonl"):
            a, b = (os.path.join(d, t, f) for t in ("seq", "lock"))
            if not (os.path.exists(a) and os.path.exists(b) and open(a, "rb").read() == open(b, "rb").read()):
                fails.append(f"cli: {f} differs between runner.py and runner.py --lockstep")
    return fails


def c_refuse(recs):
    fails = []
    try:
        LS.rows(recs[:2], FF.make("IDEAL"), "plain")
        fails.append("refuse: lockstep ran a bare fake (shared per-conversation state)")
    except TypeError:
        pass
    ns = argparse.Namespace(render="plain", lockstep=True)
    if not isinstance(R.make_responder("fake:IDEAL", ns)[0], LS.PerConv):
        fails.append("refuse: runner --lockstep does not wrap a fake in PerConv")
    return fails


def checks(recs, names=ALL, cli=True):
    fails = []
    own = [r for r in recs if r["family"] == "OWN"]
    few = sample(recs)
    for name in names:
        fails += compare(recs, name, "plain")
        fails += compare(few, name, "template")
        fails += compare(own, name, "plain", own_cf=True)
    fails += compare(recs, "SAMPLER", "plain", (None, 1, 2))
    for name in ("FORGETFUL", "HISTCHECK"):
        for render in ("plain", "template"):
            fails += compare(recs, name, render, ctx=556)
    fails += c_shape(recs)
    fails += c_serial(few)
    fails += guarded(c_refuse)(recs)
    fails += c_cli() if cli else []
    return fails


def main():
    t0 = time.time()
    recs = R.load()
    fails = checks(recs)
    lines = [f"RC-12 test_lockstep.py: {len(ALL)} fakes, {len(recs)} records, {time.time() - t0:.0f} s. No model."]
    lines += [f"FAIL {f}" for f in fails] or ["ALL LOCKSTEP CHECKS PASS (rows identical to runner.run)"]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "test_lockstep.txt"), "w") as f:
        f.write(text)
    print(text)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
