"""OOD-H play and score (oodh/DESIGN.txt s4, s6, s7): rc12/runner.py with the thread's fixed human history. No model.

FixedHistory(responder) wraps ANY rc12 responder (a fake, hf, planck, vllm through runner.run's sequential path): on
a thread user turn it returns that turn's history_reply (the OASST2 assistant turn) with stop "eos" and never asks the
inner responder; on a probe turn it asks the inner responder, which sees the history the runner built (the human
history, then its OWN earlier probe replies: s3's own-history rule inside the probe block). The runner's plain-render
cut and strip still apply to the history replies, so gate.py checks that they come back unchanged. The lockstep path
(rc12/lockstep.py) is NOT wired for it.

OOD-H cheaters (added to rc12's; the rest of their policy is rc12/fakes.Cheater's):
  LATEST_USER       every probe: the latest THREAD user turn, verbatim (the human's own words parroted). Gate:
                    0.00 on every probe.
  LATEST_USER_SWAP  the same with I/you swapped (rc12/fakes.swap_person): a shortcut, gated at the RC-12 bars.
Reference responders (must pass every probe): IDEAL (rc12/fakes.Fake; the ideal of a thread turn is its history
reply) and TERSE (the bare gold, "Obscure.", on VAL probes; IDEAL on ABS probes).
Scores: a thread's unit = the mean of its probe results (meta.unit "mean"); family OODH = mean unit over threads;
cell = the probe-kind mix; per probe kind = the pass rate over probes of that kind (diagnostic)."""
import os
import sys

sys.dont_write_bytecode = True
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "rc12"))
import fakes as F  # noqa: E402
import fakes_family as FF  # noqa: E402
import runner as RN  # noqa: E402


class FixedHistory:
    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "?")
        self.ctx = getattr(inner, "ctx", None)

    def start(self, rec, seed=None, render="plain"):
        probe_turns = {p["turn"] for p in rec["probes"]}
        self.fixed = {t["i"]: t["history_reply"] for t in rec["turns"] if t["i"] not in probe_turns}
        self.inner.start(rec, seed, render)

    def count(self, messages, render):
        return self.inner.count(messages, render)

    def reply(self, history, i):
        if i in self.fixed:
            return self.fixed[i], "eos"
        return self.inner.reply(history, i)


def latest_thread_turn(rec, p):
    thread = [t for t in rec["turns"] if t["i"] < p["turn"] and "history_reply" in t]
    return max(thread, key=lambda t: t["i"])["text"]


class LatestUser(F.Cheater):
    name = "LATEST_USER"

    def probe(self, history, t, p):
        return latest_thread_turn(self.rec, p)


class LatestUserSwap(F.Cheater):
    name = "LATEST_USER_SWAP"

    def probe(self, history, t, p):
        return F.swap_person(latest_thread_turn(self.rec, p))


class Terse(F.Fake):
    name = "TERSE"

    def probe(self, history, t, p):
        return f"{p['gold']}." if p["grader"] == "VAL" else t["ideal"]


EXTRA = {"LATEST_USER": LatestUser, "LATEST_USER_SWAP": LatestUserSwap, "TERSE": Terse}


def make(name):
    return EXTRA[name]() if name in EXTRA else FF.make(name)


def play(recs, name, render="plain"):
    """every record through rc12 runner.run (the real feedback loop and graders) with the fixed history."""
    return RN.run(recs, FixedHistory(make(name)), render, [None], None, None, name)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def scores(rows, recs):
    """family, per-cell and per-probe-kind rates, plus per-thread units and per-probe results."""
    kinds = {r["id"]: {p["turn"]: p["probe_kind"] for p in r["probes"]} for r in recs}
    graders = {r["id"]: {p["turn"]: p["grader"] for p in r["probes"]} for r in recs}
    cells, pk, probes = defaultdict(list), defaultdict(list), []
    for row in rows:
        cells[row["cell"]].append(row["unit"])
        for res in row["probes"]:
            k = kinds[row["id"]][res["turn"]]
            pk[k].append(res["ok"])
            probes.append(dict(id=row["id"], turn=res["turn"], probe_kind=k, grader=graders[row["id"]][res["turn"]],
                               ok=res["ok"], fails=res["fails"]))
    return dict(family=mean([row["unit"] for row in rows]), cells={c: mean(v) for c, v in cells.items()},
                cell_n={c: len(v) for c, v in cells.items()}, kinds={k: mean(v) for k, v in pk.items()},
                units={row["id"]: row["unit"] for row in rows}, probes=probes)
