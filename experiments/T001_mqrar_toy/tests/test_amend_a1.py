"""T001 amendment A1 tests: the outvoted / tied query flags (an independent recount from the token stream), their
scoring in train.score, and the vote-counting shortcut the amendment exists for (it must score 0 on outvoted queries
while clearing 0.9 overall, so the absence claim is tested on a fixture that really produces the shortcut)."""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M  # noqa: E402
import task  # noqa: E402
import train  # noqa: E402
from task import ASK, SET, VAL0, VOCAB, eval_rng, make_batch, make_sample  # noqa: E402

CASES = [(fam, n, seed) for fam in task.FAMILIES for n in (2, 8, 33, 96) for seed in (0, 1, 7)]


def prior_votes(toks, pos):
    """Value counts over the explicit "SET k v" / "ASK k v" events of the queried key before the query at pos."""
    key = toks[pos]
    return collections.Counter(toks[i + 2] for i in range(1, pos - 1, 3)
                               if toks[i] in (SET, ASK) and toks[i + 1] == key)


@pytest.mark.parametrize("fam,n,seed", CASES)
def test_flags_match_a_recount(fam, n, seed):
    toks, qs = make_sample(n, np.random.default_rng(seed), fam)
    for q in qs:
        c = prior_votes(toks, q["pos"])
        mine = c.get(q["ans"], 0)
        rival = max([m for v, m in c.items() if v != q["ans"]], default=0)
        assert mine + rival >= 1                                       # every key was SET explicitly first
        assert q["outvoted"] == (rival > mine)
        assert q["tied"] == (rival == mine)
        assert not (q["outvoted"] and q["tied"])
        assert q["updated"] or not (q["outvoted"] or q["tied"])       # one write: no rival value exists


def test_flags_are_not_vacuous():
    """The registered test streams do hold outvoted and tied queries (A1 counted 142 outvoted at n 8)."""
    _, _, meta = make_batch(8, 256, eval_rng("test", "mqrar", 8), "mqrar")
    qs = [q for m in meta for q in m]
    assert sum(q["outvoted"] for q in qs) == 142 and sum(q["tied"] for q in qs) == 447 and len(qs) == 2030


def mode_recent(toks, pos):
    """The critique's shortcut: the most frequent earlier value of the key, ties to the most recent."""
    key = toks[pos]
    occ = [toks[i + 2] for i in range(1, pos - 1, 3) if toks[i] in (SET, ASK) and toks[i + 1] == key]
    c = collections.Counter(occ)
    best = max(c.values())
    return next(v for v in reversed(occ) if c[v] == best)


class Shortcut(torch.nn.Module):
    """Puts the logit mass on mode_recent's value (or the true answer) at every scored position."""

    def __init__(self, n, size, oracle=False):
        super().__init__()
        self.x, _, self.meta = make_batch(n, size, eval_rng("test", "mqrar", n), "mqrar")
        self.oracle = oracle

    def forward(self, idx):
        out = torch.zeros(idx.shape[0], idx.shape[1], VOCAB)
        for r in range(idx.shape[0]):
            row = int(np.nonzero((self.x == idx[r].numpy()).all(1))[0][0])
            toks = self.x[row].tolist()
            for q in self.meta[row]:
                out[r, q["pos"], q["ans"] if self.oracle else mode_recent(toks, q["pos"])] = 1.0
        return out


def test_vote_shortcut_clears_0p9_but_scores_0_on_outvoted():
    n, size = 16, 48
    res = train.score(Shortcut(n, size), "mqrar", n, "test", size, "cpu")
    assert res["acc"] >= 0.9 and res["lw"] < 0.85                      # the shortcut exists in this fixture
    assert res["ov_q"] > 0 and res["ov_ok"] == 0 and res["acc_ov"] == 0.0
    assert res["tie_q"] > 0 and res["acc_tie"] == 1.0                   # recency breaks ties toward the answer
    ora = train.score(Shortcut(n, size, oracle=True), "mqrar", n, "test", size, "cpu")
    assert ora["acc_ov"] == 1.0 and ora["ov_q"] == res["ov_q"] and ora["acc_tie"] == 1.0


def test_vote_shortcut_has_no_capacity_under_a1():
    """Per-n numbers of the shortcut (acc 0.91-0.93, outvoted 0): the registered capacity calls it censored at 96;
    A1's capacity gives the floor, and its LWO is 0."""
    G = [8, 16, 24, 32, 48, 64, 96]
    js = dict(grid=G, diverged=False, eval={"test": {
        "mqrar": {str(n): dict(acc=0.913, acc_ov=0.0, lw=0.78) for n in G}, "ar": {"32": dict(acc=1.0)}}})
    r = M.run_metrics(js)
    assert r["cap_all"] == 96 and r["cap"] == M.CAP_FLOOR and r["lwo"] == 0.0 and not r["censored"]
