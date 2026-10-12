"""T001 train.py tests: scoring with stub models (value-restricted argmax, stale and first counts), the LR schedule,
and a tiny CPU run (labelled smoke; its numbers are not results)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402
from task import KEY0, VOCAB, eval_rng, make_batch  # noqa: E402


class Stub(torch.nn.Module):
    """Answers every query of the fixed eval set with what(q); a key token gets an even larger logit (outside the
    value range, so only a value-restricted argmax ignores it)."""

    def __init__(self, family, n, size, what):
        super().__init__()
        self.x, _, self.meta = make_batch(n, size, eval_rng("test", family, n), family)
        self.what = what

    def forward(self, idx):
        out = torch.zeros(idx.shape[0], idx.shape[1], VOCAB)
        for r in range(idx.shape[0]):
            row = int(np.nonzero((self.x == idx[r].numpy()).all(1))[0][0])
            for q in self.meta[row]:
                out[r, q["pos"], self.what(q)] = 1.0
                out[r, q["pos"], KEY0] = 5.0
        return out


def test_score_oracle_and_first_value():
    n, size = 16, 6
    res = train.score(Stub("mqrar", n, size, lambda q: q["ans"]), "mqrar", n, "test", size, "cpu", bs=4)
    assert res["acc"] == 1.0 and res["lw"] == 1.0 and res["stale"] == 0 and res["first"] == 0
    assert res["q"] == sum(len(m) for m in make_batch(n, size, eval_rng("test", "mqrar", n), "mqrar")[2])
    res = train.score(Stub("mqrar", n, size, lambda q: q["first"]), "mqrar", n, "test", size, "cpu", bs=4)
    _, _, meta = make_batch(n, size, eval_rng("test", "mqrar", n), "mqrar")
    upd = [q for m in meta for q in m if q["updated"]]
    moved = [q for q in upd if q["first"] != q["ans"]]
    assert res["upd_q"] == len(upd) and res["first"] == len(moved) == res["stale"] and len(moved) > 0
    assert res["lw"] == pytest.approx(1 - len(moved) / len(upd))


def test_score_tracked_queries():
    res = train.score(Stub("track", 16, 4, lambda q: q["ans"]), "track", 16, "test", 4, "cpu")
    assert res["trk_q"] > 0 and res["trk_acc"] == 1.0


def test_lr_schedule():
    assert train.lr_at(0, 6000, 1.0) == pytest.approx(1 / 200)
    assert train.lr_at(199, 6000, 1.0) == pytest.approx(1.0)
    assert train.lr_at(200, 6000, 1.0) == pytest.approx(1.0)
    assert train.lr_at(5999, 6000, 1.0) == pytest.approx(0.1, abs=1e-6)


def tiny(tmp_path, arm, seed=1, family="mqrar"):
    a = train.parse(["--arm", arm, "--seed", str(seed), "--lr", "1e-3", "--out", str(tmp_path / "x.json"),
                     "--steps", "3", "--n-hi", "16", "--grid", "8,16", "--eval-test", "2", "--eval-dev", "2",
                     "--log-every", "1", "--device", "cpu", "--family", family, "--smoke"])
    return train.run(a)


def test_tiny_run_record(tmp_path):
    r = tiny(tmp_path, "base")
    assert r["smoke"] and not r["diverged"] and r["steps_done"] == 3 and len(r["loss_curve"]) == 3
    assert set(r["eval"]) == {"test", "dev"} and set(r["eval"]["test"]) == {"mqrar", "ar"}
    assert set(r["eval"]["test"]["mqrar"]) == {"8", "16"}
    t = tiny(tmp_path, "base", family="track")
    assert set(t["eval"]) == {"test"} and set(t["eval"]["test"]) == {"track", "mqrar"}


def test_same_batches_across_arms(tmp_path):
    """canon starts as the base function and the data stream does not depend on the model: step 0's loss is equal."""
    b, c = tiny(tmp_path, "base"), tiny(tmp_path, "canon")
    assert b["loss_curve"][0][1] == c["loss_curve"][0][1]
    assert tiny(tmp_path, "base", seed=2)["loss_curve"][0][1] != b["loss_curve"][0][1]


def test_score_first_excludes_a_return_to_the_first_value(monkeypatch):
    """Hand-made sample: k0 = v0, then v1, then v0 again; the query's answer IS its first value. An oracle answer
    counts as correct and as neither stale nor 'first' (first counts only a wrong pick of a moved first value)."""
    from task import ASK, BOS, SET, VAL0 as V0
    k = KEY0
    x = np.array([[BOS, SET, k, V0, SET, k, V0 + 1, SET, k, V0, ASK, k, V0]])
    q = dict(pos=11, ans=V0, key=0, updated=True, stale=[V0 + 1], first=V0, tracked=False, dist=1, outvoted=False,
             tied=False)                                     # votes: V0 twice, V0 + 1 once
    monkeypatch.setattr(train, "make_batch", lambda *a, **kw: (x, None, [[q]]))

    class Oracle(torch.nn.Module):
        def forward(self, idx):
            out = torch.zeros(1, 13, VOCAB)
            out[0, 11, V0] = 1.0
            return out
    res = train.score(Oracle(), "mqrar", 1, "test", 1, "cpu")
    assert res["acc"] == 1.0 and res["upd_q"] == 1 and res["first"] == 0 and res["stale"] == 0
