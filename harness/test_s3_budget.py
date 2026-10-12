"""(c) Built models match budget.py counts exactly (total, embedding, body).

test_budget.py checks each switch on a fixed shape. Here: a seeded random sweep over every
switch at once (CPU builds, all under ~1M parameters), every solver answer for the size
curve (3M..150M, several vocabs and arms, built on 'meta', which allocates nothing), and
the n_params that train.py writes to runs.jsonl for a real run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import train
from budget import Constraints, count_analytic, solve
from config import PlanckConfig
from count_params import build_and_count
from make_fake_data import make
from testutil import read_jsonl, write_run


def random_cfg(rng: np.random.Generator) -> PlanckConfig:
    hd = int(rng.choice([8, 16, 32]))
    kv = int(rng.integers(1, 4))
    heads = kv * int(rng.integers(1, 4))
    loops = int(rng.integers(1, 4))
    return PlanckConfig(
        vocab_size=int(rng.integers(50, 700)), d_model=int(rng.choice([16, 24, 32, 48, 64])),
        n_layers=int(rng.integers(1, 5)), n_heads=heads, n_kv_heads=kv, head_dim=hd,
        mlp_hidden=int(rng.choice([0, 8, 40, 96, 170])), seq_len=32,
        tie_embeddings=bool(rng.integers(2)), qk_norm=bool(rng.integers(2)),
        attn_gate=bool(rng.integers(2)), value_residual=bool(rng.integers(2)),
        norm_scaling=bool(rng.integers(2)), n_loops=loops,
        loop_order=str(rng.choice(["cyclic", "immediate"])),
        n_prelude=int(rng.integers(0, 3)), n_coda=int(rng.integers(0, 3)),
        qk_share=int(rng.integers(1, 4)), kv_tie=bool(rng.integers(2)))


def test_random_sweep_matches_built_module():
    rng = np.random.default_rng(2026)
    seen_switches = set()
    for _ in range(150):
        c = random_cfg(rng)
        assert count_analytic(c) == build_and_count(c, "cpu"), c
        seen_switches |= {k for k in ("kv_tie", "tie_embeddings", "value_residual") if getattr(c, k)}
        seen_switches |= {"loop"} if c.n_loops > 1 else set()
        seen_switches |= {"share"} if c.qk_share > 1 else set()
    assert seen_switches == {"kv_tie", "tie_embeddings", "value_residual", "loop", "share"}


ARM_CONSTRAINTS = {
    "dense": {},
    "mlp_light": {"mlp_ratio": 4 / 3, "attn_mult": 1.5},
    "gqa": {"kv_group": 2},
    "looped2": {"base": {"n_loops": 2}},
    "shared_qk_tied_kv": {"base": {"qk_share": 2, "kv_tie": True}},
    "attn_only": {"mlp_ratio": 0.0},
}


@pytest.mark.parametrize("arm", list(ARM_CONSTRAINTS))
@pytest.mark.parametrize("vocab", [4096, 8192, 16384])
def test_size_curve_solutions_are_exact(arm, vocab):
    for target in (3e6, 5e6, 10e6, 20e6, 30e6, 60e6, 150e6):
        s = solve(target, Constraints(vocab_size=vocab, **ARM_CONSTRAINTS[arm]))  # all 126 solve
        built = build_and_count(s.cfg, "meta")
        assert (built["total"], built["embedding"], built["body"]) == (s.total, s.embedding, s.body)
        assert abs(s.total / target - 1) <= 0.02


def test_train_logs_the_exact_count(tmp_path):
    data = tmp_path / "data"
    make(str(data), docs=40, chats=40)
    model = {"vocab_size": 256, "d_model": 32, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1,
             "head_dim": 16, "mlp_hidden": 40, "seq_len": 64, "n_loops": 3, "n_prelude": 1,
             "qk_share": 2, "kv_tie": True}
    cfg = write_run(str(tmp_path / "r"), str(data), model=model, train={"total_steps": 2})
    assert train.main([cfg]) == 0
    runs = read_jsonl(os.path.join(tmp_path, "runs.jsonl"))
    assert runs[0]["n_params"] == count_analytic(PlanckConfig.from_dict(model))["total"]
