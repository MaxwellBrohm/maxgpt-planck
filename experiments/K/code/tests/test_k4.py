"""K4 shapes: harness/budget.py reproduces the registered (heads, SwiGLU width, total) at d 192 and depth 8, and the
written model blocks build models of exactly those sizes (meta device, nothing allocated)."""
import os

import yaml

import k4_shapes as S4


def test_budget_reproduces_registered_shapes():
    for r in S4.table():
        h, m, total = S4.REGISTERED[r["shape"]]
        assert (r["heads"], r["mlp_hidden"], r["total"]) == (h, m, total), r
        assert abs(r["err"]) <= 0.0025 and r["cfg"].d_model == 192 and r["cfg"].n_layers == 8
    rs = {r["shape"]: r["r"] for r in S4.table()}
    assert [round(rs[k], 2) for k in ("BASE", "H4", "H6", "H7")] == [1.91, 1.17, 0.45, 0.24]


def test_written_blocks_build_exact_models(tmp_path):
    from config import PlanckConfig
    from model import build_model
    paths = S4.write(str(tmp_path))
    assert len(paths) == 3
    for p in paths:
        cfg = PlanckConfig.from_dict(yaml.safe_load(open(p))["model"])
        n = sum(x.numel() for x in build_model(cfg, "meta").parameters())
        shape = os.path.basename(p)[6:-5]
        assert n == S4.REGISTERED[shape][2] and cfg.n_kv_heads == cfg.n_heads


def test_flops_grow_with_heads():
    f = {r["shape"]: r["flops_ctx2048"] for r in S4.table()}
    assert f["BASE"] < f["H4"] < f["H6"] < f["H7"]
