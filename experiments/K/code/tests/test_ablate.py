"""K2/K3 tools: no-op is bit-identical, each ablation equals its weight-space twin, pruning and reset keep the
model trainable and equal zero ablation, a planted fact unit is found by the scan (masks: test_masks.py), tracing hits
its two exact endpoints, random sets are layer- and stratum-matched."""
import numpy as np
import torch
import torch.nn.functional as F

import ablate as A
import masks as M
from kfix import tiny_model

IDX = torch.tensor([[3, 17, 42, 5, 9, 61, 7, 2], [8, 8, 30, 31, 32, 4, 70, 1]])


def logits(m, idx=IDX):
    with torch.no_grad():
        return m(idx)[0]


def test_empty_spec_is_bit_identical():
    m = tiny_model()
    ref = logits(m)
    with A.Ablation(m):
        assert torch.equal(logits(m), ref)
    with A.Ablation(m, mlp={0: []}, heads={1: []}) as ab:
        assert ab.handles == [] and torch.equal(logits(m), ref)
    assert torch.equal(logits(m), ref)


def test_unit_head_and_sublayer_ablations_match_weight_edits():
    m = tiny_model()
    cols = torch.tensor([3, 7])
    for spec, edit in (({"mlp": {1: [3, 7]}}, lambda w: w.blocks[1].mlp.down_proj.weight.index_fill_(1, cols, 0.0)),
                       ({"heads": {0: [1]}}, lambda w: w.blocks[0].attn.o_proj.weight[:, 16:32].zero_()),
                       ({"sublayers": [("mlp", 0)]}, lambda w: w.blocks[0].mlp.down_proj.weight.zero_()),
                       ({"sublayers": [("attn", 1)]}, lambda w: w.blocks[1].attn.o_proj.weight.zero_())):
        with A.Ablation(m, **spec):
            got = logits(m)
        m2 = tiny_model()
        with torch.no_grad():
            edit(m2)
        assert torch.allclose(got, logits(m2), atol=1e-6, rtol=0), spec
        assert not torch.allclose(got, logits(m), atol=1e-4)


def test_mean_mode_and_collected_means():
    m = tiny_model()
    zeros = {"mlp": {1: torch.zeros(24)}, "heads": {0: torch.zeros(32)}}
    with A.Ablation(m, mlp={1: [2]}, heads={0: [0]}, mode="mean", means=zeros):
        a = logits(m)
    with A.Ablation(m, mlp={1: [2]}, heads={0: [0]}):
        assert torch.equal(a, logits(m))
    prompts = [[3, 4, 5], [6, 7, 8, 9, 10]]
    means = A.collect_means(m, prompts, batch=2)
    got = []
    h = m.blocks[1].mlp.down_proj.register_forward_pre_hook(lambda mod, a: got.append(a[0][0]))
    for p in prompts:
        logits(m, torch.tensor([p]))
    h.remove()
    assert torch.allclose(means["mlp"][1], torch.cat(got).mean(0), atol=1e-6)


def test_prune_equals_zero_ablation_and_trains():
    m = tiny_model()
    with A.Ablation(m, mlp={0: [1, 5], 1: [0]}):
        ref = logits(m)
    n0 = A.n_params(m)
    A.prune_mlp(m, {0: [1, 5], 1: [0]})
    assert A.n_params(m) == n0 - 3 * 32 * 3
    assert torch.allclose(logits(m), ref, atol=1e-5)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    before = m.blocks[0].mlp.up_proj.weight.detach().clone()
    loss = F.cross_entropy(m(IDX)[0][:, :-1].reshape(-1, 97), IDX[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, m.blocks[0].mlp.up_proj.weight)


def test_reset_equals_zero_ablation_at_step0():
    m = tiny_model()
    units = {0: [2, 9], 1: [4]}
    with A.Ablation(m, mlp=units):
        ref = logits(m)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    A.reset_units(m, units, seed=7701)
    assert torch.equal(logits(m), ref)
    after = m.state_dict()
    for k in before:
        diff = (before[k] != after[k]).nonzero()
        if "gate_proj" in k or "up_proj" in k:
            layer = int(k.split(".")[1])
            assert set(diff[:, 0].tolist()) <= set(units.get(layer, []))
        elif "down_proj" in k:
            layer = int(k.split(".")[1])
            assert set(diff[:, 1].tolist()) <= set(units.get(layer, []))
            assert (after[k][:, units[layer]] == 0).all()
        else:
            assert len(diff) == 0, k
    m2 = tiny_model()
    A.reset_units(m2, units, seed=7701)
    assert torch.equal(m2.blocks[0].mlp.gate_proj.weight, m.blocks[0].mlp.gate_proj.weight)
    big = tiny_model(scramble=False, mlp_hidden=400)
    A.reset_units(big, {0: list(range(400))}, seed=7702)
    assert abs(float(big.blocks[0].mlp.up_proj.weight.detach().std()) - big.cfg.init_std) < 0.1 * big.cfg.init_std


FACT, SKILL = ([10, 11, 12], 50), ([20, 21, 22], 60)


def planted():
    """Unit (1, 7) reads token 12 and writes token 50 (the fact); unit (0, 3) reads 22 and writes 60 (the skill)."""
    m = tiny_model(scramble=False)
    E = m.tok_emb.weight
    with torch.no_grad():
        E[50] *= 5
        E[60] *= 5
        for (layer, j), (src, dst) in (((1, 7), (12, 50)), ((0, 3), (22, 60))):
            mlp = m.blocks[layer].mlp
            u = E[src] / E[src].norm()
            mlp.gate_proj.weight[j] = 2.0 * u
            mlp.up_proj.weight[j] = 2.0 * u
            mlp.down_proj.weight[:, j] = 1.5 * E[dst] / E[dst].norm()
    return m


def ce(m, item):
    p, a = item
    return F.cross_entropy(m(torch.tensor([p]))[0][:, -1], torch.tensor([a]))


def test_planted_fact_unit_is_found():
    m = planted()
    with torch.no_grad():
        eff = A.unit_effects(m, lambda mm: [float(ce(mm, FACT)), float(ce(mm, SKILL))])
    top_f = int(np.argmax(eff[:, 0]))
    assert top_f == 1 * 24 + 7 and int(np.argmax(eff[:, 1])) == 0 * 24 + 3
    assert eff[top_f, 0] > 1.0 and abs(eff[top_f, 1]) < 0.1 * eff[top_f, 0]
    assert np.sort(eff[:, 0])[-2] < 0.2 * eff[top_f, 0]        # the plant stands out
    fit = M.fit(m, lambda mm: ce(mm, FACT), lambda mm: ce(mm, SKILL), "SF", budget=1, steps=5)
    assert set(fit) >= {"units", "history", "ce0", "p_drop"} and all(p.requires_grad for p in m.parameters())
    # Masks are tested on planted sets in tests/test_masks.py (review R-2). This plant's units are the whole block
    # output (norms 87 and 182), so CE moves only once a gate is under 0.01 and a HardConcrete gate gets almost no
    # gradient: at the build defaults SF and SS found them on P(closed) margins of 0.00-0.03 (SS seed 0: 0.014
    # against 0.014); at the current defaults SF finds (1, 7) on seeds 0 and 1 (0.56, 0.45) and SS misses seed 1.
    # The single-unit scan above finds both.


def test_trace_endpoints():
    m = tiny_model()
    clean, corrupt = [5, 6, 7, 8], [9, 6, 7, 8]
    sc = A.trace(m, clean, corrupt, 20, 21)
    assert sc.shape == (3, 4)
    assert abs(sc[-1, -1] - 1) < 1e-5 and abs(sc[0, 0] - 1) < 1e-5
    assert abs(sc[0, 1]) < 1e-5 and abs(sc[0, 3]) < 1e-5


def test_random_sets_are_matched():
    chosen = {0: [1, 10, 20], 1: [10]}
    r = M.random_matched(chosen, 24, seed=7611)
    assert {k: len(v) for k, v in r.items()} == {0: 3, 1: 1}
    assert not set(r[0]) & set(chosen[0]) and not set(r[1]) & set(chosen[1])
    eff = {0: np.arange(24.0), 1: np.arange(24.0)}
    rq = M.random_matched(chosen, 24, seed=7612, effect=eff)
    q = lambda j: int(np.searchsorted(np.quantile(np.arange(24.0), [.2, .4, .6, .8]), j, side="left"))  # noqa: E731
    assert sorted(q(j) for j in rq[0]) == sorted(q(j) for j in chosen[0])
    assert M.random_matched(chosen, 24, seed=7611) == r
