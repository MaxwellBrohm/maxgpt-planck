"""attn_diag.py (SCREENS C6 SINK and GATE). CPU, tiny models.

The tool rebuilds every attention call from the module's real input with the arm's own transforms; its output
must equal the module's own output for every testutil arm (the screens' flags included) on scrambled weights, with
padded rows of several lengths, so the probabilities it reads are the model's. Closed forms: with W_q = 0 every
query attends uniformly, so SINK at query position t is 1 / (t + 1); with W_gate = 0 the gate is exactly 1.
"""
from __future__ import annotations

import math

import pytest
import torch

import attn_diag as D
import testutil as U
from blocks import Attention
from model import build_model


def rows(n_vocab=97, lens=(64, 40, 17, 64), seed=3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(8, n_vocab, (n,), generator=g).tolist() for n in lens]


@pytest.mark.parametrize("arm", sorted(U.ARMS))
def test_rebuilt_attention_is_the_modules_own(arm):
    m = U.scramble(build_model(U.tiny(**U.ARMS[arm])), seed=1)
    r = D.read(m, rows(), "cpu", pad_id=0, batch_tokens=128)
    assert r["layers"] == m.cfg.depth
    assert r["check_ok"] and r["check_max_rel_err"] < 1e-5, r["check_max_rel_err"]
    assert (r["gate_mean"] is not None) == m.cfg.attn_gate


@pytest.mark.parametrize("arm", ["base_gqa", "forget_gate", "smear_key", "canon_ac", "loop2_cyclic"])
def test_sink_is_a_probability_and_the_batch_does_not_matter(arm, monkeypatch):
    monkeypatch.setattr(D, "SINK_FROM", 8)
    m = U.scramble(build_model(U.tiny(**U.ARMS[arm])), seed=2)
    rs = rows(lens=(64, 30, 12, 50))
    a = D.read(m, rs, "cpu", batch_tokens=64)            # one row per forward
    b = D.read(m, rs, "cpu", batch_tokens=4096)          # all rows in one padded batch
    assert a["sink_queries"] == b["sink_queries"] == [sum(n - 8 for n in (64, 30, 12, 50))] * m.cfg.depth
    for la, lb in zip(a["sink"], b["sink"]):
        assert all(0.0 <= x <= 1.0 for x in la) and max(abs(x - y) for x, y in zip(la, lb)) < 1e-6


def test_sink_closed_form_with_uniform_attention(monkeypatch):
    monkeypatch.setattr(D, "SINK_FROM", 10)
    m = U.scramble(build_model(U.tiny(forget_gate=False)), seed=4)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, Attention):
                mod.q_proj.weight.zero_()                 # logits all 0: uniform over the causal prefix
    r = D.read(m, rows(lens=(40, 25)), "cpu")
    want = (sum(1 / (t + 1) for t in range(10, 40)) + sum(1 / (t + 1) for t in range(10, 25))) / 45
    assert all(abs(x - want) < 1e-6 for layer in r["sink"] for x in layer)


def test_gate_closed_form_and_percentiles():
    m = U.scramble(build_model(U.tiny()), seed=5)
    r = D.read(m, rows(), "cpu")
    assert all(lo <= mu <= hi for L in range(2) for lo, mu, hi in zip(r["gate_p10"][L], r["gate_mean"][L],
                                                                       r["gate_p90"][L]))
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, Attention):
                mod.attn_gate_proj.weight.zero_()
    r = D.read(m, rows(), "cpu")
    assert all(x == 1.0 for k in ("gate_mean", "gate_p10", "gate_p90") for layer in r[k] for x in layer)


def test_gate_only_reading_matches_the_full_reading():
    m = U.scramble(build_model(U.tiny()), seed=6)
    full = D.read(m, rows(), "cpu")
    lean = D.read(m, rows(), "cpu", sink=False, gate_values=False, check=False)
    assert lean["check_ok"] is None and "sink" not in lean
    assert all(abs(x - y) < 1e-6 for a, b in zip(full["gate_mean"], lean["gate_mean"]) for x, y in zip(a, b))


def test_hooks_removed_and_doc_path_refused():
    m = U.scramble(build_model(U.tiny()), seed=7)
    D.read(m, rows(), "cpu")
    assert not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.modules())
    with pytest.raises(ValueError, match="doc=None"):
        with D.Reader(m, torch.tensor([10])):
            m(torch.randint(8, 97, (1, 10)), doc=torch.zeros(1, 10, dtype=torch.long))


def test_sink_closed_form_with_a_fixed_forget_bias(monkeypatch):
    """q = 0 and a constant forget gate f = 0.9: p(i, j) is proportional to f^(i - j), so the first token gets
    f^i (1 - f) / (1 - f^(i + 1)) at query i (a key other than 0 gives another value)."""
    monkeypatch.setattr(D, "SINK_FROM", 3)
    m = U.scramble(build_model(U.tiny(forget_gate=True)), seed=8)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, Attention):
                mod.q_proj.weight.zero_()
                mod.forget_w.zero_()
                mod.forget_b.fill_(math.log(9.0))         # sigmoid(ln 9) = 0.9
    f = 0.9
    r = D.read(m, rows(lens=(30,)), "cpu")
    want = sum(f ** i * (1 - f) / (1 - f ** (i + 1)) for i in range(3, 30)) / 27
    assert all(abs(x - want) < 1e-5 for layer in r["sink"] for x in layer), (r["sink"], want)


def test_check_catches_a_module_that_computes_something_else(monkeypatch):
    import blocks
    m = U.scramble(build_model(U.tiny()), seed=9)
    assert D.read(m, rows(), "cpu")["check_ok"]
    real = blocks.F.scaled_dot_product_attention
    monkeypatch.setattr(blocks.F, "scaled_dot_product_attention",
                        lambda q, k, v, attn_mask=None, is_causal=False: real(q, k, v))   # not causal
    r = D.read(m, rows(), "cpu")
    assert not r["check_ok"] and r["check_max_rel_err"] > 1e-3
