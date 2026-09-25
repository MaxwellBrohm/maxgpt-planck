"""Document masking, per-document positions, summed loss, and the leak self-test.
CPU only, models under 0.1M parameters."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import blocks
import model as M
import selftest
from config import PlanckConfig
from model import build_model, document_causal_mask, positions_from_doc


def tiny(**kw) -> PlanckConfig:
    base = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
                mlp_hidden=64, seq_len=64)
    base.update(kw)
    return PlanckConfig(**base)


ARMS = [{}, {"n_loops": 2}, {"n_loops": 2, "loop_order": "immediate", "n_prelude": 1, "n_coda": 1},
        {"qk_share": 2, "kv_tie": True}, {"mlp_hidden": 0}, {"value_residual": False},
        {"attn_gate": False, "qk_norm": False, "norm_scaling": False}]


def test_positions_from_doc():
    doc = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2], [5, 5, 5, 5, 5, 5, 5, 5, 5]])
    assert positions_from_doc(doc).tolist() == [[0, 1, 2, 0, 1, 0, 1, 2, 3], list(range(9))]


def test_mask_shape_and_content():
    doc = torch.tensor([[0, 0, 1, 1, 0]])        # the second run of id 0 is a NEW document
    m = document_causal_mask(doc)[0, 0]
    assert m.shape == (5, 5)
    exp = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(m, exp)


@pytest.mark.parametrize("kw", ARMS)
def test_packed_equals_alone(kw):
    """Three documents packed in one row give the same logits as each run alone."""
    torch.manual_seed(0)
    m = build_model(tiny(**kw)).eval()
    lens = [7, 20, 13]
    docs = [torch.randint(0, 97, (1, n)) for n in lens]
    row = torch.cat(docs, dim=1)
    doc = torch.cat([torch.full((1, n), i) for i, n in enumerate(lens)], dim=1)
    with torch.no_grad():
        packed, _ = m(row, doc=doc)
        at = 0
        for d in docs:
            alone, _ = m(d)
            assert torch.allclose(packed[:, at:at + d.size(1)], alone, atol=1e-5), kw
            at += d.size(1)


def test_doc_none_is_plain_causal():
    torch.manual_seed(0)
    m = build_model(tiny()).eval()
    x = torch.randint(0, 97, (2, 30))
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x, doc=torch.zeros(2, 30, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-5)


def test_sum_reduction_matches_mean_times_count():
    torch.manual_seed(0)
    m = build_model(tiny())
    x = torch.randint(0, 97, (2, 16))
    t = torch.randint(0, 97, (2, 16))
    t[0, :5] = -100
    _, mean = m(x, t)
    _, s = m(x, t, reduction="sum")
    assert torch.allclose(s, mean * (t != -100).sum(), rtol=1e-5)
    _, zero = m(x, torch.full_like(t, -100))
    assert zero.detach().item() == 0.0


@pytest.mark.parametrize("kw", ARMS)
def test_selftest_passes_clean_model(kw):
    torch.manual_seed(0)
    out = selftest.run_selftests(build_model(tiny(**kw)), "cpu")
    assert all(v < 1e-3 for v in out.values()) and len(out) == 12     # 2 variants x 3 checks x 2


def test_selftest_catches_document_leak(monkeypatch):
    m = build_model(tiny())
    monkeypatch.setattr(M, "document_causal_mask",
                        lambda doc: torch.ones(doc.size(0), 1, doc.size(1), doc.size(1),
                                               dtype=torch.bool).tril())
    with pytest.raises(AssertionError, match="leak"):
        selftest.run_selftests(m, "cpu")


def test_selftest_catches_causal_leak(monkeypatch):
    m = build_model(tiny())
    real = F.scaled_dot_product_attention

    def leaky(q, k, v, attn_mask=None, is_causal=False):
        return real(q, k, v)                      # ignores causality and the mask
    monkeypatch.setattr(blocks.F, "scaled_dot_product_attention", leaky)
    with pytest.raises(AssertionError, match="causal"):
        selftest.run_selftests(m, "cpu", packed=False)


def test_selftest_catches_nograd_only_leak(monkeypatch):
    """The MPS failure mode: the leak appears only under torch.no_grad."""
    m = build_model(tiny())
    real = F.scaled_dot_product_attention

    def leaky(q, k, v, attn_mask=None, is_causal=False):
        if torch.is_grad_enabled():
            return real(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return real(q, k, v)
    monkeypatch.setattr(blocks.F, "scaled_dot_product_attention", leaky)
    with pytest.raises(AssertionError, match="nograd"):
        selftest.run_selftests(m, "cpu", packed=False)
