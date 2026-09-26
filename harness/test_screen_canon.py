"""S004 Canon layers, model level (model.canon, experiments/S004_canon/notes.txt; run-level tests: config,
flag off == absent and == the pre-flag commit, optimizer group, resume are in test_screen_canon_run.py).
CPU, tiny models; the CUDA cases skip without CUDA.
  isolation  every BASE parameter bitwise equal at a seed, kernels zero, torch RNG untouched, step-0 logits
             equal to BASE's (doc=None and packed).
  semantics  CanonConv == a plain-loop reference (tap j reads h[t - j], zero before the document start).
  leak       test_s3_leak's bitwise probes (exact zero gradient from other documents, causal inside one,
             padding its own document) on scrambled arms: doc=None, mask, varlen CPU stand-in.
  doc=None   model(idx) on one document == its slice of a packed row (mask and varlen; bpb.py's path).
  also       the startup self-test, budget.py counts (5M: 5,010,133 -> 5,022,421), KV decode.
"""
from __future__ import annotations

import pytest
import torch

import docattn
import selftest
from blocks import CanonConv
from budget import count_analytic
from config import PlanckConfig
from count_params import build_and_count
from decode import KVDecoder
from model import build_model, canon_skip_masks
from test_s3_docattn import BF16, ROWS, cuda_leak_probes, cuda_only, fwd_bwd, make_batches, ref_varlen, rel
from test_s3_leak import DOC, T, jacobian_check, perturb_check
from testutil import scramble, tiny

ARMS = {"ac": {"canon": "AC"}, "a_only": {"canon": "A"}, "c_only": {"canon": "C"},
        "k3_attn_only": {"canon": "AC", "canon_kernel": 3, "mlp_hidden": 0},
        "loop_share": {"canon": "AC", "n_layers": 1, "n_loops": 3, "n_prelude": 1, "n_coda": 1,
                       "qk_share": 2, "kv_tie": True}}
B5 = dict(vocab_size=8192, d_model=192, n_layers=8, n_heads=3, n_kv_heads=3, head_dim=64,
          mlp_hidden=488, seq_len=2048)                       # SCREENS C1 BASE shape


def off_kw(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if not k.startswith("canon")}


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def arm_model(arm: str, seed: int = 0):
    return scramble(build_model(tiny(**ARMS[arm])), seed)


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("arm", list(ARMS))
def test_init_isolation(arm, seed):
    torch.manual_seed(seed)
    base = build_model(tiny(**off_kw(ARMS[arm])))
    after_base = torch.rand(8)
    torch.manual_seed(seed)
    m = build_model(tiny(**ARMS[arm]))
    assert torch.equal(torch.rand(8), after_base)             # the flag drew nothing from the RNG
    pb, pm = dict(base.named_parameters()), dict(m.named_parameters())
    extra = set(pm) - set(pb)
    assert set(pb) <= set(pm) and extra and all(".canon_" in n for n in extra)
    assert all(torch.equal(p, pm[n]) for n, p in pb.items())
    assert all(not pm[n].any() for n in extra)                # zero init
    idx = torch.randint(0, 97, (2, T), generator=gen(1))
    with torch.no_grad():
        for doc in (None, torch.tensor([DOC, ROWS[1]])):
            assert torch.equal(base(idx, doc=doc)[0], m(idx, doc=doc)[0])


def reference_conv(h, w, rows):
    out = h.clone()
    for b in range(h.size(0)):
        for t in range(h.size(1)):
            for j in range(w.size(1)):
                s = t - j
                if s >= 0 and (rows is None or all(rows[b][r] == rows[b][t] for r in range(s, t))):
                    out[b, t] += w[:, j] * h[b, s]
    return out


@pytest.mark.parametrize("K", [2, 4, 5])
def test_conv_equals_plain_loop_reference(K):
    conv = CanonConv(6, K)
    h = torch.randn(len(ROWS), T, 6, generator=gen(2))
    with torch.no_grad():
        conv.weight.copy_(torch.randn(6, K, generator=gen(K)))
        w = conv.weight
        torch.testing.assert_close(conv(h, canon_skip_masks(torch.tensor(ROWS), K)), reference_conv(h, w, ROWS))
        torch.testing.assert_close(conv(h), reference_conv(h, w, None))
        torch.testing.assert_close(conv(h[:, :2]), reference_conv(h[:, :2], w, None))   # T < K


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_leak_probes(arm, packed):
    m = arm_model(arm)
    for grad in (True, False):
        perturb_check(m, DOC if packed else None, grad)
    jacobian_check(m, DOC if packed else None)


@pytest.mark.parametrize("arm", list(ARMS))
def test_leak_probes_varlen_cpu_reference_kernel(arm, monkeypatch):
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    m = arm_model(arm)
    m.doc_attn = "varlen"
    perturb_check(m, DOC, True)
    jacobian_check(m, DOC)


@pytest.mark.parametrize("impl", ["mask", "varlen"])
@pytest.mark.parametrize("arm", ["ac", "loop_share"])
def test_doc_none_equals_packed_slice(arm, impl, monkeypatch):
    """The doc=None path (data_prep/bpb.py's model(idx)) runs the convolutions too."""
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    m, zeroed = arm_model(arm), arm_model(arm)
    m.doc_attn = impl
    docs = [torch.randint(0, 97, (1, n), generator=gen(n)) for n in (7, 20, 13, 4)]
    doc = torch.cat([torch.full((1, d.size(1)), i) for i, d in enumerate(docs)], dim=1)
    with torch.no_grad():
        for n, p in zeroed.named_parameters():
            if ".canon_" in n:
                p.zero_()
        packed, at = m(torch.cat(docs, dim=1), doc=doc)[0], 0
        for d in docs:
            alone = m(d)[0]
            assert torch.allclose(packed[:, at:at + d.size(1)], alone, atol=1e-5)
            assert not torch.allclose(zeroed(d)[0], alone, atol=1e-3)   # the kernels matter here
            at += d.size(1)


def test_startup_selftest_passes():
    torch.manual_seed(0)
    out = selftest.run_selftests(build_model(tiny(canon="AC")), "cpu")
    assert len(out) == 12 and all(v < 1e-3 for v in out.values()), out


@pytest.mark.parametrize("arm", list(ARMS))
def test_budget_counts_canon(arm):
    cfg = tiny(**ARMS[arm])
    got = count_analytic(cfg)
    assert got == build_and_count(cfg, "cpu") == build_and_count(cfg, "meta")
    sites = ("A" in cfg.canon) + ("C" in cfg.canon and cfg.mlp_hidden > 0)
    extra = sites * cfg.d_model * cfg.canon_kernel * cfg.n_unique
    assert got["total"] - count_analytic(tiny(**off_kw(ARMS[arm])))["total"] == extra


def test_budget_5m_base_and_arm():
    assert count_analytic(PlanckConfig(**B5))["total"] == 5_010_133
    on = PlanckConfig(**B5, canon="AC")
    assert count_analytic(on)["total"] == build_and_count(on, "meta")["total"] == 5_022_421
    assert 5_022_421 - 5_010_133 == 2 * 8 * 192 * 4 and 5_022_421 / 5_010_133 - 1 < 0.02


@pytest.mark.parametrize("prefill", [1, 2, 5])
@pytest.mark.parametrize("arm", list(ARMS))
def test_decode_matches_full_forward(arm, prefill):
    m = arm_model(arm, 11)
    ids = torch.randint(0, 97, (20,), generator=gen(5)).tolist()
    with torch.no_grad():
        full = m(torch.tensor([ids]))[0][0]
    dec = KVDecoder(m)
    got = [dec.forward(ids[:prefill], "cpu")] + [dec.forward([t], "cpu") for t in ids[prefill:]]
    torch.testing.assert_close(torch.cat(got), full, atol=2e-5, rtol=1e-4)


# ------------------------------------------------------------------------------ CUDA (the PC)
@cuda_only
@pytest.mark.parametrize("impl", ["mask", "varlen"])
def test_cuda_leak_probes_canon(impl):
    m = arm_model("ac").cuda()
    m.doc_attn = impl
    cuda_leak_probes(m, DOC)


@cuda_only
def test_cuda_varlen_parity_canon():
    torch.manual_seed(0)
    m = arm_model("ac").cuda()
    batches = make_batches(ROWS, 97, "cuda")
    l32, s32, g32 = fwd_bwd(m, "mask", batches)
    lm, sm, gm = fwd_bwd(m, "mask", batches, BF16)
    lv, sv, gv = fwd_bwd(m, "varlen", batches, BF16)
    assert rel(lv, l32) <= 1.5 * rel(lm, l32) + 1e-4
    assert abs(sv - s32) <= 1.5 * abs(sm - s32) + 1e-4 * abs(s32)
    for n in (n for n in g32 if ".canon_" in n):
        assert rel(gv[n], g32[n]) <= 1.5 * rel(gm[n], g32[n]) + 1e-2, n
