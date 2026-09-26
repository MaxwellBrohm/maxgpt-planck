"""S007 smeared keys, model level (model.smear_key, experiments/S007_smeared_key/notes.txt; run-level tests: config,
flag off == absent and == the pre-flag commit, gradient and optimizer group, resume are in
test_screen_smear_key_run.py). CPU, tiny models; the CUDA cases skip without CUDA.
  isolation  every BASE parameter bitwise equal at a seed, alpha 0, torch RNG untouched, step-0 logits equal to
             BASE's on the same batch (doc=None, packed mask, varlen CPU stand-in).
  semantics  Attention == a plain-loop reference: raw key k_t + alpha_h * k_(t-1) inside the document, before
             QK-norm and RoPE, alpha per KV head; values (kv_tie: the key) unsmeared.
  leak       test_s3_leak's bitwise probes (exact zero gradient from other documents, causal inside one, padding
             its own document) on scrambled arms: doc=None, mask, varlen CPU stand-in, bf16 autocast.
  doc=None   model(idx) on one document == its slice of a packed row (mask and varlen; bpb.py's path).
  also       the startup self-test, budget.py counts (5M: 5,010,133 -> 5,010,157), KV decode.
"""
from __future__ import annotations

import math

import pytest
import torch

import docattn
import selftest
from budget import count_analytic
from config import PlanckConfig
from count_params import build_and_count
from decode import KVDecoder
from model import build_model, document_causal_mask, positions_from_doc
from test_s3_docattn import BF16, ROWS, cuda_leak_probes, cuda_only, fwd_bwd, make_batches, ref_varlen, rel
from test_s3_leak import DOC, T, jacobian_check, perturb_check
from testutil import allowed_matrix, scramble, tiny

ARMS = {"gqa": {"smear_key": True},                                   # 2 query heads share 1 KV head
        "mha": {"smear_key": True, "n_kv_heads": 2},
        "tie_loop_share": {"smear_key": True, "n_kv_heads": 2, "kv_tie": True, "qk_share": 2, "n_layers": 1,
                           "n_loops": 3, "n_prelude": 1, "n_coda": 1},
        "plain_attn_only": {"smear_key": True, "n_kv_heads": 2, "qk_norm": False, "attn_gate": False,
                            "value_residual": False, "norm_scaling": False, "mlp_hidden": 0}}
B5 = dict(vocab_size=8192, d_model=192, n_layers=8, n_heads=3, n_kv_heads=3, head_dim=64,
          mlp_hidden=488, seq_len=2048)                       # SCREENS C1 BASE shape


def off_kw(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if k != "smear_key"}


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def arm_model(arm: str, seed: int = 0):
    return scramble(build_model(tiny(**ARMS[arm])), seed)


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("arm", list(ARMS))
def test_init_isolation(arm, seed, monkeypatch):
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    torch.manual_seed(seed)
    base = build_model(tiny(**off_kw(ARMS[arm])))
    after_base = torch.rand(8)
    torch.manual_seed(seed)
    m = build_model(tiny(**ARMS[arm]))
    assert torch.equal(torch.rand(8), after_base)             # the flag drew nothing from the RNG
    pb, pm = dict(base.named_parameters()), dict(m.named_parameters())
    assert set(pm) - set(pb) == {f"blocks.{u}.attn.smear_alpha" for u in range(m.cfg.n_unique)}
    assert set(pb) <= set(pm) and all(torch.equal(p, pm[n]) for n, p in pb.items())
    alphas = [p for n, p in pm.items() if n not in pb]
    assert all(a.shape == (m.cfg.n_kv_heads,) and not a.any() for a in alphas)   # zero init, one per KV head
    idx = torch.randint(0, 97, (len(ROWS), T), generator=gen(1))
    with torch.no_grad():
        for impl, doc in (("mask", None), ("mask", torch.tensor(ROWS)), ("varlen", torch.tensor(ROWS))):
            base.doc_attn = m.doc_attn = impl
            assert torch.equal(base(idx, doc=doc)[0], m(idx, doc=doc)[0]), impl


def reference_attention(att, x, rows, cos_tab, sin_tab):
    """Attention by hand: raw key, the smear with plain loops, QK-norm, RoPE at in-document positions, the
    testutil.allowed_matrix mask, GQA (query head h reads KV head h // (H / Hkv)), gate, o_proj."""
    B, _, _ = x.shape
    H, Hk, hd = att.n_heads, att.n_kv_heads, att.head_dim
    q = (x @ att.q_proj.weight.T).view(B, T, H, hd)
    kr = (x @ att.k_proj.weight.T).view(B, T, Hk, hd)
    v = kr if att.kv_tie else (x @ att.v_proj.weight.T).view(B, T, Hk, hd)
    k, pos = kr.clone(), torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for t in range(1, T):
            if rows is None or rows[b][t] == rows[b][t - 1]:         # t continues its document
                pos[b, t] = pos[b, t - 1] + 1
                for h in range(Hk):
                    k[b, t, h] = kr[b, t, h] + att.smear_alpha[h] * kr[b, t - 1, h]
    if att.qk_norm:
        q, k = att.q_norm(q), att.k_norm(k)
    c, s = cos_tab[pos][:, :, None], sin_tab[pos][:, :, None]
    q, k = (z * c + torch.cat((-z[..., hd // 2:], z[..., :hd // 2]), -1) * s for z in (q, k))
    out = torch.zeros(B, T, H, hd)
    for b in range(B):
        A = torch.from_numpy(allowed_matrix(None if rows is None else rows[b], T))
        for h in range(H):
            g = h // (H // Hk)
            sc = (q[b, :, h] @ k[b, :, g].T) / math.sqrt(hd)
            out[b, :, h] = torch.softmax(sc.masked_fill(~A, float("-inf")), -1) @ v[b, :, g]
    if att.attn_gate:
        out = out * (2 * torch.sigmoid(x @ att.attn_gate_proj.weight.T))[..., None]
    return out.reshape(B, T, H * hd) @ att.o_proj.weight.T, v.transpose(1, 2)


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_attention_equals_plain_loop_reference(arm, packed):
    m = arm_model(arm)
    x = torch.randn(len(ROWS), T, m.cfg.d_model, generator=gen(2))
    doc = torch.tensor(ROWS)
    if packed:
        pos = positions_from_doc(doc)
        cos, sin = m.rope_cos[pos][:, None], m.rope_sin[pos][:, None]
        mask, starts = document_causal_mask(doc), docattn.doc_starts(doc)
    else:
        cos, sin, mask, starts = m.rope_cos[:T][None, None], m.rope_sin[:T][None, None], None, None
    with torch.no_grad():
        for blk in m.blocks:
            assert blk.attn.smear_alpha.abs().min() > 0.05          # scrambled: the smear is live
            out, v_local = blk.attn(x, cos, sin, None, mask, None, starts)
            ref, ref_v = reference_attention(blk.attn, x, ROWS if packed else None, m.rope_cos, m.rope_sin)
            torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(v_local, ref_v)             # values never smeared


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
    for grad in (True, False):
        perturb_check(m, DOC, grad)
    jacobian_check(m, DOC)


@pytest.mark.parametrize("arm", ["gqa", "tie_loop_share"])
def test_bf16_autocast_no_leak(arm):
    m = arm_model(arm)
    idx = torch.randint(0, 97, (1, T), generator=gen(7))
    A = allowed_matrix(DOC, T)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        base = m(idx, doc=torch.tensor([DOC]))[0]
        for j in range(T):
            idx2 = idx.clone()
            idx2[0, j] = (idx[0, j] + 1) % 97
            out = m(idx2, doc=torch.tensor([DOC]))[0]
            assert all(torch.equal(out[0, t], base[0, t]) for t in range(T) if not A[t, j]), j


@pytest.mark.parametrize("impl", ["mask", "varlen"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_doc_none_equals_packed_slice(arm, impl, monkeypatch):
    """The doc=None path (data_prep/bpb.py's model(idx)) smears too, and stops at the row start."""
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    m, zeroed = arm_model(arm), arm_model(arm)
    m.doc_attn = impl
    docs = [torch.randint(0, 97, (1, n), generator=gen(n)) for n in (7, 20, 13, 1, 4)]
    doc = torch.cat([torch.full((1, d.size(1)), i) for i, d in enumerate(docs)], dim=1)
    with torch.no_grad():
        for n, p in zeroed.named_parameters():
            if ".smear_" in n:
                p.zero_()
        packed, at = m(torch.cat(docs, dim=1), doc=doc)[0], 0
        for d in docs:
            alone = m(d)[0]
            assert torch.allclose(packed[:, at:at + d.size(1)], alone, atol=1e-5)
            if d.size(1) > 1:
                assert not torch.allclose(zeroed(d)[0], alone, atol=1e-3)   # alpha matters on this path
            at += d.size(1)


def test_startup_selftest_passes():
    torch.manual_seed(0)
    out = selftest.run_selftests(build_model(tiny(smear_key=True)), "cpu")
    assert len(out) == 12 and all(v < 1e-3 for v in out.values()), out


@pytest.mark.parametrize("arm", list(ARMS))
def test_budget_counts_smear(arm):
    cfg = tiny(**ARMS[arm])
    got = count_analytic(cfg)
    assert got == build_and_count(cfg, "cpu") == build_and_count(cfg, "meta")
    assert got["total"] - count_analytic(tiny(**off_kw(ARMS[arm])))["total"] == cfg.n_unique * cfg.n_kv_heads


def test_budget_5m_base_and_arm():
    assert count_analytic(PlanckConfig(**B5))["total"] == 5_010_133
    on = PlanckConfig(**B5, smear_key=True)
    assert count_analytic(on)["total"] == build_and_count(on, "meta")["total"] == 5_010_157
    assert 5_010_157 - 5_010_133 == 8 * 3 and 5_010_157 / 5_010_133 - 1 < 0.02


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
def test_cuda_leak_probes_smear(impl):
    m = arm_model("gqa").cuda()
    m.doc_attn = impl
    cuda_leak_probes(m, DOC)


@cuda_only
def test_cuda_varlen_parity_smear():
    torch.manual_seed(0)
    m = arm_model("mha").cuda()
    batches = make_batches(ROWS, 97, "cuda")
    l32, s32, g32 = fwd_bwd(m, "mask", batches)
    lm, sm, gm = fwd_bwd(m, "mask", batches, BF16)
    lv, sv, gv = fwd_bwd(m, "varlen", batches, BF16)
    assert rel(lv, l32) <= 1.5 * rel(lm, l32) + 1e-4
    assert abs(sv - s32) <= 1.5 * abs(sm - s32) + 1e-4 * abs(s32)
    for n in (n for n in g32 if ".smear_" in n):
        assert rel(gv[n], g32[n]) <= 1.5 * rel(gm[n], g32[n]) + 1e-2, n
