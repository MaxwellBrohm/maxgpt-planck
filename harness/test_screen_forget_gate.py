"""S005 forget gate, model level (model.forget_gate, experiments/S005_forget_gate/notes.txt; the rest: config,
flag off == absent and == the pre-flag commit, optimizer group, resume, varlen refusal, self-test, budget and
decode are in test_screen_forget_gate_run.py). CPU, tiny models; the CUDA cases skip without CUDA.
  isolation  every BASE parameter bitwise equal at a seed, torch RNG untouched, w = 0 and b = 7.99.
  semantics  the bias == a float64 plain-loop reference (sum over l in (j, i] of log f[l], same document, -inf
             elsewhere); every f = 1 gives the gate-free model; f < 1 on one head moves it toward recent keys.
  gradient   w and b gradients through SDPA's float mask == an eager fp32 softmax reference, and non-zero.
  precision  bf16 autocast, a 2,048-token row at init: bias(2047, 0) == 2047 log sigmoid(7.99) within 1e-4.
  leak       test_s3_leak's bitwise probes on scrambled arms (doc=None and packed with padding), bf16 too.
  doc=None   model(idx) on one document == its slice of a packed row (bpb.py's path).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import test_s3_leak
from model import build_model, document_causal_mask, forget_segments
from test_s3_docattn import BF16, ROWS, cuda_leak_probes, cuda_only, fwd_bwd, make_batches, rel
from test_s3_leak import DOC, T, jacobian_check, perturb_check
from testutil import scramble, tiny

ARMS = {"fg": {"forget_gate": True},
        "mha_attn_only": {"forget_gate": True, "n_kv_heads": 2, "mlp_hidden": 0},
        "plain_block": {"forget_gate": True, "attn_gate": False, "qk_norm": False, "value_residual": False,
                        "norm_scaling": False},
        "loop_share": {"forget_gate": True, "n_layers": 1, "n_loops": 3, "n_prelude": 1, "n_coda": 1,
                       "qk_share": 2, "kv_tie": True}}
LS_799 = F.logsigmoid(torch.tensor(7.99, dtype=torch.float64)).item()   # log f at init, float64


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def arm_model(arm: str, seed: int = 0):
    return scramble(build_model(tiny(**ARMS[arm])), seed)


def gates(m):
    return [(n, p) for n, p in m.named_parameters() if ".forget_" in n]


@torch.no_grad()
def set_gate(m, b_per_head):
    """w = 0 and b = b_per_head in every layer (b 200: log f = 0 exactly in fp32, i.e. f = 1)."""
    for n, p in gates(m):
        p.copy_(torch.tensor(b_per_head, dtype=p.dtype) if n.endswith("forget_b") else torch.zeros_like(p))


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("arm", list(ARMS))
def test_init_isolation(arm, seed):
    torch.manual_seed(seed)
    base = build_model(tiny(**{k: v for k, v in ARMS[arm].items() if k != "forget_gate"}))
    after_base = torch.rand(8)
    torch.manual_seed(seed)
    m = build_model(tiny(**ARMS[arm]))
    assert torch.equal(torch.rand(8), after_base)             # the flag drew nothing from the RNG
    pb, pm = dict(base.named_parameters()), dict(m.named_parameters())
    extra = set(pm) - set(pb)
    assert set(pb) <= set(pm) and len(extra) == 2 * m.cfg.n_unique and all(".forget_" in n for n in extra)
    assert all(torch.equal(p, pm[n]) for n, p in pb.items())
    for n in extra:
        want = torch.zeros(2, 32) if n.endswith("forget_w") else torch.full((2,), 7.99)
        assert torch.equal(pm[n], want), n
    idx = torch.randint(0, 97, (1, T), generator=gen(1))
    with torch.no_grad():                                     # registered: NOT the BASE function at init
        assert not torch.equal(base(idx)[0], m(idx)[0])


def reference_bias(x, w, b, rows):
    """float64 plain loops: bias[i, j] = sum of log sigmoid(w . x_l + b) over l in (j, i], same document."""
    logf = F.logsigmoid(x.double() @ w.double().T + b.double())               # (B, T, H)
    B, n = x.shape[:2]
    out = torch.full((B, w.size(0), n, n), float("-inf"), dtype=torch.float64)
    for r in range(B):
        for i in range(n):
            for j in range(i + 1):
                if rows is None or all(rows[r][s] == rows[r][i] for s in range(j, i + 1)):
                    out[r, :, i, j] = logf[r, j + 1:i + 1].sum(0)
    return out


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_bias_equals_plain_loop_reference(packed):
    att = arm_model("fg").blocks[1].attn
    x = torch.randn(len(ROWS), T, 32, generator=gen(2))
    causal = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    allowed = document_causal_mask(torch.tensor(ROWS)) if packed else causal
    with torch.no_grad():
        got = att.forget_bias(x, forget_segments(allowed))
    ref = reference_bias(x, att.forget_w, att.forget_b, ROWS if packed else None)
    assert got.dtype == torch.float32 and torch.equal(torch.isinf(got), torch.isinf(ref))
    fin = torch.isfinite(ref)
    assert ref[fin].min() < -1.0                              # scrambled: f about 0.7, a real bias
    torch.testing.assert_close(got[fin].double(), ref[fin], atol=2e-5, rtol=1e-5)


class Eager:
    """Explicit-softmax stand-in for F.scaled_dot_product_attention (fp32); keeps the probabilities."""
    def __init__(self):
        self.probs = []

    def __call__(self, q, k, v, attn_mask=None, is_causal=False):
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
        if is_causal:
            s = s.masked_fill(~torch.ones(s.shape[-2:], dtype=torch.bool).tril(), float("-inf"))
        elif attn_mask is not None:
            s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask
        p = torch.softmax(s, -1)
        self.probs.append(p.detach())
        return (p @ v.float()).to(v.dtype)


def test_f_one_is_gate_free_and_f_below_one_moves_toward_recent_keys(monkeypatch):
    m = arm_model("fg", 3)
    off = build_model(tiny()).eval()
    off.load_state_dict({k: v for k, v in m.state_dict().items() if ".forget_" not in k})
    idx = torch.randint(0, 97, (1, T), generator=gen(4))
    set_gate(m, [200.0, 200.0])
    with torch.no_grad():
        for doc in (None, torch.tensor([DOC])):
            torch.testing.assert_close(m(idx, doc=doc)[0], off(idx, doc=doc)[0], atol=1e-5, rtol=1e-5)
    probs = {}
    for name, b in (("free", [200.0, 200.0]), ("gated", [0.0, 200.0])):      # head 0 at f = 1/2
        set_gate(m, b)
        eager = Eager()
        monkeypatch.setattr(F, "scaled_dot_product_attention", eager)
        with torch.no_grad():
            m(idx)
        monkeypatch.undo()
        probs[name] = eager.probs[0][0]                       # layer 0: same q and k in both runs
    pos = torch.arange(T, dtype=torch.float32)
    mean_pos = {k: (p * pos).sum(-1) for k, p in probs.items()}                 # (H, T) expected key index
    assert (mean_pos["gated"][0, 1:] > mean_pos["free"][0, 1:] + 1e-4).all()
    torch.testing.assert_close(probs["gated"][1], probs["free"][1])


@pytest.mark.parametrize("arm", ["fg", "loop_share"])
def test_gradients_match_eager_fp32_reference(arm, monkeypatch):
    m = arm_model(arm, 5)
    idx, tgt = (torch.randint(0, 97, (len(ROWS), T), generator=gen(s)) for s in (6, 7))
    doc = torch.tensor(ROWS)

    def grads():
        m.zero_grad(set_to_none=True)
        m(idx, tgt, doc)[1].backward()
        return {n: p.grad.clone() if p.grad is not None else None for n, p in gates(m)}
    sdpa = grads()
    monkeypatch.setattr(F, "scaled_dot_product_attention", Eager())
    eager = grads()
    assert len(sdpa) == 2 * m.cfg.n_unique
    for n, g in sdpa.items():
        assert g is not None and eager[n] is not None and g.abs().amax() > 1e-6, n
        torch.testing.assert_close(g, eager[n], atol=1e-6, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_precision_bf16_row_2048(device):
    torch.manual_seed(0)
    m = build_model(tiny(forget_gate=True, seq_len=2048)).to(device)
    att, seen = m.blocks[0].attn, {}
    hook = att.register_forward_pre_hook(lambda mod, args: seen.update(x=args[0], fg=args[5]))
    idx = torch.randint(0, 97, (1, 2048), generator=gen(8)).to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        m(idx)
        hook.remove()
        logf = att.forget_logf(seen["x"])
        bias = att.forget_bias(seen["x"], seen["fg"])
    assert logf.dtype == bias.dtype == torch.float32
    assert (logf.double() - LS_799).abs().max() < 1e-9
    assert (bias[0, :, 2047, 0].double() - 2047 * LS_799).abs().max() < 1e-4      # -0.69348


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_leak_probes(arm, packed):
    m = arm_model(arm)
    for grad in (True, False):
        perturb_check(m, DOC if packed else None, grad)
    jacobian_check(m, DOC if packed else None)


def test_bf16_autocast_no_leak():
    test_s3_leak.test_bf16_autocast_no_leak("forget_gate")      # testutil.ARMS["forget_gate"]


@pytest.mark.parametrize("arm", ["fg", "loop_share"])
def test_doc_none_equals_packed_slice(arm):
    """The doc=None path (data_prep/bpb.py's model(idx)) builds the bias too."""
    m, free = arm_model(arm), arm_model(arm)
    set_gate(free, [200.0, 200.0])
    docs = [torch.randint(0, 97, (1, n), generator=gen(n)) for n in (7, 20, 13, 4)]
    doc = torch.cat([torch.full((1, d.size(1)), i) for i, d in enumerate(docs)], dim=1)
    with torch.no_grad():
        packed, at = m(torch.cat(docs, dim=1), doc=doc)[0], 0
        for d in docs:
            alone = m(d)[0]
            torch.testing.assert_close(packed[:, at:at + d.size(1)], alone, atol=1e-5, rtol=1e-5)
            assert not torch.allclose(free(d)[0], alone, atol=1e-3)   # the gate matters here
            at += d.size(1)


# ------------------------------------------------------------------------------ CUDA (the PC)
@cuda_only
def test_cuda_leak_probes_forget_gate():
    cuda_leak_probes(arm_model("fg").cuda(), DOC)


@cuda_only
def test_cuda_bf16_gate_gradients_near_fp32():
    m = arm_model("fg").cuda()
    batches = make_batches(ROWS, 97, "cuda")
    _, _, g32 = fwd_bwd(m, "mask", batches)
    _, _, gbf = fwd_bwd(m, "mask", batches, BF16)
    for n in (n for n in g32 if ".forget_" in n):
        assert g32[n].abs().amax() > 0 and rel(gbf[n], g32[n]) < 5e-2, (n, rel(gbf[n], g32[n]))
