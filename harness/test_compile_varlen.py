"""train.compile with doc_attn varlen, on CPU: what Dynamo does with the harness side of varlen.

Varlen (docattn.py) is CUDA-only; here its kernel is swapped for a traceable CPU stand-in (a segment
mask from cu_seqlens via searchsorted, no .tolist()), so the capture of the real forward is what is
tested: with the default Dynamo config cu_seqlens' nonzero() (a data-dependent output size) is the only
graph break, and the graph count stops growing once the document count has varied (no cache-limit
fallback). The real kernel under inductor: test_speed3_cuda.py.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import docattn
from model import build_model
from test_s3_leak import DOC, T, perturb_check
from testutil import scramble, tiny


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()


def standin_varlen(q, k, v, cu, max_len, gqa):
    """A traceable CPU stand-in for the flash varlen kernel (no .tolist(): searchsorted over cu)."""
    N = q.size(0)
    seg = torch.searchsorted(cu, torch.arange(N, dtype=cu.dtype), right=True)
    mask = (seg[:, None] == seg[None, :]) & torch.ones(N, N, dtype=torch.bool).tril()
    rep = q.size(1) // k.size(1)
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    t = lambda x: x.transpose(0, 1)[None]  # noqa: E731
    return F.scaled_dot_product_attention(t(q), t(k), t(v), attn_mask=mask)[0].transpose(0, 1)


def test_compiled_varlen_breaks_only_at_cu_seqlens(monkeypatch):
    """Default Dynamo config: every graph break of the varlen forward is cu_seqlens' nonzero()
    (measured: 4 graphs, 3 breaks, nested frames), compiled == eager, the graph count stops
    growing after the first recompile, and the compiled varlen forward passes the leak probe."""
    monkeypatch.setattr(docattn, "_flash_varlen", standin_varlen)
    torch.manual_seed(0)
    m = scramble(build_model(tiny()), 1).train()
    m.doc_attn = "varlen"
    g = torch.Generator().manual_seed(0)
    idx, tgt = torch.randint(0, 97, (2, T), generator=g), torch.randint(0, 97, (2, T), generator=g)
    doc = torch.tensor([DOC] * 2)
    ex = torch._dynamo.explain(m)(idx, tgt, doc, reduction="sum")
    assert ex.break_reasons and 1 <= ex.graph_break_count <= 3     # explain lists 2 reasons for 3
    for br in ex.break_reasons:
        assert "nonzero" in br.reason and any(f.name == "cu_seqlens" for f in br.user_stack), br.reason
    assert all("nonzero" in k for k in torch._dynamo.utils.counters["graph_break"])
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    c = torch.compile(m, backend="aot_eager")
    seen = []
    for i in range(8):
        starts = torch.rand(2, T, generator=g) < (1 + i % 4) / T
        starts[:, 0] = True
        d = torch.cumsum(starts.long(), 1)
        out = []
        for fwd in (m, c):
            m.zero_grad(set_to_none=True)
            ls = fwd(idx, tgt, d, reduction="sum")[1]
            ls.backward()
            out.append((ls.detach(), [p.grad.clone() for p in m.parameters()]))
        (le, ge), (lc, gc) = out
        assert torch.allclose(le, lc, rtol=1e-6, atol=1e-5)
        assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(ge, gc))
        seen.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    assert seen[-1] == seen[3] and seen[-1] <= 8, seen   # stable (measured 4 -> 6), no cache-limit fallback
    m.eval()
    for grad in (True, False):
        perturb_check(c, DOC, grad)
