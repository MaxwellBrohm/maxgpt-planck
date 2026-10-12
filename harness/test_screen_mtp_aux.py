"""S006 t+2 aux head (train.mtp; mtp.py), model and step level. CPU; the CUDA probes skip without CUDA.
  targets    mtp_targets == a plain-loop reference (packed rows, a reused id, padding, masked targets, doc=None);
             explicit cases: t's document ends before t + 1 (t+2 across a document start) and tgt[t + 1] = -100.
  init       building the head draws nothing (torch RNG and every BASE parameter bitwise), W_mtp == I, gain ==
             ones, no nn.Linear; z is the final norm's output; at init the aux logits are the next-token logits
             times the aux norm's per-position rescale rsqrt(mean(z^2) + eps) (a copy up to that scale).
  leak       per-position aux losses from the production loss_sum: changing token j moves aux loss t only if j
             is t's target (t + 2) or an allowed source (same document, j <= t), bitwise, grad on and off, on
             doc=None, packed rows with padding and the varlen CPU stand-in; exact zero Jacobian elsewhere.
  step       gradients, optimizer groups, the Trainer loss path and mtp_weight 0: test_screen_mtp_aux_step.py.
  budget     budget.count_mtp == the built head; 5M: deployed 5,010,133 (unchanged), training-only 37,056.
"""
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn

import docattn
from budget import count_analytic, count_mtp
from config import PlanckConfig
from model import build_model
from mtp import MTPHead, mtp_targets
from test_s3_docattn import BF16, ROWS, cuda_only, ref_varlen
from test_s3_leak import DOC, T
from testutil import ARMS, allowed_matrix, scramble, tiny

B5 = dict(vocab_size=8192, d_model=192, n_layers=8, n_heads=3, n_kv_heads=3, head_dim=64,
          mlp_hidden=488, seq_len=2048)                       # SCREENS C1 BASE shape
ITEM = [0] * 17 + [9] * 3                                     # doc=None rows: one item, then 3 pad


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def run_ids(doc: list[int]) -> list[int]:
    r = [0]
    for t in range(1, len(doc)):
        r.append(r[-1] + (doc[t] != doc[t - 1]))
    return r


def loader_tgt(idx: torch.Tensor, doc: list[int], sup: list[bool]) -> torch.Tensor:
    """packing.item_targets, per row with plain loops: tgt[t] = idx[t + 1] if t + 1 is in t's run and
    sup[t + 1]; in DOC and ITEM the last run is padding (never supervised)."""
    r, n = run_ids(doc), len(doc)
    pad = r[-1] if doc in (DOC, ITEM) else -1
    return torch.tensor([[int(idx[0, t + 1]) if t + 1 < n and r[t + 1] == r[t] != pad and sup[t + 1] else -100
                          for t in range(n)]])


def ref_t2(idx: list[int], tgt: list[int], doc: list[int] | None) -> list[int]:
    n = len(idx)
    r = run_ids(doc) if doc is not None else [0] * n
    return [idx[t + 2] if t + 2 < n and tgt[t + 1] != -100 and r[t] == r[t + 1] == r[t + 2] else -100
            for t in range(n)]


def arm(name: str = "base_gqa", seed: int = 0):
    torch.manual_seed(seed)
    m = scramble(build_model(tiny(**ARMS[name])), seed)
    return m, scramble(MTPHead(m.cfg.d_model, m.cfg.rms_eps), seed + 1)


@pytest.mark.parametrize("sup_p", [1.0, 0.6])
def test_targets_match_plain_loop_reference(sup_p):
    g = gen(1)
    for rows in (ROWS, None):
        doc = torch.tensor(rows) if rows else None
        idx = torch.randint(0, 97, (len(ROWS), T), generator=g)
        for tgt in (torch.randint(0, 97, idx.shape, generator=g), None):
            if tgt is None:                               # loader-consistent targets
                tgt = torch.cat([loader_tgt(idx[i:i + 1], rows[i] if rows else ITEM,
                                            (torch.rand(T, generator=g) < sup_p).tolist()) for i in range(len(idx))])
            else:
                tgt[torch.rand(idx.shape, generator=g) > sup_p] = -100
            got = mtp_targets(idx, tgt, doc)
            for i in range(len(idx)):
                assert got[i].tolist() == ref_t2(idx[i].tolist(), tgt[i].tolist(), rows[i] if rows else None)


def test_targets_explicit_cases():
    idx = torch.arange(1, T + 1)[None]
    ids, sup = idx[0].tolist(), [True] * T
    sup[8] = False                                            # tgt[7] = -100 inside run [5, 10]
    tgt = loader_tgt(idx, DOC, sup)
    got = mtp_targets(idx, tgt, torch.tensor([DOC]))[0].tolist()
    assert tgt[0, 11] == ids[12] and got[10] == -100          # t = 10 ends its run; t + 1, t + 2 are the next's
    assert tgt[0, 6] == ids[7] and got[6] == -100             # t + 1's target is -100
    assert got[:4] == [ids[2], ids[3], -100, -100]            # runs [0, 3], [4], [5, 10], [11, 13], [14, 16]
    assert got[4:12] == [-100, ids[7], -100, ids[9], ids[10], -100, -100, ids[13]]
    assert got[12:] == [-100, -100, ids[16]] + [-100] * 5     # then 3 pad


@pytest.mark.parametrize("seed", [0, 5])
def test_init_isolation_and_identity(seed):
    torch.manual_seed(seed)
    base = build_model(tiny())
    torch.manual_seed(seed)
    m = build_model(tiny())
    rng = torch.get_rng_state()
    head = MTPHead(m.cfg.d_model, m.cfg.rms_eps)
    assert torch.equal(torch.get_rng_state(), rng)
    a, b = base.state_dict(), m.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    d = m.cfg.d_model
    assert torch.equal(head.weight, torch.eye(d)) and torch.equal(head.norm.weight, torch.ones(d))
    assert [n for n, _ in head.named_parameters()] == ["weight", "norm.weight"]
    assert not any(isinstance(x, nn.Linear) for x in head.modules())
    idx, doc = torch.randint(0, 97, (len(ROWS), T), generator=gen(2)), torch.tensor(ROWS)
    store = []
    h = m.norm.register_forward_hook(lambda mod, i, o: store.append(o))
    logits, _, z = m(idx, doc=doc, return_hidden=True)
    h.remove()
    assert torch.equal(z, store[0]) and torch.equal(logits, m.lm_head(z)) and torch.equal(m(idx, doc=doc)[0], logits)
    s = torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + m.cfg.rms_eps)   # the aux norm's rescale of z (eps)
    torch.testing.assert_close(head.logits(z, m.lm_head), logits * s, rtol=1e-5, atol=1e-6)


def aux_per_pos(m, head, idx, tgt, doc, grad, amp=None):
    with (torch.enable_grad() if grad else torch.no_grad()), (amp() if amp else nullcontext()):
        z = m(idx, doc=doc, return_hidden=True)[2]
        t2 = mtp_targets(idx, tgt, doc)
        out = []
        for t in range(T):
            one = torch.full_like(t2, -100)
            one[0, t] = t2[0, t]
            out.append(head.loss_sum(z, m.lm_head, one, int(t2[0, t] != -100)))
    return torch.stack(out).detach(), t2


def perturb_aux(m, head, doc_list, grad, dev="cpu", amp=None):
    V, g = m.cfg.vocab_size, gen(3)
    sup = (torch.rand(T, generator=g) < 0.8).tolist()
    tdoc = doc_list or ITEM
    idx = torch.randint(0, V, (1, T), generator=g)
    doc = None if doc_list is None else torch.tensor([doc_list], device=dev)
    A = allowed_matrix(doc_list, T)
    tgt = loader_tgt(idx, tdoc, sup)
    ref = ref_t2(idx[0].tolist(), tgt[0].tolist(), doc_list)
    base, t2 = aux_per_pos(m, head, idx.to(dev), tgt.to(dev), doc, grad, amp)
    assert t2[0].tolist() == ref and sum(x != -100 for x in ref) >= 5
    for j in range(T):
        idx2 = idx.clone()
        idx2[0, j] = (idx[0, j] + 1 + int(torch.randint(0, V - 1, (1,), generator=g))) % V
        out, _ = aux_per_pos(m, head, idx2.to(dev), loader_tgt(idx2, tdoc, sup).to(dev), doc, grad, amp)
        for t in range(T):
            if ref[t] == -100 or not (A[t, j] or j == t + 2):
                assert torch.equal(out[t], base[t]), f"aux loss {t} moved when token {j} changed"
            elif j in (t, t + 2):
                assert not torch.equal(out[t], base[t]), f"aux loss {t} ignores token {j}"


@pytest.mark.parametrize("name", ["base_gqa", "prelude_loop_coda"])
@pytest.mark.parametrize("packed", [False, True], ids=["doc_none", "packed"])
def test_leak_probes_aux(name, packed):
    m, head = arm(name)
    for grad in (True, False):
        perturb_aux(m, head, DOC if packed else None, grad)


def test_leak_probes_aux_varlen_cpu_reference_kernel(monkeypatch):
    monkeypatch.setattr(docattn, "_flash_varlen", ref_varlen)
    m, head = arm()
    m.doc_attn = "varlen"
    for grad in (True, False):
        perturb_aux(m, head, DOC, grad)


@pytest.mark.parametrize("packed", [False, True], ids=["doc_none", "packed"])
def test_jacobian_aux(packed):
    m, head = arm("loop_share_tie")
    doc_list = DOC if packed else None
    idx = torch.randint(0, 97, (1, T), generator=gen(4))
    tgt = loader_tgt(idx, doc_list or ITEM, [True] * T)
    doc = None if doc_list is None else torch.tensor([doc_list])
    store = []
    h = m.tok_emb.register_forward_hook(lambda mod, i, o: store.append(o))
    z = m(idx, doc=doc, return_hidden=True)[2]
    h.remove()
    t2, A = mtp_targets(idx, tgt, doc), allowed_matrix(doc_list, T)
    for t in (t for t in range(T) if t2[0, t] != -100):
        one = torch.full_like(t2, -100)
        one[0, t] = t2[0, t]
        (gr,) = torch.autograd.grad(head.loss_sum(z, m.lm_head, one, 1), store[0], retain_graph=True)
        mag = gr[0].abs().amax(dim=-1)
        assert all((mag[s] > 0) == bool(A[t, s]) for s in range(T)), (t, mag.tolist())


@pytest.mark.parametrize("d", [32, 64, 192])
def test_budget_counts_head(d):
    cfg = tiny(d_model=d)
    assert count_mtp(cfg, 1) == sum(p.numel() for p in MTPHead(d).parameters()) and count_mtp(cfg, 0) == 0


def test_budget_5m_deployed_and_training_only():
    c = PlanckConfig(**B5)
    assert count_analytic(c)["total"] == 5_010_133 and count_mtp(c, 1) == 37_056 == 192 * 192 + 192


@cuda_only
@pytest.mark.parametrize("impl", ["mask", "varlen"])
def test_cuda_leak_probes_aux(impl):
    m, head = arm()
    m, head = m.cuda(), head.cuda()
    m.doc_attn = impl
    perturb_aux(m, head, DOC, False, "cuda", BF16)
