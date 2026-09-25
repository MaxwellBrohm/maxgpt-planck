"""(a) Causal-leak tests for every attention and looping mode (testutil.ARMS), CPU, scrambled
tiny models. Three independent probes, each checked against testutil.allowed_matrix (plain
loops, not the model's own mask code):
  perturb   change ONE token at a time; logits that may not depend on it must be BITWISE
            unchanged (grad on and under no_grad); the logit at the token itself must move.
  jacobian  d logits[t] / d embedding[s] must be exactly 0 where not allowed and nonzero
            everywhere allowed (so a model that ignores its context also fails).
  batch     changing one row never changes another row.
Each probe runs on the plain causal path (doc=None: bucket mode, inference) and on packed
rows with document ids (several documents, a one-token document, a reused id, padding).
Exactness: on CPU fp32 a masked score is exp(-inf) = 0 exactly, so no tolerance is used.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model import build_model
from testutil import ARMS, allowed_matrix, scramble, tiny

T = 20
# 5 documents: lengths 4, 1, 6, 3, then 3 pad (own doc); id 0 is reused for the 4th run,
# which must still be a NEW document (runs of equal ids, not id equality).
DOC = [0] * 4 + [1] + [2] * 6 + [0] * 3 + [3] * 3 + [9] * 3
assert len(DOC) == T


def build(arm: str, seed: int = 0):
    torch.manual_seed(seed)
    return scramble(build_model(tiny(**ARMS[arm])), seed)


def logits(m, idx, doc=None, grad=True):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out, _ = m(idx, doc=doc)
    return out.detach()


def perturb_check(m, doc_list, grad: bool) -> None:
    V = m.cfg.vocab_size
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, V, (1, T), generator=g)
    doc = None if doc_list is None else torch.tensor([doc_list])
    A = allowed_matrix(doc_list, T)
    base = logits(m, idx, doc, grad)[0]
    for j in range(T):
        idx2 = idx.clone()
        idx2[0, j] = (idx[0, j] + 1 + int(torch.randint(0, V - 1, (1,), generator=g))) % V
        out = logits(m, idx2, doc, grad)[0]
        for t in range(T):
            same = torch.equal(out[t], base[t])
            if not A[t, j]:
                assert same, f"position {t} changed when token {j} changed (doc={doc_list})"
        assert not torch.equal(out[j], base[j]), f"logit {j} ignores its own token"


def jacobian_check(m, doc_list) -> None:
    V = m.cfg.vocab_size
    idx = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(4))
    doc = None if doc_list is None else torch.tensor([doc_list])
    store = []
    h = m.tok_emb.register_forward_hook(lambda mod, i, o: store.append(o))
    try:
        out, _ = m(idx, doc=doc)
    finally:
        h.remove()
    emb = store[0]
    r = torch.randn(out.size(-1), generator=torch.Generator().manual_seed(5))
    A = allowed_matrix(doc_list, T)
    for t in range(T):
        (gr,) = torch.autograd.grad((out[0, t] * r).sum(), emb, retain_graph=True)
        mag = gr[0].abs().amax(dim=-1)                          # (T,) per source position
        for s in range(T):
            if A[t, s]:
                assert mag[s] > 0, f"logit {t} has no gradient from allowed token {s}"
            else:
                assert mag[s] == 0, f"logit {t} has gradient {mag[s]:.3g} from token {s}"


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_perturb_one_token(arm, packed):
    m = build(arm)
    for grad in (True, False):
        perturb_check(m, DOC if packed else None, grad)


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_jacobian_structure(arm, packed):
    jacobian_check(build(arm), DOC if packed else None)


@pytest.mark.parametrize("arm", ["base_gqa", "loop_share_tie", "prelude_loop_coda"])
def test_rows_are_independent(arm):
    m = build(arm)
    idx = torch.randint(0, 97, (3, T), generator=torch.Generator().manual_seed(6))
    doc = torch.tensor([DOC] * 3)
    for d in (None, doc):
        a = logits(m, idx, d)
        idx2 = idx.clone()
        idx2[1] = (idx2[1] + 1) % 97
        b = logits(m, idx2, d)
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) and not torch.equal(a[1], b[1])


def test_allowed_matrix_reference():
    """The reference itself: runs, reuse of an id, and plain causal."""
    A = allowed_matrix([5, 5, 7, 5], 4)
    assert A.tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert np.array_equal(allowed_matrix(None, 3), np.tril(np.ones((3, 3), dtype=bool)))


@pytest.mark.parametrize("arm", ["base_gqa", "loop3_immediate", "kv_tie"])
def test_bf16_autocast_no_leak(arm):
    """The MPS/CUDA precision path on CPU: masked entries stay exact zeros under bf16."""
    m = build(arm)
    idx = torch.randint(0, 97, (1, T), generator=torch.Generator().manual_seed(7))
    doc = torch.tensor([DOC])
    A = allowed_matrix(DOC, T)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        base, _ = m(idx, doc=doc)
        for j in (0, 5, 11, 14):
            idx2 = idx.clone()
            idx2[0, j] = (idx[0, j] + 1) % 97
            out, _ = m(idx2, doc=doc)
            for t in range(T):
                if not A[t, j]:
                    assert torch.equal(out[0, t], base[0, t]), (arm, t, j)
