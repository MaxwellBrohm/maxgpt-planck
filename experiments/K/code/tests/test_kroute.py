"""kroute.py (K3-R gradient routing; SPEC 14 train.grad_route): the routed loader yields the harness loader's batches
bit for bit plus a source tag that matches each item's content, through a resume; routed gradients equal an
independent masked-autograd reference, block by block, micro-batch by micro-batch; the forward is unchanged; a
routed ktrain run starts from the same weights and the same first loss as the plain run. CPU, tiny models."""
import collections
import copy
import os

os.environ.setdefault("PLANCK_TEST_THREADS", os.environ.get("K_TEST_THREADS", "4"))

import numpy as np                                     # noqa: E402
import pytest                                          # noqa: E402
import torch                                           # noqa: E402
import torch.nn.functional as F                        # noqa: E402

import kroute                                          # noqa: E402
import ktrain                                          # noqa: E402
import runio                                           # noqa: E402
from config import PlanckConfig                        # noqa: E402
from data import build_loader                          # noqa: E402
from make_fake_data import make                        # noqa: E402
from model import build_model                          # noqa: E402
from testutil import read_jsonl, write_run             # noqa: E402

CFG = dict(vocab_size=256, d_model=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16, mlp_hidden=24, seq_len=64)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("kroute")
    make(str(d / "fake"), vocab=256, docs=200, chats=150, doc_len=(5, 60))
    return str(d)


def dcfg(d):
    return {"mode": "pack", "pad_id": 0, "window_tokens": 512,
            "chat": {"role_ids": {"system": 2, "user": 3, "assistant": 4, "tool": 6}, "end_id": 5},
            "sources": [{"name": "text", "kind": "tokens", "paths": [f"{d}/fake/text_*.bin"], "eot_id": 1, "share": 0.6},
                        {"name": "chat", "kind": "chat", "paths": [f"{d}/fake/chat_*.jsonl"], "share": 0.4}]}


def loaders(d, seed=3):
    return build_loader(dcfg(d), 64, 2, seed=seed), kroute.as_routed(build_loader(dcfg(d), 64, 2, seed=seed))


def test_loader_same_batches_and_tags_match_content_through_a_resume(data):
    plain, routed = loaders(data)
    n_chat = n_text = 0
    for i in range(40):
        a, b = plain.next_batch(), routed.next_batch()
        for k in ("idx", "tgt", "doc", "pos"):
            assert torch.equal(a[k], b[k]), (i, k)
        for row in range(b["idx"].shape[0]):
            idx, doc, src = b["idx"][row], b["doc"][row], b["src"][row]
            for dd in doc.unique().tolist():
                at = (doc == dd).nonzero().flatten()
                s = set(src[at].tolist())
                assert len(s) == 1, (i, row, dd)
                (s,) = s
                if s == -1:                                  # padding: pad ids only
                    assert bool((idx[at] == 0).all())
                    continue
                first = int(idx[at[0]])
                assert (s == 1) == (first in (2, 3)), (i, row, dd, s, first)   # chats open with a role id
                n_chat, n_text = n_chat + (s == 1), n_text + (s == 0)
        if i == 23:
            st = routed.state_dict()
    assert n_chat > 20 and n_text > 20
    assert list(routed.k_queue) and len(routed.k_queue) == 40
    _, again = loaders(data)
    again.load_state_dict(st)
    _, ref = loaders(data)
    for _ in range(24):
        ref.next_batch()
    for i in range(12):
        x, y = ref.next_batch(), again.next_batch()
        for k in ("idx", "tgt", "doc", "pos", "src"):
            assert torch.equal(x[k], y[k]), (i, k)


def ref_grads(model, batches, reserved_n):
    """Independent reference: SwiGLU out = linear(h * A) + linear(h * (1 - A)).detach(), A[t, j] = fact[t] == res[j]."""
    m = copy.deepcopy(model)
    H = m.blocks[0].mlp.gate_proj.out_features
    res = torch.zeros(H, dtype=torch.bool)
    res[H - reserved_n:] = True
    state = {}

    def fwd(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        A = (state["fact"][..., None] == res).to(h.dtype)
        return F.linear(h * A, self.down_proj.weight) + F.linear(h * (1 - A), self.down_proj.weight).detach()
    for b in m.blocks:
        b.mlp.forward = fwd.__get__(b.mlp)
    for b in batches:
        state["fact"] = b["src"] == 1
        _, loss = m(b["idx"], b["tgt"], b["doc"], b["pos"], reduction="sum")
        loss.backward()
    return {n: p.grad.clone() for n, p in m.named_parameters()}


def test_routed_gradients_equal_the_reference_per_micro_batch(data):
    torch.manual_seed(0)
    model = build_model(PlanckConfig(**CFG))
    _, ld = loaders(data, seed=5)
    b1, b2 = ld.next_batch(), ld.next_batch()
    f1, f2 = b1["src"] == 1, b2["src"] == 1
    assert f1.any() and (~f1 & (b1["src"] >= 0)).any() and not torch.equal(f1, f2)
    plain = copy.deepcopy(model)
    ctx = kroute.RouteCtx()
    units = kroute.install(model, 8, ctx)
    assert units == {0: list(range(16, 24)), 1: list(range(16, 24))}
    rf = kroute.RoutedForward(model, ld, ctx, [1])
    losses = []
    for b in (b1, b2):
        _, loss = rf(b["idx"], b["tgt"], b["doc"], b["pos"], reduction="sum")
        losses.append(loss)
        loss.backward()
    assert ctx.mask is None and not ld.k_queue
    for b, loss in zip((b1, b2), losses):                     # the forward is the plain model's, bit for bit
        assert torch.equal(loss, plain(b["idx"], b["tgt"], b["doc"], b["pos"], reduction="sum")[1])
    ref = ref_grads(plain, (b1, b2), 8)
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, ref[n], rtol=1e-5, atol=1e-6), n


def test_fact_only_and_nonfact_only_positions_touch_their_own_block(data):
    torch.manual_seed(1)
    model = build_model(PlanckConfig(**CFG))
    ctx = kroute.RouteCtx()
    kroute.install(model, 8, ctx)
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(8, 256, (2, 64), generator=g)
    tgt = torch.randint(8, 256, (2, 64), generator=g)
    for fact in (True, False):
        model.zero_grad(set_to_none=True)
        ctx.mask = torch.full((2, 64), fact)
        model(idx, tgt, reduction="sum")[1].backward()
        ctx.mask = None
        for blk in model.blocks:
            mlp = blk.mlp
            on, off = (slice(16, 24), slice(0, 16)) if fact else (slice(0, 16), slice(16, 24))
            for w in (mlp.gate_proj.weight, mlp.up_proj.weight):
                assert torch.count_nonzero(w.grad[off]) == 0 and torch.count_nonzero(w.grad[on]) > 0
            assert torch.count_nonzero(mlp.down_proj.weight.grad[:, off]) == 0
            assert torch.count_nonzero(mlp.down_proj.weight.grad[:, on]) > 0
        for blk in model.blocks:                             # attention learns from every position
            assert torch.count_nonzero(blk.attn.q_proj.weight.grad) > 0
    with pytest.raises(ValueError):
        kroute.reserved_units(24, 24)


def test_routed_run_starts_like_the_plain_run(tmp_path):
    make(str(tmp_path / "data"), vocab=256, docs=120, chats=80)
    src = [{"name": "text", "kind": "tokens", "paths": [f"{tmp_path}/data/text_*.bin"], "eot_id": 1, "share": 0.7},
           {"name": "chat", "kind": "chat", "paths": [f"{tmp_path}/data/chat_*.jsonl"], "share": 0.3}]
    outs = {}
    for name, k in (("plain", None), ("route", {"grad_route": {"sources": ["chat"], "reserved": 16}})):
        p = write_run(str(tmp_path / name), str(tmp_path / "data"), data={"sources": src},
                      train={"total_steps": 8, "ckpt_every": 0}, **({"k": k} if k else {}))
        assert ktrain.main([p, "--device", "cpu", "--max-steps", "0"]) == 0
        outs[name] = os.path.join(os.path.dirname(p), "out")
        ck0 = runio.load_checkpoint(runio.latest_checkpoint(outs[name]))
        outs[name + "0"] = ck0["model"]
        assert ktrain.main([p, "--device", "cpu"]) == 0
    assert all(torch.equal(outs["plain0"][k], outs["route0"][k]) for k in outs["plain0"])   # init: no RNG drawn
    a, b = (read_jsonl(os.path.join(outs[n], "log.jsonl")) for n in ("plain", "route"))
    assert a[0]["step"] == b[0]["step"] == 1 and a[0]["loss"] == b[0]["loss"]               # forward unchanged
    assert [r["loss"] for r in a[2:]] != [r["loss"] for r in b[2:]]                          # gradients are not
    st = read_jsonl(os.path.join(outs["route"], "k_start.jsonl"))
    assert st[-1]["installed"]["grad_route"] == {"reserved": {"0": [48, 63], "1": [48, 63]}, "fact_sources": [1]}
    assert np.isfinite(b[-1]["loss"])


def test_a_misaligned_shadow_packing_is_refused(data, monkeypatch):
    real = kroute.pack_rows
    monkeypatch.setattr(kroute, "pack_rows", lambda items, L, pad, rng: real(items[::-1], L, pad, rng))
    _, ld = loaders(data)
    with pytest.raises(RuntimeError, match="align"):
        for _ in range(10):
            ld.next_batch()


@pytest.mark.parametrize("field", ["doc", "pos"])
def test_a_shadow_row_with_other_docs_or_positions_is_refused(data, monkeypatch, field):
    """REVIEW 14c D6: the shadow check compares doc ids and positions too, not only token ids. A shadow packing whose
    every id agrees with the real rows but whose doc ids (or positions) differ is refused (mutants
    r14c_shadow_doc_unchecked, s14c_shadow_pos_unchecked); the unbent packing is not."""
    real = kroute.pack_rows

    def bent(items, L, pad, rng):
        rows = real(items, L, pad, rng)
        for r in rows:
            r[field] = r[field] + 1
        return rows
    _, ld = loaders(data)
    ld.next_batch()                                   # the plain shadow aligns
    monkeypatch.setattr(kroute, "pack_rows", bent)
    _, ld = loaders(data)
    with pytest.raises(RuntimeError, match="align"):
        for _ in range(10):
            ld.next_batch()


def test_a_src_of_another_shape_than_the_batch_is_refused():
    """K3 round 15 (REVIEW 16b T-6): the routed forward refuses a fact mask whose shape is not the batch's (a src
    one column short would mask the wrong positions or broadcast), before the model runs, and leaves no mask on
    (mutant r16_route_shape_unchecked)."""
    calls = []

    class Ld:
        k_queue = collections.deque([torch.zeros(2, 7, dtype=torch.long)])
    ctx = kroute.RouteCtx()
    rf = kroute.RoutedForward(lambda *a, **kw: calls.append(a), Ld, ctx, [1])
    with pytest.raises(RuntimeError, match="disagree in shape"):
        rf(torch.zeros(2, 8, dtype=torch.long))
    assert calls == [] and ctx.mask is None and not Ld.k_queue
