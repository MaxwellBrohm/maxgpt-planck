"""SPEED V3: train.compile and train.ce_chunk_rows, both opt-in, alone and together (CPU).

Default path: a run without the keys equals one with them set off, bit for bit; in a fresh process
it never calls torch.compile, compiles no graph, never imports chunked_ce, and hands the trainer the
plain module (default_parity.py runs the same check across two trees). Validation of the
ce_chunk_rows value. Chunked loss switched on then off across a resume, alone and with compile.
Both together: the compiled chunked forward is one graph (aot_eager, fullgraph) and matches the
eager chunked and reference paths; with the S006 aux head the chunked trainer step matches the
reference one. CUDA versions: test_speed3_cuda.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

import runio
import schedule as S
import train
from make_fake_data import make
from model import build_model, ce_chunk_rows
from mtp import MTPHead
from optim import make_optimizer
from test_s3_leak import DOC, T
from test_s3_resume import state_equal
from testutil import read_jsonl, scramble, tiny, write_run

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def fresh_dynamo():
    yield
    if "torch._dynamo" in sys.modules:
        torch._dynamo.reset()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("speed3_data")
    make(str(d), docs=200, chats=150)
    return str(d)


def ck(run_dir) -> dict:
    return runio.load_checkpoint(runio.latest_checkpoint(os.path.join(str(run_dir), "out")))


def test_ce_chunk_rows_values():
    assert ce_chunk_rows(None) == ce_chunk_rows(0) == ce_chunk_rows(False) == 0
    assert ce_chunk_rows(2048) == 2048
    for bad in (True, -1, 1.5, "2048"):
        with pytest.raises(ValueError):
            ce_chunk_rows(bad)


PROBE = """
import json, sys, torch
def refuse(*a, **k):
    raise AssertionError("torch.compile called on the default path")
torch.compile = refuse
import trainer
seen = []
init = trainer.Trainer.__init__
def spy(self, *a, **k):
    init(self, *a, **k)
    seen.append(self.fwd is self.model and "ce_chunk" not in self._hid)
trainer.Trainer.__init__ = spy
import train
rc = train.main(sys.argv[2:])
from torch._dynamo.utils import counters     # torch.optim imports Dynamo on the default path too
json.dump({"rc": rc, "plain": seen, "chunked_ce": "chunked_ce" in sys.modules,
           "graphs": counters["stats"]["unique_graphs"]}, open(sys.argv[1], "w"))
"""


def test_default_run_is_the_reference_and_builds_nothing_new(tmp_path, data):
    """No speed keys == both keys explicitly off, bit for bit (log records, weights, optimizer,
    loader); neither start record names them; a fresh default process calls no torch.compile,
    compiles no Dynamo graph, never imports chunked_ce, and the trainer runs the plain model.
    (torch.optim itself imports torch._dynamo on every path; default_parity.py compares the whole
    module set against the previous tree.)"""
    tc = {"total_steps": 12, "grad_accum": 2}
    a = write_run(str(tmp_path / "a"), data, train=tc)
    b = write_run(str(tmp_path / "b"), data, train={**tc, "compile": False, "ce_chunk_rows": 0})
    out = tmp_path / "probe.json"
    p = subprocess.run([sys.executable, "-c", PROBE, str(out), a], cwd=HERE, capture_output=True,
                       text=True, timeout=600, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert p.returncode == 0, p.stderr[-3000:]
    assert json.load(open(out)) == {"rc": 0, "plain": [True], "chunked_ce": False, "graphs": 0}
    assert train.main([b]) == 0
    runs = read_jsonl(tmp_path / "runs.jsonl")
    assert len(runs) == 4 and not {"compile", "compile_backend", "ce_chunk_rows"} & {k for r in runs for k in r}
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in "ab")
    drop = lambda r: {k: v for k, v in r.items() if k not in ("tok_per_s", "time")}  # noqa: E731
    assert [drop(r) for r in la] == [drop(r) for r in lb] and len(la) == 12
    ca, cb = ck(tmp_path / "a"), ck(tmp_path / "b")
    for k in ("model", "optimizer", "data_state"):
        assert state_equal(ca[k], cb[k]) == [], k


@pytest.mark.parametrize("both", [False, True], ids=["chunk", "chunk+compile"])
def test_resume_switch_on_then_off(tmp_path, data, both):
    """20 steps with ce_chunk_rows (and compile) on, then 20 with them off in the same run dir:
    loads cleanly, the start records say what ran, and the weights land within fp32 rounding of
    40 reference steps (CPU fp32: chunked = reference to ~1e-6 per step)."""
    on = {"ce_chunk_rows": 50, **({"compile": "default", "compile_backend": "aot_eager"} if both else {})}
    ref = write_run(str(tmp_path / "ref"), data)
    assert train.main([ref]) == 0
    mixed = write_run(str(tmp_path / "mixed"), data, train=on)
    assert train.main([mixed, "--max-steps", "20"]) == 0
    write_run(str(tmp_path / "mixed"), data)                     # same run dir, both keys gone
    assert train.main([mixed]) == 0
    a, b = ck(tmp_path / "ref"), ck(tmp_path / "mixed")
    assert b["step"] == 40 and list(a["model"]) == list(b["model"])
    assert [(g["name"], len(g["params"])) for g in a["optimizer"]["param_groups"]] == \
        [(g["name"], len(g["params"])) for g in b["optimizer"]["param_groups"]]
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], rtol=1e-3, atol=1e-4), k
    st = [r for r in read_jsonl(tmp_path / "runs.jsonl") if r["event"] == "start" and r["run"] == "mixed"]
    assert [r.get("ce_chunk_rows") for r in st] == [50, None]
    assert [r.get("compile") for r in st] == (["default", None] if both else [None, None])
    la = read_jsonl(tmp_path / "ref" / "out" / "log.jsonl")
    lb = read_jsonl(tmp_path / "mixed" / "out" / "log.jsonl")
    assert [r["step"] for r in lb] == list(range(1, 41))
    assert max(abs(x["loss"] - y["loss"]) for x, y in zip(la, lb)) < 1e-3


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_compiled_chunked_forward_is_one_graph(packed, reduction):
    """compile + ce_chunk: the whole forward, the chunk loop and the autograd.Function included,
    is one graph (fullgraph), and its loss and grads equal the eager chunked path and the
    reference path to fp32 rounding; no logits come back."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(0)
    m = scramble(build_model(tiny(n_loops=2)), 3).train()
    g = torch.Generator().manual_seed(5)
    idx, tgt = torch.randint(0, 97, (3, T), generator=g), torch.randint(0, 97, (3, T), generator=g)
    tgt[torch.rand(3, T, generator=g) < 0.4] = -100
    doc = torch.tensor([DOC] * 3) if packed else None
    c = torch.compile(m, backend="aot_eager", fullgraph=True)
    out = {}
    for name, fwd, kw in (("ref", m, {}), ("chunk", m, {"ce_chunk": 7}), ("both", c, {"ce_chunk": 7})):
        m.zero_grad(set_to_none=True)
        lg, ls = fwd(idx, tgt, doc, reduction=reduction, **kw)
        ls.backward()
        out[name] = (lg, ls.detach(), [p.grad.clone() for p in m.parameters()])
    assert out["ref"][0] is not None and out["chunk"][0] is None and out["both"][0] is None
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1
    for name in ("chunk", "both"):
        assert float(out[name][1]) == pytest.approx(float(out["ref"][1]), rel=2e-6), name
        for a, b in zip(out[name][2], out["ref"][2]):
            assert (a - b).norm() <= 2e-4 * b.norm() + 1e-12, name


@pytest.mark.parametrize("arm", ["chunk", "compile", "chunk+compile"])
def test_trainer_mtp_with_switches(tmp_path, arm):
    """S006 aux head with ce_chunk and/or compile (aot_eager, fullgraph): the head still reads z
    (return_hidden) through its own full logits; the step's loss, mtp_loss, n_sup and every
    gradient (model and head) match the reference step."""
    from test_s3_loss_mask import ListLoader
    from trainer import Trainer
    g = torch.Generator().manual_seed(7)
    bs = []
    for f in (0.9, 0.2):
        tgt = torch.randint(1, 97, (2, T), generator=g)
        tgt[torch.rand(2, T, generator=g) > f] = -100
        bs.append({"idx": torch.randint(1, 97, (2, T), generator=g), "tgt": tgt,
                   "doc": torch.tensor([DOC] * 2), "pos": None})
    res = []
    for on in (False, True):
        chunk = 9 if on and "chunk" in arm else 0
        torch._dynamo.reset()
        torch.manual_seed(0)
        m = scramble(build_model(tiny()), 2).train()
        head = MTPHead(m.cfg.d_model, m.cfg.rms_eps)
        opt = make_optimizer(m, {"lr": 1e-3}, extra=head)
        grads, step = {}, opt.step
        opt.step = lambda _s=step: (grads.update({i: p.grad.clone() for i, p in enumerate(
            [*m.parameters(), *head.parameters()])}), _s())[1]
        tr = Trainer(model=m, optimizer=opt, loader=ListLoader(bs), sched=S.full(10), device="cpu",
                     amp=None, cfg={}, out_dir=str(tmp_path), grad_accum=2, grad_clip=1e9, log_every=1,
                     ckpt_every=0, keep_last=1, stable_points=set(), meta={}, mtp=head, mtp_weight=0.5,
                     ce_chunk=chunk, forward=torch.compile(m, backend="aot_eager", fullgraph=True)
                     if on and "compile" in arm else None)
        res.append((tr.train_step(), grads))
    (o_r, g_r), (o_c, g_c) = res
    assert o_r["n_sup"] == o_c["n_sup"] and o_r["mtp_n"] == o_c["mtp_n"] > 0
    assert o_c["loss"] == pytest.approx(o_r["loss"], rel=2e-6)
    assert o_c["mtp_loss"] == pytest.approx(o_r["mtp_loss"], rel=2e-6)
    assert g_r.keys() == g_c.keys()
    for i in g_r:
        assert (g_c[i] - g_r[i]).norm() <= 2e-4 * g_r[i].norm() + 1e-12, i
