"""train.compile on CUDA with the real backend (inductor, mode default). Skipped without CUDA.

Every test runs with torch.use_deterministic_algorithms(True) (CUBLAS_WORKSPACE_CONFIG
:4096:8) so eager is reproducible and any compiled-vs-eager gap is compile's own.
Tolerance for compiled vs eager under bf16 autocast: the eager bf16 step's own rounding
error, measured here as the distance from eager bf16 to eager fp32 on the same batch.
Inductor fuses elementwise chains and keeps their intermediates in fp32 where eager rounds
to bf16 between kernels, so compiled bf16 should sit no further from eager bf16 than fp32
does; the bound is 1.5x that distance.
The step and leak tests build models directly, so they run the mask path (model.doc_attn default);
the resume test goes through train.main and so runs the cuda defaults (optim.batched, doc_attn
varlen). Compile with varlen at the step level, and with train.ce_chunk_rows: test_speed3_cuda.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402
import torch  # noqa: E402

import runio  # noqa: E402
import train  # noqa: E402
from make_fake_data import make  # noqa: E402
from model import build_model, compile_forward  # noqa: E402
from optim import make_optimizer  # noqa: E402
from selftest import causal_leak, document_leak, scrambled_copy  # noqa: E402
from test_compile import FixedLoader, FlatSched, GradSpy  # noqa: E402
from test_s3_leak import DOC, perturb_check  # noqa: E402
from test_s3_resume import state_equal  # noqa: E402
from testutil import read_jsonl, scramble, tiny, write_run  # noqa: E402
from trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CFG = dict(vocab_size=2048, d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=32,
           mlp_hidden=352, seq_len=256)
AMP = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731


@pytest.fixture(autouse=True)
def deterministic():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)
    torch._dynamo.reset()


def batches(packed: bool, T: int = 256, B: int = 4):
    """grad_accum 3, supervised fractions about 0.9, 0.1 and 0 (unequal counts)."""
    g = torch.Generator().manual_seed(21)
    out = []
    for frac in (0.9, 0.1, 0.0):
        idx = torch.randint(1, CFG["vocab_size"], (B, T), generator=g)
        tgt = torch.randint(0, CFG["vocab_size"], (B, T), generator=g)
        tgt[torch.rand(B, T, generator=g) >= frac] = -100
        doc = torch.cumsum(torch.rand(B, T, generator=g) < 6 / T, dim=1) if packed else None
        out.append({"idx": idx, "tgt": tgt, "doc": doc, "pos": None})
    return out


def one_step(packed: bool, amp, compiled: bool):
    torch.manual_seed(0)
    m = build_model(tiny(**CFG), "cpu").to("cuda")
    opt = GradSpy(make_optimizer(m, {}, "cuda"))
    tr = Trainer(model=m, optimizer=opt, loader=FixedLoader(batches(packed)), sched=FlatSched(),
                 device="cuda", amp=amp, cfg={}, out_dir="/nonexistent", grad_accum=3,
                 grad_clip=1e9, log_every=1, ckpt_every=0, keep_last=1, stable_points=set(),
                 meta={}, forward=compile_forward(m, "default") if compiled else None)
    out = tr.train_step()
    return out["loss"], torch.cat([g.float().flatten() for g in opt.grads])


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_compiled_step_within_eager_bf16_rounding(packed):
    lb, gb = one_step(packed, AMP, False)
    lb2, gb2 = one_step(packed, AMP, False)
    assert lb == lb2 and torch.equal(gb, gb2), "eager is not reproducible in deterministic mode"
    lf, gf = one_step(packed, None, False)
    lc, gc = one_step(packed, AMP, True)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    floor_l, floor_g = abs(lf - lb), rel(gf, gb)
    got_l, got_g = abs(lc - lb), rel(gc, gb)
    print(f"\n[{'packed' if packed else 'causal'}] loss eager bf16 {lb:.6f} fp32 {lf:.6f} "
          f"compiled {lc:.6f}; grad rel err vs eager bf16: fp32 {floor_g:.2e} compiled {got_g:.2e}"
          f"; graphs {graphs} breaks {breaks}")
    assert (graphs, breaks) == (1, 0)
    assert got_g <= 1.5 * floor_g
    assert got_l <= 1.5 * max(floor_l, 1e-5 * lb)


class OnCuda:
    """perturb_check builds CPU inputs: run them through a CUDA module, logits back on CPU."""

    def __init__(self, fwd, cfg):
        self.fwd, self.cfg = fwd, cfg

    def __call__(self, idx, doc=None):
        logits, _ = self.fwd(idx.cuda(), doc=None if doc is None else doc.cuda())
        return logits.cpu(), None


@pytest.mark.parametrize("packed", [False, True], ids=["causal", "packed"])
def test_compiled_forward_no_leak_cuda(packed):
    """Bitwise one-token perturbation probe (fp32) and the startup self-test's ratio checks
    (bf16 autocast, grad and no_grad) through the inductor-compiled forward."""
    torch.manual_seed(0)
    m = scramble(build_model(tiny()), 0).to("cuda")
    perturb_check(OnCuda(compile_forward(m, "default"), m.cfg), DOC if packed else None, grad=False)
    torch._dynamo.reset()
    big = scrambled_copy(build_model(tiny(**CFG), "cpu").to("cuda")).eval()
    c = compile_forward(big, "default")
    for grad in (True, False):
        ratios = [causal_leak(c, 256, "cuda", AMP, grad, masked=packed)]
        if packed:
            ratios.append(document_leak(c, 256, "cuda", AMP, grad))
        assert max(ratios) < 3e-2, ratios


def test_resume_with_compile_cuda(tmp_path):
    """20 + 20 compiled steps == 40 compiled steps on CUDA (bf16, pack mode, on the train.py cuda
    defaults: batched NorMuon and doc_attn varlen, which the start record must show)."""
    d = str(tmp_path / "data")
    make(d, docs=150, chats=120)
    over = {"train": {"device": "cuda", "precision": "bf16", "compile": "default"}}
    straight = write_run(str(tmp_path / "straight"), d, **over)
    assert train.main([straight]) == 0
    split = write_run(str(tmp_path / "split"), d, **over)
    assert train.main([split, "--max-steps", "20"]) == 0
    torch._dynamo.reset()
    assert train.main([split]) == 0
    a = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "straight" / "out")))
    b = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "split" / "out")))
    assert a["step"] == b["step"] == 40
    assert not any(k.startswith("_orig_mod.") for k in a["model"])
    for k in ("model", "optimizer", "data_state"):
        assert state_equal(a[k], b[k]) == [], k
    la = read_jsonl(tmp_path / "straight" / "out" / "log.jsonl")
    lb = read_jsonl(tmp_path / "split" / "out" / "log.jsonl")
    assert [(r["step"], r["loss"]) for r in la] == [(r["step"], r["loss"]) for r in lb]
    starts = [r for r in read_jsonl(tmp_path / "runs.jsonl") if r["event"] == "start"]
    assert len(starts) == 3 and all((r.get("compile"), r.get("doc_attn"), r.get("optim_batched")) ==
                                    ("default", "varlen", True) for r in starts), starts
