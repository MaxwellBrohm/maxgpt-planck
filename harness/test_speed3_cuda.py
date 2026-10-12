"""SPEED V3 on CUDA (skipped without it): train.compile and train.ce_chunk_rows on top of the cuda
defaults (doc_attn varlen, batched NorMuon). Deterministic algorithms, CUBLAS_WORKSPACE_CONFIG
:4096:8, inductor (mode default).

  step    Trainer, 2 micro-batches of packed rows with unequal supervised counts: the step-1 loss
          and gradients of varlen alone, + compile, + ce_chunk, + both, may be no further from an
          independent fp32 sum / n_sup than the bf16 reference path (mask, per-matrix, eager) is
          (1.5x, as test_speed_combined).
  graphs  torch._dynamo.explain of the varlen forward (with and without ce_chunk): every graph
          break is cu_seqlens' nonzero() (a data-dependent size), none inside attention, the
          varlen kernel or the chunked loss; over batches with varying document counts the
          graph count stops growing (no cache-limit fallback to eager). Printed, for the notes.
  resume  train.main with both switches: 20 + resume + 20 == 40 with no tolerance; then the same
          run switched off after 20 loads cleanly, finishes, and its start records say what ran.
"""
from __future__ import annotations

import copy
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before cuBLAS starts

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import runio  # noqa: E402
import schedule as S  # noqa: E402
from model import build_model, compile_forward  # noqa: E402
from optim import make_optimizer  # noqa: E402
from test_s3_loss_mask import ListLoader  # noqa: E402
from test_s3_resume import state_equal  # noqa: E402
from test_speed_combined import BF16, MODEL, fake, packed_batches, rel, run_train  # noqa: E402,F401
from testutil import read_jsonl, scramble, tiny, write_run  # noqa: E402
from trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CHUNK = 200            # 512 rows per micro-batch: chunks split rows, the last one is short


@pytest.fixture
def deterministic():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch._dynamo.reset()


def step1(base, bs, tmp, impl="varlen", compiled=False, chunk=0, batched=True):
    m = copy.deepcopy(base)
    m.doc_attn = impl
    opt = make_optimizer(m, {"lr": 3e-3, "batched": batched}, device_type="cuda")
    grads, step = [], opt.step
    opt.step = lambda: (grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()])),
                        step())[1]
    tr = Trainer(model=m, forward=compile_forward(m, "default") if compiled else None, optimizer=opt,
                 loader=ListLoader(bs), sched=S.full(1000, warmup_steps=1), device="cuda", amp=BF16,
                 cfg={}, out_dir=str(tmp), grad_accum=2, grad_clip=1e9, log_every=1, ckpt_every=0,
                 keep_last=1, stable_points=set(), meta={}, ce_chunk=chunk)
    return tr.train_step()["loss"], grads[0]


def test_cuda_switches_step1_against_fp32(tmp_path, deterministic):
    torch.manual_seed(0)
    base = scramble(build_model(tiny(**MODEL)), 0).cuda().train()
    bs = packed_batches(MODEL["vocab_size"], MODEL["seq_len"])
    n = sum(int((b["tgt"] != -100).sum()) for b in bs)
    m32 = copy.deepcopy(base)
    l32 = sum(F.cross_entropy(m32(b["idx"].cuda(), doc=b["doc"].cuda())[0].flatten(0, 1),
                              b["tgt"].cuda().flatten(), ignore_index=-100, reduction="sum") for b in bs) / n
    l32.backward()
    l32, g32 = float(l32), torch.cat([p.grad.flatten() for p in m32.parameters()])
    arms = {"ref_bf16": dict(impl="mask", batched=False), "varlen": {},
            "varlen_compile": dict(compiled=True), "varlen_chunk": dict(chunk=CHUNK),
            "varlen_compile_chunk": dict(compiled=True, chunk=CHUNK)}
    err = {}
    for k, kw in arms.items():
        torch._dynamo.reset()
        lo, g = step1(base, bs, tmp_path, **kw)
        err[k] = (abs(lo - l32) / l32, rel(g, g32))
    print("\n(loss rel err, grad rel err) vs fp32:", {k: f"{a:.2e} {b:.2e}" for k, (a, b) in err.items()})
    le, ge = err.pop("ref_bf16")
    assert le < 1e-3 and ge < 5e-2, (le, ge)
    for k, (lo, gr) in err.items():
        assert lo <= 1.5 * le + 1e-4 and gr <= 1.5 * ge + 1e-4, (k, lo, gr, le, ge)


def break_frames(ex) -> list[str]:
    return [f"{f.filename.split('/')[-1]}:{f.name}" for br in ex.break_reasons for f in br.user_stack]


@pytest.mark.parametrize("chunk", [0, CHUNK], ids=["plain", "chunked"])
def test_cuda_compiled_varlen_graph_breaks(chunk, deterministic):
    torch.manual_seed(0)
    m = scramble(build_model(tiny(**MODEL)), 0).cuda().train()
    m.doc_attn = "varlen"
    b = packed_batches(MODEL["vocab_size"], MODEL["seq_len"])[0]
    idx, tgt, doc = (b[k].cuda() for k in ("idx", "tgt", "doc"))
    kw = {"ce_chunk": chunk} if chunk else {}
    with BF16():
        ex = torch._dynamo.explain(m)(idx, tgt, doc, reduction="sum", **kw)
    print(f"\n[{'chunked' if chunk else 'plain'}] graphs {ex.graph_count} breaks {ex.graph_break_count}: "
          f"{[br.reason.splitlines()[0] for br in ex.break_reasons]} at {sorted(set(break_frames(ex)))}")
    for br in ex.break_reasons:
        frames = [f.name for f in br.user_stack]
        assert "cu_seqlens" in frames and "nonzero" in br.reason, (br.reason, frames)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    c = compile_forward(m, "default")
    g, seen = torch.Generator().manual_seed(9), []
    for i in range(6):
        starts = torch.rand(2, MODEL["seq_len"], generator=g) < (2 + 3 * i) / MODEL["seq_len"]
        starts[:, 0] = True
        with BF16():
            ls = c(idx, tgt, torch.cumsum(starts.long(), 1).cuda(), reduction="sum", **kw)[1]
        ls.backward()
        seen.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    breaks = dict(torch._dynamo.utils.counters["graph_break"])
    print(f"unique graphs after each batch {seen}; break counters {[(k.splitlines()[0], v) for k, v in breaks.items()]}")
    assert seen[-1] == seen[3] and seen[-1] <= 8, seen
    assert all("nonzero" in k for k in breaks), breaks


def test_cuda_resume_both_switches(tmp_path, fake):
    """compile + ce_chunk_rows on the cuda defaults: 20 + 20 == 40 bitwise (fresh processes,
    deterministic); switched off after 20 in the same run dir: resumes and finishes."""
    tc = {"device": "cuda", "precision": "bf16", "micro_batch": 8, "grad_accum": 2, "total_steps": 40,
          "log_every": 5, "ckpt_every": 10, "compile": "default", "ce_chunk_rows": 1000}
    over = dict(runs_jsonl="runs.jsonl", model=MODEL, train=tc)
    straight = write_run(str(tmp_path / "straight"), fake, **over)
    run_train(straight)
    split = write_run(str(tmp_path / "split"), fake, **over)
    run_train(split, "--max-steps", "20")
    run_train(split)
    a, b = (runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / r / "out"))) for r in ("straight", "split"))
    assert a["step"] == b["step"] == 40 and not any(k.startswith("_orig_mod.") for k in a["model"])
    for k in ("model", "optimizer", "data_state"):
        assert state_equal(a[k], b[k]) == [], k
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in ("straight", "split"))
    keys = ("step", "loss", "gnorm", "lr_factor", "tokens", "sup_tokens")
    assert [tuple(r[k] for k in keys) for r in la] == [tuple(r[k] for k in keys) for r in lb]
    toggle = write_run(str(tmp_path / "toggle"), fake, **over)
    run_train(toggle, "--max-steps", "20")
    write_run(str(tmp_path / "toggle"), fake, **{**over, "train": {k: v for k, v in tc.items()
                                                                   if k not in ("compile", "ce_chunk_rows")}})
    run_train(toggle)
    lt = read_jsonl(tmp_path / "toggle" / "out" / "log.jsonl")
    assert [r["step"] for r in lt] == list(range(5, 41, 5)) and abs(lt[-1]["loss"] - la[-1]["loss"]) < 0.05 * la[-1]["loss"]
    st = [r for r in read_jsonl(tmp_path / "toggle" / "runs.jsonl") if r["event"] == "start"]
    assert [(r.get("compile"), r.get("ce_chunk_rows"), r.get("doc_attn")) for r in st] == \
        [("default", 1000, "varlen"), (None, None, "varlen")]
