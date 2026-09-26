"""The speed switches together, against the reference (Integrate, 2026-09-26).

Defaults since 2026-09-26 (optim.resolve_batched, docattn.resolve_doc_attn): on cuda with bf16
the optimizer is optim_batched and packed rows use doc_attn varlen; cpu and mps keep the
reference (never measured there). train.lazy_metrics stays opt-in. torch.compile is not a
train.py option (pc/bench_micro.py --compile only), so the compiled checks drive Trainer.
CPU: the auto rule; a default cpu run equals an explicit reference run bit for bit.
CUDA (skipped without it; deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8):
  resume   train.py defaults + lazy_metrics (pack with chats, bf16, accum 2, log_every 5), each
           run a fresh process: 20 + resume + 20 == 40 with no tolerance (weights, optimizer
           state, loader state, every log record); the start record shows what ran.
  accum    Trainer on fixed packed batches, 2 micro-batches with unequal supervised counts:
           step-1 loss and gradients of the combined path (varlen + batched + lazy, eager and
           compiled; accum 2 and the same rows as 1 micro-batch) may be no further from an
           independent fp32 sum / n_sup (plain forward + autograd, no Trainer) than the bf16
           reference path is (1.5x, as test_s3_docattn); the reference must itself be close.
  compile  30 steps compiled vs eager (combined), within 3x the eager run's own floor
           (weights x (1 + 1e-7 N)); bitwise leak probes through the compiled varlen forward.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before cuBLAS starts

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import runio  # noqa: E402
import schedule as S  # noqa: E402
import train  # noqa: E402
from docattn import resolve_doc_attn, varlen_available  # noqa: E402
from make_fake_data import make  # noqa: E402
from model import build_model  # noqa: E402
from optim import make_optimizer, resolve_batched  # noqa: E402
from test_s3_leak import DOC  # noqa: E402
from test_s3_loss_mask import ListLoader  # noqa: E402
from test_s3_resume import state_equal  # noqa: E402
from testutil import allowed_matrix, read_jsonl, scramble, tiny, write_run  # noqa: E402
from trainer import Trainer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="needs CUDA")
MODEL = {"vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
         "head_dim": 32, "mlp_hidden": 128, "seq_len": 256}
BF16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_speed")
    make(str(d), docs=300, chats=300)
    return str(d)


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def test_auto_rule():
    assert resolve_batched(None, "cuda") and resolve_batched("auto", "cuda")
    assert not resolve_batched("auto", "mps") and not resolve_batched("auto", "cpu")
    assert resolve_batched(True, "cpu") and not resolve_batched(False, "cuda")
    with pytest.raises(ValueError):
        resolve_batched("yes", "cuda")
    assert resolve_doc_attn("auto", "cuda", "bf16") == ("varlen" if varlen_available() else "mask")
    for dev, prec in (("cuda", "fp32"), ("mps", "bf16"), ("cpu", "fp32"), ("cpu", "bf16")):
        assert resolve_doc_attn("auto", dev, prec) == "mask"
    assert resolve_doc_attn("varlen", "cpu", "fp32") == "varlen"      # set_doc_attn refuses it


def test_cpu_default_run_equals_explicit_reference(tmp_path, fake):
    tc = {"total_steps": 12, "grad_accum": 2}
    a = write_run(str(tmp_path / "a"), fake, train=tc)
    b = write_run(str(tmp_path / "b"), fake, optim={"lr": 3e-3, "batched": False},
                  train={**tc, "doc_attn": "mask", "lazy_metrics": False})
    assert train.main([a]) == 0 and train.main([b]) == 0
    runs = read_jsonl(tmp_path / "runs.jsonl")
    assert not {"doc_attn", "optim_batched", "lazy_metrics"} & {k for r in runs for k in r}
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in "ab")
    drop = lambda r: {k: v for k, v in r.items() if k not in ("tok_per_s", "time")}  # noqa: E731
    assert [drop(r) for r in la] == [drop(r) for r in lb] and len(la) == 12
    ca, cb = (runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / r / "out"))) for r in "ab")
    assert state_equal(ca["model"], cb["model"]) == []


# ------------------------------------------------------------------------------ CUDA
def run_train(cfg: str, *extra: str) -> None:
    code = ("import sys, torch; torch.use_deterministic_algorithms(True); import train; "
            "sys.exit(train.main(sys.argv[1:]))")
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    p = subprocess.run([sys.executable, "-c", code, cfg, *extra], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]


@cuda_only
def test_cuda_defaults_resume_exact(tmp_path, fake):
    over = dict(runs_jsonl="runs.jsonl", model=MODEL,
                train={"device": "cuda", "precision": "bf16", "micro_batch": 8, "grad_accum": 2,
                       "total_steps": 40, "log_every": 5, "ckpt_every": 10, "lazy_metrics": True})
    straight = write_run(str(tmp_path / "straight"), fake, **over)
    run_train(straight)
    split = write_run(str(tmp_path / "split"), fake, **over)
    run_train(split, "--max-steps", "20")
    run_train(split)
    for r in ("straight", "split"):
        for st in (x for x in read_jsonl(tmp_path / r / "runs.jsonl") if x["event"] == "start"):
            assert (st.get("doc_attn"), st.get("optim_batched"), st.get("lazy_metrics")) == \
                ("varlen", True, True), st
    a, b = (runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / r / "out")))
            for r in ("straight", "split"))
    assert a["step"] == b["step"] == 40
    for k in ("model", "optimizer", "data_state"):
        assert state_equal(a[k], b[k]) == [], k
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in ("straight", "split"))
    keys = ("step", "loss", "gnorm", "lr_factor", "tokens", "sup_tokens")
    assert [r["step"] for r in la] == list(range(5, 41, 5))
    assert [tuple(r[k] for k in keys) for r in la] == [tuple(r[k] for k in keys) for r in lb]
    assert la[-1]["loss"] < la[0]["loss"]


def packed_batches(V: int, T: int, sup=(0.9, 0.1), seed: int = 0) -> list[dict]:
    """2 micro-batches of 2 packed rows (about 6 docs a row), unequal supervised counts."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for f in sup:
        starts = torch.rand(2, T, generator=g) < 6 / T
        starts[:, 0] = True
        tgt = torch.randint(1, V, (2, T), generator=g)
        tgt[torch.rand(2, T, generator=g) > f] = -100
        out.append({"idx": torch.randint(1, V, (2, T), generator=g), "tgt": tgt,
                    "doc": torch.cumsum(starts.long(), 1), "pos": None})
    return out


def one_batch(bs: list[dict]) -> dict:
    return {k: None if bs[0][k] is None else torch.cat([b[k] for b in bs]) for k in bs[0]}


def trainer(base, tmp, batches, impl="varlen", batched=True, lazy=True, compiled=False,
            amp=BF16, accum=2):
    m = copy.deepcopy(base)
    m.doc_attn = impl
    opt = make_optimizer(m, {"lr": 3e-3, "batched": batched}, device_type="cuda")
    assert opt.batched == batched
    grads, step = [], opt.step

    def spy():
        grads.append(torch.cat([p.grad.float().flatten() for p in m.parameters()]))
        step()
    opt.step = spy
    tr = Trainer(model=torch.compile(m) if compiled else m, optimizer=opt,
                 loader=ListLoader(batches if accum > 1 else [one_batch(batches)]),
                 sched=S.full(1000, warmup_steps=1), device="cuda", amp=amp, cfg={}, out_dir=str(tmp),
                 grad_accum=accum, grad_clip=1e9, log_every=1, ckpt_every=0, keep_last=1,
                 stable_points=set(), meta={}, lazy_metrics=lazy)
    return tr, grads


def losses(tr, n: int) -> list[float]:
    return [tr.read_metrics(tr.train_step())["loss"] for _ in range(n)]


def rel(a, b) -> float:
    return float((a - b).norm() / b.norm())


@cuda_only
def test_cuda_accum_and_compile_step1_against_fp32(tmp_path, deterministic):
    torch.manual_seed(0)
    base = scramble(build_model(tiny(**MODEL)), 0).cuda().train()
    bs = packed_batches(MODEL["vocab_size"], MODEL["seq_len"])
    n1, n2 = (int((b["tgt"] != -100).sum()) for b in bs)
    assert n1 > 5 * n2 > 0, (n1, n2)
    m32 = copy.deepcopy(base)                       # independent of Trainer: fp32, mask,
    l32 = sum(F.cross_entropy(m32(b["idx"].cuda(), doc=b["doc"].cuda())[0].flatten(0, 1),
                              b["tgt"].cuda().flatten(), ignore_index=-100, reduction="sum")
              for b in bs) / (n1 + n2)             # the whole step's sum / its supervised count
    l32.backward()
    l32, g32 = float(l32), torch.cat([p.grad.flatten() for p in m32.parameters()])
    arms = {"ref_bf16": dict(impl="mask", batched=False, lazy=False),
            "comb": {}, "comb_1mb": dict(accum=1), "comb_compiled": dict(compiled=True),
            "comb_compiled_1mb": dict(compiled=True, accum=1)}
    err = {}
    for k, kw in arms.items():
        tr, g = trainer(base, tmp_path, bs, **kw)
        lo = losses(tr, 1)[0]
        err[k] = (abs(lo - l32) / l32, rel(g[0], g32))
    print("\n(loss rel err, grad rel err) vs fp32:", {k: f"{a:.2e} {b:.2e}" for k, (a, b) in err.items()})
    le, ge = err.pop("ref_bf16")
    assert le < 1e-3 and ge < 5e-2, (le, ge)        # the reference path itself is right
    for k, (lo, gr) in err.items():
        assert lo <= 1.5 * le + 1e-4 and gr <= 1.5 * ge + 1e-4, (k, lo, gr, le, ge)


@cuda_only
def test_cuda_compiled_curve_and_leak(tmp_path, deterministic):
    torch.manual_seed(0)
    base = build_model(tiny(**MODEL)).cuda().train()
    bs = packed_batches(MODEL["vocab_size"], MODEL["seq_len"])
    pert = copy.deepcopy(base)
    g = torch.Generator(device="cuda").manual_seed(123)
    with torch.no_grad():
        for p in pert.parameters():
            p.mul_(1 + 1e-7 * torch.randn(p.shape, generator=g, device="cuda"))
    ref = losses(trainer(base, tmp_path, bs)[0], 30)
    comp = losses(trainer(base, tmp_path, bs, compiled=True)[0], 30)
    flo = losses(trainer(pert, tmp_path, bs)[0], 30)
    dc, df = ([abs(x - y) for x, y in zip(c, ref)] for c in (comp, flo))
    print(f"\nloss {ref[0]:.4f} -> {ref[-1]:.4f}; max|d| compiled {max(dc):.2e} floor {max(df):.2e}")
    assert ref[-1] < 0.8 * ref[0]
    assert dc[0] <= 2e-3 * ref[0] and max(dc) <= 3 * max(df) + 2e-3 * ref[0]

    m = scramble(copy.deepcopy(base), 0)
    m.doc_attn = "varlen"
    fwd, n, V = torch.compile(m), len(DOC), MODEL["vocab_size"]
    A = allowed_matrix(DOC, n)
    idx = torch.randint(0, V, (1, n), generator=torch.Generator().manual_seed(3)).cuda()
    doc = torch.tensor([DOC]).cuda()
    with torch.no_grad(), BF16():
        out0 = fwd(idx, doc=doc)[0][0]
        for j in range(n):
            idx2 = idx.clone()
            idx2[0, j] = (idx[0, j] + 1) % V
            out = fwd(idx2, doc=doc)[0][0]
            for t in range(n):
                assert A[t, j] or torch.equal(out[t], out0[t]), (t, j)
            assert not torch.equal(out[j], out0[j]), j
