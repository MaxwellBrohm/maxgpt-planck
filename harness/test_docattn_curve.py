"""train.doc_attn wiring (CPU) and the 200-step loss-curve comparison (CUDA).

CPU: train.py refuses doc_attn varlen off CUDA at startup (before any step), and
an explicit doc_attn mask trains bit for bit like a config without the key.
CUDA (skipped without it, about a minute): one small packed run (fake data, pack mode with
chat items, bf16, grad_accum 2, so micro-batches carry unequal supervised counts) trained for
200 steps, each arm in a fresh process with torch.use_deterministic_algorithms(True) and
CUBLAS_WORKSPACE_CONFIG=:4096:8 unless noted:
  mask, mask2   the deterministic reference, twice: must be identical
  mask_nd       deterministic mode OFF: CUDA eager's run-to-run floor (0 at this size so far)
  mask_math     the same masked attention on SDPA's math kernel instead of mem-efficient:
                what swapping one exact attention kernel for another does to the curve
  varlen        the candidate
The candidate may drift from the reference only about as far as those floors do (3x), with a
small absolute allowance (a floor can be 0).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import train
from make_fake_data import make
from test_train import read_jsonl, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
CURVE_MODEL = {"vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
               "head_dim": 32, "mlp_hidden": 128, "seq_len": 256}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    make(str(d), docs=300, chats=300)
    return str(d)


def test_varlen_refused_off_cuda(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake, train={"doc_attn": "varlen"})
    with pytest.raises(ValueError, match="cuda"):
        train.main([cfg])


def test_explicit_mask_is_the_default_path(tmp_path, fake):
    a = write_run(str(tmp_path / "a"), fake)
    b = write_run(str(tmp_path / "b"), fake, train={"doc_attn": "mask"})
    assert train.main([a]) == 0 and train.main([b]) == 0
    la, lb = (read_jsonl(tmp_path / r / "out" / "log.jsonl") for r in ("a", "b"))
    assert [r["loss"] for r in la] == [r["loss"] for r in lb] and len(la) == 12


MATH_ONLY = ("torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp"
             "(False); torch.backends.cuda.enable_cudnn_sdp(False); ")


def curve(tmp_path, fake, name: str, doc_attn: str, deterministic: bool = True,
          prelude: str = "") -> list[float]:
    cfg = write_run(str(tmp_path / name), fake, runs_jsonl="runs.jsonl", model=CURVE_MODEL,
                    train={"device": "cuda", "precision": "bf16", "micro_batch": 8,
                           "grad_accum": 2, "total_steps": 200, "log_every": 1,
                           "ckpt_every": 0, "doc_attn": doc_attn})
    code = ("import sys, torch; torch.use_deterministic_algorithms(%s); %simport train; "
            "sys.exit(train.main([sys.argv[1]]))" % (deterministic, prelude))
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    p = subprocess.run([sys.executable, "-c", code, cfg], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    start = read_jsonl(tmp_path / name / "runs.jsonl")[0]
    assert start.get("doc_attn", "mask") == doc_attn
    log = read_jsonl(tmp_path / name / "out" / "log.jsonl")
    assert [r["step"] for r in log] == list(range(1, 201))
    return [r["loss"] for r in log]


def diffs(a: list[float], b: list[float]) -> dict:
    d = [abs(x - y) for x, y in zip(a, b)]
    return {"first": d[0], "max": max(d), "mean_last50": sum(d[-50:]) / 50}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (flash varlen)")
def test_cuda_200_step_curve_varlen_vs_mask(tmp_path, fake):
    ref = curve(tmp_path, fake, "mask", "mask")
    ref2 = curve(tmp_path, fake, "mask2", "mask")
    nd = curve(tmp_path, fake, "mask_nd", "mask", deterministic=False)
    math = curve(tmp_path, fake, "mask_math", "mask", prelude=MATH_ONLY)
    cand = curve(tmp_path, fake, "varlen", "varlen")
    det, fnd, fmath, got = diffs(ref, ref2), diffs(nd, ref), diffs(math, ref), diffs(cand, ref)
    print(f"\nloss step1 {ref[0]:.5f} step200 mask {ref[-1]:.5f} math {math[-1]:.5f} "
          f"varlen {cand[-1]:.5f}\n  mask vs mask (deterministic) {det}\n  mask_nd vs mask "
          f"{fnd}\n  mask_math vs mask {fmath}\n  varlen vs mask {got}")
    assert det["max"] == 0.0, "the deterministic reference is not reproducible"
    assert ref[-1] < 0.8 * ref[0], "the run did not learn"
    assert got["first"] <= 2e-3 * ref[0]                     # step 1: bf16 rounding only
    for k, allow in (("mean_last50", 1e-3 * ref[-1]), ("max", 2e-3 * ref[0])):
        assert got[k] <= 3 * max(fnd[k], fmath[k]) + allow, (k, got[k], fnd[k], fmath[k])
