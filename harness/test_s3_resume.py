"""(f) Train 20 steps, save, resume, and match an uninterrupted 40-step run BIT FOR BIT.

Tolerance: none. Weights, every optimizer state tensor, the loader state, and the logged
loss of every step 21..40 must be identical (torch.equal / ==). CPU is deterministic for
these ops; on MPS or CUDA this must be re-checked (kernels may not be).
Cases: pack mode fp32; bucket mode with a looped, shared, K=V arm; pack mode under bf16
autocast with prelude + looped core + coda and grad_accum 3; the speed switches (optim
batched + lazy_metrics, log_every 5). The same on CUDA with the cuda defaults (varlen doc
attention + batched) and lazy_metrics, deterministic: test_speed_combined.py.
"""
from __future__ import annotations

import os

import pytest
import torch

import runio
import train
from make_fake_data import make
from testutil import read_jsonl, write_run

LOOPED = {"vocab_size": 256, "d_model": 32, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1,
          "head_dim": 16, "mlp_hidden": 48, "seq_len": 64}

CASES = {
    "pack_fp32": {},
    "bucket_looped": {"data": {"mode": "bucket", "tokens_per_micro": 256},
                      "model": {**LOOPED, "n_layers": 2, "n_loops": 2, "qk_share": 2, "kv_tie": True}},
    "pack_bf16_prelude_coda": {"model": {**LOOPED, "n_loops": 3, "n_prelude": 1, "n_coda": 1},
                               "train": {"precision": "bf16", "grad_accum": 3}},
    "pack_batched_lazy": {"optim": {"lr": 3e-3, "batched": True},
                          "train": {"lazy_metrics": True, "log_every": 5}},
}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("resume_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def state_equal(a, b, path="") -> list[str]:
    """Paths where two nested checkpoint structures differ (tensors compared exactly)."""
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [path + "{keys}"]
        return [x for k in a for x in state_equal(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + "[len]"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in state_equal(u, v, f"{path}[{i}]")]
    return [] if a == b else [path]


@pytest.mark.parametrize("case", list(CASES))
def test_resume_20_plus_20_equals_40(tmp_path, data, case):
    over = CASES[case]
    straight = write_run(str(tmp_path / "straight"), data, **over)
    assert train.main([straight]) == 0
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([split, "--max-steps", "20"]) == 0
    mid = runio.latest_checkpoint(str(tmp_path / "split" / "out"))
    assert os.path.basename(mid) == "ckpt_00000020.pt"
    assert train.main([split]) == 0

    a = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "straight" / "out")))
    b = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "split" / "out")))
    assert a["step"] == b["step"] == 40
    assert state_equal(a["model"], b["model"]) == []
    assert state_equal(a["optimizer"], b["optimizer"]) == []
    assert state_equal(a["data_state"], b["data_state"]) == []
    assert (a["tokens"], a["sup_tokens"]) == (b["tokens"], b["sup_tokens"])

    la = read_jsonl(tmp_path / "straight" / "out" / "log.jsonl")
    lb = read_jsonl(tmp_path / "split" / "out" / "log.jsonl")
    keys = ("step", "loss", "gnorm", "lr_factor", "tokens", "sup_tokens")
    assert [tuple(r[k] for k in keys) for r in la] == [tuple(r[k] for k in keys) for r in lb]
    runs = read_jsonl(tmp_path / "runs.jsonl")
    starts = [r for r in runs if r["event"] == "start"]
    assert starts[-1]["resumed_from"] == mid and starts[-1]["step"] == 20
    if case == "pack_bf16_prelude_coda":
        assert starts[-1]["precision"] == "bf16"


def test_mid_run_state_really_differs_from_start(tmp_path, data):
    """Guard against a vacuous pass: the step-20 checkpoint must differ from init and from
    step 40 in weights, optimizer moments and loader position."""
    cfg = write_run(str(tmp_path / "r"), data)
    assert train.main([cfg]) == 0
    out = str(tmp_path / "r" / "out")
    c10 = runio.load_checkpoint(os.path.join(out, "ckpt_00000010.pt"))
    c20 = runio.load_checkpoint(os.path.join(out, "ckpt_00000020.pt"))
    c40 = runio.load_checkpoint(runio.latest_checkpoint(out))
    for x, y in ((c10, c20), (c20, c40)):
        assert state_equal(x["model"], y["model"]) != []
        assert state_equal(x["optimizer"]["state"], y["optimizer"]["state"]) != []
        assert x["data_state"] != y["data_state"]
