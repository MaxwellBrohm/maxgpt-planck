"""End-to-end trainer tests on CPU with a ~40k-parameter model and synthetic data:
the prereg gate, a full run, exact resume, trunk -> decay branch, bucket mode."""
from __future__ import annotations

import json
import os

import pytest
import torch
import yaml

import runio
import train
from make_fake_data import SPECIAL, make

PREREG = {"id": "T-1", "hypothesis": "h", "metric": "m", "decision_rule": "d"}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    make(str(d), docs=200, chats=150)
    return str(d)


def write_run(dirpath, fake, prereg=PREREG, **over):
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "name": os.path.basename(dirpath), "seed": 0, "out_dir": "out", "runs_jsonl": "../runs.jsonl",
        "model": {"vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
                  "head_dim": 16, "mlp_hidden": 64, "seq_len": 64},
        "data": {"mode": "pack", "pad_id": 0, "window_tokens": 1024, "buckets": [16, 32, 64],
                 "chat": {"role_ids": SPECIAL["role_ids"], "end_id": SPECIAL["end_id"]},
                 "sources": [{"kind": "tokens", "paths": [f"{fake}/text_*.bin"], "eot_id": 1, "share": 0.7},
                             {"kind": "chat", "paths": [f"{fake}/chat_*.jsonl"], "share": 0.3}]},
        "optim": {"lr": 3e-3},
        "schedule": {"mode": "full", "warmup_frac": 0.1, "decay_frac": 0.2},
        "train": {"device": "cpu", "precision": "fp32", "micro_batch": 4, "grad_accum": 2,
                  "total_steps": 12, "log_every": 1, "ckpt_every": 4, "keep_last": 2},
    }
    for k, v in over.items():
        if isinstance(v, dict):
            cfg[k] = {**cfg.get(k, {}), **v}
        else:
            cfg[k] = v
    path = os.path.join(dirpath, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    if prereg is not None:
        with open(os.path.join(dirpath, "prereg.yaml"), "w") as f:
            yaml.safe_dump(prereg, f)
    return path


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def final_weights(run_dir):
    return runio.load_checkpoint(runio.latest_checkpoint(os.path.join(run_dir, "out")))["model"]


def test_refuses_without_prereg(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake, prereg=None)
    assert train.main([cfg]) == 2
    assert not os.path.exists(tmp_path / "r" / "out") and not os.path.exists(tmp_path / "runs.jsonl")


@pytest.mark.parametrize("drop", ["id", "hypothesis", "metric", "decision_rule"])
def test_refuses_incomplete_prereg(tmp_path, fake, drop):
    pr = {k: v for k, v in PREREG.items() if k != drop}
    assert train.main([write_run(str(tmp_path / "r"), fake, prereg=pr)]) == 2


def test_refuses_uncommitted_prereg(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake)       # tmp_path is not a git repo
    assert train.main([cfg, "--require-committed"]) == 2


def test_full_run_learns_and_logs(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake, train={"total_steps": 40, "log_every": 5})
    assert train.main([cfg]) == 0
    log = read_jsonl(tmp_path / "r" / "out" / "log.jsonl")
    assert log[-1]["step"] == 40 and log[-1]["loss"] < 0.8 * log[0]["loss"]
    runs = read_jsonl(tmp_path / "runs.jsonl")
    assert [r["event"] for r in runs] == ["start", "end"]
    assert runs[0]["prereg_sha256"] == runio.sha256_file(str(tmp_path / "r" / "prereg.yaml"))
    assert runs[0]["n_params"] < 100_000 and runs[1]["step"] == 40
    assert os.path.basename(runs[1]["checkpoint"]) == "final_00000040.pt"
    kept = sorted(f for f in os.listdir(tmp_path / "r" / "out") if f.startswith("ckpt_"))
    assert kept == ["ckpt_00000032.pt", "ckpt_00000036.pt"]    # keep_last 2


def test_resume_is_bitwise_exact(tmp_path, fake):
    straight = write_run(str(tmp_path / "a"), fake)
    assert train.main([straight]) == 0
    paused = write_run(str(tmp_path / "b"), fake)
    assert train.main([paused, "--max-steps", "5"]) == 0
    assert runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "b" / "out")))["step"] == 5
    assert train.main([paused]) == 0
    wa, wb = final_weights(str(tmp_path / "a")), final_weights(str(tmp_path / "b"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    ev = [r["event"] for r in read_jsonl(tmp_path / "runs.jsonl")]
    assert ev == ["start", "end", "start", "pause", "start", "end"]


def test_resume_refuses_changed_schedule(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake)
    assert train.main([cfg, "--max-steps", "4"]) == 0
    cfg = write_run(str(tmp_path / "r"), fake, schedule={"decay_frac": 0.5})
    with pytest.raises(AssertionError, match="schedule"):
        train.main([cfg])


def test_trunk_then_decay_branch(tmp_path, fake):
    trunk = write_run(str(tmp_path / "trunk"), fake,
                      schedule={"mode": "trunk", "warmup_steps": 2, "branch_points": [8]},
                      train={"total_steps": 16})
    assert train.main([trunk]) == 0
    stable = tmp_path / "trunk" / "out" / "stable_00000008.pt"
    ck = runio.load_checkpoint(str(stable))
    assert ck["step"] == 8 and ck["phase"] == "stable"
    br = write_run(str(tmp_path / "br"), fake,
                   schedule={"mode": "branch", "init_from": str(stable), "decay_frac": 0.5})
    assert train.main([br]) == 0
    fin = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "br" / "out")))
    assert fin["step"] == 16 and fin["schedule"]["decay_start"] == 8
    log = read_jsonl(tmp_path / "br" / "out" / "log.jsonl")
    assert [r["step"] for r in log] == list(range(9, 17))            # continues at step 8
    assert log[0]["lr_factor"] == 1.0 and log[-1]["lr_factor"] == pytest.approx(1 / 8)
    trunk_log = read_jsonl(tmp_path / "trunk" / "out" / "log.jsonl")
    assert all(r["lr_factor"] == 1.0 for r in trunk_log if r["step"] > 2)
    # the branch's first step sees the same data as the trunk's step 9
    assert log[0]["tokens"] == trunk_log[8]["tokens"]


def test_branch_refuses_decay_phase_checkpoint(tmp_path, fake):
    full = write_run(str(tmp_path / "full"), fake)
    assert train.main([full]) == 0
    last = runio.latest_checkpoint(str(tmp_path / "full" / "out"))    # final, step 12, decayed
    br = write_run(str(tmp_path / "br"), fake, schedule={"mode": "branch", "init_from": last})
    with pytest.raises(AssertionError, match="stable"):
        train.main([br])


def test_bucket_mode_and_looped_arm_train(tmp_path, fake):
    cfg = write_run(str(tmp_path / "r"), fake,
                    data={"mode": "bucket", "tokens_per_micro": 256},
                    model={"vocab_size": 256, "d_model": 32, "n_layers": 1, "n_heads": 2,
                           "n_kv_heads": 1, "head_dim": 16, "mlp_hidden": 64, "seq_len": 64,
                           "n_loops": 3, "qk_share": 1},
                    train={"total_steps": 60, "log_every": 10})
    assert train.main([cfg]) == 0
    log = read_jsonl(tmp_path / "r" / "out" / "log.jsonl")
    assert log[-1]["loss"] < 0.8 * log[0]["loss"]


def test_grad_accum_matches_one_big_batch(tmp_path, fake):
    """Token-normalized loss: 2 micro-batches of 4 rows == 1 micro-batch of 8 rows (the
    packed row stream is the same), so the weights agree to float precision."""
    a = write_run(str(tmp_path / "a"), fake, train={"micro_batch": 4, "grad_accum": 2, "total_steps": 6})
    b = write_run(str(tmp_path / "b"), fake, train={"micro_batch": 8, "grad_accum": 1, "total_steps": 6})
    assert train.main([a]) == 0 and train.main([b]) == 0
    wa, wb = final_weights(str(tmp_path / "a")), final_weights(str(tmp_path / "b"))
    assert all(torch.allclose(wa[k], wb[k], atol=2e-5, rtol=1e-4) for k in wa)


def test_bf16_autocast_path_runs_on_cpu(tmp_path, fake):
    """The MPS/CUDA precision path (bf16 autocast, fp32 master weights), exercised on CPU."""
    cfg = write_run(str(tmp_path / "r"), fake, train={"precision": "bf16", "total_steps": 20, "log_every": 5})
    assert train.main([cfg]) == 0
    runs = read_jsonl(tmp_path / "runs.jsonl")
    assert runs[0]["precision"] == "bf16"
    ck = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "r" / "out")))
    assert all(v.dtype == torch.float32 for v in ck["model"].values())
    log = read_jsonl(tmp_path / "r" / "out" / "log.jsonl")
    assert log[-1]["loss"] < log[0]["loss"]
