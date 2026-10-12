"""The trainer's RC-12 eval hook (rc12_eval.py) and the runner's planck: responder on a REAL harness checkpoint.

Two 20-step runs of train.py on the fake shards, identical except that one has eval.rc12 (every 10 steps, T0 +
one BIND pair, greedy and seed 1). The hook must log both evals, write transcripts, and leave training bitwise
unchanged (weights, optimizer, loader state, torch RNG, per-step log). The final checkpoint then goes through
rc12/runner.py --responder planck:... in a subprocess and must reproduce the hook's greedy replies at step 20.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import rc12_eval as E
import runio
import toy_tokenizer as TT
import train
from make_fake_data import make
from testutil import read_jsonl, write_run

EVAL = {"every": 10, "per_family": 1, "families": ["T0", "BIND"], "seeds": ["greedy", 1]}


def run_dir(root, name, data, **over):
    return write_run(str(root / name), str(data), model={"vocab_size": TT.VOCAB, "seq_len": 512},
                     data={"tokenizer": "../tok.json"}, train={"total_steps": 20, "ckpt_every": 10}, **over)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("hook")
    make(str(root / "data"), vocab=TT.VOCAB, docs=120, chats=60)
    TT.write(str(root / "tok.json"))
    on = run_dir(root, "on", root / "data", eval={"rc12": EVAL})
    off = run_dir(root, "off", root / "data")
    assert train.main([on, "--device", "cpu"]) == 0
    assert train.main([off, "--device", "cpu"]) == 0
    return dict(root=root, on=os.path.dirname(on), off=os.path.dirname(off), on_cfg=on)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def test_hook_off_by_default():
    assert E.make_hook({}, ".", ".", "cpu", None, 64) is None
    assert E.make_hook({"eval": {"rc12": {"every": 0}}}, ".", ".", "cpu", None, 64) is None


def test_hook_logs_each_eval(runs):
    log = read_jsonl(os.path.join(runs["on"], "out", "rc12_eval.jsonl"))
    assert [x["step"] for x in log] == [10, 20] and not any("error" in x for x in log)
    for x in log:
        assert x["n_conv"] == 6 and {"greedy", "sampled"} <= set(x)
        assert x["greedy"]["R"] is None and x["greedy"]["families"]["BIND"] is not None
        assert x["greedy"]["t0"] is not None and 0 <= x["sampled"]["loop_rate"] <= 1
        rows = read_jsonl(os.path.join(runs["on"], "out", "rc12", f"step_{x['step']:08d}", "transcripts.jsonl"))
        assert len(rows) == 6 and {r["seed"] for r in rows} == {None, 1}
        assert {r["id"] for r in rows} == {"rc12-dev-t0-001", "rc12-dev-bind-001a", "rc12-dev-bind-001b"}
    assert not os.path.exists(os.path.join(runs["off"], "out", "rc12_eval.jsonl"))
    assert not os.path.exists(os.path.join(runs["off"], "out", "rc12"))


def test_hook_leaves_training_bitwise_unchanged(runs):
    a = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(runs["on"], "out")))
    b = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(runs["off"], "out")))
    assert a["step"] == b["step"] == 20
    for k in ("model", "optimizer", "data_state", "rng_torch", "tokens", "sup_tokens"):
        assert same(a[k], b[k]), k
    drop = lambda r: {k: v for k, v in r.items() if k not in ("time", "tok_per_s")}   # noqa: E731
    la = [drop(r) for r in read_jsonl(os.path.join(runs["on"], "out", "log.jsonl"))]
    lb = [drop(r) for r in read_jsonl(os.path.join(runs["off"], "out", "log.jsonl"))]
    assert la == lb and len(la) == 20


def test_hook_error_is_logged_not_raised(runs, tmp_path):
    cfg = runio.load_yaml(runs["on_cfg"])
    hook = E.make_hook(cfg, runs["on"], str(tmp_path), "cpu", None, 512)

    def boom(model, step):
        raise RuntimeError("eval exploded")
    hook.evaluate = boom
    trainer = type("T", (), {"step": 30, "model": None})()
    assert hook(trainer)["error"] == "RuntimeError: eval exploded"
    assert read_jsonl(str(tmp_path / "rc12_eval.jsonl"))[0]["step"] == 30
    trainer.step = 31
    assert hook(trainer) is None


def cli(runs, out, *extra):
    ck = runio.latest_checkpoint(os.path.join(runs["on"], "out"))
    args = [sys.executable, "-B", "runner.py", "--responder", f"planck:{ck}", "--seeds", "greedy",
            "--families", "T0", "--limit", "1", "--out", str(out), *extra]
    return subprocess.run(args, cwd=E.RC12, capture_output=True, text=True, timeout=100)


def test_runner_cli_reproduces_the_hook(runs, tmp_path):
    r = cli(runs, tmp_path / "cli", "--planck-config", runs["on_cfg"])
    assert r.returncode == 0, r.stderr[-800:]
    got = read_jsonl(str(tmp_path / "cli" / "transcripts.jsonl"))
    hook = [x for x in read_jsonl(os.path.join(runs["on"], "out", "rc12", "step_00000020", "transcripts.jsonl"))
            if x["id"] == "rc12-dev-t0-001" and x["seed"] is None]
    assert len(got) == 1 and len(hook) == 1
    assert [(t["reply"], t["stop"], t["dropped"]) for t in got[0]["turns"]] == \
        [(t["reply"], t["stop"], t["dropped"]) for t in hook[0]["turns"]]
    tok = runs["root"] / "tok.json"
    r2 = cli(runs, tmp_path / "cli2", "--planck-tokenizer", str(tok))
    assert r2.returncode == 0 and read_jsonl(str(tmp_path / "cli2" / "transcripts.jsonl"))[0]["turns"] == \
        got[0]["turns"]


def test_checkpoint_config_must_match(runs, tmp_path):
    E.use_rc12()
    import planck_responder as PR
    ck = runio.latest_checkpoint(os.path.join(runs["on"], "out"))
    other = write_run(str(tmp_path / "other"), str(runs["root"] / "data"), model={"vocab_size": TT.VOCAB,
                      "seq_len": 512, "d_model": 48}, data={"tokenizer": str(runs["root"] / "tok.json")})
    with pytest.raises(ValueError, match="different models"):
        PR.from_checkpoint(ck, config=other)
    r = PR.from_checkpoint(ck, config=runs["on_cfg"])
    assert (r.ctx, r.step, r.tmpl.end_id, sorted(r.eos_ids)) == (512, 20, 5, [1])


def test_runner_cli_refusals(runs, tmp_path):
    r = cli(runs, tmp_path / "x")                    # relative tokenizer path and no config: refused
    assert r.returncode != 0 and "relative" in r.stderr
    r = cli(runs, tmp_path / "y", "--planck-config", runs["on_cfg"], "--ctx", "4096")
    assert r.returncode != 0 and "exceeds the checkpoint's seq_len 512" in r.stderr
    assert not os.path.exists(tmp_path / "y" / "transcripts.jsonl")
