"""SCREENS ORDER 0 code on CUDA (skipped without it; the PC runs these, the Mac skips them).

  - attn_diag's rebuilt attention equals each Attention module's own output on CUDA, fp32, for every testutil arm
    (SDPA picks other kernels on CUDA than on the CPU, so the check is redone where the SINK readings are made).
  - ind_score on CUDA (fp32) equals the CPU value on the same model and sequences.
  - A CUDA bf16 run (varlen doc_attn, batched optimizer: the screens' engine) with eval.induction on equals the
    same run with it off bitwise under deterministic algorithms: final weights, optimizer, loader state, torch and
    CUDA RNG states and every log record (each run in its own process, as default_parity.py runs them).
"""
from __future__ import annotations

import os

import pytest
import torch

import attn_diag as D
import induction as I
import runio
import testutil as U
import toy_tokenizer as TT
from default_parity import run_tree, same
from make_fake_data import make
from model import build_model
from test_induction import chat_evalset, write_seqs
from testutil import read_jsonl, write_run

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("arm", sorted(U.ARMS))
def test_cuda_rebuild_equals_the_module(arm):
    m = U.scramble(build_model(U.tiny(**U.ARMS[arm])), seed=1).to("cuda")
    g = torch.Generator().manual_seed(3)
    rows = [torch.randint(8, 97, (n,), generator=g).tolist() for n in (64, 40, 17, 64)]
    r = D.read(m, rows, "cuda", batch_tokens=128)
    assert r["check_ok"] and r["check_max_rel_err"] < 1e-4 and r["layers"] == m.cfg.depth, r["check_max_rel_err"]


def test_cuda_ind_equals_cpu(tmp_path):
    write_seqs(tmp_path / "s.json", n=16, L=32, vocab=97)
    seqs = I.load_seqs(str(tmp_path / "s.json"), None)
    m = U.scramble(build_model(U.tiny()), seed=2)
    a = I.ind_score(m, seqs, "cpu")
    b = I.ind_score(m.to("cuda"), seqs, "cuda")
    assert abs(a["ind"] - b["ind"]) < 1e-4 and abs(a["nll_copy1"] - b["nll_copy1"]) < 1e-4


def test_cuda_hook_on_equals_off_bitwise(tmp_path):
    make(str(tmp_path / "data"), vocab=TT.VOCAB, docs=300, chats=300)
    TT.write(str(tmp_path / "tok.json"))
    chat_evalset(str(tmp_path / "ev"))
    sha = write_seqs(tmp_path / "seqs.json")
    ind = {"every": 5, "file": str(tmp_path / "seqs.json"), "sha256": sha, "gate_evalset": str(tmp_path / "ev"),
           "gate_set": "oasst2", "gate_windows": 4, "gate_tokenizer": str(tmp_path / "tok.json")}
    model = {"vocab_size": TT.VOCAB, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1, "head_dim": 32,
             "mlp_hidden": 128, "seq_len": 256}
    train = {"device": "cuda", "precision": "bf16", "micro_batch": 8, "grad_accum": 2, "total_steps": 24,
             "log_every": 1, "ckpt_every": 0, "doc_attn": "varlen"}
    out = {}
    for tag, extra in (("on", {"eval": {"induction": ind}}), ("off", {})):
        d = str(tmp_path / tag / "run")                # same run name: the two configs differ only in eval
        cfg = write_run(d, str(tmp_path / "data"), runs_jsonl="runs.jsonl", model=model, train=train, **extra)
        run_tree(HERE, cfg, os.path.join(d, "modules.json"))
        ck = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(d, "out")))
        out[tag] = {"ck": {k: ck[k] for k in ("model", "optimizer", "data_state", "rng_torch", "rng_cuda", "step")},
                    "log": [{k: v for k, v in r.items() if k not in ("tok_per_s", "time")}
                            for r in read_jsonl(os.path.join(d, "out", "log.jsonl"))],
                    "start": [r for r in read_jsonl(os.path.join(d, "runs.jsonl")) if r["event"] == "start"][-1],
                    "diag": os.path.join(d, "out", "diag.jsonl")}
    assert out["on"]["ck"]["step"] == 24 and len(out["on"]["log"]) == 24
    assert same(out["on"]["ck"], out["off"]["ck"]) == [] and same(out["on"]["log"], out["off"]["log"]) == []
    assert out["on"]["start"]["doc_attn"] == "varlen" and out["on"]["start"]["optim_batched"] is True
    diag = read_jsonl(out["on"]["diag"])
    assert [r["step"] for r in diag] == [5, 10, 15, 20, 24] and all(len(r["gate"]) == 2 for r in diag)
    assert not os.path.exists(out["off"]["diag"])
