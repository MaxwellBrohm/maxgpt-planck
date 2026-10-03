"""eval.induction (induction.py; SCREENS C6 IND and the GATE curve). CPU, tiny models, the toy tokenizer.

Fixtures: a uniform model (all logits 0) scores IND 0 exactly; a hand-built copy model (it predicts the token that
followed the last earlier occurrence of the current token) scores near ln V. Two 12-step train.py runs on the fake
shards, identical except that one has eval.induction (every 5, GATE on 4 windows of a hand-made chat eval set):
the hook must log steps 5, 10 and 12 (the last) and leave training bitwise unchanged (weights, optimizer, loader
state, torch RNG, every log record). The last reading equals ind_score on the final checkpoint's plain model, also
on a train.mtp run (IND reads the next-token logits, never S006's aux head). A changed file is refused.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import pytest
import torch

import induction as I
import runio
import toy_tokenizer as TT
import train
from config import PlanckConfig
from make_fake_data import make
from model import build_model
from testutil import read_jsonl, write_run

DP = os.environ.get("PLANCK_DATA_PREP") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "data_prep")
if DP not in sys.path:
    sys.path.insert(0, DP)
import evalwin as EW  # noqa: E402

WORDS = "the dog Pearl likes long walks near the river while it rains and the cat sleeps all day".split()


def write_seqs(path, n=8, L=16, vocab=TT.VOCAB, seed=0):
    g = torch.Generator().manual_seed(seed)          # no repeat inside a sequence: the copy model is then exact
    seqs = [(8 + torch.randperm(vocab - 8, generator=g)[:L]).tolist() for _ in range(n)]
    raw = json.dumps({"seqs": seqs}).encode()
    open(path, "wb").write(raw)
    return hashlib.sha256(raw).hexdigest()


def chat_evalset(d, n_docs=6):
    """A one-set chat eval set in eval_sets.py's layout (manifest + docs + windows; evalwin's cut rule)."""
    os.makedirs(d, exist_ok=True)
    docs, wins = [], []
    for i in range(n_docs):
        turns = [{"role": "user" if j % 2 == 0 else "assistant",
                  "text": " ".join(WORDS[(i + j + k) % len(WORDS)] for k in range(5 + 3 * j))} for j in range(4)]
        docs.append({"id": f"c{i}", "turns": turns})
        for k, w in enumerate(EW.chat_windows(turns, 2048, 3072)):
            wins.append({**w, "d": i, "k": k, "h": EW.window_hash("oasst2", f"c{i}", k)})
    files = {}
    for kind, recs in (("docs", docs), ("windows", wins)):
        p = os.path.join(d, f"oasst2.{kind}.jsonl")
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in recs))
        files[kind] = {"path": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    man = {"evalset_sha256": "test", "config": {}, "sets": {"oasst2": {"kind": "chat", "files": files}}}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))


class CopyModel(torch.nn.Module):
    """logits[t] = +40 on the token that followed the last earlier occurrence of idx[t]; else all 0."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, idx):
        B, T = idx.shape
        out = torch.zeros(B, T, self.vocab)
        for b in range(B):
            last = {}
            for t in range(T):
                tok = int(idx[b, t])
                if tok in last and last[tok] + 1 <= t:
                    out[b, t, int(idx[b, last[tok] + 1])] = 40.0
                last[tok] = t
        return out, None


def test_uniform_model_scores_zero(tmp_path):
    m = build_model(PlanckConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
                                 mlp_hidden=64, seq_len=64))
    with torch.no_grad():
        m.tok_emb.weight.zero_()                     # tied head: every logit is exactly 0
    write_seqs(tmp_path / "s.json", vocab=97)
    r = I.ind_score(m, I.load_seqs(str(tmp_path / "s.json"), None), "cpu")
    assert r["ind"] == 0.0 and abs(r["nll_copy1"] - math.log(97)) < 1e-6


def test_copy_model_scores_near_ln_v(tmp_path):
    write_seqs(tmp_path / "s.json", n=16, L=32, vocab=500, seed=1)
    seqs = I.load_seqs(str(tmp_path / "s.json"), None)
    r = I.ind_score(CopyModel(500), seqs, "cpu", rows_per_batch=5)
    assert abs(r["nll_copy2"]) < 1e-6 and r["ind"] > 0.95 * math.log(500), r


def test_off_by_default_and_file_pinned(tmp_path):
    assert I.make_hook({}, ".", ".", "cpu", None) is None
    assert I.make_hook({"eval": {"induction": {"every": 0}}}, ".", ".", "cpu", None) is None
    write_seqs(tmp_path / "s.json")
    with pytest.raises(ValueError, match="sha256"):
        I.load_seqs(str(tmp_path / "s.json"), "0" * 64)


def run_dir(root, name, model=None, train=None, **over):
    return write_run(str(root / name), str(root / "data"), model={"vocab_size": TT.VOCAB, "seq_len": 256, **(model or {})},
                     train={"total_steps": 12, "ckpt_every": 6, **(train or {})}, **over)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ind")
    make(str(root / "data"), vocab=TT.VOCAB, docs=120, chats=60)
    TT.write(str(root / "tok.json"))
    sha = write_seqs(root / "seqs.json")
    chat_evalset(str(root / "ev"))
    ind = {"every": 5, "file": "../seqs.json", "sha256": sha, "gate_evalset": "../ev", "gate_set": "oasst2",
           "gate_windows": 4, "gate_tokenizer": "../tok.json"}
    paths = {"on": run_dir(root, "on", eval={"induction": ind}), "off": run_dir(root, "off"),
             "nogate": run_dir(root, "nogate", eval={"induction": ind}, model={"attn_gate": False}),
             "mtp": run_dir(root, "mtp", eval={"induction": ind}, train={"mtp": 1})}
    for p in paths.values():
        assert train.main([p, "--device", "cpu"]) == 0
    return {k: os.path.join(os.path.dirname(p), "out") for k, p in paths.items()}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def test_hook_leaves_training_bitwise_unchanged(runs):
    a = runio.load_checkpoint(runio.latest_checkpoint(runs["on"]))
    b = runio.load_checkpoint(runio.latest_checkpoint(runs["off"]))
    assert a["step"] == b["step"] == 12
    for k in ("model", "optimizer", "data_state", "rng_torch", "tokens", "sup_tokens"):
        assert same(a[k], b[k]), k
    drop = lambda r: {k: v for k, v in r.items() if k not in ("time", "tok_per_s")}   # noqa: E731
    assert [drop(r) for r in read_jsonl(os.path.join(runs["on"], "log.jsonl"))] == \
        [drop(r) for r in read_jsonl(os.path.join(runs["off"], "log.jsonl"))]
    assert not os.path.exists(os.path.join(runs["off"], "diag.jsonl"))


def test_readings_at_every_and_last_step_with_gate(runs):
    log = read_jsonl(os.path.join(runs["on"], "diag.jsonl"))
    assert [r["step"] for r in log] == [5, 10, 12] and not any("error" in r for r in log)
    for r in log:
        assert len(r["gate"]) == 2 and all(len(h) == 2 and all(0 < g < 2 for g in h) for h in r["gate"])
        assert math.isfinite(r["ind"]) and r["ind"] == r["nll_copy1"] - r["nll_copy2"]
    off = read_jsonl(os.path.join(runs["nogate"], "diag.jsonl"))
    assert [r["step"] for r in off] == [5, 10, 12] and all(r["gate"] is None for r in off)
    starts = {r["run"]: r for r in read_jsonl(os.path.join(os.path.dirname(os.path.dirname(runs["on"])), "runs.jsonl"))
              if r["event"] == "start"}
    assert starts["on"]["induction"]["gate"] is True and starts["nogate"]["induction"]["gate"] is False
    assert "induction" not in starts["off"] and starts["on"]["induction"]["every"] == 5


@pytest.mark.parametrize("which", ["on", "mtp"])
def test_last_reading_is_the_plain_models_next_token_ind(runs, which):
    ck = runio.load_checkpoint(runio.latest_checkpoint(runs[which]))
    m = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    cfg = runio.load_yaml(os.path.join(os.path.dirname(runs[which]), "config.yaml"))
    seqs = I.load_seqs(os.path.join(os.path.dirname(runs[which]), cfg["eval"]["induction"]["file"]), None)
    want = I.ind_score(m, seqs, "cpu")
    got = read_jsonl(os.path.join(runs[which], "diag.jsonl"))[-1]
    assert got["step"] == 12 and abs(got["ind"] - want["ind"]) < 1e-9 and ("mtp" in ck) == (which == "mtp")


def test_a_reading_that_draws_from_the_rng_stops_the_run(tmp_path, monkeypatch):
    sha = write_seqs(tmp_path / "s.json")
    m = build_model(PlanckConfig(vocab_size=TT.VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
                                 mlp_hidden=64, seq_len=64))
    hook = I.make_hook({"eval": {"induction": {"every": 1, "file": "s.json", "sha256": sha}}}, str(tmp_path),
                       str(tmp_path), "cpu", m)

    class T:
        step, model, sched = 1, m, type("S", (), {"total_steps": 9})()
    assert hook(T())["step"] == 1
    real = I.ind_score
    monkeypatch.setattr(I, "ind_score", lambda *a, **k: (torch.rand(1), real(*a, **k))[1])
    with pytest.raises(RuntimeError, match="RNG"):
        hook(T())
