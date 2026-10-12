"""S005's micro 8 x accum 2 row-order loader test (S005 notes ENGINE: "Either way the step is 16 rows
(auto_micro_on_oom stays false); a loader test checks whether a step's rows match micro 16's in order"). No model
above the tiny CPU shape; the data are synthetic (harness make_fake_data) in S005's own data block.
  1 Every S005 config (its own BASE and the forget arm) runs micro 8 x accum 2 = 16 rows per step, the shared BASE's
    16 x 1, with auto_micro_on_oom false, the mask engine and the shared BASE's data block.
  2 The harness loader on that data block at full length (seq_len 2048, window_tokens 131072, max_item_len 2048,
    pack): at seeds 1, 101 and 102, the two micro-8 batches of each step, concatenated in order, equal the one
    micro-16 batch row for row (idx, tgt, doc, pos), with equal loader state after every step, across window and
    epoch boundaries.
  3 Through harness train.main (tiny shape, CPU, fp32): the rows the forward sees in each step, in order, at
    micro 8 x 2 equal those at micro 16 x 1, and the step's loss and the weights after 3 steps agree to float precision.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_s005_row_order.py
"""
import copy
import json
import os
import shutil
import sys

import pytest
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_lib as L  # noqa: E402

import runio  # noqa: E402
import train  # noqa: E402
from data import build_loader  # noqa: E402
from make_fake_data import make  # noqa: E402

S005 = sorted(p for p in L.all_configs() if os.sep + "S005_forget_gate" + os.sep in p)
KEYS = ("idx", "tgt", "doc", "pos")
TINY = {"vocab_size": 264, "d_model": 48, "n_layers": 2, "n_heads": 3, "n_kv_heads": 3, "head_dim": 16,
        "mlp_hidden": 64, "seq_len": 64}


def test_s005_step_is_16_rows_at_micro_8_accum_2():
    from test_screens_configs import stage2_seed_configs      # S005's seed configs once plans/stage2_seeds.txt exists
    seeds = {n for n in stage2_seed_configs() if n.startswith("s005_")}
    assert {os.path.basename(p)[:-5] for p in S005} == {"s005_base_s101", "s005_base_s102", "s005_forget_g0.5_s1",
                                                         "s005_forget_g1_s1", "s005_forget_g2_s1"} | seeds
    assert L.engine("S005") == {"train.doc_attn": "mask", "train.micro_batch": 8, "train.grad_accum": 2}
    shared = L.resolve(L.find("base_s101")[0])
    assert (shared["train"]["micro_batch"], shared["train"]["grad_accum"]) == (16, 1)
    for p in S005:
        c = L.resolve(p)
        t = c["train"]
        assert (t["micro_batch"], t["grad_accum"], t["auto_micro_on_oom"], t["doc_attn"]) == (8, 2, False, "mask"), p
        assert t["micro_batch"] * t["grad_accum"] == shared["train"]["micro_batch"] * shared["train"]["grad_accum"]
        assert c["data"] == shared["data"] and c["model"]["seq_len"] == shared["model"]["seq_len"] == 2048, p


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """Full-size synthetic shards (ids up to 8191; documents up to 3,000 tokens, cut at max_item_len)."""
    d = str(tmp_path_factory.mktemp("s005fake"))
    make(d, vocab=8192, docs=150, chats=200, doc_len=(5, 3000), chat_turn_len=(2, 200))
    return d


@pytest.fixture(scope="module")
def fake_tiny(tmp_path_factory):
    """Synthetic shards for the tiny model's 264-id vocabulary (the screens' 2-step smoke uses the same)."""
    d = str(tmp_path_factory.mktemp("s005tiny"))
    make(d, docs=200, chats=150)
    return d


def s005_data(fake: str, **over) -> dict:
    """S005's data block (resolved from its forget config) with only each source's paths swapped for synthetic
    shards of the same kind; every other key (mode, window_tokens, max_item_len, shares, chat ids) as written."""
    d = copy.deepcopy(L.resolve(L.find("s005_forget_g1_s1")[0])["data"])
    for s in d["sources"]:
        s["paths"] = [os.path.join(fake, "text_*.bin" if s["kind"] == "tokens" else "chat_*.jsonl")]
    d.update(over)
    return d


@pytest.mark.parametrize("seed", [1, 101, 102])
def test_loader_micro_8_x_2_rows_equal_micro_16_rows_in_order(fake, seed):
    d = s005_data(fake)
    assert (d["mode"], d["window_tokens"], d["max_item_len"], len(d["sources"])) == ("pack", 131072, 2048, 7)
    a, b = build_loader(d, 2048, 8, fake, seed), build_loader(d, 2048, 16, fake, seed)
    epochs = 0
    for step in range(24):
        parts, whole = [a.next_batch() for _ in range(2)], b.next_batch()
        for k in KEYS:
            got = torch.cat([x[k] for x in parts])
            assert got.shape == whole[k].shape == (16, 2048) and torch.equal(got, whole[k]), (seed, step, k)
        assert a.state_dict() == b.state_dict(), (seed, step)
        epochs = max(epochs, max(s.epoch for s in a.sources))
    assert a.state_dict()["window"] >= 4 and epochs >= 1     # several windows and a source epoch were crossed
    assert int((whole["doc"] > 0).sum()) > 0                  # packed rows hold more than one document


def tiny_run(cfg: dict, fake: str, work: str, micro: int, accum: int) -> str:
    c = copy.deepcopy(cfg)
    c["model"].update(TINY)
    c["data"] = s005_data(fake, window_tokens=1024, max_item_len=64)
    c["eval"].pop("induction")                               # the C6 hook is not under test here
    c["train"].update(device="cpu", precision="fp32", compile=False, micro_batch=micro, grad_accum=accum,
                      total_steps=3, ckpt_every=3, log_every=1)
    c["schedule"]["warmup_steps"] = 1
    c.update(out_dir=os.path.join(work, "out"), runs_jsonl=os.path.join(work, "runs.jsonl"))
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "config.yaml"), "w") as f:
        yaml.safe_dump(c, f)
    shutil.copy(os.path.join(os.path.dirname(L.find("s005_forget_g1_s1")[0]), "prereg.yaml"), work)
    return os.path.join(work, "config.yaml")


@pytest.mark.parametrize("name", ["s005_forget_g1_s1", "s005_base_s101"])
def test_trainer_steps_see_the_micro_16_rows_in_order(fake_tiny, tmp_path, monkeypatch, name):
    seen = []
    orig = train.compile_forward

    def recording(*a, **k):
        fwd = orig(*a, **k)

        def f(idx, tgt, doc, pos, **kw):
            seen[-1].append({"idx": idx.clone(), "tgt": tgt.clone(), "doc": doc.clone(), "pos": pos.clone()})
            return fwd(idx, tgt, doc, pos, **kw)
        return f
    monkeypatch.setattr(train, "compile_forward", recording)
    cfg, out = L.resolve(L.find(name)[0]), {}
    for micro, accum in ((8, 2), (16, 1)):
        seen.append([])
        work = str(tmp_path / f"m{micro}")
        assert train.main([tiny_run(cfg, fake_tiny, work, micro, accum)]) == 0
        log = [json.loads(x) for x in open(os.path.join(work, "out", "log.jsonl")) if '"loss"' in x]
        ck = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(work, "out")))
        out[micro] = {"log": log, "model": ck["model"], "cfg": ck["model_cfg"]}
    m8, m16 = seen
    assert len(m8) == 6 and len(m16) == 3 and all(x["idx"].shape == (8, 64) for x in m8)
    for step in range(3):
        for k in KEYS:
            assert torch.equal(torch.cat([m8[2 * step][k], m8[2 * step + 1][k]]), m16[step][k]), (name, step, k)
    assert out[8]["cfg"] == out[16]["cfg"] and out[8]["cfg"].get("forget_gate", False) == ("forget" in name)
    assert [r["step"] for r in out[8]["log"]] == [r["step"] for r in out[16]["log"]] == [1, 2, 3]
    for r8, r16 in zip(out[8]["log"], out[16]["log"]):
        assert r8["loss"] == pytest.approx(r16["loss"], abs=2e-5)          # log.jsonl rounds to 5 decimals
        assert (r8["tokens"], r8["sup_tokens"]) == (r16["tokens"], r16["sup_tokens"])
    w8, w16 = out[8]["model"], out[16]["model"]
    assert all(torch.allclose(w8[k], w16[k], atol=2e-5, rtol=1e-4) for k in w8)
