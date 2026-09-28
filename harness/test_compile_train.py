"""train.compile through train.main on CPU (backend aot_eager; unit checks: test_compile.py).

Off by default (no Dynamo graph, no start-record field); on: resume 20 + 20 == 40 bitwise in pack
and bucket mode with the plain module's checkpoint keys and the eager run's optimizer groups;
switched on then off across a resume: loads cleanly, lands within float tolerance of 40 eager
steps, and each start record says what ran. Follow-ups (notes.txt SPEED V3 FOLLOW-UPS): the start
record carries compile_dynamic; train.main refuses compile_dynamic false with varlen and the
unvalidated mode max-autotune-no-cudagraphs.
"""
from __future__ import annotations

import os

import pytest
import torch

import runio
import train
from config import PlanckConfig
from make_fake_data import make
from model import build_model
from train import check_compile
from test_s3_resume import state_equal
from testutil import read_jsonl, write_run


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("compile_data")
    make(str(d), docs=150, chats=120)
    return str(d)


COMPILED = {"compile": "default", "compile_backend": "aot_eager"}


def last_ck(run_dir) -> dict:
    return runio.load_checkpoint(runio.latest_checkpoint(os.path.join(str(run_dir), "out")))


def groups(ck: dict) -> list:
    """The optimizer's param-group layout (name and parameter count per group)."""
    return [(g["name"], len(g["params"])) for g in ck["optimizer"]["param_groups"]]


def test_compile_is_off_by_default(tmp_path, data):
    """No train.compile key: the trainer runs the plain module (no Dynamo graph at all) and
    the start record has no compile field, i.e. the reference path is unchanged."""
    run = write_run(str(tmp_path / "run"), data)
    assert train.main([run, "--max-steps", "3"]) == 0
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 0
    starts = [r for r in read_jsonl(tmp_path / "runs.jsonl") if r["event"] == "start"]
    assert starts and all(not {"compile", "compile_backend", "compile_dynamic"} & set(r) for r in starts)


@pytest.mark.parametrize("mode", ["pack", "bucket"])
def test_resume_with_compile_20_plus_20_equals_40(tmp_path, data, mode):
    """test_s3_resume with train.compile on: bitwise weights, optimizer, loader and logged
    losses; checkpoints hold the plain module's keys (no _orig_mod.) and the optimizer groups
    an eager run has (an optimizer built on the wrapper would put the tied embedding, named
    _orig_mod.tok_emb.weight, into NorMuon)."""
    over = {"train": COMPILED}
    if mode == "bucket":      # varying (B, T) per bucket: recompiles, dynamic shapes
        over["data"] = {"mode": "bucket", "tokens_per_micro": 256}
    straight = write_run(str(tmp_path / "straight"), data, **over)
    assert train.main([straight]) == 0
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([split, "--max-steps", "20"]) == 0
    torch._dynamo.reset()
    assert train.main([split]) == 0
    a, b = last_ck(tmp_path / "straight"), last_ck(tmp_path / "split")
    assert a["step"] == b["step"] == 40
    for k in ("model", "optimizer", "data_state"):
        assert state_equal(a[k], b[k]) == [], k
    plain = build_model(PlanckConfig.from_dict(a["model_cfg"]), "meta").state_dict()
    assert list(a["model"]) == list(plain)                       # no _orig_mod. prefix
    eager = write_run(str(tmp_path / "eager"), data, **({"data": over["data"]} if "data" in over else {}))
    assert train.main([eager, "--max-steps", "1"]) == 0
    assert groups(a) == groups(last_ck(tmp_path / "eager"))
    la = read_jsonl(tmp_path / "straight" / "out" / "log.jsonl")
    lb = read_jsonl(tmp_path / "split" / "out" / "log.jsonl")
    keys = ("step", "loss", "gnorm", "tokens", "sup_tokens")
    assert [tuple(r[k] for k in keys) for r in la] == [tuple(r[k] for k in keys) for r in lb]
    runs = read_jsonl(tmp_path / "runs.jsonl")
    starts = [r for r in runs if r["event"] == "start" and r["run"] != "eager"]
    assert len(starts) == 3 and all(r.get("compile") == "default" for r in starts)
    assert all(r.get("compile_backend") == "aot_eager" for r in starts)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] >= 1   # the trainer used it


def test_compiled_checkpoint_resumes_eager_and_back(tmp_path, data):
    """Toggling train.compile across a resume: 20 compiled + 20 eager steps load cleanly and
    land within float tolerance of 40 eager steps (aot_eager is not bitwise eager)."""
    eager = write_run(str(tmp_path / "eager"), data)
    assert train.main([eager]) == 0
    mixed = write_run(str(tmp_path / "mixed"), data, train=COMPILED)
    assert train.main([mixed, "--max-steps", "20"]) == 0
    write_run(str(tmp_path / "mixed"), data)                     # same run dir, compile off
    assert train.main([os.path.join(tmp_path, "mixed", "config.yaml")]) == 0
    a, b = last_ck(tmp_path / "eager"), last_ck(tmp_path / "mixed")
    assert b["step"] == 40 and set(a["model"]) == set(b["model"])
    assert groups(a) == groups(b)
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], rtol=1e-3, atol=1e-4), k
    st = [r for r in read_jsonl(tmp_path / "runs.jsonl") if r["event"] == "start" and r["run"] == "mixed"]
    assert [r.get("compile") for r in st] == ["default", None]
    assert st[1]["resumed_from"] and st[1]["step"] == 20


def test_compile_dynamic_is_recorded(tmp_path, data):
    """A compiled run records train.compile_dynamic in its start record, null when the key is absent
    (Dynamo's automatic dynamic shapes); false is allowed with the mask engine (CPU) and recorded."""
    for name, extra in (("auto", {}), ("static", {"compile_dynamic": False})):
        run = write_run(str(tmp_path / name), data, train={**COMPILED, **extra})
        assert train.main([run, "--max-steps", "1"]) == 0
        torch._dynamo.reset()
    st = {r["run"]: r for r in read_jsonl(tmp_path / "runs.jsonl") if r["event"] == "start"}
    assert "compile_dynamic" in st["auto"] and st["auto"]["compile_dynamic"] is None
    assert st["static"]["compile_dynamic"] is False and st["static"]["compile"] == "default"


def test_compile_dynamic_false_refused_with_varlen(tmp_path, data, monkeypatch):
    """compile_dynamic false with doc_attn varlen recompiles per document count and falls back to eager
    at Dynamo's limit (notes.txt SPEED V3 FOLLOW-UPS), so train.main refuses it before any work. varlen
    needs CUDA, so the doc_attn resolution is stubbed: the refusal comes before any forward."""
    monkeypatch.setattr(train, "resolve_doc_attn", lambda *a: "varlen")
    monkeypatch.setattr(train, "set_doc_attn", lambda m, impl, device=None: setattr(m, "doc_attn", impl))
    run = write_run(str(tmp_path / "run"), data, train={**COMPILED, "compile_dynamic": False, "selftest": False})
    with pytest.raises(ValueError, match="compile_dynamic false with doc_attn varlen"):
        train.main([run, "--max-steps", "1"])
    for ok in ((None, False, "varlen"), ("default", None, "varlen"), ("default", True, "varlen"),
               ("default", False, "mask")):
        check_compile(*ok)                                   # eager, unset, true, or the mask engine


def test_train_refuses_the_unvalidated_compile_mode(tmp_path, data):
    """max-autotune-no-cudagraphs failed one 20M parity check on the old tree and was never re-proved:
    model.compile_mode accepts it (a pc/bench_micro.py arm), train.main refuses it before any work;
    "default" (the validated mode) runs (test_compile_dynamic_is_recorded and the resume tests)."""
    run = write_run(str(tmp_path / "run"), data,
                    train={**COMPILED, "compile": "max-autotune-no-cudagraphs"})
    with pytest.raises(ValueError, match="not validated for training"):
        train.main([run, "--max-steps", "1"])
    assert not os.path.exists(tmp_path / "runs.jsonl")                 # refused before the start record
    check_compile("default", None, "varlen")
    with pytest.raises(ValueError, match="bench_micro"):
        check_compile("max-autotune-no-cudagraphs", None, "mask")
