"""SCREENS 2-step CPU smoke and init isolation (tiny models only; nothing loads a real-size model or a checkpoint
from a run). Every config on disk is resolved, then run for 2 steps by harness train.main with only these
substitutions: tiny shape (every block and screen flag kept), synthetic data (make_fake_data), cpu, doc_attn mask
(varlen needs CUDA), micro 2 (grad_accum kept), 2 steps, warmup 1, S003 trunk points [1] and its branches off the
tiny trunk, and the C6 eval.induction hook kept ON with tiny inputs (every step; 8 x 16 IND sequences and a 2-window
GATE set with the toy tokenizer, harness/test_induction.py's fixtures), so every arm runs the hook.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_smoke.py
"""
import copy
import json
import math
import os
import shutil
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_lib as L  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import runio  # noqa: E402
import train  # noqa: E402
from budget import count_analytic  # noqa: E402
from config import PlanckConfig  # noqa: E402
from make_fake_data import make  # noqa: E402
from model import build_model  # noqa: E402
import toy_tokenizer as TT  # noqa: E402
from test_induction import chat_evalset, write_seqs  # noqa: E402  (harness/test_induction.py's fixtures)

HOOK: dict = {}

CONFIGS = L.all_configs()
TINY = {"vocab_size": 264, "d_model": 48, "n_layers": 2, "n_heads": 3, "n_kv_heads": 3, "head_dim": 16,
        "mlp_hidden": 64, "seq_len": 64}


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    make(str(d), docs=200, chats=150)
    TT.write(str(d / "tok.json"))
    chat_evalset(str(d / "ev"))
    sha = write_seqs(d / "seqs.json")
    HOOK.update(every=1, file=str(d / "seqs.json"), sha256=sha, gate_evalset=str(d / "ev"), gate_set="oasst2",
                gate_windows=2, gate_tokenizer=str(d / "tok.json"))
    return str(d)


def tiny(cfg: dict, fake: str, work: str, trunk_out: str | None = None) -> str:
    c = copy.deepcopy(cfg)
    c["model"].update(TINY)
    c["data"].update(window_tokens=1024, max_item_len=64, sources=[
        {"name": "text", "kind": "tokens", "eot_id": 1, "share": 0.7, "paths": [f"{fake}/text_*.bin"]},
        {"name": "chat", "kind": "chat", "share": 0.3, "paths": [f"{fake}/chat_*.jsonl"]}])
    assert set(c["eval"]["induction"]) == set(HOOK)          # BASE's hook, every key replaced by a tiny input
    c["eval"]["induction"] = dict(HOOK)
    c["train"].update(device="cpu", compile=False, doc_attn="mask", micro_batch=2, total_steps=2, ckpt_every=1,
                      log_every=1)
    sc = c["schedule"]
    sc["warmup_steps"] = 1
    if sc["mode"] == "trunk":
        sc["branch_points"] = [1]
    if sc["mode"] == "branch":
        sc.update(init_from=os.path.join(trunk_out, "stable_00000001.pt"), decay_steps=1)
        c["train"]["total_steps"] = 2
    c.update(out_dir=os.path.join(work, "out"), runs_jsonl=os.path.join(work, "runs.jsonl"))
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "config.yaml"), "w") as f:
        yaml.safe_dump(c, f)
    return os.path.join(work, "config.yaml")


def run_tiny(path: str, fake: str, work: str, trunk_out: str | None = None) -> dict:
    cfg = L.resolve(path)
    p = tiny(cfg, fake, work, trunk_out)
    shutil.copy(os.path.join(os.path.dirname(path), "prereg.yaml"), work)
    assert train.main([p]) == 0
    recs = [json.loads(x) for x in open(os.path.join(work, "runs.jsonl"))]
    log = [json.loads(x) for x in open(os.path.join(work, "out", "log.jsonl")) if '"loss"' in x]
    ck = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(work, "out")))
    diag = [json.loads(x) for x in open(os.path.join(work, "out", "diag.jsonl"))]
    return {"cfg": cfg, "start": recs[0], "end": recs[-1], "log": log, "ck": ck, "diag": diag}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(p)[:-5])
def test_two_step_cpu_smoke(path, fake, tmp_path):
    m = L.NAME_RE.match(os.path.basename(path)[:-5])
    trunk_out = None
    if m["tag"] and m["tag"] != "trunk":                 # an S003 branch starts from its own tiny trunk
        trunk_out = str(tmp_path / "trunk" / "out")
        run_tiny(L.find(f"s003_adamw_e{m['e']}_r{m['r']}_trunk")[0], fake, str(tmp_path / "trunk"))
    r = run_tiny(path, fake, str(tmp_path / "run"), trunk_out)
    cfg, start, ck = r["cfg"], r["start"], r["ck"]
    want_model = PlanckConfig.from_dict({**cfg["model"], **TINY})
    assert start["n_params"] == count_analytic(want_model)["total"]
    assert ck["model_cfg"] == want_model.to_dict()
    flags = {k: v for sid in L.SCREENS for a in L.SCREENS[sid]["arms"].values() for k, v in a.items()}
    for k, v in flags.items():                        # every screen flag reached the harness as written, or is off
        sec, key = k.split(".")
        if sec == "model":
            got = ck["model_cfg"].get(key, PlanckConfig.__dataclass_fields__[key].default)
            assert got == cfg["model"].get(key, PlanckConfig.__dataclass_fields__[key].default)
    assert ("mtp" in start) == (cfg["train"].get("mtp", 0) == 1)
    if "mtp" in start:
        assert start["mtp"]["heads"] == 1 and start["mtp"]["weight"] == 1.0 and ck["mtp_params"] == 48 * 48 + 48
    assert ck["config"]["optim"]["kind"] == cfg["optim"]["kind"] and ck["config"]["seed"] == cfg["seed"]
    assert start["batch_tokens"] == 2 * cfg["train"]["grad_accum"] * 64 and "compile" not in start
    assert r["end"]["event"] == "end" and r["end"]["step"] == 2
    assert all(math.isfinite(x["loss"]) for x in r["log"]) and r["log"][-1]["step"] == 2
    assert [d["step"] for d in r["diag"]] == list(range(start["step"] + 1, 3))      # a branch starts at step 1
    assert r["diag"] and all(math.isfinite(d["ind"]) for d in r["diag"])
    assert all((d["gate"] is None) == (not want_model.attn_gate) for d in r["diag"])
    assert start["induction"]["gate"] == want_model.attn_gate


def init_state(model_cfg: dict, seed: int) -> dict:
    """As train.py builds a model: torch.manual_seed(seed), np.random.seed(seed), build_model on cpu."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return build_model(PlanckConfig.from_dict({**model_cfg, **TINY}), "cpu").state_dict()


def test_init_isolation_matches_each_screens_class():
    """C4: SIA arms keep every BASE draw bitwise (S002 asks for this test with the generator); S001's gate is a
    drawn nn.Linear, so removing it shifts later draws (NEW INIT); S003 changes no parameter (same init)."""
    base = L.resolve(L.find("base_s101")[0])["model"]
    seen = {}
    for sid, s in L.SCREENS.items():
        for a in s["arms"]:
            name = f"s003_adamw_e3_r1_trunk" if sid == "S003" else f"{sid.lower()}_{a}_g1_s1"
            arm = L.resolve(L.find(name)[0])["model"]
            for seed in (101, 102):
                b, x = init_state(base, seed), init_state(arm, seed)
                shared = sorted(set(b) & set(x))
                assert shared and len(shared) >= len(b) - 4 * TINY["n_layers"]
                same = all(torch.equal(b[k], x[k]) for k in shared)
                seen[(sid, a, seed)] = same
                assert same == (s["cls"] != "new init"), (sid, a, seed, [k for k in shared if not torch.equal(b[k], x[k])][:3])
    assert len(seen) == 2 * sum(len(s["arms"]) for s in L.SCREENS.values())
