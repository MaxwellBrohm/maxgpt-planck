"""preflight.py against a scratch code root (a copy of harness/, tokenizer/v0 and this experiment) and a scratch
planck root (the committed copies of the shard and eval manifests, empty shard files). Imports torch through
harness/runio.py and saves one small dict with torch.save; builds no model. PC only (the Mac runs no torch):
  ~/planck/venv/bin/python -m pytest -q experiments/E2_lr_transfer/tests/test_preflight.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(E2))
SRC_KIND = {"cccc": "bin", "dolly": "jsonl", "gutenberg": "bin", "irc": "bin", "oasst2": "jsonl",
            "stackexchange": "bin", "wikimedia": "bin"}


@pytest.fixture
def c(tmp_path):
    code, planck = tmp_path / "code", tmp_path / "planck"
    shutil.copytree(os.path.join(REPO, "harness"), code / "harness",
                    ignore=shutil.ignore_patterns("__pycache__", "runs", ".pytest_cache", "test_*"))
    (code / "tokenizer" / "v0").mkdir(parents=True)
    for f in ("tok_v0_8k.json", "tok_v0_manifest.json"):
        shutil.copy(os.path.join(REPO, "tokenizer", "v0", f), code / "tokenizer" / "v0" / f)
    shutil.copytree(E2, code / "experiments" / "E2_lr_transfer", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    sh = planck / "data" / "shards" / "starter_v0_8k"
    for s, ext in SRC_KIND.items():
        (sh / s).mkdir(parents=True)
        (sh / s / f"{s}-00000.{ext}").write_bytes(b"")
    shutil.copy(os.path.join(REPO, "data_prep", "stats", "starter_v0_8k", "manifest.json"), sh / "manifest.json")
    (planck / "data" / "evalsets" / "v0").mkdir(parents=True)
    shutil.copy(os.path.join(REPO, "data_prep", "stats", "evalsets_v0", "manifest.json"),
                planck / "data" / "evalsets" / "v0" / "manifest.json")
    os.symlink(planck, code / "planck_root")
    return {"code": code, "planck": planck, "xd": code / "experiments" / "E2_lr_transfer", "tmp": tmp_path}


def pf(c, cfg, *flags):
    out = c["tmp"] / "rep.json"
    p = subprocess.run([sys.executable, str(c["xd"] / "preflight.py"), str(c["xd"] / cfg), "--code", str(c["code"]),
                        "--out", str(out), *flags], capture_output=True, text=True, timeout=100)
    assert out.exists(), p.stderr[-2000:]
    rep = json.load(open(out))["runs"][0]
    assert (p.returncode == 0) == rep["ok"]
    return rep


def edit(path, old, new):
    t = open(path).read()
    assert t.count(old) == 1, (path, old)
    open(path, "w").write(t.replace(old, new))


def refused(rep, text):
    return any(text in r for r in rep["refusals"])


def git_commit(c):
    g = ["git", "-C", str(c["code"])]
    subprocess.run(g + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A", "experiments"], check=True)
    subprocess.run(g + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"], check=True)


def test_committed_configs_pass_non_strict(c):
    for cfg in ("smoke/trunk.yaml", "configs/5m_e3_r1_trunk.yaml"):
        rep = pf(c, cfg)
        assert rep["ok"], rep["refusals"]
        assert rep["n_params"] == 5010133 and rep["engine"]["optim.batched"] is True
    assert pf(c, "configs/5m_e3_r1_b250M.yaml", "--no-init-check")["ok"]
    assert refused(pf(c, "configs/5m_e3_r1_b250M.yaml"), "does not exist")


def test_strict_needs_commit_and_fixed_engine_and_no_auto(c):
    rep = pf(c, "configs/5m_e3_r1_trunk.yaml", "--strict")
    assert refused(rep, "engine_fixed is null") and refused(rep, "prereg")
    edit(c["xd"] / "configs" / "engine.yaml", "\nengine_fixed: null\n", "\nengine_fixed: 2026-09-27 test\n")
    git_commit(c)
    assert pf(c, "configs/5m_e3_r1_trunk.yaml", "--strict")["ok"]
    edit(c["xd"] / "configs" / "engine.yaml", "doc_attn: varlen", "doc_attn: auto")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml", "--strict"), "doc_attn is auto")


@pytest.mark.parametrize("what", ["tokenizer", "shard_manifest", "eval_manifest", "evalset_field"])
def test_provenance_mismatch_refused(c, what):
    if what == "tokenizer":
        with open(c["code"] / "tokenizer" / "v0" / "tok_v0_8k.json", "a") as f:
            f.write(" ")
    elif what == "shard_manifest":
        with open(c["planck"] / "data" / "shards" / "starter_v0_8k" / "manifest.json", "a") as f:
            f.write(" ")
    else:
        edit(c["xd"] / "configs" / "base5m.yaml",
             *(("evalset_manifest_sha256: b", "evalset_manifest_sha256: c") if what == "eval_manifest"
               else ("evalset_sha256: fe1b", "evalset_sha256: ee1b")))
    rep = pf(c, "configs/5m_e3_r1_trunk.yaml")
    assert not rep["ok"] and all("provenance" in r for r in rep["refusals"]), rep["refusals"]


def test_data_block_checks(c):
    base = c["xd"] / "configs" / "base5m.yaml"
    edit(base, "share: 0.0120,", "")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "oasst2: share missing")
    edit(base, "oasst2/oasst2-*.jsonl", "oasst2/nothing-*.jsonl")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "oasst2: no file matches")


def test_source_order_and_ids_must_match_the_manifest(c):
    base = c["xd"] / "configs" / "base5m.yaml"
    t = open(base).read()
    a, b = re.search(r"    - \{name: dolly.*?\]\}\n", t, re.S).group(0), re.search(
        r"    - \{name: gutenberg.*?\]\}\n", t, re.S).group(0)
    open(base, "w").write(t.replace(a + b, b + a))
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "differ from the shard manifest")
    open(base, "w").write(t.replace("end_id: 5", "end_id: 6"))
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "pad/end/role ids")
    open(base, "w").write(t.replace("vocab_size: 8192", "vocab_size: 4096"))
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "model vocab 4096")


def test_engine_key_read_by_harness_but_unset(c):
    with open(c["code"] / "harness" / "train.py", "a") as f:
        f.write('\n_probe = {}.get("compile", False)\n')
    edit(c["xd"] / "configs" / "engine.yaml", "  compile: false ", "  # compile removed ")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "harness reads train.compile but the config does not set it")


def test_engine_key_on_but_not_read_and_data_seed(c):
    for p in (c["code"] / "harness").glob("*.py"):
        t = p.read_text()
        p.write_text(t.replace('.get("ce_chunk_rows"', '.get("x_ce"').replace('.get("data_seed"', '.get("x_ds"'))
    assert pf(c, "configs/5m_e3_r1_trunk.yaml")["ok"]                 # 0 = off: harmless when unread
    edit(c["xd"] / "configs" / "engine.yaml", "ce_chunk_rows: 0 ", "ce_chunk_rows: 64 ")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "train.ce_chunk_rows = 64 but this harness does not read")
    edit(c["xd"] / "configs" / "engine.yaml", "ce_chunk_rows: 64 ", "ce_chunk_rows: 0 ")
    with open(c["xd"] / "configs" / "5m_e3_r1_trunk.yaml", "a") as f:
        f.write("data_seed: 2\n")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "data_seed is set but")


def test_schedule_and_scorer_checks(c):
    cfg = c["xd"] / "configs" / "5m_e3_r1_trunk.yaml"
    edit(cfg, "total_steps: 6105", "total_steps: 6104")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "not inside [warmup, total_steps - 1")
    edit(cfg, "total_steps: 6104", "total_steps: 6105")
    edit(c["xd"] / "configs" / "base5m.yaml", "  warmup_steps: 76 ", "  # no warmup_steps ")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "warmup_steps not written")
    edit(c["xd"] / "configs" / "base5m.yaml", "  # no warmup_steps ", "  warmup_steps: 76 ")
    edit(c["xd"] / "configs" / "base5m.yaml", "precision: fp32", "precision: bf16")
    assert refused(pf(c, "configs/5m_e3_r1_trunk.yaml"), "not the scorer invocation")


def test_branch_init_checks(c):
    import torch
    sys.path.insert(0, str(c["code"] / "harness"))
    import runio
    from config import PlanckConfig
    trunk_cfg = runio.load_yaml(str(c["xd"] / "configs" / "5m_e3_r1_trunk.yaml"))
    d = c["planck"] / "runs" / "E2" / "5m_e3_r1_trunk"
    d.mkdir(parents=True)
    ck = {"model_cfg": PlanckConfig.from_dict(trunk_cfg["model"]).to_dict(), "step": 6104, "config": trunk_cfg}
    torch.save(dict(ck, model_cfg=dict(ck["model_cfg"], d_model=256)), d / "stable_00006104.pt")
    assert refused(pf(c, "configs/5m_e3_r1_b250M.yaml"), "another model shape")
    torch.save(ck, d / "stable_00006104.pt")
    assert pf(c, "configs/5m_e3_r1_b250M.yaml")["ok"]
    edit(c["xd"] / "configs" / "5m_e3_r1_b250M.yaml", "embed_lr: 0.003", "embed_lr: 0.006")
    assert refused(pf(c, "configs/5m_e3_r1_b250M.yaml"), "optim.embed_lr")
    edit(c["xd"] / "configs" / "5m_e3_r1_b250M.yaml", "embed_lr: 0.006", "embed_lr: 0.003")
    edit(c["xd"] / "configs" / "5m_e3_r1_b250M.yaml", "total_steps: 7630", "total_steps: 7631")
    assert refused(pf(c, "configs/5m_e3_r1_b250M.yaml"), "!= init step")
    edit(c["xd"] / "configs" / "5m_e3_r1_b250M.yaml", "total_steps: 7631", "total_steps: 7630")
    with open(c["xd"] / "configs" / "5m_e3_r1_b250M.yaml", "a") as f:
        f.write("seed: 2\n")
    assert refused(pf(c, "configs/5m_e3_r1_b250M.yaml"), "seed differs")

