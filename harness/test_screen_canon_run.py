"""S004 Canon layers, run level (model.canon; the model-level tests are test_screen_canon.py). CPU.
  config     values refused, to_dict drops the off flag (so an off run's model_cfg is the pre-flag dict).
  off        canon absent == canon "" (a canon_kernel given is ignored): 20 CPU steps, bitwise log records,
             weights, optimizer, loader, model_cfg; and == the pre-flag commit's harness (git show; skipped
             without git or the commit, e.g. in a mutation_check scratch copy unless GIT_DIR is set).
  optimizer  every tap of every kernel gets a gradient at zero init; the kernels are in the AdamW 'scalar'
             group (scalar_lr, no decay), not NorMuon; one step moves them.
  resume     20 + resume + 20 == 40 bitwise with canon AC, per-matrix and batched optimizer.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

import runio
import train
from config import PlanckConfig
from make_fake_data import make
from model import build_model
from optim import make_optimizer
from test_s3_docattn import ROWS
from test_s3_leak import DOC, T
from test_s3_resume import state_equal
from testutil import read_jsonl, tiny, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
PREFLAG = "5c5a30a"   # harness commit before any screen flag (SCREENS C4: BASE pairs across commits)
CK_KEYS = ("model", "optimizer", "data_state", "model_cfg", "step", "tokens", "sup_tokens",
           "schedule", "rng_torch", "n_params", "seed", "precision")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("canon_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def outcome(cfg_path: str):
    out = os.path.join(os.path.dirname(cfg_path), "out")
    log = [{k: v for k, v in r.items() if k not in ("time", "tok_per_s")}
           for r in read_jsonl(os.path.join(out, "log.jsonl"))]
    return runio.load_checkpoint(runio.latest_checkpoint(out)), log


def assert_same_run(a, b, steps: int) -> None:
    assert a[1] == b[1] and len(a[1]) == steps
    for k in CK_KEYS:
        assert state_equal(a[0][k], b[0][k]) == [], k


def test_config_values_and_dict():
    for bad in ("B", "D", "ACD", "CA", "ac"):
        with pytest.raises(AssertionError):
            tiny(canon=bad)
    with pytest.raises(AssertionError):
        tiny(canon="AC", canon_kernel=1)
    off, on = tiny(canon_kernel=3), tiny(canon="AC", canon_kernel=3)
    assert set(off.to_dict()) == set(on.to_dict()) - {"canon", "canon_kernel"}
    assert PlanckConfig.from_dict(on.to_dict()) == on and on.to_dict()["canon"] == "AC"
    assert PlanckConfig.from_dict(off.to_dict()) == tiny()
    assert tiny(canon="AC", mlp_hidden=0).canon_sites() == "A" and tiny().canon_sites() == ""


def test_off_equals_absent_20_steps(tmp_path, data):
    tr = {"total_steps": 20}
    a = write_run(str(tmp_path / "absent"), data, train=tr)
    b = write_run(str(tmp_path / "off"), data, model={"canon": "", "canon_kernel": 3}, train=tr)
    c = write_run(str(tmp_path / "on"), data, model={"canon": "AC"}, train=tr)
    assert train.main([a]) == 0 and train.main([b]) == 0 and train.main([c]) == 0
    ra, rb, rc = outcome(a), outcome(b), outcome(c)
    assert_same_run(ra, rb, 20)
    assert "canon" not in ra[0]["model_cfg"] and rc[0]["model_cfg"]["canon"] == "AC"
    assert [r["loss"] for r in rc[1]] != [r["loss"] for r in ra[1]]    # the probe can see a change


def git_tree(commit: str, dest: str) -> str | None:
    """harness/*.py at commit written into dest; None when git or the commit is unavailable."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", HERE, *a], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
    ls = git("ls-tree", "--name-only", "--full-tree", commit, "harness/")
    if ls is None or ls.returncode != 0:
        return None
    for name in (n for n in ls.stdout.decode().split() if n.endswith(".py")):
        blob = git("show", f"{commit}:{name}")
        if blob is None or blob.returncode != 0:
            return None
        with open(os.path.join(dest, os.path.basename(name)), "wb") as f:
            f.write(blob.stdout)
    return dest


def run_in(code_dir: str, cfg: str) -> None:
    code = "import sys, torch; torch.set_num_threads(2); import train; sys.exit(train.main(sys.argv[1:]))"
    p = subprocess.run([sys.executable, "-c", code, cfg], cwd=code_dir, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=100)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-3000:]


def test_off_equals_preflag_commit_20_steps(tmp_path, data):
    old = tmp_path / "preflag_harness"
    old.mkdir()
    if git_tree(PREFLAG, str(old)) is None:
        pytest.skip(f"git or commit {PREFLAG} unavailable")
    assert "canon" not in (old / "config.py").read_text()
    a = write_run(str(tmp_path / "preflag"), data, train={"total_steps": 20})
    b = write_run(str(tmp_path / "now_off"), data, model={"canon": ""}, train={"total_steps": 20})
    run_in(str(old), a)
    run_in(HERE, b)
    assert_same_run(outcome(a), outcome(b), 20)


def test_gradient_reaches_every_tap_and_scalar_group():
    torch.manual_seed(0)
    m = build_model(tiny(canon="AC", n_loops=2))
    idx, tgt = (torch.randint(0, 97, (2, T), generator=torch.Generator().manual_seed(s)) for s in (3, 4))
    m(idx, tgt, torch.tensor([DOC, ROWS[1]]))[1].backward()
    ks = {n: p for n, p in m.named_parameters() if ".canon_" in n}
    assert len(ks) == 2 * m.cfg.n_unique
    assert all(p.grad is not None and (p.grad != 0).all() for p in ks.values())
    opt = make_optimizer(m, {"lr": 1e-2, "embed_lr": 5e-3, "scalar_lr": 2e-3, "weight_decay": 0.1})
    g = {x["name"]: x for x in opt.param_groups}
    ids = {name: {id(p) for p in x["params"]} for name, x in g.items()}
    assert all(id(p) in ids["scalar"] and id(p) not in ids["matrix"] for p in ks.values())
    assert g["scalar"]["weight_decay"] == 0.0 and g["scalar"]["base_lr"] == 2e-3
    assert not g["scalar"]["use_muon"]
    opt.step()
    assert all(p.any() for p in ks.values())


@pytest.mark.parametrize("optim", [{"lr": 3e-3}, {"lr": 3e-3, "batched": True}], ids=["ref", "batched"])
def test_resume_20_plus_20_equals_40_with_canon(tmp_path, data, optim):
    over = dict(model={"canon": "AC"}, optim=optim)
    straight = write_run(str(tmp_path / "straight"), data, **over)
    split = write_run(str(tmp_path / "split"), data, **over)
    assert train.main([straight]) == 0 and train.main([split, "--max-steps", "20"]) == 0
    assert train.main([split]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["step"] == 40 and a[0]["model_cfg"]["canon"] == "AC"
    assert_same_run(a, b, 40)
    assert all(v.any() for n, v in a[0]["model"].items() if ".canon_" in n)   # the kernels trained
