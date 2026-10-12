"""data_seed (E3 Part 3: experiments/E3_seed_noise/notes.txt, PART 3 REOPENED; notes.txt DATA SEED). CPU.
  values     absent = the run seed; set = a non-negative int (bool, float, str, None, negative refused).
  off        absent == data_seed equal to seed: 20 CPU steps, the same batches, log records, weights, optimizer, loader,
             RNG state; the checkpoint and start record differ only by the data_seed key, which only the set run has.
  pre-patch  absent == the harness at 93bc10b (the commit every SCREENS BASE run and stage 2 pair ran from), 20 CPU
             steps: log records, the whole checkpoint (every key) and the start record identical (SCREENS C4's
             flag-off test; git show, skipped without git or the commit, e.g. in a mutation_check scratch copy).
  split      seed 3 / data 3 vs seed 4 / data 3: identical batches and loader state, different init;
             seed 3 / data 3 vs seed 3 / data 4: identical init, different batches (the loader gets data_seed).
  resume     20 + resume + 20 == 40 bitwise with data_seed 5 != seed 3; a resume under another data seed is refused.
  branch     a branch continues its trunk's stream: another data seed is refused, the trunk's is accepted; with
             data_seed absent on config and checkpoint the new check is never consulted (no new refusal).
  preflight  experiments/E2_lr_transfer/preflight.py's harness_reads sees data_seed (it refuses an unread one).
"""
from __future__ import annotations

import hashlib
import os
import re

import pytest
import torch
import yaml

import runio
import train
from make_fake_data import make
from test_s3_resume import state_equal
from test_screen_canon_run import assert_same_run, git_tree, outcome, run_in
from testutil import read_jsonl, write_run

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = "93bc10b"     # harness/ here equals 93bc10b's (SCREENS AMENDMENT BASE-REPLAY, SAME CODE); the patch is the next
START_SKIP = ("time", "config", "out_dir", "env", "resumed_from")
REAL_MODEL, REAL_LOADER = train.build_model, train.build_loader     # spy() wraps these, never an earlier spy


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def spy(monkeypatch):
    """Record train.main's init weights (build_model), the seed it gives build_loader and a digest of every batch."""
    rec = {"init": None, "loader_seed": None, "batches": []}
    real_model, real_loader = REAL_MODEL, REAL_LOADER

    def build_model(*a, **k):
        m = real_model(*a, **k)
        rec["init"] = {n: t.detach().clone() for n, t in m.state_dict().items()}
        return m

    def build_loader(dcfg, seq_len, micro, base, seed=0):
        rec["loader_seed"] = seed
        ld = real_loader(dcfg, seq_len, micro, base, seed)
        nb = ld.next_batch

        def next_batch():
            b = nb()
            h = hashlib.sha256()
            for k in ("idx", "tgt", "doc", "pos"):
                if b.get(k) is not None:
                    h.update(k.encode() + b[k].numpy().tobytes())
            rec["batches"].append(h.hexdigest())
            return b
        ld.next_batch = next_batch
        return ld
    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(train, "build_loader", build_loader)
    return rec


def run_spied(monkeypatch, cfg, *args):
    rec = spy(monkeypatch)
    assert train.main([cfg, *args]) == 0
    return rec


def start_of(run_dir) -> dict:
    s = [r for r in read_jsonl(os.path.join(os.path.dirname(run_dir), "runs.jsonl"))
         if r["event"] == "start" and r["run"] == os.path.basename(run_dir)][-1]
    return {k: v for k, v in s.items() if k not in START_SKIP}


def ck_of(cfg) -> dict:
    return runio.load_checkpoint(runio.latest_checkpoint(os.path.join(os.path.dirname(cfg), "out")))


def test_values():
    assert train.data_seed_of({}, 7) == 7 and train.data_seed_of({"seed": 2}, 2) == 2
    assert train.data_seed_of({"data_seed": 3}, 7) == 3 and train.data_seed_of({"data_seed": 0}, 7) == 0
    for bad in (True, False, 3.0, "3", None, -1, [3]):
        with pytest.raises(ValueError, match="data_seed must be"):
            train.data_seed_of({"data_seed": bad}, 7)


def test_absent_equals_data_seed_equal_to_seed_20_steps(tmp_path, data, monkeypatch):
    tr = {"total_steps": 20}
    a = write_run(str(tmp_path / "absent"), data, seed=3, train=tr)
    b = write_run(str(tmp_path / "equal"), data, seed=3, data_seed=3, train=tr)
    ra, rb = run_spied(monkeypatch, a), run_spied(monkeypatch, b)
    assert ra["loader_seed"] == rb["loader_seed"] == 3 and len(ra["batches"]) == 40
    assert ra["batches"] == rb["batches"] and state_equal(ra["init"], rb["init"]) == []
    oa, ob = outcome(a), outcome(b)
    assert_same_run(oa, ob, 20)
    assert "data_seed" not in oa[0] and ob[0]["data_seed"] == 3
    assert set(ob[0]) - set(oa[0]) == {"data_seed"} and not set(oa[0]) - set(ob[0])
    for k in ("data_state", "rng_torch", "optimizer", "phase", "schedule"):
        assert state_equal(oa[0][k], ob[0][k]) == [], k
    sa, sb = start_of(os.path.dirname(a)), start_of(os.path.dirname(b))
    assert "data_seed" not in sa and sb.pop("data_seed") == 3
    sa.pop("run"), sb.pop("run"), sa.pop("config_sha256"), sb.pop("config_sha256")
    assert sa == sb


def test_absent_equals_pre_patch_commit_20_steps(tmp_path, data):
    old = tmp_path / "pre_harness"
    old.mkdir()
    if git_tree(PRE, str(old)) is None:
        pytest.skip(f"git or commit {PRE} unavailable")
    assert '"data_seed"' not in (old / "train.py").read_text()
    a = write_run(str(tmp_path / "pre" / "run"), data, seed=3, train={"total_steps": 20})
    b = write_run(str(tmp_path / "now" / "run"), data, seed=3, train={"total_steps": 20})
    assert open(a).read() == open(b).read()           # the same name and bytes: the config hashes agree
    run_in(str(old), a)
    run_in(HERE, b)
    (ca, la), (cb, lb) = outcome(a), outcome(b)
    assert la == lb and len(la) == 20
    assert state_equal(ca, cb) == [] and "data_seed" not in cb
    assert start_of(os.path.dirname(a)) == start_of(os.path.dirname(b))


def test_init_and_stream_split(tmp_path, data, monkeypatch):
    tr = {"total_steps": 6}
    base = run_spied(monkeypatch, write_run(str(tmp_path / "s3d3"), data, seed=3, data_seed=3, train=tr))
    init = run_spied(monkeypatch, write_run(str(tmp_path / "s4d3"), data, seed=4, data_seed=3, train=tr))
    strm = run_spied(monkeypatch, write_run(str(tmp_path / "s3d4"), data, seed=3, data_seed=4, train=tr))
    assert (base["loader_seed"], init["loader_seed"], strm["loader_seed"]) == (3, 3, 4)
    assert init["batches"] == base["batches"] and len(base["batches"]) == 12      # same stream, new init
    assert state_equal(init["init"], base["init"]) != []
    assert state_equal(strm["init"], base["init"]) == []                          # same init, new stream
    assert strm["batches"] != base["batches"] and strm["batches"][0] != base["batches"][0]
    ck = {n: ck_of(str(tmp_path / n / "config.yaml")) for n in ("s3d3", "s4d3", "s3d4")}
    assert state_equal(ck["s4d3"]["data_state"], ck["s3d3"]["data_state"]) == []
    assert (ck["s3d3"]["seed"], ck["s4d3"]["seed"], ck["s3d4"]["seed"]) == (3, 4, 3)
    assert (ck["s3d3"]["data_seed"], ck["s4d3"]["data_seed"], ck["s3d4"]["data_seed"]) == (3, 3, 4)
    # the start record (the provenance line a result cites) names the loader's seed, not the run seed
    assert [start_of(str(tmp_path / n))["data_seed"] for n in ("s3d3", "s4d3", "s3d4")] == [3, 3, 4]


def test_resume_20_plus_20_equals_40_with_data_seed(tmp_path, data):
    straight = write_run(str(tmp_path / "straight"), data, seed=3, data_seed=5)
    split = write_run(str(tmp_path / "split"), data, seed=3, data_seed=5)
    assert train.main([straight]) == 0 and train.main([split, "--max-steps", "20"]) == 0
    assert train.main([split]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["step"] == 40 and a[0]["data_seed"] == b[0]["data_seed"] == 5
    assert_same_run(a, b, 40)


def set_data_seed(cfg: str, value) -> None:
    c = yaml.safe_load(open(cfg))
    if value is None:
        c.pop("data_seed", None)
    else:
        c["data_seed"] = value
    yaml.safe_dump(c, open(cfg, "w"))


@pytest.mark.parametrize("new", [6, None], ids=["another", "removed"])
def test_resume_under_another_data_seed_is_refused(tmp_path, data, new):
    cfg = write_run(str(tmp_path / "r"), data, seed=3, data_seed=5)
    assert train.main([cfg, "--max-steps", "20"]) == 0
    set_data_seed(cfg, new)
    with pytest.raises(ValueError, match=r"trained on data seed 5; this config gives data seed (6|3)"):
        train.main([cfg])


def trunk_ckpt(tmp_path, data, **over) -> str:
    t = write_run(str(tmp_path / "trunk"), data, schedule={"mode": "trunk", "warmup_steps": 4, "branch_points": [10]},
                  train={"total_steps": 30}, **over)
    assert train.main([t, "--max-steps", "12"]) == 0
    return str(tmp_path / "trunk" / "out" / "stable_00000010.pt")


def branch(tmp_path, data, name, init, **over) -> str:
    return write_run(str(tmp_path / name), data, schedule={"mode": "branch", "init_from": init, "decay_steps": 4},
                     **over)


def test_branch_continues_its_trunks_stream(tmp_path, data):
    st = trunk_ckpt(tmp_path, data, seed=3, data_seed=5)
    assert runio.load_checkpoint(st)["data_seed"] == 5
    with pytest.raises(ValueError, match="init_from checkpoint trained on data seed 5"):
        train.main([branch(tmp_path, data, "b_other", st, seed=3, data_seed=6)])
    with pytest.raises(ValueError, match="init_from checkpoint trained on data seed 5"):
        train.main([branch(tmp_path, data, "b_absent", st, seed=3)])
    ok = branch(tmp_path, data, "b_same", st, seed=3, data_seed=5)
    assert train.main([ok]) == 0 and ck_of(ok)["step"] == 14


def test_without_data_seed_the_check_is_never_consulted(tmp_path, data):
    # absent on config and checkpoint: no new refusal, whatever the seeds (what the loader then does is the pre-patch
    # harness's own business, e.g. its "resume replay drew different data" assertion)
    for ck_seed, seed in ((3, 3), (3, 4), (None, 4)):
        assert train.check_data_seed({} if ck_seed is None else {"seed": ck_seed}, seed, {"seed": seed}, "x") is None
    st = trunk_ckpt(tmp_path, data, seed=3)            # the pre-patch path end to end: trunk, then its branch
    br = branch(tmp_path, data, "b", st, seed=3)
    assert train.main([br]) == 0
    assert ck_of(br)["step"] == 14 and "data_seed" not in ck_of(br) and "data_seed" not in runio.load_checkpoint(st)


READ_RE = re.compile(r"""\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")    # preflight.py's READ_RE


def test_preflight_sees_the_read():
    pf = os.path.join(os.path.dirname(HERE), "experiments", "E2_lr_transfer", "preflight.py")
    if os.path.exists(pf):          # the repo; a mutation_check scratch copy has only harness/
        assert READ_RE.pattern in open(pf).read()
    readers = []
    for b in sorted(os.listdir(HERE)):
        if b.endswith(".py") and not b.startswith(("test_", "mutant", "mutation_")) and b != "conftest.py":
            if "data_seed" in READ_RE.findall(open(os.path.join(HERE, b)).read()):
                readers.append(b)
    assert readers == ["train.py"]
