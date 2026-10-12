"""data_seed, the cases the 2026-10-11 verify found unpinned (E3 Part 3: experiments/E3_seed_noise/notes.txt, PART 3
REOPENED; notes.txt DATA SEED). CPU. Split from test_data_seed.py for the file size; its helpers are reused.
  fallback   a checkpoint without data_seed trained on its seed's stream: a config that adds data_seed resumes or
             branches only when the value equals that seed (check_data_seed's "else its seed" branch, read because the
             config, not the checkpoint, carries data_seed).
  zero       data_seed 0 is a set value: the loader gets 0, the checkpoints and the start record record 0, and the run
             resumes under it (a truthiness test would drop the key and then refuse the run's own resume).
"""
from __future__ import annotations

import pytest

import runio
import train
from make_fake_data import make
from test_data_seed import branch, ck_of, run_spied, set_data_seed, start_of, trunk_ckpt
from test_screen_canon_run import assert_same_run, outcome
from testutil import write_run


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds_more_data")
    make(str(d), docs=150, chats=120)
    return str(d)


def test_resume_adding_data_seed_needs_the_checkpoints_seed(tmp_path, data):
    cfg = write_run(str(tmp_path / "r"), data, seed=3)                  # flag off: the checkpoint has no data_seed
    assert train.main([cfg, "--max-steps", "20"]) == 0
    assert "data_seed" not in ck_of(cfg) and ck_of(cfg)["seed"] == 3
    set_data_seed(cfg, 4)
    with pytest.raises(ValueError, match=r"trained on data seed 3; this config gives data seed 4"):
        train.main([cfg])
    assert ck_of(cfg)["step"] == 20                                        # refused before any step
    set_data_seed(cfg, 3)                                                  # the same stream: accepted, runs to the end
    assert train.main([cfg]) == 0
    ck = ck_of(cfg)
    assert ck["step"] == 40 and ck["data_seed"] == 3


def test_branch_adding_data_seed_needs_the_trunks_seed(tmp_path, data):
    st = trunk_ckpt(tmp_path, data, seed=3)
    assert "data_seed" not in runio.load_checkpoint(st)
    with pytest.raises(ValueError, match=r"init_from checkpoint trained on data seed 3; this config gives data seed 4"):
        train.main([branch(tmp_path, data, "b_other", st, seed=3, data_seed=4)])
    ok = branch(tmp_path, data, "b_same", st, seed=3, data_seed=3)
    assert train.main([ok]) == 0 and ck_of(ok)["step"] == 14 and ck_of(ok)["data_seed"] == 3


def test_data_seed_zero_is_recorded_and_resumes(tmp_path, data, monkeypatch):
    straight = write_run(str(tmp_path / "straight"), data, seed=3, data_seed=0)
    split = write_run(str(tmp_path / "split"), data, seed=3, data_seed=0)
    rec = run_spied(monkeypatch, split, "--max-steps", "20")
    assert rec["loader_seed"] == 0
    assert ck_of(split)["data_seed"] == 0 and start_of(str(tmp_path / "split"))["data_seed"] == 0
    assert train.main([split]) == 0 and train.main([straight]) == 0
    a, b = outcome(straight), outcome(split)
    assert a[0]["data_seed"] == b[0]["data_seed"] == 0
    assert_same_run(a, b, 40)
