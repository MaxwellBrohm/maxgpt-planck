"""diag_runner.py (SCREENS C6 diagnostics, E3 Part 4d): the registry, resume, the time budget, the replay's own process,
refusals, pins, and that it never opens a run's result files. CPU, tiny models (diag_fixtures.py).
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_diag_runner.py
"""
from __future__ import annotations

import builtins
import glob
import os
import sys

import pytest
import torch

import diag_fixtures as FX
import diag_lib as DL
import diag_runner as DR

SX, ROOT = FX.SX, FX.ROOT


def test_registry_names_the_kept_verdict_runs_and_e3():
    rows = DR.jobs()
    assert list(DR.GROUPS) == ["e3p1", "stage2", "basediag", "stage1", "replay", "e3p2"]
    runs = {g: sorted({r for gg, _, r, _, _ in rows if gg == g}) for g in DR.GROUPS}
    assert runs["stage1"] == sorted(["base_s101", "base_s102"] + [f"{a}_s{s}" for s in (101, 102) for a in (
        "s001_nogate_g1", "s002_novres_g1", "s002_noqknorm_g1", "s002_nonormscale_g1", "s003_adamw_e3_r2")])
    assert runs["stage2"] == sorted(f"{a}_s{s}" for s in (101, 102) for a in (
        "s005_base", "s005_forget_g1", "s004_canonac_g1", "s007_smear_g1", "s006_mtp_g2"))
    assert runs["basediag"] == ["base_s103", "base_s104", "base_s105"] and runs["replay"] == ["e3_5m_A_s1"]
    assert len(runs["e3p1"]) == 18 and len(runs["e3p2"]) == 6
    assert not any(r.startswith("s006_mtp_g1") for g in runs.values() for r in g)    # IND-only runs (C6): no HD
    for g, (exp, rs, _, _) in DR.GROUPS.items():                                    # every run has a config
        for r in rs:
            pat = (f"{ROOT}/experiments/E3_seed_noise/configs/{r}.yaml" if exp == "E3"
                   else f"{ROOT}/experiments/*/configs/{r}.yaml")
            assert len(glob.glob(pat)) == 1, r
    jobs = {(r, j) for _, _, r, j, _ in rows}
    assert {j for r, j in jobs if r == "s005_forget_g1_s101"} == {"modules", "ind", "hd", "attn", "halflife"}
    assert {j for r, j in jobs if r == "s005_base_s101"} == {"ind", "hd", "attn"}
    assert {j for r, j in jobs if r.startswith("e3_")} == {"ind", "hd"}
    assert {r for r, j in jobs if j == "modules"} == {f"{a}_s{s}" for s in (101, 102)
                                                     for a in ("s004_canonac_g1", "s005_forget_g1", "s007_smear_g1")}
    assert {r for r, j in jobs if j == "halflife"} == {"s005_forget_g1_s101", "s005_forget_g1_s102"}
    assert len(rows) == 133


def test_pins_and_steps_equal_the_registered_ones():
    import yaml
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E3_seed_noise"))
    import e3analyze
    import runio
    base = yaml.safe_load(open(os.path.join(SX, "configs", "screens_base.yaml")))
    assert DL.IND_SHA == base["eval"]["induction"]["sha256"] == DR.sha256(DL.IND_FILE)
    assert DL.TOK_SHA == e3analyze.FIXED["tokenizer"] == DR.sha256(DL.TOK_FILE)
    assert DL.EVALSET_SHA == e3analyze.FIXED["evalset"]
    assert (DL.WINDOWS, DL.BPB_BATCH) == (200, 16384)              # C6's 200 windows; E2 METRIC's batch tokens
    for _, (exp, rs, _, _) in DR.GROUPS.items():
        for r in rs:
            pat = f"{ROOT}/experiments/{'E3_seed_noise' if exp == 'E3' else '*'}/configs/{r}.yaml"
            c = runio.load_yaml(glob.glob(pat)[0])
            step, d, want = DR.expected(r)
            assert (c["train"]["total_steps"], c["model"]["d_model"]) == (step, d), r
            assert all(c["model"].get(k, DR.OFF.get(k, 4)) == v for k, v in want.items()), r


@pytest.fixture
def world(tmp_path):
    return FX.root_with(tmp_path, FX.SMALL), FX.evalset(str(tmp_path / "ev")), tmp_path


def mtimes(root):
    return {p: os.stat(p).st_mtime_ns for p in glob.glob(f"{root}/*/*/c6/*.json")}


def test_done_jobs_are_skipped_and_a_lost_one_is_redone(world, capsys):
    root, ev, _ = world
    assert FX.cli(root, ev) == 3 and FX.cli(root, ev) == 0
    before = mtimes(root)
    assert len(before) == 23
    capsys.readouterr()
    assert FX.cli(root, ev) == 0 and mtimes(root) == before and "done " not in capsys.readouterr().out
    lost = os.path.join(root, "SCREENS", "s007_smear_g1_s101", "c6", "attn.json")
    os.remove(lost)
    assert FX.cli(root, ev) == 0
    after = mtimes(root)
    assert set(after) == set(before) and [p for p in after if after[p] != before[p]] == [lost]


def test_budget_stops_between_runs_and_the_result_equals_one_pass(world, tmp_path_factory):
    root, ev, _ = world
    calls, rc = 0, 3
    while rc == 3:
        rc = FX.cli(root, ev, "--budget-s", "0")
        calls += 1
        assert calls < 20
    assert rc == 0 and calls == 7                                   # one run per call, then the replay's own call
    one = tmp_path_factory.mktemp("one")
    root1 = FX.root_with(one, FX.SMALL)
    assert FX.cli(root1, ev) == 3 and FX.cli(root1, ev) == 0
    strip = lambda o: {k: {**v, "seconds": 0} for k, v in o.items()}   # noqa: E731
    assert strip(FX.outputs(root)) == strip(FX.outputs(root1))


@pytest.mark.parametrize("case", ["DIVERGED", "GAP", "rc12_eval.jsonl", "two_finals", "wrong_step_name", "ck_step",
                                  "flag_off", "flag_on", "d_model", "missing"])
def test_refused_runs_write_nothing_and_the_rest_runs(world, case):
    root, ev, _ = world
    d = os.path.join(root, "SCREENS", "s004_canonac_g1_s101")
    if case in ("DIVERGED", "GAP", "rc12_eval.jsonl"):
        open(os.path.join(d, case), "w").close()
    elif case == "two_finals":
        FX.save(root, "SCREENS", "s004_canonac_g1_s101", FX.model("s004_canonac"), name="final_00009999.pt")
    elif case == "wrong_step_name":
        os.rename(os.path.join(d, "final_00007630.pt"), os.path.join(d, "final_00007629.pt"))
    elif case == "ck_step":
        FX.save(root, "SCREENS", "s004_canonac_g1_s101", FX.model("s004_canonac"), step=7629, name="final_00007630.pt")
    elif case == "flag_off":
        FX.save(root, "SCREENS", "s004_canonac_g1_s101", FX.model("base"))
    elif case == "flag_on":
        FX.save(root, "SCREENS", "base_s101", FX.model("s007_smear"))
    elif case == "d_model":
        FX.save(root, "SCREENS", "s004_canonac_g1_s101", FX.model("s004_canonac", d_model=160, head_dim=48))
    else:
        import shutil
        shutil.rmtree(d)
    assert FX.cli(root, ev) == 2
    out = FX.outputs(root)
    bad = "base_s101" if case == "flag_on" else "s004_canonac_g1_s101"
    assert not any(r == bad for _, r, _ in out)
    assert ("SCREENS", "s007_smear_g1_s101", "attn.json") in out and ("E3", "e3_5m_A_s1", "hd.json") in out


def test_pins_refuse_another_eval_set_or_tokenizer_and_another_ind_file_fails_the_job(world, monkeypatch, capsys):
    root, ev, tmp = world
    other = FX.evalset(str(tmp / "ev2"), sha="0" * 64)
    assert FX.cli(root, other) == 2 and FX.outputs(root) == {}
    monkeypatch.setattr(DL, "TOK_SHA", "1" * 64)
    assert FX.cli(root, ev) == 2 and FX.outputs(root) == {}
    monkeypatch.undo()
    monkeypatch.setattr(DL, "IND_SHA", "2" * 64)
    assert FX.cli(root, ev) == 1
    out = FX.outputs(root)
    assert out and not any(f.startswith("ind") for _, _, f in out) and "FAILED" in capsys.readouterr().out


def test_a_failed_job_writes_nothing_and_the_others_run(world, monkeypatch):
    root, ev, _ = world
    import attn_diag
    real = attn_diag.read
    monkeypatch.setattr(attn_diag, "read", lambda *a, **k: {**real(*a, **k), "check_ok": False})
    assert FX.cli(root, ev) == 1
    out = FX.outputs(root)
    assert not any(f == "attn.json" for _, _, f in out) and ("SCREENS", "base_s101", "hd.json") in out


def test_cpu_only_runs_the_modules_jobs_and_builds_no_model(world, monkeypatch):
    root, ev, _ = world
    import bpb
    monkeypatch.setattr(bpb, "load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model built")))
    assert DR.main(["run", "--runs-root", root, "--cpu-only", "--only", ",".join(r for _, r in FX.SMALL)]) == 0
    assert sorted(FX.outputs(root)) == [("SCREENS", r, "modules.json") for r in
                                        ("s004_canonac_g1_s101", "s005_forget_g1_s101", "s007_smear_g1_s101")]


def test_never_opens_a_result_file_of_a_run_and_writes_no_machine_path(world, monkeypatch):
    root, ev, _ = world
    for p in glob.glob(f"{root}/*/*"):
        for f in ("bpb.jsonl", "diag.jsonl", "log.jsonl", "SCORED"):
            open(os.path.join(p, f), "w").write("{}\n")
        os.makedirs(os.path.join(p, "bpb"))
        open(os.path.join(p, "bpb", "final_00007630.json"), "w").write("{}\n")
    seen, real_open, real_load = [], builtins.open, torch.load
    def spy(f, *a, **k):
        seen.append(os.path.abspath(f) if isinstance(f, (str, os.PathLike)) else "")
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(torch, "load", lambda f, *a, **k: (seen.append(os.path.abspath(f)), real_load(f, *a, **k))[1])
    assert FX.cli(root, ev) == 3 and FX.cli(root, ev) == 0
    monkeypatch.undo()
    inside = [p for p in seen if p.startswith(os.path.abspath(root) + os.sep)]
    assert inside and all(os.path.basename(p) == "final_00007630.pt" or "/c6/" in p for p in inside), inside
    for p in glob.glob(f"{root}/*/*/c6/*.json"):
        txt = open(p).read()
        assert root not in txt and os.path.expanduser("~") not in txt and ev not in txt, p


def test_list_and_argument_refusals(capsys):
    assert DR.main(["list", "--groups", "replay"]) == 0
    assert capsys.readouterr().out.splitlines() == ["replay\tE3/e3_5m_A_s1\tind\tind.replay.json",
                                                    "replay\tE3/e3_5m_A_s1\thd\thd.replay.json"]
    assert DR.main(["list", "--groups", "nope"]) == 2
    assert DR.main(["run", "--runs-root", "/nonexistent"]) == 2
