"""diag_runner.py's robust group (SCREENS.txt AMENDMENT SEEDS-103-105, C6 RUNNER): the 30 runs it scores (every new run of
plans/robustness_k5.txt but S006's three g 1 IND-only runs), its 102 jobs, the flags each checkpoint must carry against
the run's own config, and that it is never in the default selection (batch 1's six groups and their 133 jobs unchanged).
No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_diag_robust.py
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(SX))
for _p in (os.path.join(ROOT, "data_prep"), os.path.join(ROOT, "harness"), SX, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import diag_runner as DR  # noqa: E402
import robust_lib as RB  # noqa: E402
import screens_lib as L  # noqa: E402

WANT = [r for r in RB.NEW if not r.startswith("s006_mtp_g1_")]      # C6: IND-only runs are not scored


def rows(groups=None):
    return DR.jobs(groups)


def test_the_robust_group_scores_every_new_run_but_the_ind_only_ones():
    r = rows(["robust"])
    runs = sorted({x[2] for x in r})
    assert runs == sorted(WANT) and len(runs) == 30 and {x[0] for x in r} == {"robust"} and {x[1] for x in r} == {"SCREENS"}
    assert set(DR.LATER_GROUPS) == {"robust"} and DR.LATER_GROUPS["robust"][1] == DR.ROBUST and DR.ROBUST == WANT
    jobs = {(x[2], x[3]) for x in r}
    assert {j for n, j in jobs if n == "s005_forget_g1_s104"} == {"modules", "ind", "hd", "attn", "halflife"}
    assert {j for n, j in jobs if n == "s005_base_s104"} == {"ind", "hd", "attn"}
    assert {n for n, j in jobs if j == "modules"} == {f"{a}_s{s}" for s in RB.SEEDS
                                                     for a in ("s004_canonac_g1", "s005_forget_g1", "s007_smear_g1")}
    assert {n for n, j in jobs if j == "halflife"} == {f"s005_forget_g1_s{s}" for s in RB.SEEDS}
    assert {j for n, j in jobs if j not in ("modules", "halflife")} == {"ind", "hd", "attn"}
    assert len(r) == 102 and all(fn == f"{j}.json" for _, _, _, j, fn in r)


def test_each_run_has_one_config_and_the_checkpoint_check_wants_its_flags():
    keys = {"canon": "model.canon", "canon_kernel": "model.canon_kernel", "forget_gate": "model.forget_gate",
            "smear_key": "model.smear_key", "attn_gate": "model.attn_gate", "value_residual": "model.value_residual",
            "qk_norm": "model.qk_norm", "norm_scaling": "model.norm_scaling"}
    for r in WANT:
        p = L.find(r)
        assert len(p) == 1, r
        f = L.flat(L.resolve(p[0]))
        step, d, want = DR.expected(r)
        assert (step, d) == (7630, 192), r
        for k, v in want.items():
            assert f.get(keys[k], DR.OFF.get(k, 4 if k == "canon_kernel" else None)) == v, (r, k, v, f.get(keys[k]))
        for k, fk in keys.items():                 # every flag the config moves from BASE is one the check reads
            if fk in f and k in DR.OFF and f[fk] != DR.OFF[k]:
                assert want[k] == f[fk], (r, k)


def test_never_in_the_default_selection():
    d = rows()
    assert len(d) == 133 and "robust" not in {x[0] for x in d} and list(DR.GROUPS) == ["e3p1", "stage2", "basediag",
                                                                                       "stage1", "replay", "e3p2"]
    both = rows(["stage1", "robust"])
    assert len(both) == 102 + 36 and [x[0] for x in both][:36] == ["stage1"] * 36


def test_the_cli_takes_the_group_and_refuses_an_unknown_one():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert DR.main(["list", "--groups", "robust"]) == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == 102 and all(ln.startswith("robust\tSCREENS/") for ln in lines)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert DR.main(["list", "--groups", "robust2"]) == 2
    assert "unknown groups ['robust2']" in out.getvalue() and "'robust'" in out.getvalue()


def test_a_run_not_yet_on_disk_is_refused_not_skipped(tmp_path):
    class A:
        groups, only, cpu_only, budget_s, device = ["robust"], ["s004_canonac_g1_s103"], True, 1500.0, "cpu"
        runs_root, evalset = str(tmp_path), None
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert DR.run(A) == 2
    assert "REFUSED s004_canonac_g1_s103: no run directory" in out.getvalue()
