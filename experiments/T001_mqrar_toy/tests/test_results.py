"""T001 results.py and rederive.py tests on fake output directories (tests/fakeout.py; never results). rederive.py is
the independent reading: it must agree with runplan.py + metrics.py on every per-run metric, C0, every LR pick, every
helps / passes flag and R1-R3, over fakes that reach every branch of the rules. results.py must print the module's
picks and every registered reported-only item."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
T001 = os.path.dirname(HERE)
sys.path.insert(0, T001)
sys.path.insert(0, HERE)
import fakeout  # noqa: E402
import metrics as M  # noqa: E402
import rederive  # noqa: E402
import results  # noqa: E402
import runplan as R  # noqa: E402

# (fake seed, build options): plain seeds, C0 (a) and (b), censored caps, and the tie branches of R1 and R2
CASES = [(0, {}), (1, {}), (2, {}), (5, {}), (11, {}), (14, {}), (3, dict(fire_a=True)),
         (4, dict(fire_a=True, fire_b=True)), (5, dict(fire_b=True, ceiling=("sb", "gdn"))),
         (1, dict(p_div=0.0, ceiling=("gdn", "canon", "sb", "alibi"))),
         (8, dict(p_div=0.0, ceiling=("gdn", "canon", "sb", "alibi"))),
         (1, dict(p_div=0.0, ceiling={"gdn": 96, "canon": 32})), (1, dict(p_div=0.0, ceiling={"gdn": 32, "canon": 96})),
         (0, dict(p_div=0.0, ceiling={"base": 48, "gdn": 64})),        # gdn helps by capacity alone (gm 1.33)
         (0, dict(p_div=0.0, ar32={"base": 0.9, "smear1": 0.8, "base1": 0.5})),
         (0, dict(fire_b="acc_only")), (2, dict(lr_tie=("fg", "smear")))]


@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    root = tmp_path_factory.mktemp("t001_fakes")
    return [fakeout.build(str(root / f"f{i}"), s, **kw) for i, (s, kw) in enumerate(CASES)]


def module_view(out):
    st, lrs, tab = R.table(out)
    return st, lrs, tab


def test_fakes_reach_every_branch(fakes):
    r1, r2, r3, flags = set(), set(), set(), set()
    for out in fakes:
        st, lrs, tab = module_view(out)
        r1.add(M.r1(tab)["why"].split(":")[0])
        r2.add(M.r2(tab)["why"].split(":")[0])
        r3.add(M.r3(tab)["reading"])
        flags |= {f for f, on in (("a", st["a"]), ("b", st["b"]), ("ext", any(v in (1e-4, 1e-2) for v in lrs.values())),
                                  ("div", any(tab[a][s]["diverged"] for a in tab for s in M.SEEDS)),
                                  ("cens", any(tab[a][s]["censored"] for a in tab for s in M.SEEDS))) if on}
        if len(M.r2(tab)["why"].split(":")[1].split(",")) > 1:
            flags.add("r2_several")
        gap = M.diffs(tab, "smear1", "base1", "ar32")
        m1, m2 = (sum(tab[a][s]["ar32"] for s in M.SEEDS) / 5 for a in ("smear1", "base"))
        if sum(g >= 0.2 for g in gap) >= 4 and m2 - 0.15 <= m1 < m2 - 0.05:
            flags.add("r3_margin")                     # YES only by its margin: R3 reads PARTIAL
        if not st["b"] and min(v["acc"] for v in R.dev_acc(R.load(out, R.name(
                "base", "mqrar", 0, 1e-3, st))).values()) >= 0.9:
            flags.add("b_needs_ov")                    # C0 (b) not fired although every dev acc >= 0.9
        if any(lrs[a] == 1e-4 and len({round(sum(R._get(R.dev_acc(R.load(out, R.name(a, "mqrar", 0, lr, st))), n, k)
                                                  for n in st["grid"] for k in ("acc", "acc_ov")), 9)
                                       for lr in (1e-4, 3e-4, 1e-3, 3e-3)}) == 1 for a in lrs if a != "base"):
            flags.add("lr_tie")                        # equal LR scores: the smaller LR, then its extension
    assert r1 == {"gdn helps, canon does not", "canon helps, gdn does not", "neither helps",
                  "both help; head to head on LW", "both help; LW tied; head to head on capacity", "both help and tie"}
    assert r2 == {"passing variants", "none passes; helping variants", "no variant helps"}
    assert r3 == {"YES", "NO", "PARTIAL"}
    assert flags == {"a", "b", "ext", "div", "cens", "r2_several", "r3_margin", "b_needs_ov", "lr_tie"}


def test_rederive_agrees_with_module(fakes):
    for out in fakes:
        st, lrs, tab = module_view(out)
        ind = json.loads(json.dumps(rederive.main(out), default=str))
        assert (st["steps"], st["n_hi"], st["grid"], st["a"], st["b"]) == tuple(
            ind["c0"][k] for k in ("steps", "n_hi", "grid", "a", "b"))
        assert {a: float(v) for a, v in lrs.items()} == {a: float(v) for a, v in ind["lr"].items()}
        for a in tab:
            for s in M.SEEDS:
                mine, theirs = tab[a][s], ind["table"][a][str(s)]
                for k in ("cap", "cap_all", "censored"):
                    assert mine[k] == theirs[k], (out, a, s, k)
                for k in ("lwo", "lw", "ar32"):
                    assert mine[k] == pytest.approx(theirs[k], abs=1e-12), (out, a, s, k)
        assert (M.r1(tab)["pick"], M.r2(tab)["pick"], M.r3(tab)["reading"]) == (ind["R1"], ind["R2"], ind["R3"])
        assert {x: M.helps(tab, x) for x in ind["helps"]} == ind["helps"]
        assert {x: M.passes(tab, x) for x in ind["passes"]} == ind["passes"]


def test_rederive_imports_nothing_from_the_module():
    with open(os.path.join(T001, "rederive.py")) as f:
        tree = ast.parse(f.read())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"__future__", "glob", "json", "math", "os", "sys"}


def test_results_prints_the_module_picks(fakes):
    for out in fakes:
        st, lrs, tab = module_view(out)
        lines, picks = results.build(out)
        assert picks == dict(R1=M.r1(tab)["pick"], R2=M.r2(tab)["pick"], R3=M.r3(tab)["reading"])
        text = "\n".join(lines)
        assert f"R1 PICK: {M.r1(tab)['pick']} ({M.r1(tab)['why']})" in text
        assert f"R2 PICK: {M.r2(tab)['pick']} ({M.r2(tab)['why']})" in text
        assert f"R3 READING: {M.r3(tab)['reading']}" in text
        assert len(lines) < 200


def test_results_reports_registered_items(fakes):
    out = fakes[0]
    st, lrs, tab = module_view(out)
    lines, _ = results.build(out)
    text = "\n".join(lines)
    assert "Accuracy by distance (registered as reported only): NOT RECORDED" in text
    js = [R.load(out, R.name("base", "mqrar", s, lrs["base"], st)) for s in M.SEEDS]
    ok = [j for j in js if j["eval"]]

    def share(key):
        vals = []
        for j in ok:
            t = j["eval"]["test"]["mqrar"]
            vals.append(sum(t[str(n)][key] for n in (8, 16, 24, 32)) / sum(t[str(n)]["upd_q"] for n in (8, 16, 24, 32)))
        return sum(vals) / len(vals)
    row = [ln for ln in lines if ln.startswith("  base ") and len(ln.split()) == 6][0].split()
    assert row[1] == str(js[0]["params"])
    assert row[2] == f"{share('stale'):.3f}" and row[3] == f"{share('first'):.3f}"



def test_results_cap_rows(fakes):
    """Every arm's cap row, censored caps starred (fakes with a ceiling have them)."""
    for out in fakes:
        st, lrs, tab = module_view(out)
        lines, _ = results.build(out)
        for a in R.PICK_ARMS + R.DIAG_ARMS:
            caps = " ".join(f"{tab[a][s]['cap']}{'*' if tab[a][s]['censored'] else ''}" for s in M.SEEDS)
            assert any(ln.startswith(f"  {a:7s}  {caps}") for ln in lines), (out, a)


def test_results_write_equals_stdout(fakes, tmp_path):
    dst = tmp_path / "R.txt"
    p = subprocess.run([sys.executable, os.path.join(T001, "results.py"), "--out", fakes[0], "--write", str(dst)],
                       capture_output=True, text=True, check=True)
    assert dst.read_text() == p.stdout and p.stdout.splitlines()[-1].startswith("PICKS {")
