"""e3analyze_p3.py (notes.txt PART 3 REOPENED, ANALYZER) on tests/p3fixture.py's synthetic record set, before any Part 3
value exists: the checks fail exactly on the five expected code checks; every registered statistic is recomputed here
from the raw fixture files by another route (statistics.stdev, closed-form k_needed, scipy 1.18.1 quantiles typed in);
GAP handling; and the DATA mutants (e3analyze_p3_mutants.py --fixture) all killed. E3P3_CODE (e3analyze_p3_mutants.py
--code) points at a folder holding a mutant analyzer. Needs git (commit 4962426's code list).
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_e3analyze_p3.py
"""
import json
import math
import os
import statistics as st
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import p3fixture as F  # noqa: E402

CODE = os.environ.get("E3P3_CODE") or E3
EXPECTED = {"code digest harness+data_prep .py == FIXED 2 (92deaef0)", "harness/train.py == FIXED 2",
            "harness/trainer.py == FIXED 2", "harness/model.py == FIXED 2", "data_prep/bpb.py == FIXED 2"}
Q = {"factor4": 3.717409682432975, "factor3": 4.160918617647012, "lo4": 0.5991331391341409, "hi4": 2.873555634078219,
     "ub4": 1.55757542255663, "F975_4_7": 5.5225943453085495, "F025_4_7": 0.1102032682010376}   # scipy 1.18.1
INIT = ["e3_5m_A_s1_P3"] + [f"e3_5m_A_init{k}" for k in range(2, 6)]
DATA = ["e3_5m_A_s1_P3"] + [f"e3_5m_A_data{k}" for k in range(2, 6)]


@pytest.fixture(scope="module")
def codelist():
    try:
        return F.code_list()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git or commit 4962426 unavailable")


def make(tmp_path, codelist, gaps=()):
    d = F.build(str(tmp_path / "E3"), code=CODE, codelist=codelist)
    for g in gaps:
        open(os.path.join(d, "out", g, "GAP"), "w").close()
    return d


def run(d, *a):
    r = subprocess.run([sys.executable, os.path.join(d, "e3analyze_p3.py"), *a], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def raw(d, run_, s, sp="all", kind="F"):
    """A metric straight from the fixture's files (not through e3analyze)."""
    if kind == "T":
        return st.fmean(json.loads(x)["loss"] for x in open(os.path.join(d, "out", run_, "train_tail.jsonl")) if json.loads(x)["step"] > 7477.4)
    rows = [json.loads(x) for x in open(os.path.join(d, "out", run_, "bpb.jsonl"))]
    v = {r["step"]: r["bpb"] for r in rows if r["set"] == s and r["split"] == sp}
    return v[7630] if kind == "F" else st.fmean(v[k] for k in (7344, 7497, 7630))


def test_checks_fail_only_the_expected_five(tmp_path, codelist):
    d = make(tmp_path, codelist)
    ck = json.loads(run(d, "--checks-only"))
    assert {c["check"] for c in ck if not c["ok"]} == EXPECTED and len(ck) == 32


@pytest.mark.parametrize("m,args", [("F(CHAT)", ("CHAT",)), ("F(PROSE)", ("PROSE",)), ("M(CHAT)", ("CHAT", "all", "M")),
                                    ("CHAT user", ("oasst2", "user")), ("train loss last 2%", (None, None, "T"))])
def test_statistics_by_another_route(tmp_path, codelist, m, args):
    d = make(tmp_path, codelist)
    run(d)
    r = json.load(open(os.path.join(d, "results_part3.json")))
    p1 = json.load(open(os.path.join(d, "results.json")))
    assert r["unexpected_failures"] == [] and r["gaps"] == []
    for fam, names, cls in (("INIT", INIT, "same stream, new init"), ("DATA", DATA, "same init, new stream")):
        v = [raw(d, n, *args) for n in names]
        assert r["values"][m][fam] == dict(zip(names, v))
        z, sd, mean = r["families"][m][fam], st.stdev(v), st.fmean(v)
        assert (z["n"], z["df"]) == (5, 4) and z["sd"] == pytest.approx(sd, rel=1e-12) and z["rel"] == pytest.approx(sd / mean, rel=1e-12)
        assert z["rel_ci95"] == pytest.approx([sd / mean * Q["lo4"], sd / mean * Q["hi4"]], rel=1e-6)
        assert z["rel_ub80"] == pytest.approx(sd / mean * Q["ub4"], rel=1e-6)
        u, ref = r["rule1"][m][cls], math.sqrt(2) * sd / mean
        assert u["df"] == 4 and u["factor"] == pytest.approx(Q["factor4"], rel=1e-8) and u["ref_rel"] == pytest.approx(ref, rel=1e-12)
        for k in (1, 2, 3, 5):
            assert u["MDE_rel"][str(k)] == pytest.approx(Q["factor4"] * sd / mean * math.sqrt(2 / k), rel=1e-8)
            assert u["MDE_bpb"][str(k)] == pytest.approx(Q["factor4"] * sd * math.sqrt(2 / k), rel=1e-8)
        for dl, key in ((0.005, "0.500%"), (0.01, "1.000%"), (0.02, "2.000%")):
            kn = max(1, math.ceil((Q["factor4"] * ref / dl) ** 2 - 1e-9))
            assert u["k_needed"][key] == kn and u["k_rule2"][key] == max(2, kn) and u["underpowered"][key] == (kn > 5)
        s1 = p1["noise_5M_250M"][m]["sigma_seed_rel"]
        c = r["compare_part1_sigma_seed"][m][fam]
        assert c["ratio"] == pytest.approx(sd / mean / s1, rel=1e-12)
        assert c["ratio_95ci_F"] == pytest.approx([sd / mean / s1 / math.sqrt(Q["F975_4_7"]), sd / mean / s1 / math.sqrt(Q["F025_4_7"])], rel=1e-6)
    zi, zd = r["families"][m]["INIT"], r["families"][m]["DATA"]
    assert r["compare_part1_sigma_seed"][m]["sqrt_init2_plus_data2_over_seed"] == pytest.approx(
        math.sqrt(zi["rel"] ** 2 + zd["rel"] ** 2) / p1["noise_5M_250M"][m]["sigma_seed_rel"], rel=1e-12)
    rp = [raw(d, n, *args) for n in ("e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3")]
    x, v = r["replay_vs_part1_A1"][m], raw(d, "e3_5m_A_s1_P3", *args)
    assert x["minus_A1"] == pytest.approx(v - rp[0], abs=1e-12) and x["minus_rep_mean"] == pytest.approx(v - st.fmean(rp), abs=1e-12)
    assert x["in_sigma_rep"] == pytest.approx((v - st.fmean(rp)) / p1["replay"][m]["sigma_rep"], rel=1e-9)
    assert r["replay_stream_equal_A1"] is True
    assert "E3 PART 3 TABLES" in open(os.path.join(d, "tables_part3.txt")).read()


def test_gaps(tmp_path, codelist):
    d = make(tmp_path, codelist, gaps=["e3_5m_A_data2"])
    run(d)
    r = json.load(open(os.path.join(d, "results_part3.json")))
    assert r["gaps"] == ["e3_5m_A_data2"] and r["unexpected_failures"] == ["e3_5m_A_data2: FIXED hashes, settings, one clean run"]
    z, u = r["families"]["F(CHAT)"]["DATA"], r["rule1"]["F(CHAT)"]["same init, new stream"]
    v = [raw(d, n, "CHAT") for n in DATA if n != "e3_5m_A_data2"]
    assert (z["n"], z["df"], u["df"]) == (4, 3, 3) and z["sd"] == pytest.approx(st.stdev(v), rel=1e-12)
    assert u["factor"] == pytest.approx(Q["factor3"], rel=1e-8) and r["families"]["F(CHAT)"]["INIT"]["n"] == 5
    d2 = make(tmp_path / "two", codelist, gaps=["e3_5m_A_data2", "e3_5m_A_data5"])
    run(d2)
    r2 = json.load(open(os.path.join(d2, "results_part3.json")))
    assert r2["families"]["F(CHAT)"]["DATA"] is None and "keeps sqrt(2) x sigma_seed" in r2["rule1"]["F(CHAT)"]["same init, new stream"]
    assert r2["families"]["F(PROSE)"]["INIT"]["n"] == 5


def test_quantiles_recorded(tmp_path, codelist):
    d = make(tmp_path, codelist)
    run(d)
    q = json.load(open(os.path.join(d, "results_part3.json")))["quantiles"]
    assert q["factor_df4"] == pytest.approx(Q["factor4"], rel=1e-8) and q["ub80_df4"] == pytest.approx(Q["ub4"], rel=1e-6)
    assert (q["chi2_sd_lo_df4"], q["chi2_sd_hi_df4"]) == (pytest.approx(Q["lo4"], rel=1e-6), pytest.approx(Q["hi4"], rel=1e-6))
    assert (q["F_0.025_df4_7"], q["F_0.975_df4_7"]) == (pytest.approx(Q["F025_4_7"], rel=1e-6), pytest.approx(Q["F975_4_7"], rel=1e-6))


@pytest.mark.skipif(bool(os.environ.get("E3P3_CODE")), reason="the data mutants test the committed analyzer only")
def test_data_mutants_on_the_fixture_all_killed(codelist):
    r = subprocess.run([sys.executable, os.path.join(E3, "e3analyze_p3_mutants.py"), "--fixture"], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "baseline failing: 5 " in r.stdout and "SURVIVED" not in r.stdout
