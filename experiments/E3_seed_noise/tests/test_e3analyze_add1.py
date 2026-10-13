"""e3analyze_p1b.py and e3analyze_p3_20m.py (notes.txt ADDENDUM 2026-10-12, ANALYZERS) on tests/add1fixture.py's synthetic
record set, before any Part 1b or Part 3 at 20M value exists: the checks fail exactly on Part 3's five expected code
checks; every registered statistic is recomputed here from the raw fixture files by another route (statistics.stdev,
closed-form pooling and k_needed, a t density integrated numerically for p, scipy 1.18.1 quantiles typed in); GAPs; and
the DATA mutants (e3analyze_add1_mutants.py --fixture) all killed. E3A1_CODE (e3analyze_add1_mutants.py --code) points
at a folder holding mutant analyzers. No git, no model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_e3analyze_add1.py
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
import add1fixture as F  # noqa: E402

CODE = os.environ.get("E3A1_CODE") or E3
EXPECTED = {"code digest harness+data_prep .py == FIXED 2 (92deaef0)", "harness/train.py == FIXED 2",
            "harness/trainer.py == FIXED 2", "harness/model.py == FIXED 2", "data_prep/bpb.py == FIXED 2"}
Q = {"f4": 3.717409682432975, "f3": 4.160918617647012, "f7": 3.2606538959065494, "f15": 2.9976945187547277,
     "f16": 2.984572301019546, "chi": {4: (0.5991331391341409, 2.873555634078219, 1.55757542255663),
     9: (0.6878352067489997, 1.8256101825195739, 1.2933854444729525), 16: (0.744769836860888, 1.521928948871894, 1.1977918206163645)},
     "t975": {13: 2.1603686564627913, 16: 2.1199052992212546}, "F44": (0.10411753745392764, 9.604529884722858),
     "F42": (0.0939045553889328, 39.248417658131466)}   # scipy 1.18.1
METRICS = [("F(CHAT)", ("CHAT",)), ("F(PROSE)", ("PROSE",)), ("M(CHAT)", ("CHAT", "all", "M")), ("CHAT user", ("oasst2", "user")),
           ("train loss last 2%", (None, None, "T"))]
P1B = [f"e3_5m_A_s{s}" for s in range(12, 22)]
INIT = ["e3_20m_A_s1_P3"] + [f"e3_20m_A_init{k}" for k in range(2, 6)]
DATA = ["e3_20m_A_s1_P3"] + [f"e3_20m_A_data{k}" for k in range(2, 6)]


def make(tmp_path, gaps=()):
    d = F.build(str(tmp_path / "root"), code=CODE)
    for g in gaps:
        open(os.path.join(d, "out", g, "GAP"), "w").close()
    return d


def run(d, script, *a):
    r = subprocess.run([sys.executable, os.path.join(d, script), *a], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def raw(path, s, sp="all", kind="F", size="5m"):
    """A metric straight from a run directory's files (not through any analyzer)."""
    total, cut = (7630, 7477.4) if size == "5m" else (15259, 14953.82)
    if kind == "T":
        return st.fmean(json.loads(x)["loss"] for x in open(os.path.join(path, "train_tail.jsonl")) if json.loads(x)["step"] > cut)
    rows = [json.loads(x) for x in open(path if path.endswith(".jsonl") else os.path.join(path, "bpb.jsonl"))]
    v = {r["step"]: r["bpb"] for r in rows if r["set"] == s and r["split"] == sp}
    return v[total] if kind == "F" else st.fmean(v.values())


def p_two(t, df, n=4000):
    """Two-sided p of t on df by Simpson's rule over the t density (independent of e3analyze's incomplete beta)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)   # noqa: E731
    h = abs(t) / n
    s = f(0) + f(abs(t)) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, n))
    return 2 * (0.5 - s * h / 3)


def kneed(f, ref, dl): return max(1, math.ceil((f * ref / dl) ** 2 - 1e-9))
def ss(x): return sum((v - st.fmean(x)) ** 2 for v in x)


def test_checks_fail_only_the_expected_five(tmp_path):
    d = make(tmp_path)
    for script, n in (("e3analyze_p1b.py", 32), ("e3analyze_p3_20m.py", 33)):
        ck = json.loads(run(d, script, "--checks-only"))
        assert {c["check"] for c in ck if not c["ok"]} == EXPECTED and len(ck) == n, script


@pytest.mark.parametrize("m,args", METRICS)
def test_part1b_statistics_by_another_route(tmp_path, m, args):
    d = make(tmp_path)
    run(d, "e3analyze_p1b.py")
    r = json.load(open(os.path.join(d, "results_part1b.json")))
    s1 = json.load(open(os.path.join(d, "results.json")))["noise_5M_250M"][m]
    assert r["unexpected_failures"] == [] and r["gaps"] == [] and r["commit"] == F.COMMIT
    b = [raw(os.path.join(d, "out", n), *args) for n in P1B]
    a = [raw(os.path.join(d, "out", f"e3_5m_A_s{s}"), *args) for s in range(1, 9)]
    assert list(r["values"][m]["part1b"].values()) == b and list(r["values"][m]["part1_A"].values()) == a
    z, sd = r["part1b"][m], st.stdev(b)
    lo, hi, u = Q["chi"][9]
    assert (z["n"], z["df"]) == (10, 9) and z["rel"] == pytest.approx(sd / st.fmean(b), rel=1e-12)
    assert z["rel_ci95"] == pytest.approx([sd / st.fmean(b) * lo, sd / st.fmean(b) * hi], rel=1e-6)
    assert z["rel_ub80"] == pytest.approx(sd / st.fmean(b) * u, rel=1e-6)
    p, sp = r["pooled"][m], math.sqrt((ss(a) + ss(b)) / 16)
    lo, hi, u = Q["chi"][16]
    rel = sp / st.fmean(a + b)
    assert (p["n"], p["df"]) == (18, 16) and p["sd"] == pytest.approx(sp, rel=1e-12) and p["rel"] == pytest.approx(rel, rel=1e-12)
    assert p["rel_ci95"] == pytest.approx([rel * lo, rel * hi], rel=1e-6) and p["rel_ub80"] == pytest.approx(rel * u, rel=1e-6)
    assert r["pooled_over_part1_sigma_seed"][m] == pytest.approx(rel / s1["sigma_seed_rel"], rel=1e-12)
    for key, sdr, f in (("unpaired_pooled", rel, Q["f16"]), ("unpaired_part1", s1["sigma_seed_rel"], Q["f7"])):
        w = r[key][m]
        assert w["factor"] == pytest.approx(f, rel=1e-8) and w["ref_rel"] == pytest.approx(math.sqrt(2) * sdr, rel=1e-12)
        for k in (1, 2, 3, 5):
            assert w["MDE_rel"][str(k)] == pytest.approx(f * sdr * math.sqrt(2 / k), rel=1e-8)
        for dl, k in ((0.005, "0.500%"), (0.01, "1.000%"), (0.02, "2.000%")):
            kn = kneed(f, math.sqrt(2) * sdr, dl)
            assert w["k_needed"][k] == kn and w["k_rule2"][k] == max(2, kn) and w["underpowered"][k] == (kn > 5)
    for k in ("1", "2", "3", "5"):     # Part 1's unpaired MDE, as results.json recorded it
        assert r["unpaired_part1"][m]["MDE_rel"][k] == pytest.approx(s1["MDE"][k]["unpaired_rel"], rel=1e-6)
    lv = r["level_vs_part1"][m]
    diff, se = st.fmean(b) - st.fmean(a), sp * math.sqrt(1 / 10 + 1 / 8)
    assert lv["diff"] == pytest.approx(diff, abs=1e-12) and lv["t"] == pytest.approx(diff / se, rel=1e-9) and lv["df"] == 16
    assert lv["p_two_sided"] == pytest.approx(p_two(diff / se, 16), abs=1e-7)
    assert lv["ci95"] == pytest.approx([diff - Q["t975"][16] * se, diff + Q["t975"][16] * se], rel=1e-6, abs=1e-9)
    if args[2:] == ("T",):
        assert r["values"][m]["base"] == {} and m not in r["level_vs_base"]
        return
    bs = [raw(os.path.join(d, "..", "..", F.BASE[n]), *args) for n in F.BASE]
    lb, sb = r["level_vs_base"][m], math.sqrt((ss(b) + ss(bs)) / 13)
    diff, se = st.fmean(b) - st.fmean(bs), sb * math.sqrt(1 / 10 + 1 / 5)
    assert list(r["values"][m]["base"].values()) == bs and lb["df"] == 13 and lb["diff"] == pytest.approx(diff, abs=1e-12)
    assert lb["t"] == pytest.approx(diff / se, rel=1e-9) and lb["p_two_sided"] == pytest.approx(p_two(diff / se, 13), abs=1e-7)
    assert lb["ci95"] == pytest.approx([diff - Q["t975"][13] * se, diff + Q["t975"][13] * se], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("m,args", METRICS)
def test_part3_20m_statistics_by_another_route(tmp_path, m, args):
    d = make(tmp_path)
    run(d, "e3analyze_p3_20m.py")
    r = json.load(open(os.path.join(d, "results_part3_20m.json")))
    r3 = json.load(open(os.path.join(d, "results_part3.json")))["families"][m]
    s2 = json.load(open(os.path.join(d, "results_part2.json")))["noise_20M_500M"][m]["sigma_seed_rel"]
    assert r["unexpected_failures"] == [] and r["gaps"] == [] and r["commit"] == F.COMMIT
    rels = {}
    for fam, names, cls in (("INIT", INIT, "same stream, new init"), ("DATA", DATA, "same init, new stream")):
        v = [raw(os.path.join(d, "out", n), *args, size="20m") for n in names]
        assert r["values"][m][fam] == dict(zip(names, v))
        z, sd, mean = r["families"][m][fam], st.stdev(v), st.fmean(v)
        rels[fam] = sd / mean
        assert (z["n"], z["df"]) == (5, 4) and z["rel"] == pytest.approx(sd / mean, rel=1e-12)
        assert z["rel_ci95"] == pytest.approx([sd / mean * Q["chi"][4][0], sd / mean * Q["chi"][4][1]], rel=1e-6)
        u = r["rule1_20M"][m][cls]
        assert u["factor"] == pytest.approx(Q["f4"], rel=1e-8) and u["ref_rel"] == pytest.approx(math.sqrt(2) * sd / mean, rel=1e-12)
        w, r20, r5 = r["rule4_20M"][m][cls], math.sqrt(2) * sd / mean, math.sqrt(2) * r3[fam]["rel"]
        used = max(r20, r5)
        assert w["used"] == ("20M" if r20 >= r5 else "5M") and w["ref_rel"] == pytest.approx(used, rel=1e-12) and w["df"] == 4
        assert w["rel_20M"] == pytest.approx(r20, rel=1e-12) and w["rel_5M"] == pytest.approx(r5, rel=1e-12)
        for k in (1, 2, 3, 5):
            assert w["MDE_rel"][str(k)] == pytest.approx(Q["f4"] * used / math.sqrt(k), rel=1e-8)
        for dl, k in ((0.005, "0.500%"), (0.01, "1.000%"), (0.02, "2.000%")):
            kn = kneed(Q["f4"], used, dl)
            assert w["k_needed"][k] == kn and w["k_rule2"][k] == max(2, kn) and w["underpowered"][k] == (kn > 5)
        c5, c2 = r["compare_5M"][m][fam], r["compare_part2_sigma_seed"][m][fam]
        q5, q2 = sd / mean / r3[fam]["rel"], sd / mean / s2
        assert c5["ratio"] == pytest.approx(q5, rel=1e-12) and c2["ratio"] == pytest.approx(q2, rel=1e-12)
        assert c5["ratio_95ci_F"] == pytest.approx([q5 / math.sqrt(Q["F44"][1]), q5 / math.sqrt(Q["F44"][0])], rel=1e-6)
        assert c2["ratio_95ci_F"] == pytest.approx([q2 / math.sqrt(Q["F42"][1]), q2 / math.sqrt(Q["F42"][0])], rel=1e-6)
    assert r["compare_part2_sigma_seed"][m]["sqrt_init2_plus_data2_over_seed"] == pytest.approx(
        math.sqrt(rels["INIT"] ** 2 + rels["DATA"] ** 2) / s2, rel=1e-12)
    x = r["replay_vs_part2_A_s1"][m]
    v, w = raw(os.path.join(d, "out", "e3_20m_A_s1_P3"), *args, size="20m"), raw(os.path.join(d, "out", "e3_20m_A_s1"), *args, size="20m")
    assert x["diff"] == pytest.approx(v - w, abs=1e-12)


def test_gaps(tmp_path):
    d = make(tmp_path / "a", gaps=["e3_5m_A_s15", "e3_20m_A_data2"])
    run(d, "e3analyze_p1b.py"); run(d, "e3analyze_p3_20m.py")
    r1 = json.load(open(os.path.join(d, "results_part1b.json")))
    assert r1["gaps"] == ["e3_5m_A_s15"] and r1["unexpected_failures"] == ["e3_5m_A_s15: FIXED hashes, settings, one clean run"]
    assert (r1["part1b"]["F(CHAT)"]["df"], r1["pooled"]["F(CHAT)"]["df"], r1["unpaired_pooled"]["F(CHAT)"]["factor"]) == (
        8, 15, pytest.approx(Q["f15"], rel=1e-8))
    r2 = json.load(open(os.path.join(d, "results_part3_20m.json")))
    z, w = r2["families"]["F(CHAT)"]["DATA"], r2["rule4_20M"]["F(CHAT)"]["same init, new stream"]
    assert (z["n"], z["df"], r2["families"]["F(CHAT)"]["INIT"]["n"]) == (4, 3, 5) and w["df_20M"] == 3
    assert (w["df"], w["factor"]) == ((3, pytest.approx(Q["f3"], rel=1e-8)) if w["used"] == "20M" else (4, pytest.approx(Q["f4"], rel=1e-8)))
    assert r2["rule1_20M"]["F(CHAT)"]["same init, new stream"]["factor"] == pytest.approx(Q["f3"], rel=1e-8)
    d = make(tmp_path / "b", gaps=[f"e3_5m_A_s{s}" for s in (12, 13, 14, 15, 16)] + ["e3_20m_A_data2", "e3_20m_A_data5"])
    run(d, "e3analyze_p1b.py"); run(d, "e3analyze_p3_20m.py")
    r1 = json.load(open(os.path.join(d, "results_part1b.json")))
    assert r1["pooled"]["F(CHAT)"].startswith("not computed: 5 Part 1b runs") and r1["part1_A"]["F(CHAT)"]["n"] == 8
    r2 = json.load(open(os.path.join(d, "results_part3_20m.json")))
    assert r2["families"]["F(CHAT)"]["DATA"] is None and r2["rule4_20M"]["F(CHAT)"]["same init, new stream"].startswith("no 20M SD")
    assert r2["families"]["F(PROSE)"]["INIT"]["n"] == 5


@pytest.mark.skipif(bool(os.environ.get("E3A1_CODE")), reason="the data mutants test the committed analyzers only")
def test_data_mutants_on_the_fixture_all_killed():
    r = subprocess.run([sys.executable, os.path.join(E3, "e3analyze_add1_mutants.py"), "--fixture"], capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert r.stdout.count("baseline failing: 5 ") == 2 and "SURVIVED" not in r.stdout and "INVALID" not in r.stdout
