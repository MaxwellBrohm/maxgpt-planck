"""experiments/E3_seed_noise/e3analyze_4d.py (E3 PART 4d ADDENDUM) on fixture runner JSON in a scratch out/ tree.
Every statistic is recomputed here with the statistics module; the role and key reads, the pair-dropping rule, the
6-pair floor, the replays and the absolute units are checked on values chosen so that a swap shows.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_e3_part4d.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "experiments", "E3_seed_noise")
sys.path.insert(0, E3)
import e3analyze_4d as A  # noqa: E402


def write(out, run, job, result, suffix=""):
    d = out / run / "c6"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job}{suffix}.json").write_text(json.dumps({"run": run, "job": job, "result": result}))


def hd(asst, user, asst_n, user_n):
    return {"sets": {"oasst2": {"hd": {"by_role": {"assistant": {"HD": asst, "HD_none": asst_n},
                                                     "user": {"HD": user, "HD_none": user_n}}}}}}


def fill(out, skip=(), bad=()):
    """IND A_s = 2 + s/10, B_s = A_s + 0.03 + s/1000; HD by role and key on separate scales (1, 10, 100, 1000)."""
    vals = {}
    for size, seeds in (("5m", range(1, 9)), ("20m", (1, 2, 3))):
        for s in seeds:
            for arm, off in (("A", 0.0), ("B", 0.03 + s / 1000)):
                run = f"e3_{size}_{arm}_s{s}"
                if run in skip:
                    continue
                x = 2 + s / 10 + off + (s % 3) * 0.007 * (arm == "B")
                vals[run] = x
                write(out, run, "ind", {"ind": float("nan") if run in bad else x, "nll_copy1": 9.0, "nll_copy2": 9 - x})
                write(out, run, "hd", hd(x, 10 * x + s / 7, 100 * x - s / 5, 1000 * x))
    for i, r in enumerate(("e3_5m_A_s1_R2", "e3_5m_A_s1_R3")):
        vals[r] = 2.1 + (i + 1) * 0.002
        write(out, r, "ind", {"ind": vals[r]})
        write(out, r, "hd", hd(vals[r], 0, 0, 0))
    write(out, "e3_5m_A_s1", "ind", {"ind": 2.1 + 1e-6}, ".replay")
    return vals


def test_part1_statistics_recomputed(tmp_path):
    vals = fill(tmp_path)
    r = A.analyze(str(tmp_path))
    z = r["metrics"]["IND"]["part1_5M"]["noise"]
    a = [vals[f"e3_5m_A_s{s}"] for s in range(1, 9)]
    b = [vals[f"e3_5m_B_s{s}"] for s in range(1, 9)]
    d = [y - x for x, y in zip(a, b)]
    assert z["n"] == 8 and z["df"] == 7 and math.isclose(z["SD_d"], st.stdev(d))
    assert math.isclose(z["sigma_seed"], math.sqrt((st.variance(a) + st.variance(b)) / 2))
    assert math.isclose(z["mean_d"], st.fmean(d)) and math.isclose(z["mean_a"], st.fmean(a))
    assert abs(z["factor"] - 3.261) < 1e-3                               # E3 STATISTICS: 3.261 at df 7
    assert math.isclose(z["MDE"]["2"]["paired"], z["factor"] * st.stdev(d) / math.sqrt(2))
    assert math.isclose(z["MDE"]["1"]["unpaired_ub80"], z["factor"] * z["sigma_seed"] * math.sqrt(2) * 1.3533, rel_tol=1e-3)
    s2 = r["metrics"]["IND"]["part1_5M"]["noise_seeds2to8"]
    assert s2["n"] == 7 and math.isclose(s2["SD_d"], st.stdev(d[1:]))
    assert not any("rel" in k or "k_needed" in k for k in z)              # absolute units only
    rep = r["metrics"]["IND"]["part1_5M"]["replay"]
    assert rep["values"] == [vals["e3_5m_A_s1"], vals["e3_5m_A_s1_R2"], vals["e3_5m_A_s1_R3"]]
    assert math.isclose(rep["sigma_rep"], st.stdev(rep["values"])) and rep["df"] == 2
    assert math.isclose(r["metrics"]["IND"]["part1_5M"]["scorer_replay"], 1e-6, rel_tol=1e-6)
    assert r["metrics"]["HD_asst"]["part1_5M"]["scorer_replay"] == "not computed"
    p2 = r["metrics"]["IND"]["part2_20M"]["noise"]
    assert p2["n"] == 3 and p2["df"] == 2


def test_each_metric_reads_its_own_role_and_key(tmp_path):
    vals = fill(tmp_path)
    r = A.analyze(str(tmp_path))
    for m, f in (("HD_asst", lambda x, s: x), ("HD_user", lambda x, s: 10 * x + s / 7),
                 ("HDn_asst", lambda x, s: 100 * x - s / 5), ("HDn_user", lambda x, s: 1000 * x)):
        a = [f(vals[f"e3_5m_A_s{s}"], s) for s in range(1, 9)]
        z = r["metrics"][m]["part1_5M"]["noise"]
        assert math.isclose(z["mean_a"], st.fmean(a)), m
    assert r["ruled"] == ["IND", "HD_asst", "HD_user"]


def test_a_missing_or_non_finite_value_drops_its_pair(tmp_path):
    vals = fill(tmp_path, skip={"e3_5m_B_s3"}, bad={"e3_5m_A_s5"})
    p = A.analyze(str(tmp_path))["metrics"]["IND"]["part1_5M"]
    assert p["pairs"] == 6 and p["dropped"] == ["s3", "s5"] and p["noise"]["df"] == 5
    keep = [s for s in range(1, 9) if s not in (3, 5)]
    d = [vals[f"e3_5m_B_s{s}"] - vals[f"e3_5m_A_s{s}"] for s in keep]
    assert math.isclose(p["noise"]["SD_d"], st.stdev(d))
    hdp = A.analyze(str(tmp_path))["metrics"]["HD_asst"]["part1_5M"]
    assert hdp["pairs"] == 7 and hdp["dropped"] == ["s3"]                # the NaN was IND's only


def test_under_six_pairs_gives_no_sd(tmp_path):
    fill(tmp_path, skip={"e3_5m_B_s1", "e3_5m_B_s2", "e3_5m_B_s3"})
    p = A.analyze(str(tmp_path))["metrics"]["IND"]["part1_5M"]
    assert p["pairs"] == 5 and p["noise"] is None and "no SD" in p["why"] and p["replay"] == "not computed"


def test_main_writes_both_files_where_told(tmp_path):
    fill(tmp_path / "out")
    dest = tmp_path / "dest"
    dest.mkdir()
    assert A.main(["--out", str(tmp_path / "out"), "--dest", str(dest)]) == 0
    res = json.loads((dest / "results_part4d.json").read_text())
    txt = (dest / "tables_part4d.txt").read_text().splitlines()
    assert set(res["metrics"]) == set(A.METRICS) and len(txt) == 2 + 2 * len(A.METRICS)
    assert txt[2].split()[:3] == ["IND", "5M", "8"]
