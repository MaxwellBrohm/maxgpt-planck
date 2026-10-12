"""AMENDMENT C6-DIAG's READING RULE, CHECKS and NEAR ITS INIT (SCREENS.txt) through c6read_lib.py and c6read.py, on
synthetic values and fixture run trees only. Written 2026-10-11 before any real C6 value was read (09:47:53 EDT),
EXCEPT, written after it: the base_by_seed and base_s103 to 105 assertions in test_readings_on_a_fixture_tree (09:52)
and, after the audit (10:45), test_reread_at_a_smaller_sd and the sd_ref_registered / sensitivity assertions.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_c6read.py
"""
import json, math, os, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import c6read as R  # noqa: E402
import c6read_lib as C  # noqa: E402
from test_analyze import write_run  # noqa: E402

NZ = {"SD_d": 0.1, "sigma_seed": 0.2, "df": 7}
T7 = AL.t_ppf(0.975, 7)


def test_reading_both_directions_and_every_seed_clause():
    assert C.reading([0.5, 0.6], 0.1, 7)["label"] == "HIGHER"
    assert C.reading([-0.5, -0.6], 0.1, 7)["label"] == "LOWER"
    assert C.reading([1.5, -0.01], 0.1, 7)["label"] == "UNRESOLVED"        # mean far above thr, one seed below 0
    assert C.reading([-1.5, 0.01], 0.1, 7)["label"] == "UNRESOLVED"
    r = C.reading([0.1, 0.1], 0.1, 7)
    assert r["thr"] == pytest.approx(T7 * 0.1 / math.sqrt(2)) and r["label"] == "UNRESOLVED"   # 0.1 < thr 0.167
    assert r["p_two_sided"] == pytest.approx(2 * (1 - AL.t_cdf(0.1 / (0.1 / math.sqrt(2)), 7)))


def test_sd_ref_per_class():
    assert C.u_of(7) == pytest.approx(1.3533, abs=1e-4)
    assert C.sd_ref("SIA", NZ) == pytest.approx(1.3533 * 0.1, abs=1e-5)
    assert C.sd_ref("same init", NZ) == 0.1
    assert C.sd_ref("new init", NZ) == pytest.approx(math.sqrt(2) * 0.2)
    assert C.sd_ref("SIA", {**NZ, "df": 6}) == pytest.approx(C.u_of(6) * 0.1) and C.u_of(6) > C.u_of(7)


def test_descriptive_when_no_sd_or_a_value_missing():
    c = C.contrast({101: 1.0, 102: 2.0}, {101: 0.5, 102: 2.5}, "SIA", None)
    assert c["reading"] == "descriptive" and c["higher"] == 1 and c["lower"] == 1 and c["dbar"] == 0.0
    c = C.contrast({101: 1.0, 102: None}, {101: 0.5, 102: 2.5}, "SIA", NZ)
    assert c["reading"] == "descriptive" and c["d"] == [0.5] and "102" in str(c["why"])
    c = C.contrast({101: 1.0, 102: float("nan")}, {101: 0.5, 102: 2.5}, "SIA", NZ)
    assert c["reading"] == "descriptive"
    assert C.contrast({101: 1.0, 102: 1.0}, {101: 0.5, 102: 0.5}, "SIA", NZ)["reading"] == "rule"


def vals(spread):
    """Four SIA arms and one same-init arm; spread widens the SIA arms' seed difference."""
    v = {f"S00{i}.x": ({101: 0.5 + spread, 102: 0.5 - spread}, {101: 0.0, 102: 0.0}) for i in (4, 5, 6, 7)}
    v["S003.adamw"] = ({101: 0.3, 102: 0.31}, {101: 0.0, 102: 0.0})
    return v, {**{k: "SIA" for k in v}, "S003.adamw": "same init"}


def test_pooled_sia_check_and_reread():
    v, cls = vals(0.001)
    r = C.read_metric(v, cls, NZ)
    assert r["noise_checks"]["pooled_sia"]["ok"] and r["noise_checks"]["pooled_sia"]["df"] == 4
    assert all(c["class"] == "SIA" and c["labels"] == [] for k, c in r["contrasts"].items() if k != "S003.adamw")
    assert "sensitivity" not in r                       # nothing re-read, nothing to report
    v, cls = vals(0.3)
    r = C.read_metric(v, cls, NZ)
    pc = r["noise_checks"]["pooled_sia"]
    assert not pc["ok"] and pc["limit"] == pytest.approx(C.u_of(7) * 0.1 * AL.sd_bound(4))
    for k in ("S004.x", "S005.x", "S006.x", "S007.x"):
        c = r["contrasts"][k]
        assert c["class"] == "new init" and c["class_registered"] == "SIA"
        assert c["labels"] == ["noise check failed (pooled SIA)", "underpowered"]   # 0.283 > SIA's 0.135: larger
        assert c["sd_ref"] == pytest.approx(math.sqrt(2) * 0.2)
        assert c["sd_ref_registered"] == pytest.approx(C.u_of(7) * 0.1)
    assert r["contrasts"]["S003.adamw"]["class"] == "same init"
    assert r["sensitivity"]["reread"] == ["S004.x", "S005.x", "S006.x", "S007.x"]
    assert r["sensitivity"]["smaller_than_registered"] == []


def test_reread_at_a_smaller_sd():
    """C4 labels "underpowered" only "an arm re-read with a larger SD". sigma_seed 0.05 puts the new-init SD
    (sqrt(2) x 0.05 = 0.0707) BELOW the SIA SD (1.3533 x 0.1 = 0.135) whose pooled check fails: no "underpowered", the
    replaced SD is recorded, and the sensitivity block reads the re-read arms at the SIA SD and at the observed pooled
    SD (0.2263, df 4), each with Holm over the whole family of 6 (S001 new init, S003 same init beside them)."""
    nz, z = {**NZ, "sigma_seed": 0.05}, {101: 0.0, 102: 0.0}
    v = {"S004.x": ({101: 0.66, 102: 0.34}, z), "S005.x": ({101: 0.36, 102: 0.04}, z),
         "S006.x": ({101: 0.46, 102: 0.14}, z), "S007.x": ({101: -0.14, 102: -0.46}, z),
         "S001.x": ({101: 0.4, 102: 0.41}, z), "S003.adamw": ({101: 0.3, 102: 0.31}, z)}
    cls = {"S004.x": "SIA", "S005.x": "SIA", "S006.x": "SIA", "S007.x": "SIA", "S001.x": "new init",
           "S003.adamw": "same init"}
    r = C.read_metric(v, cls, nz)
    pc, sia = r["noise_checks"]["pooled_sia"], ["S004.x", "S005.x", "S006.x", "S007.x"]
    assert not pc["ok"] and pc["df"] == 4 and pc["sd"] == pytest.approx(0.16 * math.sqrt(2))
    for k in sia:
        c = r["contrasts"][k]
        assert c["labels"] == ["noise check failed (pooled SIA)"] and c["class"] == "new init"
        assert c["sd_ref"] == pytest.approx(math.sqrt(2) * 0.05) and c["sd_ref_registered"] == pytest.approx(C.u_of(7) * 0.1)
        assert c["holm"] == "holds"
    assert {k: c["label"] for k, c in r["contrasts"].items()} == {**{k: "HIGHER" for k in v}, "S007.x": "LOWER"}
    s = r["sensitivity"]
    assert s["reread"] == sia and s["smaller_than_registered"] == sia
    reg, po = s["at_registered_sd"], s["at_pooled_sd"]
    assert sorted(reg["readings"]) == sia and sorted(po["readings"]) == sia
    for k in sia:
        x, y, d = reg["readings"][k], po["readings"][k], v[k][0][101] / 2 + v[k][0][102] / 2
        assert x["sd_ref"] == pytest.approx(C.u_of(7) * 0.1) and x["df"] == 7
        assert x["thr"] == pytest.approx(T7 * C.u_of(7) * 0.1 / math.sqrt(2))
        assert x["p_two_sided"] == pytest.approx(2 * (1 - AL.t_cdf(abs(d) / (C.u_of(7) * 0.1 / math.sqrt(2)), 7)))
        assert y["sd_ref"] == pytest.approx(pc["sd"]) and y["df"] == 4
        assert y["thr"] == pytest.approx(AL.t_ppf(0.975, 4) * pc["sd"] / math.sqrt(2))
    assert {k: x["label"] for k, x in reg["readings"].items()} == {"S004.x": "HIGHER", "S005.x": "UNRESOLVED",
                                                                   "S006.x": "HIGHER", "S007.x": "LOWER"}
    assert {k: x["label"] for k, x in po["readings"].items()} == {"S004.x": "HIGHER", **{k: "UNRESOLVED" for k in sia[1:]}}
    assert reg["holm"] == {**{k: "holds" for k in v}, "S005.x": "does not hold"}          # family of 6, not of 4
    assert po["holm"] == {**{k: "does not hold" for k in sia}, "S001.x": "holds", "S003.adamw": "holds"}
    assert all(reg["readings"][k]["holm"] == reg["holm"][k] for k in sia)


def test_same_init_check():
    v = {"S003.adamw": ({101: 0.9, 102: 0.1}, {101: 0.0, 102: 0.0})}
    r = C.read_metric(v, {"S003.adamw": "same init"}, NZ)
    c = r["contrasts"]["S003.adamw"]
    assert not r["noise_checks"]["S003.adamw"]["ok"] and c["class"] == "new init"
    assert c["labels"] == ["noise check failed", "underpowered"] and c["sd_ref_registered"] == 0.1
    assert "at_pooled_sd" not in r["sensitivity"] and r["sensitivity"]["at_registered_sd"]["readings"]["S003.adamw"][
        "sd_ref"] == 0.1
    v, cls = vals(0.3)                                   # both checks fail: only the SIA arms get the pooled-SD reading
    v["S003.adamw"] = ({101: 0.9, 102: 0.1}, {101: 0.0, 102: 0.0})
    s = C.read_metric(v, cls, NZ)["sensitivity"]
    assert "S003.adamw" in s["reread"] and sorted(s["at_pooled_sd"]["readings"]) == ["S004.x", "S005.x", "S006.x", "S007.x"]
    v = {"S003.adamw": ({101: 0.5, 102: 0.51}, {101: 0.0, 102: 0.0})}
    r = C.read_metric(v, {"S003.adamw": "same init"}, NZ)
    assert r["noise_checks"]["S003.adamw"]["ok"] and r["contrasts"]["S003.adamw"]["sd_ref"] == 0.1


def test_holm_per_metric_steps_down():
    h = C.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert h == {"a": True, "b": True, "c": True}
    h = C.holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert h == {"a": True, "b": False, "c": False}
    v, cls = vals(0.001)
    r = C.read_metric(v, cls, NZ)
    assert r["holm_family"] == sorted(v) and all(c["holm"] in ("holds", "does not hold") for c in r["contrasts"].values())
    r = C.read_metric(v, cls, None)
    assert r["holm_family"] == [] and all(c["holm"] is None for c in r["contrasts"].values())


def test_s005_flag():
    assert C.s005_flag("BETTER", {"label": "LOWER"})
    assert not C.s005_flag("BETTER", {"label": "UNRESOLVED"}) and not C.s005_flag("EQUAL", {"label": "LOWER"})
    assert not C.s005_flag("BETTER", None)


def test_near_init():
    can = lambda a, c: {"canon": [{"layer": i, "site": s, "mean_abs": v} for i in range(8) for s, v in (("A", a), ("C", c))]}  # noqa: E731
    r = C.near_init("S004", {101: can(0.005, 0.2), 102: can(0.009, 0.3)})
    assert r["fires"] and r["sites_near_init"] == ["A"]
    assert not C.near_init("S004", {101: can(0.005, 0.2), 102: can(0.011, 0.3)})["fires"]   # both seeds needed
    sm = lambda a: {"smear_alpha": [{"layer": i, "alpha": [a, -a / 2, 0.0]} for i in range(8)]}  # noqa: E731
    assert C.near_init("S007", {101: sm(0.04), 102: sm(-0.049)})["fires"]
    assert not C.near_init("S007", {101: sm(0.04), 102: sm(0.06)})["fires"]
    hl = lambda h: {"by_layer": [{"half_life": [h, h * 3, 5000.0]} for _ in range(8)]}  # noqa: E731
    assert C.near_init("S005", {101: hl(1100.0), 102: hl(1024.0)})["fires"]
    assert not C.near_init("S005", {101: hl(1100.0), 102: hl(900.0)})["fires"]


SEEDS = (101, 102)
RUNS = {"S001.nogate": ("s001_nogate_g1_s{}", "base_s{}"), "S004.canonac": ("s004_canonac_g1_s{}", "base_s{}"),
        "S005.forget": ("s005_forget_g1_s{}", "s005_base_s{}")}
CODE = {"harness/model.py": "c0de"}


def put(runs, name, chat, ind, hd_a, sink, sha="ck", code=CODE, hd_chat=None, hook=None):
    write_run(runs, name, chat, 1.38)
    d = os.path.join(runs, name, "c6")
    os.makedirs(d, exist_ok=True)
    meta = {"checkpoint_sha256": sha, "code_sha256": code}
    role = lambda h: {"HD": h, "HD_none": 2 * h, "own": 1.0, "foreign": 1.0 + h, "none": 1.0 + 2 * h}  # noqa: E731
    hd = {"sets": {"oasst2": {"bpb": hd_chat or chat, "hd": {"by_role": {"assistant": role(hd_a), "user": role(hd_a / 2)},
                                                            "by_turn": {}, "windows": 9, "kept": 5}}}}
    for job, res in (("ind", {"ind": ind}), ("hd", hd), ("attn", {"sink": [[sink, sink]] * 2, "gate_mean": None})):
        json.dump({**meta, "result": res}, open(os.path.join(d, f"{job}.json"), "w"))
    with open(os.path.join(runs, name, "diag.jsonl"), "w") as f:
        f.write(json.dumps({"step": 153, "ind": -1.0}) + "\n")
        f.write(json.dumps({"step": 7630, "ind": ind if hook is None else hook}) + "\n")


def tree(tmp, arm_hd=0.03, **kw):
    runs = str(tmp)
    for key, (a, b) in RUNS.items():
        for s in SEEDS:
            put(runs, b.format(s), 1.15, 8.0, 0.10, 0.5, sha=f"ck{b.format(s)}")
            put(runs, a.format(s), 1.14, 8.5, arm_hd, 0.4 if key == "S001.nogate" else 0.6, sha=f"ck{a.format(s)}", **kw)
    for s in (103, 104, 105):
        put(runs, f"base_s{s}", 1.15, 7.0 + s - 103, 0.09, 0.5, sha=f"ckbase_s{s}")
    rec = {"ckpt_sha256": {a.format(s): f"ck{a.format(s)}" for v in RUNS.values() for a in v for s in SEEDS}
                   | {f"base_s{s}": f"ckbase_s{s}" for s in (103, 104, 105)},
           "code_sha256": CODE}
    picks = {k: {"decided": True, "pick": 1.0} for k in RUNS} | {"S003": {}}
    ver = {"picks": picks, "contrasts": {k: {"readings": {"CHAT": {"label": "BETTER"}}} for k in RUNS}}
    p4 = {"metrics": {m: {"part1_5M": {"noise": {"SD_d": 0.01, "sigma_seed": 0.02, "df": 7}, "scorer_replay": 0.0,
                                     "values": {"s1": [0.5, 0.6], "s2": [0.7, 0.8]}}}
                       for m in R.P4.METRICS}}
    return runs, ver, p4, rec


def test_readings_on_a_fixture_tree(tmp_path):
    runs, ver, p4, rec = tree(tmp_path)
    r = R.readings(runs, ver, p4, rec)
    assert all(c["ok"] for c in r["checks"].values()) and len(r["checks"]) == 13
    bb = r["base_by_seed"]["IND"]
    assert bb["base"] == {"base_s101": 8.0, "base_s102": 8.0, "base_s103": 7.0, "base_s104": 8.0, "base_s105": 9.0}
    assert bb["e3_arm_a"] == [0.5, 0.7] and r["base_by_seed"]["HD_asst"]["base"]["base_s104"] == 0.09
    hd = r["metrics"]["HD_asst"]["contrasts"]
    assert {k: c["label"] for k, c in hd.items()} == {k: "LOWER" for k in RUNS}
    assert hd["S004.canonac"]["d"] == pytest.approx([-0.07, -0.07])
    assert r["metrics"]["IND"]["contrasts"]["S001.nogate"]["class"] == "new init"
    assert r["metrics"]["HDn_asst"]["contrasts"]["S005.forget"]["dbar"] == pytest.approx(-0.14)
    for m in ("HDn_asst", "HDn_user"):                  # HD_none: reported beside HD, never read by the rule (C6)
        assert all("label" not in c and "holm" not in c for c in r["metrics"][m]["contrasts"].values())
    assert r["s005_flag"]["flagged"]
    sp = r["reported"]["s001_sink_paired"]
    assert sp["gate_on_minus_off_per_seed"][101] == pytest.approx([0.1, 0.1]) and sp["seeds_negative_per_layer"] == [[], []]
    assert r["checks"]["base_s101"]["check3_ind_minus_hook_abs"] == 0.0


def test_checks_gate_the_readings(tmp_path):
    runs, ver, p4, rec = tree(tmp_path, hd_chat=1.14 * (1 + 2e-5))       # check 1 off by 2e-5 relative on the arms
    r = R.readings(runs, ver, p4, rec)
    assert not r["checks"]["s004_canonac_g1_s101"]["ok"] and "check 1" in r["checks"]["s004_canonac_g1_s101"]["why"][0]
    assert r["metrics"]["HD_asst"]["contrasts"]["S004.canonac"]["reading"] == "descriptive"
    runs, ver, p4, rec = tree(tmp_path / "b", hd_chat=1.14 * (1 + 5e-6))
    assert R.readings(runs, ver, p4, rec)["checks"]["s004_canonac_g1_s101"]["ok"]
    rec["ckpt_sha256"]["s004_canonac_g1_s102"] = "other"
    r = R.readings(runs, ver, p4, rec)
    assert not r["checks"]["s004_canonac_g1_s102"]["ok"]
    assert r["metrics"]["IND"]["contrasts"]["S004.canonac"]["reading"] == "descriptive"
    rec["ckpt_sha256"]["s004_canonac_g1_s102"] = "cks004_canonac_g1_s102"
    rec["code_sha256"] = {"harness/model.py": "other"}
    r = R.readings(runs, ver, p4, rec)
    assert not any(c["ok"] for c in r["checks"].values())


def test_check3_reported_not_gating(tmp_path):
    runs, ver, p4, rec = tree(tmp_path, hook=8.25)
    r = R.readings(runs, ver, p4, rec)
    assert r["checks"]["s004_canonac_g1_s101"]["check3_ind_minus_hook_abs"] == pytest.approx(0.25)
    assert r["checks"]["s004_canonac_g1_s101"]["ok"]
