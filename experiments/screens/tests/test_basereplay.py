"""AMENDMENT BASE-REPLAY's READINGS and LABEL ON THE STAGE 2 PAIRS, and the BASE LEVEL reading over BASE seeds 101 to
105 (AMENDMENT BASE-DIAG), through basereplay.py on synthetic values and fixture run trees only (written 2026-10-11
before any value of the replay, hook-off or base_s103..105 runs was read).
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_basereplay.py
"""
import json, math, os, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import basereplay as B  # noqa: E402
from test_analyze import write_run  # noqa: E402

E3 = json.load(open(os.path.join(AL.E3D, "results.json")))
E3REP = E3["replay"]
CHAT_REL = E3REP["F(CHAT)"]["sigma_rep_rel"]


def test_sd_rep_pools_e3_and_the_same_day_pair():
    assert B.sd_rep(0.0015, 1.0, 1.0, 1.0) == pytest.approx(math.sqrt(2 * 0.0015 ** 2 / 3))
    pair = 0.01 / math.sqrt(2) / 1.005
    assert B.sd_rep(0.0015, 1.01, 1.0, 2.0) == pytest.approx(math.sqrt((2 * 0.0015 ** 2 + pair ** 2) / 3) * 2.0)


def test_smallest_d_beyond_replay_noise_is_the_entrys_0_00555():
    """BASE-REPLAY READINGS: 0.00555 bpb (0.48% of base_s101's 1.15250) when the same-day pair agrees exactly."""
    sd = B.sd_rep(CHAT_REL, 1.0, 1.0, 1.15250)
    x = AL.t_ppf(0.975, 3) * sd * math.sqrt(1.5)
    assert round(x, 5) == 0.00555 and round(100 * x / 1.15250, 2) == 0.48 and round(AL.t_ppf(0.975, 3), 3) == 3.182
    assert B.shift(x * 1.001, sd)["beyond"] and not B.shift(x * 0.999, sd)["beyond"]
    assert B.shift(-x * 1.001, sd)["reads"] == "beyond replay noise"
    assert B.shift(x * 0.5, sd)["reads"] == "not separated from replay noise"


def vals(b, r1, r2, h, prose=1.38):
    mk = lambda c: None if c is None else {"CHAT": c, "PROSE": prose, "cccc": 1.3, "gutenberg": 1.5, "wikimedia": 1.3}  # noqa: E731
    return {B.B101: mk(b), B.REP1: mk(r1), B.REP2: mk(r2), B.HOOK: mk(h)}


def test_d_and_h_signs_and_missing_runs():
    r = B.readings(vals(1.150, 1.140, 1.142, 1.130), E3REP)["CHAT"]
    assert r["D"]["value"] == pytest.approx(1.141 - 1.150) and r["H"]["value"] == pytest.approx(1.130 - 1.141)
    assert r["rep1_minus_rep2"] == pytest.approx(-0.002) and r["df"] == 3
    assert B.readings(vals(1.15, 1.14, None, 1.13), E3REP)["CHAT"]["D"] == "not available"
    assert B.readings(vals(1.15, 1.14, 1.14, None), E3REP)["CHAT"]["H"] == "not available"
    assert B.readings(vals(1.15, 1.14, 1.14, None), E3REP)["CHAT"]["D"]["value"] == pytest.approx(-0.01)


def contrast(d_chat, verdict, sd=0.00378, d_prose=(0.0, 0.0)):
    rd = {}
    for k in B.KEYS:
        d = list(d_chat) if k == "CHAT" else list(d_prose) if k == "PROSE" else [0.0, 0.0]
        rd[k] = AL.metric_reading(d, sd, 7, 1.15 if k == "CHAT" else 1.38)
        rd[k]["base_mean"] = 1.15 if k == "CHAT" else 1.38
    v, _ = AL.arm_verdict(rd)
    assert v == verdict
    return {"verdict": v, "readings": rd}


def test_relabel_moves_d_by_minus_d():
    c = contrast([-0.002, 0.001], "TIE")
    zero = {k: 0.0 for k in B.KEYS}
    assert B.relabel(c, zero) == "BASE replay: stable"
    up = {**zero, "CHAT": 0.012}             # BASE at the replay level is 0.012 higher: d - D is 0.012 lower
    assert B.relabel(c, up) == "BASE replay: sensitive (NOMINATE at the replay level)"
    down = {**zero, "CHAT": -0.012}
    assert B.relabel(c, down) == "BASE replay: sensitive (REJECT at the replay level)"
    assert B.relabel(c, {**zero, "PROSE": -0.02}) == "BASE replay: sensitive (GUARD-FAIL at the replay level)"
    assert B.relabel(c, {**zero, "PROSE": None}) == "BASE replay: not available"
    assert B.relabel({"verdict": "INCONCLUSIVE", "readings": {}}, zero) == "BASE replay: not available"


def test_relabel_uses_the_registered_sd():
    c = contrast([0.006, 0.007], "INCONCLUSIVE", sd=0.004)
    zero = {k: 0.0 for k in B.KEYS}
    assert B.relabel(c, {**zero, "CHAT": -0.003}) == "BASE replay: sensitive (REJECT at the replay level)"
    c2 = contrast([0.006, 0.007], "INCONCLUSIVE", sd=0.02)
    assert B.relabel(c2, {**zero, "CHAT": -0.003}) == "BASE replay: stable"


def test_relabel_onset():
    o = {**AL.onset_reading([-153, -153], 50.0, 4), "reading": "rule"}
    assert o["label"] == "EARLIER"
    assert B.relabel_onset(o, 0.0) == "BASE replay: stable"
    assert B.relabel_onset(o, -306.0) == "BASE replay: sensitive (LATER at the replay level)"
    assert B.relabel_onset(o, None) == "not available"
    assert B.relabel_onset({"reading": "descriptive"}, 0.0) == "not applicable (onset descriptive)"


def test_base_level_n5_n3_and_the_seed_draw_reading(tmp_path):
    e3 = [1.140, 1.141, 1.142, 1.143, 1.144, 1.145, 1.146, 1.147]
    b = {101: 1.150, 102: 1.152, 103: 1.149, 104: 1.150, 105: 1.151}
    r = B.base_level(b, e3, 0.003, 7, (101, 102, 103, 104, 105))
    gap = 1.1504 - 1.1435
    assert r["gap"] == pytest.approx(gap) and r["t"] == pytest.approx(gap / (0.003 * math.sqrt(1 / 5 + 1 / 8)))
    assert r["p_two_sided"] == pytest.approx(2 * (1 - AL.t_cdf(r["t"], 7))) and all(r["above_all_8"])
    r3 = B.base_level(b, e3, 0.003, 7, (103, 104, 105))
    assert r3["t"] == pytest.approx((1.150 - 1.1435) / (0.003 * math.sqrt(1 / 3 + 1 / 8)))
    e3o, runs = str(tmp_path / "e3"), str(tmp_path / "runs")
    for s, c in enumerate(e3, 1):
        write_run(e3o, f"e3_5m_A_s{s}", c, 1.38)
    for s, c in b.items():
        write_run(runs, f"base_s{s}", c, 1.39)
    lv = B.level(runs, e3o, E3["noise_5M_250M"])
    assert lv["seed_draw"] == "the gap is not the seed draw" and lv["CHAT"]["n3"]["seeds"] == [103, 104, 105]
    write_run(runs, "base_s104", 1.146, 1.39)            # one fresh seed inside E3's range
    lv = B.level(runs, e3o, E3["noise_5M_250M"])
    assert lv["seed_draw"] == "the seed draw is not excluded" and lv["CHAT"]["n3"]["above_all_8"] == [True, False, True]
    write_run(runs, "base_s105", 1.146, 1.39)
    write_run(runs, "base_s101", 1.140, 1.39)            # seed 101 inside E3's range does not enter the n 3 rule
    write_run(runs, "base_s104", 1.150, 1.39)
    write_run(runs, "base_s105", 1.151, 1.39)
    assert B.level(runs, e3o, E3["noise_5M_250M"])["seed_draw"] == "the gap is not the seed draw"
    os.remove(os.path.join(runs, "base_s103", "bpb.jsonl"))
    assert B.level(runs, e3o, E3["noise_5M_250M"])["reading"] == "not available"


def test_analyze_labels_only_the_shared_base_pairs(tmp_path):
    runs = str(tmp_path)
    for n, c in ((B.B101, 1.150), (B.REP1, 1.150), (B.REP2, 1.150), (B.HOOK, 1.150)):
        write_run(runs, n, c, 1.38)
    c = contrast([-0.002, 0.001], "TIE")
    ver = {"contrasts": {k: c for k in ("S004.canonac", "S005.forget", "S006.mtp", "S007.smear", "S002.novres")},
           "onsets": {"contrasts": {}}}
    r = B.analyze(runs, ver, str(tmp_path / "noe3"), E3)
    assert r["labels"] == {"S004.canonac": "BASE replay: stable", "S005.forget": "not applicable",
                           "S006.mtp": "BASE replay: stable", "S007.smear": "BASE replay: stable",
                           "S002.novres": "not applicable"}
    os.remove(os.path.join(runs, B.REP2, "bpb.jsonl"))
    r = B.analyze(runs, ver, str(tmp_path / "noe3"), E3)
    assert r["labels"]["S004.canonac"] == "BASE replay: not available" and r["readings"]["CHAT"]["D"] == "not available"
