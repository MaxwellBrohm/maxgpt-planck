"""SCREENS analyzer (analyze_lib.py, analyze.py): every C7 verdict branch in both directions, the every-seed
clause, delta, DIVERGENCE (+inf, never dropped; a diverged BASE seed dropped), CCCC-ONLY, the SIA u factor, C4's
noise fallbacks (same init per arm, pooled SIA), Holm, the cap and cut order, C3 picks through E2's rule, and the
RC-12 run-directory refusal. No model; fixtures are numbers and fake run directories.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_analyze.py
"""
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze as A  # noqa: E402
import analyze_lib as AL  # noqa: E402
import screens_lib as L  # noqa: E402

NOISE = {m: {"SD_d_rel": 0.002, "sigma_seed_rel": 0.003, "df": 7} for m in AL.METRICS.values()}
U = 1.35
THR1 = AL.t_ppf(0.975, 7) * 0.002 / math.sqrt(2)          # same init, BASE mean 1.0, k = 2: about 0.00334


def runs_of(dchat, dprose=None, src=None, base=1.0, diverged=()):
    """Arm and BASE values at seeds 101.. with BASE at `base` on every metric and arm = BASE + d."""
    dprose = dprose or [0.0] * len(dchat)
    src = src or {}
    arm, bas = {}, {}
    for i, (c, p) in enumerate(zip(dchat, dprose)):
        s = 101 + i
        bas[s] = {k: base for k in AL.METRICS}
        arm[s] = {"diverged": True} if s in diverged else {
            "CHAT": base + c, "PROSE": base + p, **{k: base + src.get(k, [0.0] * len(dchat))[i]
                                                    for k in ("cccc", "gutenberg", "wikimedia")}}
    return arm, bas


def verdict(dchat, dprose=None, cls="same init", **kw):
    arm, base = runs_of(dchat, dprose, **kw)
    return AL.contrast(arm, base, cls, NOISE, U)


@pytest.mark.parametrize("dchat,dprose,want", [
    ([-0.006, -0.005], None, "NOMINATE"), ([0.006, 0.005], None, "REJECT"),
    ([0.0005, -0.0003], None, "TIE"), ([0.02, -0.001], None, "INCONCLUSIVE"), ([-0.02, 0.001], None, "INCONCLUSIVE"),
    ([-0.006, -0.005], [0.006, 0.005], "GUARD-FAIL"), ([0.0005, -0.0003], [0.006, 0.005], "GUARD-FAIL"),
    ([0.006, 0.005], [-0.006, -0.005], "REJECT")])
def test_verdict_branches(dchat, dprose, want):
    src = {"gutenberg": dprose, "wikimedia": dprose} if dprose else None
    assert verdict(dchat, dprose, src=src)["verdict"] == want


def test_metric_labels_both_directions_and_thresholds():
    r = AL.metric_reading([-0.004, -0.004], 0.002, 7, 1.0)
    assert r["label"] == "BETTER" and abs(r["thr"] - THR1 * math.sqrt(2) / math.sqrt(2)) < 1e-12
    assert AL.metric_reading([0.004, 0.004], 0.002, 7, 1.0)["label"] == "WORSE"
    assert AL.metric_reading([0.004, -0.0001], 0.002, 7, 1.0)["label"] == "EQUAL"       # not every d_s > 0
    assert AL.metric_reading([0.009, 0.004], 0.002, 7, 1.0)["label"] == "WORSE"
    assert AL.metric_reading([0.009, -0.0001], 0.002, 7, 1.0)["label"] == "EQUAL"       # 0.00445 + thr < 1%
    assert AL.metric_reading([0.02, -0.0001], 0.002, 7, 1.0)["label"] == "UNRESOLVED"   # |dbar| + thr > delta
    assert AL.metric_reading([0.001, 0.0], 0.002, 7, 2.0)["delta"] == 0.02                # delta = 1% of BASE


def test_cccc_only_does_not_block_a_nomination_but_a_tie_keeps_the_guard():
    src = {"cccc": [0.02, 0.02], "gutenberg": [0.0, 0.0], "wikimedia": [0.001, 0.0]}
    v = verdict([-0.006, -0.005], [0.006, 0.005], src=src)
    assert v["verdict"] == "NOMINATE" and "CCCC-ONLY" in v["labels"]
    v = verdict([0.0005, -0.0003], [0.006, 0.005], src=src)
    assert v["verdict"] == "GUARD-FAIL" and "CCCC-ONLY" in v["labels"]
    v = verdict([-0.006, -0.005], [0.006, 0.005], src={**src, "wikimedia": [0.006, 0.005]})
    assert v["verdict"] == "GUARD-FAIL" and "CCCC-ONLY" not in v["labels"]


def test_divergence_scores_inf_and_is_never_dropped():
    v = verdict([0.0, 0.006, 0.005], diverged=(101,))
    assert v["verdict"] == "REJECT" and "UNSTABLE" in v["labels"] and v["readings"]["CHAT"]["dbar"] == AL.INF
    v = verdict([0.0, -0.009, -0.008], diverged=(101,))          # dropping the pair would read NOMINATE
    assert v["verdict"] == "INCONCLUSIVE" and "UNSTABLE" in v["labels"] and v["seeds"] == [101, 102, 103]
    arm, base = runs_of([-0.009, -0.008, -0.007])
    base[101] = {"diverged": True}
    v = AL.contrast(arm, base, "same init", NOISE, U)
    assert v["seeds"] == [102, 103] and v["dropped_base_diverged"] == [101] and v["verdict"] == "NOMINATE"
    base[102] = {"diverged": True}
    assert AL.contrast(arm, base, "same init", NOISE, U)["verdict"] == "INCONCLUSIVE"


def test_sia_uses_u_times_sd_d_and_new_init_sqrt2_sigma_seed():
    assert verdict([-0.004, -0.004], cls="same init")["verdict"] == "NOMINATE"
    v = verdict([-0.004, -0.004], cls="SIA")                       # thr x 1.35 > 0.004: not BETTER
    assert v["verdict"] == "TIE" and abs(v["readings"]["CHAT"]["sd_ref"] - U * 0.002) < 1e-12
    assert abs(verdict([0.0, 0.0], cls="new init")["readings"]["CHAT"]["sd_ref"] - math.sqrt(2) * 0.003) < 1e-12


def test_same_init_noise_check_rereads_with_the_new_init_sd():
    c = verdict([-0.02, 0.01])
    chk = AL.same_init_check(c, NOISE)
    assert not chk["CHAT"]["ok"] and abs(chk["CHAT"]["limit_rel"] - 0.002 * math.sqrt(AL.chi2_ppf(0.95, 1))) < 1e-9
    assert AL.same_init_check(verdict([-0.006, -0.005]), NOISE)["CHAT"]["ok"]
    r = AL.reread_new_init(c, *runs_of([-0.02, 0.01]), NOISE, U, "noise check failed")
    assert r["class"] == "new init" and r["class_registered"] == "same init"
    assert {"noise check failed", "underpowered"} <= set(r["labels"])


def test_pooled_sia_check():
    quiet = [verdict([-0.004, -0.0035], cls="SIA"), verdict([0.001, 0.0015], cls="SIA")]
    loud = quiet + [verdict([-0.03, 0.02], cls="SIA")]
    q, l_ = AL.pooled_sia_check(quiet, NOISE, U), AL.pooled_sia_check(loud, NOISE, U)
    assert q["CHAT"]["ok"] and q["CHAT"]["df"] == 2 and not l_["CHAT"]["ok"] and l_["CHAT"]["df"] == 3
    assert abs(q["CHAT"]["limit_rel"] - U * 0.002 * math.sqrt(AL.chi2_ppf(0.95, 2) / 2)) < 1e-12


def test_holm_step_down():
    assert AL.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert AL.holm({"a": 0.02, "b": 0.024, "c": 0.001}) == {"c": True, "a": True, "b": True}
    assert AL.holm({"a": 0.02, "b": 0.024, "c": 0.04}) == {"a": False, "b": False, "c": False}
    assert AL.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}


def test_cap_and_cut_order():
    rem = {"BASE": 2.0, "S001": 3.0, "S002": 4.0, "S003": 5.0, "S004": 3.0, "S005": 4.0, "S006": 2.0, "S007": 3.0}
    assert AL.cap_cut(rem)["cut"] == ["S006"] and AL.cap_cut(rem)["fits"]               # 26 h: S006 off -> 24
    assert AL.cap_cut(rem, measured=-1.5)["cut"] == []                                    # 24.5 h fits
    r = AL.cap_cut(rem, measured=1.5)
    assert r["cut"] == ["S006", "S003"] and r["fits"] and abs(r["total_after"] - 20.5) < 1e-12
    r = AL.cap_cut(rem, measured=14.0)
    assert r["cut"] == ["S006", "S003", "S007", "S002"] and not r["fits"]
    assert AL.cap_cut(rem, measured=1.5, toy_favours_p022=True)["cut"] == ["S004"]
    assert AL.cap_cut(rem, measured=0.6)["cut"] == ["S006"]                               # 26.6 - 2 = 24.6
    r = AL.cap_cut(rem, measured=0.6, current={"S006": 1.0})     # S006 keeps its current set: 25.6, then S003
    assert r["cut"] == ["S006", "S003"] and abs(r["total_after"] - 20.6) < 1e-12
    r = AL.cap_cut({"BASE": 5.0, "S001": 5.0, "S004": 5.0, "S005": 5.0}, measured=8.0)
    assert r["cut"] == [] and not r["fits"]


def test_onsets():
    curve = [(153, 0.1), (306, 0.9), (459, 1.6), (7630, 2.0)]
    assert AL.onsets(curve, 3.0) == {"onset_half": 459, "onset_abs": 459, "level_abs": 1.5, "final": 2.0}
    assert AL.onsets(curve, 1.0)["onset_abs"] == 306 and AL.onsets(curve, 5.0)["onset_abs"] == "not reached"


def write_run(runs, name, chat=None, prose=None, diverged=False, rc12=False, src=1.3):
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    if diverged:
        open(os.path.join(d, "DIVERGED"), "w").close()
    if rc12:
        open(os.path.join(d, "rc12_eval.jsonl"), "w").close()
    if chat is not None:
        vals = {"CHAT": chat, "PROSE": prose, "cccc": src, "gutenberg": src, "wikimedia": src}
        with open(os.path.join(d, "bpb.jsonl"), "w") as f:
            for ck in ("ckpt_00007497.pt", "final_00007630.pt"):
                for k, v in vals.items():
                    f.write(json.dumps({"ckpt": ck, "set": k, "split": "all", "bpb": v + (0.1 if ck[0] == "c" else 0)}) + "\n")


def test_c3_pick_uses_e2s_rule(tmp_path):
    runs = str(tmp_path)
    for g, c in ((0.5, 1.150), (1.0, 1.140), (2.0, 1.130)):
        write_run(runs, f"s001_nogate_g{L.num(g)}_s1", c, 1.38)
    r = A.c3_pick(runs, "S001", "nogate")
    assert r["pick"] == 2.0 and r["at_edge"] == "high" and r["extend_with"] == [4.0] and not r["decided"]
    write_run(runs, "s001_nogate_g4_s1", diverged=True)
    r = A.c3_pick(runs, "S001", "nogate")
    assert r["axis"] == [0.5, 1.0, 2.0, 4.0] and r["pick"] == 2.0 and r["decided"]
    write_run(runs, "s002_novres_g0.5_s1", 1.14004, 1.38)                 # tie at 4 decimals: the lower LR
    write_run(runs, "s002_novres_g1_s1", 1.13996, 1.38)
    write_run(runs, "s002_novres_g2_s1", 1.15, 1.38)
    assert A.c3_pick(runs, "S002", "novres")["pick"] == 0.5


def test_rc12_run_directory_is_refused(tmp_path):
    runs = str(tmp_path)
    write_run(runs, "s001_nogate_g1_s1", 1.14, 1.38)
    A.rc12_guard(runs)
    write_run(runs, "base_s101", 1.14, 1.38, rc12=True)
    with pytest.raises(A.Refused, match="RC-12"):
        A.rc12_guard(runs)
    with pytest.raises(A.Refused, match="RC-12"):
        A.final(runs, "base_s101")
    assert A.main(["picks", "--runs", runs]) == 3


def test_end_to_end_verdicts_on_fake_runs(tmp_path, capsys):
    runs = str(tmp_path)
    for g, c in ((0.5, 1.150), (1.0, 1.140), (2.0, 1.145)):
        write_run(runs, f"s001_nogate_g{L.num(g)}_s1", c, 1.38)
    for s, (b, a) in {101: (1.1467, 1.1300), 102: (1.1470, 1.1310)}.items():
        write_run(runs, f"base_s{s}", b, 1.381)
        write_run(runs, f"s001_nogate_g1_s{s}", a, 1.381)
    assert A.main(["verdicts", "--runs", runs, "--out", str(tmp_path / "v.json")]) == 0
    v = json.load(open(tmp_path / "v.json"))
    c = v["contrasts"]["S001.nogate"]
    assert c["class"] == "new init" and c["seeds"] == [101, 102] and c["verdict"] == "NOMINATE"
    assert c["holm"] == "holds" and v["holm_family"] == ["S001.nogate"] and v["picks"]["S001.nogate"]["pick"] == 1.0


def test_c3_extensions_stop_at_two(tmp_path):
    runs = str(tmp_path)
    for g, c in ((0.125, 1.120), (0.25, 1.125), (0.5, 1.130), (1.0, 1.140), (2.0, 1.150)):
        write_run(runs, f"s004_canonac_g{L.num(g)}_s1", c, 1.38)
    r = A.c3_pick(runs, "S004", "canonac")
    assert r["extensions_used"] == 2 and r["pick"] == 0.125 and r["at_edge"] == "low"
    assert r["extend_with"] == [] and r["decided"]


def test_verdicts_apply_the_noise_fallbacks(tmp_path):
    """S003 (same init) with a seed spread far above SD_d is re-read with the new-init SD; three SIA arms whose
    pooled residual SD exceeds u x SD_d send every SIA arm to the new-init SD."""
    runs, p = str(tmp_path), L.params()
    noise = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]
    for s, b in ((101, 1.1467), (102, 1.1470)):
        write_run(runs, f"base_s{s}", b, 1.381)
    write_run(runs, "s003_adamw_e3_r1_s101", 1.1467 - 0.03, 1.381)
    write_run(runs, "s003_adamw_e3_r1_s102", 1.1470 + 0.02, 1.381)
    for a, (x, y) in {"novres": (-0.02, 0.02), "noqknorm": (0.015, -0.015), "nonormscale": (-0.001, -0.0012)}.items():
        write_run(runs, f"s002_{a}_g1_s101", 1.1467 + x, 1.381)
        write_run(runs, f"s002_{a}_g1_s102", 1.1470 + y, 1.381)
    g = {"S002.novres": 1.0, "S002.noqknorm": 1.0, "S002.nonormscale": 1.0}
    v = A.verdicts(runs, p, noise, g, (0.003, 1.0))
    c = v["contrasts"]
    assert c["S003.adamw"]["class"] == "new init" and not c["S003.adamw"]["noise_check"]["CHAT"]["ok"]
    assert {"noise check failed", "underpowered"} <= set(c["S003.adamw"]["labels"])
    assert not v["pooled_sia_check"]["CHAT"]["ok"] and v["pooled_sia_check"]["CHAT"]["df"] == 3
    for a in ("novres", "noqknorm", "nonormscale"):
        assert c[f"S002.{a}"]["class"] == "new init" and c[f"S002.{a}"]["class_registered"] == "SIA"
        assert "noise check failed (pooled SIA)" in c[f"S002.{a}"]["labels"]
    assert sorted(v["holm_family"]) == sorted(c) and all(x["holm"] in ("holds", "does not hold") for x in c.values())


def test_sds_and_residuals_scale_with_a_real_base_mean():
    """E3 carries SDs as a fraction of A's mean (C7: times BASE's mean). Every fixture above sits at BASE 1.0, where
    a fraction and a bpb value are the same number; here BASE is 1.15 (verification 2026-10-04: dropping the scaling
    from sd_ref, or reading the noise-check residuals in bpb, passed every test above)."""
    b = 1.15
    c = verdict([-0.004, -0.002], cls="same init", base=b)
    r = c["readings"]["CHAT"]
    assert abs(r["sd_ref"] - 0.002 * b) < 1e-12
    assert abs(r["thr"] - AL.t_ppf(0.975, 7) * 0.002 * b / math.sqrt(2)) < 1e-12
    assert abs(AL.same_init_check(c, NOISE)["CHAT"]["sd_rel"] - math.sqrt(2) * 0.001 / b) < 1e-12
    sia = [verdict([-0.004, -0.002], cls="SIA", base=b), verdict([0.001, 0.003], cls="SIA", base=b)]
    pooled = AL.pooled_sia_check(sia, NOISE, U)["CHAT"]
    assert pooled["df"] == 2 and abs(pooled["sd_rel"] - math.sqrt(2) * 0.001 / b) < 1e-12


def test_holm_reads_a_two_sided_p():
    """C7's Holm reading uses the two-sided p of dbar / (SD_ref / sqrt(k)): at t(0.975, df) it is 0.05."""
    d = -AL.t_ppf(0.975, 7) * 0.002 / math.sqrt(2)
    assert abs(AL.metric_reading([d, d], 0.002, 7, 1.0)["p_two_sided"] - 0.05) < 1e-9


def test_s005_pairs_with_its_own_mask_engine_base(tmp_path):
    """C4: S005 pairs with its own BASE (s005_base_s<seed>, mask engine, micro 8 x accum 2), never the shared one."""
    runs, p = str(tmp_path), L.params()
    noise = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]
    for s, (shared, own, arm) in {101: (1.1467, 1.2000, 1.1950), 102: (1.1470, 1.2004, 1.1951)}.items():
        write_run(runs, f"base_s{s}", shared, 1.381)
        write_run(runs, f"s005_base_s{s}", own, 1.39)
        write_run(runs, f"s005_forget_g1_s{s}", arm, 1.39)
    c = A.verdicts(runs, p, noise, {"S005.forget": 1.0}, None)["contrasts"]["S005.forget"]
    assert c["class"] == "SIA" and c["seeds"] == [101, 102]
    assert abs(c["readings"]["CHAT"]["base_mean"] - 1.2002) < 1e-12
    assert [round(x, 6) for x in c["readings"]["CHAT"]["d"]] == [-0.005, -0.0053]
    assert c["verdict"] == "TIE"           # against the shared BASE every d_s would be about +0.048: REJECT
