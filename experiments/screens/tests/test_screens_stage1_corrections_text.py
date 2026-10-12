"""SCREENS STAGE 1 VERDICTS, corrected 2026-10-06: the BASE level against E3 Part 1 arm A (recomputed from E3's out/ and
this batch's out/ copies), S003's F(CHAT) per GPU-hour, the registered wording (S002's parenthetical, S003's "at larger
size", E3 AUDIT Q3's qualifier, S003's does_not_show, S001's GATE reading), the tok/s medians and the p column.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_corrections_text.py
"""
import json, math, os, re, statistics as st, sys  # noqa: E401
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import screens_lib as L  # noqa: E402
from e3analyze import t_cdf  # noqa: E402
from test_screens_s003_stage_c import section  # noqa: E402
from test_screens_stage1_verdicts import EXP, SIDS, flat, res, text  # noqa: E402

E3 = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]
OUT = os.path.join(EXP, "S001_attn_gate", "out")
SEC = flat(section("STAGE 1 VERDICTS"))


def fix(sid):
    return flat(text(sid, "notes.txt").partition("\nCORRECTIONS (2026-10-06")[2])


def final(path):
    return {r["set"]: r["bpb"] for r in map(json.loads, open(path)) if r["ckpt"].startswith("final_") and r["split"] == "all"}


def jl(path):
    return [json.loads(x) for x in open(path)]


def test_base_level_against_e3_arm_a_recomputed():
    a = [final(os.path.join(L.E3D, "out", f"e3_5m_A_s{s}", "bpb.jsonl")) for s in range(1, 9)]
    b = [final(os.path.join(OUT, f"base_s{s}", "bpb.jsonl")) for s in (101, 102)]
    for sid in SIDS:
        lv = res(sid)["base_level_vs_e3_part1"]
        for m in ("CHAT", "PROSE"):
            xa, xb, x = [r[m] for r in a], [r[m] for r in b], lv["metrics"][m]
            gap = st.fmean(xb) - st.fmean(xa)
            t = gap / (E3[f"F({m})"]["sigma_seed"] * math.sqrt(1 / 2 + 1 / 8))
            assert (x["e3_arm_a"], x["base"], x["base_above_all_8"]) == (xa, xb, True) and min(xb) > max(xa)
            assert (x["gap"], x["t"], x["p_two_sided"]) == pytest.approx((gap, t, 2 * (1 - t_cdf(t, 7))), rel=1e-12)
        ratio = {m: st.fmean(r[m] for r in a) / st.fmean(r[m] for r in b) for m in ("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia")}
        for k, c in res(sid)["contrasts"].items():
            rd = {m: AL.metric_reading(y["d"], y["sd_ref"] * ratio[m], y["df"], y["base_mean"] * ratio[m]) for m, y in c["readings"].items()}
            assert lv["verdicts_with_sd_ref_and_delta_at_e3_mean"][k] == AL.arm_verdict(rd)[0] == c["verdict"]
        g = lv["lr_check_gnorm_last2pct"]
        tail = [r for r in jl(os.path.join(OUT, "base_s101", "log.jsonl")) if r["step"] >= 7480]
        assert g["base"][0] == pytest.approx(st.fmean(r["gnorm"] for r in tail)) and len(tail) == 16
        assert max(g["e3_arm_a"]) < min(g["e3_arm_b"]) and max(g["base"]) < min(g["e3_arm_b"])
    x = res("S001")["base_level_vs_e3_part1"]["metrics"]
    c, p = x["CHAT"], x["PROSE"]
    assert (f"both BASE seeds sit above all 8 of E3 Part 1 arm A's seeds on F(CHAT) ({c['base'][0]:.5f} / {c['base'][1]:.5f} "
            f"against {min(c['e3_arm_a']):.5f} to {max(c['e3_arm_a']):.5f}; gap {c['gap']:+.5f}, {100 * c['gap_rel']:+.2f}%; "
            f"t {c['t']:.2f} with E3's sigma_seed") in SEC
    assert f"(gap {p['gap']:+.5f}, {100 * p['gap_rel']:+.2f}%; t {p['t']:.2f}, p {p['p_two_sided']:.3f})" in SEC
    assert "Cause not separated: the seed draw" in SEC and "at E3 arm A's mean all five verdicts are unchanged" in SEC
    for sid in SIDS:
        assert f"gap {c['gap']:+.5f}, {100 * c['gap_rel']:+.2f}%; t {c['t']:.2f}" in fix(sid)
        assert "Not separated" in fix(sid) or "are not separated" in fix(sid)


def test_s003_f_chat_per_gpu_hour_is_formed_from_the_measured_hours():
    r = res("S003")
    x, f, h = r["optimizer_cost"]["f_chat_per_gpu_hour"], r["c5_reported"]["adamw"]["F(CHAT)"], r["c6"]["hours_lock_train"]
    want = {f"base_s{s}": f["base"][i] / h[f"base_s{s}"][1] for i, s in enumerate((101, 102))}
    want |= {f"s003_adamw_e3_r2_s{s}": f["arm"][i] / h[f"s003_adamw_e3_r2_s{s}"][1] for i, s in enumerate((101, 102))}
    assert x["per_run"] == pytest.approx(want, rel=1e-12) and x["formula"].startswith("F(CHAT) / h_train per run")
    v = [x["per_run"][n] for n in sorted(want)]
    assert (f"NorMuon (BASE) {v[0]:.3f} / {v[1]:.3f}, mean {x['mean_normuon_base']:.3f}; AdamW {v[2]:.3f} / {v[3]:.3f}, "
            f"mean {x['mean_adamw']:.3f} bpb per GPU-hour") in fix("S003")
    assert "not formed" not in json.dumps(r) and "F(CHAT) per GPU-hour = F(CHAT) / train hours per run" in text("S003", "tables.txt")


def test_the_registered_wording():
    r2, r3 = res("S002"), res("S003")
    paren = re.search(r"\(row 19's 30M check[^)]*\)", r2["consequence_registered"]["REJECT"])[0]
    for where in (r2["consequence"], text("S002", "tables.txt"), fix("S002"), SEC):
        assert paren in flat(where)
    flip = "A later optimizer flip at larger size is then not looked for"
    assert flip in flat(r3["consequence_registered"])
    for where in (r3["consequence"], text("S003", "tables.txt"), fix("S003")):
        assert flip in flat(where) and "at 30M would go unseen" not in flat(where)
    assert "A later optimizer flip at larger size is then not looked for (the registered REJECT; S003 KNOWN RISKS names " \
           "30M)" in SEC and "flip at 30M then goes unseen" not in SEC
    assert not [d for d in r3["does_not_show"] if "SINK" in d or "attention or block" in d]
    q3 = "confirmed against B only, not against 6e-3"
    assert q3 in flat(section("FIXED, BATCH PARAMETERS"))
    for where in (r3["asymmetry_as_run"], fix("S003")):
        assert "confirmed E2's pick against E3's arm B only, not against 6e-3 (E3 AUDIT.md Q3" in flat(where)
    assert "No run here measures how much of this REJECT the asymmetry produced" in fix("S003")
    assert res("S001")["c6"]["gate_reading"].startswith("no threshold is registered for gate openness")
    assert "did not stay open" not in SEC and "\"gate stays open\" case is not decided here" in SEC
    assert "is not decided here" in fix("S001")


def test_tok_s_medians_and_the_p_column():
    for sid in SIDS:
        r, t = res(sid), text(sid, "tables.txt")
        out = os.path.join(EXP, L.SCREENS[sid]["dir"], "out")
        for n, v in r["c6"]["tok_per_s_median"].items():
            toks = [x["tok_per_s"] for x in jl(os.path.join(out, n, "log.jsonl"))]
            assert len(toks) == 763 and v == st.median(toks[2:]) and r["c6"]["tok_per_s_median_all_records"][n] == st.median(toks)
        assert "median tok/s\nover log.jsonl records 3 to 763: the first two records" in t
        rows = [ln.split() for ln in t.splitlines() if re.match(r"^\w+\s+(CHAT|PROSE|cccc|gutenberg|wikimedia)\s", ln)]
        assert rows and all(float(x[9]) > 0 and x[9] != "0.0000" for x in rows)


SIA_S1 = {"S002.novres": "s002_novres_g1_s1", "S002.noqknorm": "s002_noqknorm_g1_s1", "S002.nonormscale": "s002_nonormscale_g1_s1"}


def test_same_seed_check_at_seed_1():
    """N-B at one seed: S002's SIA arms keep every BASE draw and read the seed's stream (C4), so each seed-1 g 1 run
    minus its paired dbar (seeds 101, 102) estimates the screens' BASE at seed 1, against E3 A_s1 and its replays."""
    from test_screens_stage1_next import RECORDED
    con = res("S002")["contrasts"]
    for sid in SIDS:
        lv = res(sid)["base_level_vs_e3_part1"]
        x = lv["same_seed_check_seed1"]
        assert x["arm_runs"] == SIA_S1 and x["e3_runs"] == ["e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3"]
        assert x == res("S001")["base_level_vs_e3_part1"]["same_seed_check_seed1"]
        for i, m in enumerate(("CHAT", "PROSE")):
            y, sdd = x["metrics"][m], E3[f"F({m})"]["SD_d"]
            e3 = [final(os.path.join(L.E3D, "out", n, "bpb.jsonl"))[m] for n in x["e3_runs"]]
            assert y["e3"] == e3 and y["e3_mean"] == pytest.approx(st.fmean(e3), rel=1e-12)
            for k, n in SIA_S1.items():                          # the stored seed-1 values are the recorded ones
                assert round(y["arm_seed1"][k], 5) == RECORDED[n][i] and y["dbar"][k] == con[k]["readings"][m]["dbar"]
                assert y["base_s1_est"][k] == pytest.approx(y["arm_seed1"][k] - y["dbar"][k], rel=1e-12)
            shift = st.fmean(y["base_s1_est"].values()) - st.fmean(e3)
            se = math.sqrt(sdd ** 2 * 1.5 / 3 + st.stdev(e3) ** 2 / 3)
            gap = lv["metrics"][m]["gap"]
            assert (y["shift"], y["rough_se"], y["gap_101_102"]) == pytest.approx((shift, se, gap), rel=1e-12)
            assert (y["z_vs_0"], y["z_vs_gap"], y["share_of_gap"]) == pytest.approx((shift / se, (shift - gap) / se, shift / gap), rel=1e-12)
            assert 0 < shift < gap and shift / se < 2 < (gap - shift) / se       # between the two readings, as stated
    c, p = x["metrics"]["CHAT"], x["metrics"]["PROSE"]
    phrase = (f"{c['shift']:+.5f} CHAT ({100 * c['shift_rel']:+.2f}%), {p['shift']:+.5f} PROSE ({100 * p['shift_rel']:+.2f}%), "
              f"{100 * c['share_of_gap']:.0f}% and {100 * p['share_of_gap']:.0f}% of the 101/102 gap; {c['z_vs_0']:.1f} and "
              f"{p['z_vs_0']:.1f} rough se above 0, {-c['z_vs_gap']:.1f} and {-p['z_vs_gap']:.1f} below that gap")
    assert phrase in SEC and all(phrase in fix(sid) for sid in SIDS)
    for sid in SIDS:
        t = text(sid, "tables.txt")
        assert f"same seed (seed 1, S002's SIA arms minus dbar, against E3 A_s1 and 2 replays): CHAT {c['shift']:+.5f}" in t
        dna = res(sid)["base_level_vs_e3_part1"]["does_not_affect"]
        assert "would show as a spread" not in dna and "only if it varies across seeds" in dna
    for where in [SEC] + [fix(sid) for sid in SIDS]:
        assert "would show as a spread" not in where and "would spread d across seeds" not in where
