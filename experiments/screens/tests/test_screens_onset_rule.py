"""AMENDMENT BASE-DIAG's ONSET RULE (SCREENS.txt; code: ONSET RULE CODE, 2026-10-10), on synthetic runs only: sigma_seed
= the sample SD of BASE's onset_half over seeds 101-105 (df 4); S004 canonac and S007 smear at g 1 and S006 mtp's g 1
IND-only runs against base_s101/102: d = arm - BASE in steps, SD_ref = sqrt(2) sigma_seed, thr = t(0.975, 4) sigma_seed
at k 2; EARLIER / LATER mirrored with the every-seed clause, no EQUAL; Holm over the contrasts read; the descriptive
fallbacks; no bpb verdict moves. Also AMENDMENT CAP-FLAG's printed flag. No model; no real run is read.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_onset_rule.py
"""
import contextlib, io, json, math, os, statistics as st, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze as A  # noqa: E402
import analyze_lib as AL  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_analyze import write_run  # noqa: E402

P = L.params()
STEPS = list(range(153, L.STEPS, 153)) + [L.STEPS]
BASE5 = {101: 612, 102: 765, 103: 612, 104: 918, 105: 765}           # sigma_seed 128.0 steps (sample SD, df 4)
G1 = {"S004.canonac": 1.0, "S006.mtp": 2.0, "S007.smear": 1.0}      # S006 picks g 2: its onset reads the g 1 IND runs


def diag(runs, name, onset, final=8.0, steps=STEPS, extra=()):
    with open(os.path.join(runs, name, "diag.jsonl"), "w") as f:
        for s in steps:
            f.write(json.dumps({"step": s, "ind": final if s >= onset else final / 4, "gate": None}) + "\n")
        for s, v in extra:
            f.write(json.dumps({"step": s, "ind": v}) + "\n")


def run(runs, name, onset, chat=1.15, **kw):
    write_run(runs, name, chat, 1.38)
    diag(runs, name, onset, **kw)


def fixture(tmp, base=BASE5, arms=None):
    runs = str(tmp)
    for s, o in base.items():
        run(runs, f"base_s{s}", o)
    arms = arms or {"s004_canonac_g1": (153, 306), "s006_mtp_g1": (459, 918), "s007_smear_g1": (1071, 1224)}
    for n, (a, b) in arms.items():
        run(runs, f"{n}_s101", a)
        run(runs, f"{n}_s102", b)
    return runs


def test_sigma_thresholds_both_directions_and_holm(tmp_path):
    o = A.onset_readings(fixture(tmp_path), P, G1)
    sig = st.stdev(BASE5.values())
    assert o["sigma"]["df"] == 4 and o["sigma"]["sigma_seed"] == pytest.approx(sig) and round(sig, 1) == 128.0
    c = o["contrasts"]
    thr = AL.t_ppf(0.975, 4) * sig
    assert {k: (x["reading"], x["label"]) for k, x in c.items()} == {
        "S004.canonac": ("rule", "EARLIER"), "S006.mtp": ("rule", "UNRESOLVED"), "S007.smear": ("rule", "LATER")}
    assert c["S004.canonac"]["d"] == [-459, -459] and c["S007.smear"]["d"] == [459, 459] and c["S006.mtp"]["d"] == [-153, 153]
    for x in c.values():
        assert x["thr"] == pytest.approx(thr) and x["sd_ref"] == pytest.approx(math.sqrt(2) * sig) and x["k"] == 2
        se = math.sqrt(2) * sig / math.sqrt(2)
        assert x["p_two_sided"] == pytest.approx(2 * (1 - AL.t_cdf(abs(x["dbar"]) / se, 4)))
    assert c["S006.mtp"]["runs"] == ["s006_mtp_g1_s101", "s006_mtp_g1_s102"]
    ps = {k: x["p_two_sided"] for k, x in c.items()}
    assert o["holm_family"] == sorted(c) and {k: x["holm"] for k, x in c.items()} == \
        {k: ("holds" if v else "does not hold") for k, v in AL.holm(ps).items()}
    assert c["S006.mtp"]["holm"] == "does not hold" and ps["S004.canonac"] < 0.05


def test_no_equal_and_the_every_seed_clause():
    for d in ([0, 0], [153, -153], [-900, 10], [900, -10]):
        assert AL.onset_reading(d, 128.0, 4)["label"] == "UNRESOLVED", d
    assert AL.onset_reading([-900, -10], 128.0, 4)["label"] == "EARLIER"     # every d_s < 0, mean past thr
    assert AL.onset_reading([900, 10], 128.0, 4)["label"] == "LATER"


def test_one_logged_interval_reads_below_55_1_steps():
    """The amendment: thr = 2.776 x sigma_seed, so one 153-step interval at both seeds reads when sigma_seed < 55.1."""
    assert round(153 / AL.t_ppf(0.975, 4), 1) == 55.1
    assert AL.onset_reading([-153, -153], 55.0, 4)["label"] == "EARLIER"
    assert AL.onset_reading([-153, -153], 55.2, 4)["label"] == "UNRESOLVED"
    assert AL.onset_reading([153, 153], 55.0, 4)["label"] == "LATER"


def test_sigma_zero(tmp_path):
    """Five equal BASE onsets: thr 0, so one interval at both seeds reads; p is 0 for any shift, 1 for none."""
    runs = fixture(tmp_path, {s: 612 for s in BASE5}, {"s004_canonac_g1": (459, 459), "s006_mtp_g1": (612, 612),
                                                       "s007_smear_g1": (459, 612)})
    c = A.onset_readings(runs, P, G1)["contrasts"]
    assert (c["S004.canonac"]["label"], c["S004.canonac"]["thr"], c["S004.canonac"]["p_two_sided"]) == ("EARLIER", 0.0, 0.0)
    assert (c["S006.mtp"]["label"], c["S006.mtp"]["p_two_sided"]) == ("UNRESOLVED", 1.0)
    assert c["S007.smear"]["label"] == "UNRESOLVED"


@pytest.mark.parametrize("damage,why", [
    ("base_s104 missing", "BASE onset_half missing or undefined at seeds [104]"),
    ("base_s103 diverged", "BASE onset_half missing or undefined at seeds [103]"),
    ("base_s105 unscored", "BASE onset_half missing or undefined at seeds [105]"),
    ("base_s102 ind", "base_s102 final IND <= 0")])
def test_a_missing_diverged_unscored_or_undefined_base_makes_every_onset_descriptive(tmp_path, damage, why):
    runs = fixture(tmp_path)
    name = damage.split()[0]
    if damage.endswith("missing"):
        for f in os.listdir(os.path.join(runs, name)):
            os.remove(os.path.join(runs, name, f))
        os.rmdir(os.path.join(runs, name))
    elif damage.endswith("diverged"):
        open(os.path.join(runs, name, "DIVERGED"), "w").close()
    elif damage.endswith("unscored"):
        os.remove(os.path.join(runs, name, "bpb.jsonl"))
    else:
        diag(runs, name, 612, final=-0.01)
    o = A.onset_readings(runs, P, G1)
    assert o["sigma"]["sigma_seed"] is None and o["holm_family"] == []
    for x in o["contrasts"].values():
        assert x["reading"] == "descriptive" and any(why in w for w in x["why"]) and x["holm"] is None
        assert set(x) >= {"d", "dbar", "earlier", "later", "same"}
    assert o["base_states"][int(name[-3:])] == {"missing": "missing", "diverged": "diverged", "unscored": "missing",
                                                 "ind": "final IND <= 0"}[damage.split()[1]]


def test_one_arm_undefined_or_incomplete_leaves_the_others_read(tmp_path):
    runs = fixture(tmp_path)
    diag(runs, "s007_smear_g1_s102", 1224, final=0.0)                             # final IND <= 0
    diag(runs, "s006_mtp_g1_s101", 459, steps=STEPS[:-2])                          # stops at 7,344: no last reading
    o = A.onset_readings(runs, P, G1)
    c = o["contrasts"]
    assert (c["S007.smear"]["reading"], c["S007.smear"]["why"]) == ("descriptive", ["s007_smear_g1_s102 final IND <= 0"])
    assert c["S007.smear"]["d"] == [459] and c["S007.smear"]["later"] == 1 and c["S007.smear"]["dbar"] == 459
    assert c["S006.mtp"]["why"] == ["s006_mtp_g1_s101 diag incomplete"]
    assert c["S004.canonac"]["reading"] == "rule" and o["holm_family"] == ["S004.canonac"]


def test_s004_off_g1_is_lr_confounded_and_s006_reads_its_g1_runs(tmp_path):
    runs = fixture(tmp_path)
    for s in (101, 102):
        run(runs, f"s006_mtp_g2_s{s}", 153)                                        # the verdict runs: not read here
    o = A.onset_readings(runs, P, {**G1, "S004.canonac": 2.0})
    assert o["contrasts"]["S004.canonac"]["reading"] == "LR-confounded, not read"
    assert o["contrasts"]["S006.mtp"]["d"] == [-153, 153]                          # g 1 runs, never g 2's 153


def test_a_resumed_runs_repeated_step_keeps_its_last_reading(tmp_path):
    runs = fixture(tmp_path)
    diag(runs, "s004_canonac_g1_s101", 1530, extra=[(306, 8.0)])     # a pre-crash reading at 306, then the resumed 2.0
    lines = open(os.path.join(runs, "s004_canonac_g1_s101", "diag.jsonl")).read().splitlines()
    open(os.path.join(runs, "s004_canonac_g1_s101", "diag.jsonl"), "w").write("\n".join([lines[-1]] + lines[:-1]) + "\n")
    assert A.onset_half(runs, "s004_canonac_g1_s101") == ("ok", 1530)


def picks_runs(runs):
    for sid, arm in (("s004", "canonac"), ("s006", "mtp"), ("s007", "smear")):
        for g, c in ((0.5, 1.150), (1.0, 1.140), (2.0, 1.145)):
            write_run(runs, f"{sid}_{arm}_g{L.num(g)}_s1", c, 1.38)


def test_end_to_end_onsets_beside_unchanged_verdicts(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, with_diag in ((a, True), (b, False)):
        runs = fixture(root)
        picks_runs(runs)
        if not with_diag:
            for d in os.listdir(runs):
                if os.path.exists(os.path.join(runs, d, "diag.jsonl")):
                    os.remove(os.path.join(runs, d, "diag.jsonl"))
        with contextlib.redirect_stdout(io.StringIO()):
            assert A.main(["verdicts", "--runs", runs, "--out", str(root / "v.json")]) == 0
    va, vb = (json.load(open(r / "v.json")) for r in (a, b))
    assert va["contrasts"] == vb["contrasts"] and sorted(va["contrasts"]) == ["S004.canonac", "S006.mtp", "S007.smear"]
    assert {k: x.get("label") for k, x in va["onsets"]["contrasts"].items()} == \
        {"S004.canonac": "EARLIER", "S006.mtp": "UNRESOLVED", "S007.smear": "LATER"}
    assert all(x["reading"] == "descriptive" for x in vb["onsets"]["contrasts"].values())
    assert "S005.forget" not in va["onsets"]["contrasts"] and va["onsets"]["rule"].endswith("BASE-DIAG, ONSET RULE")


def test_the_hours_report_flags_an_overrun_and_cuts_nothing(tmp_path):
    t = tmp_path / "m.tsv"
    t.write_text(open(H.TSV).read() + "other\tpretend\t6.0\t\t\ta test line\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = H.report(P, str(t), cap=28.0)   # AMENDMENT CAP-FLAG's 28 h (31 h since AMENDMENT CAP-31)
    line = next(ln for ln in out.getvalue().splitlines() if ln.startswith("ORDER cap"))
    assert r["flag"]["flag_to_max"] and r["flag"]["cut"] == [] and r["cap"]["cap"] == 28.0
    assert "against 28 h: OVER by" in line and "FLAG TO MAX, nothing is cut or queued without his decision" in line
    assert f"the cut order would take {', '.join(r['cap']['cut'])} -> " in line and r["cap"]["cut"]
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P, cap=28.0)
    assert not r["flag"]["flag_to_max"] and r["flag"]["proposal"] is None and r["cap"]["cap"] == 28.0
