"""ONSET RULE CODE's CORRECTION 1 (SCREENS.txt, CORRECTIONS AFTER THE W1 VERIFY, 2026-10-10): what makes a run's
diag.jsonl "diag incomplete" for AMENDMENT BASE-DIAG's ONSET RULE. A step's last line that is induction.py's error line
({"step", "error"}), or whose ind is null, NaN, infinite, missing or not a number; a grid step (every 153 steps and
7,630) with no reading; a step off the grid; a line that is not a JSON object with an integer step. NUL bytes are
dropped. Before this, an error line raised KeyError out of analyze.py verdicts (the bpb verdicts too), a null raised
TypeError, a missing middle step moved onset_half silently, and no test covered a NaN. Synthetic runs only.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_onset_diag.py
"""
import contextlib, io, json, os, sys  # noqa: E401

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze as A  # noqa: E402
import analyze_lib as AL  # noqa: E402
import screens_lib as L  # noqa: E402
from test_analyze import write_run  # noqa: E402
from test_screens_onset_rule import G1, P, STEPS, diag, fixture, picks_runs  # noqa: E402

SMEAR, B3 = "s007_smear_g1_s101", "base_s103"        # SMEAR's onset is step 1,071 in fixture()


def path(runs, name):
    return os.path.join(runs, name, "diag.jsonl")


def put(runs, name, step, raw, append=False):
    """Replace step's line by raw (bytes or str), or append raw as a later line (a resumed run's re-reading)."""
    raw = raw.encode() if isinstance(raw, str) else raw
    lines = open(path(runs, name), "rb").read().split(b"\n")
    if append:
        lines.insert(-1, raw)
    else:
        i = next(i for i, ln in enumerate(lines) if ln and json.loads(ln)["step"] == step)
        lines[i:i + 1] = [raw] if raw else []
    open(path(runs, name), "wb").write(b"\n".join(lines))


def readings(runs):
    return A.onset_readings(runs, P, G1)


def test_the_grid_is_the_hooks_logging_steps():
    base = yaml.safe_load(open(os.path.join(L.HERE, "configs", "screens_base.yaml")))
    assert A.IND_EVERY == base["eval"]["induction"]["every"] == 153 and base["train"]["total_steps"] == L.STEPS
    assert A.IND_GRID == set(STEPS) and len(A.IND_GRID) == 50 and max(A.IND_GRID) == 7630 and min(A.IND_GRID) == 153


@pytest.mark.parametrize("name,step", [(SMEAR, 1071), (SMEAR, 7630), (B3, 7630), (B3, 153)])
def test_an_error_line_is_diag_incomplete_and_verdicts_still_run(tmp_path, name, step):
    runs = fixture(tmp_path)
    put(runs, name, step, json.dumps({"step": step, "error": "RuntimeError: CUDA error"}), append=True)
    assert A.ind_curve(runs, name)[0] == "diag incomplete"
    o = readings(runs)
    assert o["diag_problems"] == {name: [f"step {step}: error line"]}
    if name == SMEAR:
        c = o["contrasts"]["S007.smear"]
        assert (c["reading"], c["why"], c["d"]) == ("descriptive", [f"{SMEAR} diag incomplete"], [459])
        assert o["contrasts"]["S004.canonac"]["label"] == "EARLIER" and o["holm_family"] == ["S004.canonac", "S006.mtp"]
    else:
        assert o["sigma"]["sigma_seed"] is None and o["base_states"][103] == "diag incomplete" and o["holm_family"] == []
        assert all(c["reading"] == "descriptive" for c in o["contrasts"].values())


def test_the_whole_verdicts_command_survives_an_error_line_and_its_bpb_verdicts_do_not_move(tmp_path):
    out = {}
    for tag in ("clean", "error"):
        runs = fixture(tmp_path / tag)
        picks_runs(runs)
        if tag == "error":
            put(runs, SMEAR, 1071, json.dumps({"step": 1071, "error": "ValueError: x"}))
        with contextlib.redirect_stdout(io.StringIO()):
            assert A.main(["verdicts", "--runs", runs, "--out", str(tmp_path / f"{tag}.json")]) == 0
        out[tag] = json.load(open(tmp_path / f"{tag}.json"))
    assert out["clean"]["contrasts"] == out["error"]["contrasts"] and out["clean"]["onsets"]["diag_problems"] == {}
    assert out["error"]["onsets"]["contrasts"]["S007.smear"]["reading"] == "descriptive"
    assert out["error"]["onsets"]["diag_problems"] == {SMEAR: ["step 1071: error line"]}


@pytest.mark.parametrize("raw,why", [
    ('{"step": 1071, "ind": NaN}', "no finite ind"), ('{"step": 1071, "ind": Infinity}', "no finite ind"),
    ('{"step": 1071, "ind": -Infinity}', "no finite ind"), ('{"step": 1071, "ind": null}', "no finite ind"),
    ('{"step": 1071, "nll_copy1": 3.0}', "no finite ind"), ('{"step": 1071, "ind": "8.0"}', "no finite ind"),
    ('{"step": 1071, "ind": true}', "no finite ind"), ('{"step": 1071, "ind": 8.0, "error": "x"}', "error line")])
def test_a_non_finite_or_absent_ind_at_the_onset_step_is_diag_incomplete(tmp_path, raw, why):
    """F2: a NaN at the onset step would otherwise move onset_half from 1,071 to 1,224 (NaN >= x is False)."""
    runs = fixture(tmp_path)
    put(runs, SMEAR, 1071, raw)
    assert A.onset_half(runs, SMEAR) == ("diag incomplete", None)
    assert readings(runs)["diag_problems"] == {SMEAR: [f"step 1071: {why}"]}


@pytest.mark.parametrize("raw", ['{"step": 7630, "ind": NaN}', '{"step": 7630, "ind": null}'])
def test_a_non_finite_final_reading_is_diag_incomplete_not_final_ind_le_0(tmp_path, raw):
    runs = fixture(tmp_path)
    put(runs, SMEAR, 7630, raw)
    c = readings(runs)["contrasts"]["S007.smear"]
    assert c["why"] == [f"{SMEAR} diag incomplete"] and c["arm_onset_half"][101] is None


def test_a_missing_middle_reading_or_a_step_off_the_grid_is_diag_incomplete(tmp_path):
    runs = fixture(tmp_path)
    put(runs, SMEAR, 1071, b"")                                            # its onset step: onset_half would be 1,224
    put(runs, B3, 0, json.dumps({"step": 1000, "ind": 8.0}), append=True)
    o = readings(runs)
    assert o["diag_problems"] == {B3: ["steps off the grid [1000]"], SMEAR: ["no reading at steps [1071]"]}
    assert o["sigma"]["sigma_seed"] is None and o["contrasts"]["S007.smear"]["arm_onset_half"][101] is None
    runs = fixture(tmp_path / "b")
    diag(runs, "base_s104", 918, steps=[s for s in STEPS if s != 4590])
    assert readings(runs)["base_states"][104] == "diag incomplete"


@pytest.mark.parametrize("raw", [b'{"step": 1530, "ind": 8.', b"\x00\x00{oops}", b'["step", 1530]',
                                 b'{"step": "1530", "ind": 8.0}', b'{"step": 1530.0, "ind": 8.0}', b"\xff\xfe"])
def test_a_line_that_is_not_a_reading_is_diag_incomplete(tmp_path, raw):
    runs = fixture(tmp_path)
    put(runs, SMEAR, 0, raw, append=True)
    assert readings(runs)["diag_problems"] == {SMEAR: [f"line {len(STEPS) + 1} is not a reading"]}


def test_nul_bytes_and_a_superseded_error_line_leave_the_run_ok(tmp_path):
    runs = fixture(tmp_path)
    put(runs, SMEAR, 306, json.dumps({"step": 306, "error": "x"}))           # the pre-crash attempt errored
    put(runs, SMEAR, 0, json.dumps({"step": 306, "ind": 2.0}), append=True)   # the resumed run read it again
    put(runs, SMEAR, 0, b"\x00" * 300 + json.dumps({"step": 459, "ind": 2.0}).encode(), append=True)
    open(path(runs, SMEAR), "ab").write(b"\x00" * 4096)                       # a power cut's NUL tail
    assert A.ind_curve(runs, SMEAR)[0] == "ok" and A.onset_half(runs, SMEAR) == ("ok", 1071)
    o = readings(runs)
    assert o["diag_problems"] == {} and o["contrasts"]["S007.smear"]["label"] == "LATER"


def test_no_diag_file_is_reported(tmp_path):
    runs = fixture(tmp_path)
    os.remove(path(runs, SMEAR))
    assert readings(runs)["diag_problems"] == {SMEAR: ["no diag.jsonl"]}


def test_stage_1s_recorded_base_curves_read_ok_with_their_recorded_onsets(tmp_path):
    """The rule against real data already read (S001 results.json, STAGE 1 VERDICTS): base_s101 and base_s102's 50
    readings each pass, and onset_half is the one recorded then."""
    r = json.load(open(os.path.join(os.path.dirname(L.HERE), "S001_attn_gate", "results.json")))["c6"]
    for n in ("base_s101", "base_s102"):
        write_run(str(tmp_path), n, 1.15, 1.38)
        (tmp_path / n / "diag.jsonl").write_text("".join(json.dumps({"step": s, "ind": x, "gate": None}) + "\n"
                                              for s, x in r["ind_curves"][n]))
        state, curve = A.ind_curve(str(tmp_path), n)
        assert state == "ok" and len(curve) == 50
        assert A.onset_half(str(tmp_path), n) == ("ok", r["ind_onsets"][n]["onset_half"])
        assert AL.onsets(curve, None)["onset_half"] == r["ind_onsets"][n]["onset_half"]
