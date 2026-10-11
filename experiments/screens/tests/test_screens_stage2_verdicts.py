"""SCREENS.txt STAGE 2 VERDICTS (2026-10-11): the C7 verdicts of S004, S005, S006 and S007 at seeds 101 and 102, the onset
readings, the C6 readings of all 9 contrasts, BASE-REPLAY's readings and labels and the BASE level follow from the
files the repo holds (out/ copies, base_out/, c6_out/, the seed-1 records) through analyze.py, c6read.py and
basereplay.py; the writer reproduces every results.json and table; the copies hash as recorded on the PC; each RESULT
block quotes its registration; no private identifier. A 5M loss reading only (C8). No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_verdicts.py
"""
import glob, hashlib, json, os, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import basereplay  # noqa: E402
import c6read  # noqa: E402
import privacy  # noqa: E402
import stage2_report  # noqa: E402
import stage2_tables  # noqa: E402
from s2v_lib import (EXP, HOLDS, KEYS, STAGE1, STAGE2, SX, VERDICT, build, flat, jload, res, roundtrip, text,  # noqa: E402
                     verdicts)

E3D = os.path.join(EXP, "E3_seed_noise")


@pytest.fixture(scope="module")
def T(tmp_path_factory):
    root = build(tmp_path_factory.mktemp("runs"))
    return root, verdicts(root)


def test_recorded_runs_give_the_verdicts(T):
    _, V = T
    assert roundtrip(V) == jload(SX, "stage2_verdicts.json")
    c = V["contrasts"]
    assert {k: x["verdict"] for k, x in c.items()} == {**STAGE1, **VERDICT}
    assert {k for k, x in c.items() if x["holm"] == "holds"} == HOLDS
    assert all(x["labels"] == [] and x["seeds"] == [101, 102] and x["dropped_base_diverged"] == [] for x in c.values())
    p = V["pooled_sia_check"]
    assert [(m, p[m]["df"], p[m]["arms"], p[m]["ok"]) for m in ("CHAT", "PROSE")] == [("CHAT", 7, 7, True), ("PROSE", 7, 7, True)]
    assert {k: V["picks"][k]["pick"] for k in VERDICT} == {"S004.canonac": 1.0, "S005.forget": 1.0, "S006.mtp": 2.0,
                                                          "S007.smear": 1.0}
    on = V["onsets"]
    assert on["sigma"]["base_onset_half"] == [612, 459, 459, 612, 459] and on["diag_problems"] == {}
    assert {k: (x["d"], x["label"], x["holm"]) for k, x in on["contrasts"].items()} == {
        "S004.canonac": ([0, 153], "UNRESOLVED", "does not hold"), "S006.mtp": ([-153, 765], "UNRESOLVED", "does not hold"),
        "S007.smear": ([-153, 153], "UNRESOLVED", "does not hold")}
    assert on["contrasts"]["S006.mtp"]["runs"] == ["s006_mtp_g1_s101", "s006_mtp_g1_s102"]   # the IND-only runs


def test_each_label_of_the_four_arms(T):
    _, V = T
    for k, want in VERDICT.items():
        r = V["contrasts"][k]["readings"]
        lab = {"NOMINATE": "BETTER", "REJECT": "WORSE", "TIE": "EQUAL"}[want]
        assert {m: x["label"] for m, x in r.items()} == {m: lab for m in r}          # so no CCCC-ONLY either


def test_results_json_carry_the_analyzer_readings(T):
    _, V = T
    for sid, key in KEYS.items():
        r = res(sid)
        assert r["contrasts"] == {key: roundtrip(V["contrasts"][key])} and r["verdicts"] == {key: VERDICT[key]}
        assert r["holm"]["family"] == V["holm_family"] and len(r["holm"]["family"]) == 9
        assert r["pooled_sia_check"] == roundtrip(V["pooled_sia_check"])
        assert r["verdict_at_e3_arm_a_mean"] == VERDICT[key]
        assert r["base_replay"]["label"] == ("not applicable" if sid == "S005" else "BASE replay: stable")


def test_c6_and_replay_readings_recomputed(T):
    root, V = T
    c6 = c6read.readings(root, V, jload(E3D, "results_part4d.json"), jload(SX, "c6_record.json"))
    assert roundtrip(c6) == jload(SX, "c6_readings.json")
    assert all(x["ok"] for x in c6["checks"].values()) and len(c6["checks"]) == 25
    hd = c6["metrics"]["HD_asst"]
    assert not hd["noise_checks"]["pooled_sia"]["ok"]
    assert {k for k, x in hd["contrasts"].items() if x["label"] == "LOWER"} == {
        "S001.nogate", "S002.nonormscale", "S002.novres", "S003.adamw", "S004.canonac", "S005.forget"}
    assert all(x["label"] == "UNRESOLVED" for m in ("IND", "HD_user") for x in c6["metrics"][m]["contrasts"].values())
    assert c6["s005_flag"]["flagged"] and not any(x["fires"] for x in c6["reported"]["near_init"].values())
    sia = ["S002.nonormscale", "S002.noqknorm", "S002.novres", "S004.canonac", "S005.forget", "S006.mtp", "S007.smear"]
    for k in sia:      # C4: "underpowered" only for a LARGER re-read SD; HD_asst's new-init SD is smaller (audit, 10:45)
        x = hd["contrasts"][k]
        assert x["labels"] == ["noise check failed (pooled SIA)"] and x["sd_ref"] < x["sd_ref_registered"]
    s = hd["sensitivity"]
    assert s["reread"] == sia and s["smaller_than_registered"] == sia and "sensitivity" not in c6["metrics"]["IND"]
    reg, po = s["at_registered_sd"], s["at_pooled_sd"]
    assert {k for k, h in reg["holm"].items() if h == "holds"} == {k for k, h in po["holm"].items() if h == "holds"} \
        == {"S001.nogate"}
    assert {k for k, x in reg["readings"].items() if x["label"] == "LOWER"} == {
        "S002.nonormscale", "S002.novres", "S004.canonac", "S005.forget"}
    assert all(x["label"] == "UNRESOLVED" and x["sd_ref"] == hd["noise_checks"]["pooled_sia"]["sd"]
               for x in po["readings"].values())
    br = basereplay.analyze(root, V, os.path.join(E3D, "out"), jload(E3D, "results.json"))
    assert roundtrip(br) == jload(SX, "base_replay.json")
    assert all(not x[s]["beyond"] for x in br["readings"].values() for s in ("D", "H"))
    assert br["base_level_101_105"]["seed_draw"] == "the seed draw is not excluded"


def test_at_e3_mean_scales_sd_thr_and_delta():
    """STAGE 1 VERDICTS' BASE LEVEL check, as the writer applies it: SD_ref, thr and delta at E3 arm A's mean."""
    import analyze_lib as AL
    e3 = jload(E3D, "results.json")["noise_5M_250M"]
    rd = {k: {**AL.metric_reading([-0.008, -0.008] if k == "CHAT" else [0.0, 0.0], 0.00378, 7, 0.5), "base_mean": 0.5}
          for k in AL.METRICS}
    c = {"readings": rd, "verdict": AL.arm_verdict(rd)[0]}
    assert c["verdict"] == "NOMINATE" and stage2_report.at_e3_mean(c, e3) == "INCONCLUSIVE"


def test_c7_reread_sd_is_larger_than_every_registered_sd():
    """C4 gives "underpowered" only to a re-read at a LARGER SD. analyze_lib.reread_new_init attaches it to every C7
    re-read; that is right only while E3's new-init SD exceeds the SIA and same-init SDs on every C7 metric, which
    E3's committed noise gives (pinned 2026-10-11 after the audit found the C6 case where it does not)."""
    import analyze_lib as AL
    import screens_lib as L
    e3, u = jload(E3D, "results.json")["noise_5M_250M"], L.params()["u"]
    for m in AL.METRICS.values():
        new = AL.sd_ref("new init", m, 1.0, e3, u)[0]
        assert new > AL.sd_ref("SIA", m, 1.0, e3, u)[0] and new > AL.sd_ref("same init", m, 1.0, e3, u)[0], m


def test_part4d_recomputed():
    sys.path.insert(0, E3D)
    import e3analyze_4d as P4
    assert roundtrip(P4.analyze(os.path.join(E3D, "out"))) == jload(E3D, "results_part4d.json")


def test_the_writer_reproduces_every_output(T, tmp_path):
    root, _ = T
    dest = tmp_path / "exp"
    for d in list(STAGE2.values()) + ["screens"]:
        os.makedirs(dest / d)
    src = jload(SX, "stage2_source.json")
    pcsha = tmp_path / "pcsha.txt"
    pcsha.write_text("".join(f"{h}  SCREENS/{p}\n" for p, h in src.pop("pc_files_sha256").items()))
    (tmp_path / "src.json").write_text(json.dumps(src))
    args = ["--runs", root, "--dest", str(dest), "--source", str(tmp_path / "src.json"), "--pcsha", str(pcsha)]
    for k, f in (("--verdicts", "stage2_verdicts.json"), ("--c6", "c6_readings.json"), ("--replay", "base_replay.json"),
                 ("--record", "c6_record.json")):
        args += [k, os.path.join(SX, f)]
    assert stage2_report.main(args) == 0 and stage2_tables.main(["--dest", str(dest)]) == 0
    for d in STAGE2.values():
        for f in ("results.json", "tables.txt"):
            assert (dest / d / f).read_text() == text(f"{d}/{f}"), (d, f)
    for f in ("c6_tables.txt", "base_tables.txt", "stage2_source.json"):
        assert (dest / "screens" / f).read_text() == text(f"screens/{f}"), f


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_copies_hash_as_on_the_pc():
    pc = jload(SX, "stage2_source.json")["pc_files_sha256"]
    seen = 0
    for d in STAGE2.values():
        rec = res(next(k for k, v in STAGE2.items() if v == d))["source"]["files_sha256"]
        files = sorted(glob.glob(os.path.join(EXP, d, "out", "*", "*")))
        assert sorted(os.path.relpath(f, os.path.join(EXP, d, "out")) for f in files) == sorted(rec)
        for f in files:
            rel = os.path.relpath(f, os.path.join(EXP, d, "out"))
            assert sha(f) == rec[rel] == pc[rel]
            seen += 1
    for sub, pre in (("base_out", ""), ("c6_out", "c6/")):
        for f in glob.glob(os.path.join(SX, sub, "*", "*")):
            run, name = f.split(os.sep)[-2:]
            assert sha(f) == pc[f"{run}/{pre}{name}"], f
            seen += 1
    assert seen == 54 + 17 + 83


def test_result_blocks_quote_their_registration():
    for sid, d in STAGE2.items():
        r = res(sid)
        reg = r["consequence_registered"]
        assert reg.startswith(VERDICT[KEYS[sid]] + " ") and reg in flat(text(f"{d}/notes.txt").split("METRIC AND DECISION RULE")[1])
        blk = flat(text(f"{d}/notes.txt").split("\nRESULT (2026-10-11")[1])
        assert f'Registered consequence (PRE-REGISTRATION, verbatim): "{reg}"' in blk
        assert f"VERDICT {KEYS[sid]}: {VERDICT[KEYS[sid]]} (raw C7)" in blk and "A 5M loss reading only (C8)" in flat(
            text(f"{d}/notes.txt").split("\nRESULT (2026-10-11")[1][:400])
    s6 = flat(text("S006_mtp_aux/notes.txt").split("\nRESULT (2026-10-11")[1])
    assert 'so the ledger row says "failed at 5M in this form (one t+2 head, weight decaying with the LR)", never ' \
           '"P-105 failed"' in s6 and "Q2 read UNRESOLVED, so it does not fire" in s6
    s5 = flat(text("S005_forget_gate/notes.txt").split("\nRESULT (2026-10-11")[1])
    assert res("S005")["s005_flag"]["flagged"] and "an arm that is BETTER on CHAT with lower HD is flagged\"): FLAGGED." in s5


def test_the_entry():
    t = text("SCREENS.txt")
    sec = t[t.index("\nSTAGE 2 VERDICTS (2026-10-11"):]
    assert sec.count("\nSTAGE 2 VERDICTS (") == 1 and t.rstrip().endswith(sec.rstrip())
    f = flat(sec)
    for k, v in VERDICT.items():
        assert f"{k} SIA {v}" in f
    assert "the first value read was analyze.py verdicts at 09:47:53 EDT" in f and "40 mutants" in f
    assert "the first logged run with all 40 killed is from 10:07, after it" in f and "Written AFTER values were read" in f
    assert "THE RE-READ SD IS SMALLER" in f and "\"underpowered\" (C4: a re-read at a LARGER SD) does not apply" in f
    assert "Holm over the 9 holds only for S001" in f and "the five LOWER readings against the shared BASE" in f
    assert "only if the PLAN 3d toy favours it over P-022" in f
    for d in ("S002_block_ablations", "S004_canon", "S005_forget_gate"):
        blk = flat(text(f"{d}/notes.txt").split("\nRESULT (2026-10-11" if d != "S002_block_ablations" else
                                                 "\nC6 READINGS (2026-10-11")[1])
        assert "is SMALLER than the SIA SD whose check failed" in blk, d
        assert not any(x in blk for x in ('(pooled SIA)" and "underpowered"', '(pooled SIA)", "underpowered"')), d
    assert not any(x in f for x in ('(pooled SIA)" and "underpowered"', '(pooled SIA)", "underpowered"'))
    assert "FOLLOW-UP TRIGGER does not fire" in f and "the seed draw is not excluded" in f
    assert "A 5M loss reading only (C8)" in f and "S005 FLAG (S005 DIAGNOSTICS): BETTER on CHAT with LOWER HD_asst: FLAGGED" in f


def test_no_private_identifier_in_any_new_file():
    pats = privacy.patterns()
    files = [f for d in STAGE2.values() for f in glob.glob(os.path.join(EXP, d, "**"), recursive=True) if os.path.isfile(f)]
    files += [f for s in ("base_out", "c6_out") for f in glob.glob(os.path.join(SX, s, "**"), recursive=True)
              if os.path.isfile(f)]
    files += [os.path.join(SX, x) for x in ("stage2_verdicts.json", "c6_readings.json", "base_replay.json", "c6_record.json",
                                            "stage2_source.json", "c6_tables.txt", "base_tables.txt")]
    files += glob.glob(os.path.join(E3D, "out", "*", "c6", "*.json")) + [os.path.join(E3D, "results_part4d.json")]
    assert len(files) > 200
    for f in files:
        assert privacy.hits(open(f, errors="replace").read(), pats) == [], os.path.relpath(f, EXP)
