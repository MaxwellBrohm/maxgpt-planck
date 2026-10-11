"""The 2026-10-10 entries (SCREENS.txt AMENDMENT CAP-FLAG, DEVIATION 1, ONSET RULE CODE; S006 notes FOLLOW-UP TRIGGER)
say what the code does and were written in the order the rules need: each pinned phrase is a rule a later step reads.
POSITION NOW's numbers are recomputed from a copy of the PC's queue files when SCREENS_PC_COPY names one. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_saturday_text.py
"""
import contextlib, io, os, re, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402

TXT = open(os.path.join(L.ROOT, "experiments", "SCREENS.txt")).read()
S006 = open(os.path.join(L.ROOT, "experiments", "S006_mtp_aux", "notes.txt")).read()


def entry(text, head):
    i = text.index("\n" + head) + 1
    j = text.find("\n\n", i)
    return " ".join(text[i:j if j > 0 else len(text)].split())


CAPF, DEV1, ONS = (entry(TXT, h) for h in ("AMENDMENT CAP-FLAG (2026-10-10", "DEVIATION 1 (2026-10-10",
                                             "ONSET RULE CODE (2026-10-10"))
TRIG = entry(S006, "FOLLOW-UP TRIGGER (2026-10-10")


def test_order_and_dates():
    pos = [TXT.index("\n" + h) for h in ("AMENDMENT BASE-DIAG (2026-10-06", "AMENDMENT CAP-FLAG (2026-10-10",
                                         "DEVIATION 1 (2026-10-10", "ONSET RULE CODE (2026-10-10")]
    assert pos == sorted(pos) and TXT.count("\nAMENDMENT CAP-FLAG (") == TXT.count("\nDEVIATION 1 (") == 1
    for e in (CAPF, DEV1, ONS):
        assert "nothing committed" in e and "\u2014" not in e
    assert "Written before any stage 2 onset, HD or verdict is read" in CAPF and "Written before any affected run" in DEV1
    assert S006.index("\nONSET READING (2026-10-06") < S006.index("\nFOLLOW-UP TRIGGER (2026-10-10")


def test_cap_flag_says_flag_never_cut_and_the_code_agrees():
    for phrase in ('"if something can be helpful even a little we should give it its proper chance"',
                   "The cap is 28 GPU-hours (AMENDMENT BASE-DIAG's number)", "nothing is cut and nothing more is queued",
                   "brings Max a FLAG with the total, the overrun and the proposal", "a screen is cut only on that answer",
                   "the E4 GATE (a scientific gate, not a budget rule) is unchanged", "analyze_lib.CAP = 28.0",
                   "its \"cut\" is always empty", "OVER by ... h: FLAG TO MAX"):
        assert phrase in CAPF, phrase
    r = AL.cap_check({"S006": 30.0}, cap=28.0)       # CAP-FLAG's 28 h; the code's cap is 31 h since AMENDMENT CAP-31
    assert r["flag_to_max"] and r["cut"] == [] and AL.CUT_ORDER == ["S006", "S003", "S007", "S002"] and AL.CAP >= 28.0


def test_deviation_names_each_change_and_what_did_not_change():
    for phrase in ("(1) SCORED AFTER ITS FILES", "(2) AN UNPARSABLE SCORE FILE IS SCORED AGAIN",
                   "(3) A CUT COUNTS AS THE RUN'S ONE CRASH", "(4) HOURS: EACH LOCK HOLD ONCE",
                   "<ckpt>.json.unparsable_<time> (kept, never deleted)", "A QRUN beside a final checkpoint is not counted",
                   "the old and the new parse give the same 96 runs", "measured_hours.tsv is not regenerated here",
                   "Not changed: queue_lib.sh, train.py, bpb.py, every config and plan"):
        assert phrase in DEV1, phrase


def test_onset_code_entry_and_the_s006_trigger():
    for phrase in ("thr = t(0.975, 4) x SD_ref / sqrt(k), 2.776 x sigma_seed at k 2", "no EQUAL",
                   "S006 mtp from its g 1 IND-only runs (s006_mtp_g1_s101, s006_mtp_g1_s102), never its g 2 verdict runs",
                   "S005 is not read (its own BASE)", "(b) A resumed run's diag.jsonl can repeat a step: the last reading",
                   "(c) sigma_seed 0", "--no-stage1-earlier"):
        assert phrase in ONS, phrase
    assert f"{AL.t_ppf(0.975, 4):.3f}" == "2.776" and round(153 / AL.t_ppf(0.975, 4), 1) == 55.1
    for phrase in ("Trigger: Q2 reads EARLIER", "whatever the bpb verdict (NOMINATE, TIE, REJECT, GUARD-FAIL or INCONCLUSIVE)",
                   "(1) NTP-to-MTP CURRICULUM at 5M", "(2) TWO AUX HEADS at 5M", "(3) 20M",
                   "each in its own entry written and committed before any of its runs",
                   "LATER, UNRESOLVED, LR-confounded or descriptive do not fire it", "never a silent cut of these arms"):
        assert phrase in TRIG, phrase


PC = os.environ.get("SCREENS_PC_COPY", "")


@pytest.mark.skipif(not os.path.isfile(os.path.join(PC, "queue_screens.txt")), reason="needs a copy of the PC queue files")
def test_position_now_is_the_queue_copy(tmp_path):
    q, st = (open(os.path.join(PC, f)).read() for f in ("queue_screens.txt", "status.jsonl"))
    assert H.L.sha256(os.path.join(PC, "queue_screens.txt")).startswith("93d1847aec949d30")
    t = tmp_path / "now.tsv"
    t.write_text(H.tsv_text(q, st))
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(L.params(), str(t), cap=28.0)
    runs, other = H.load(str(t))
    v = (len(runs), f"{sum(runs.values()):.3f}", f"{sum(other.values()):.3f}", f"{r['flag']['total']:.3f}",
         f"{28 - r['flag']['total']:.3f}", sum(r["tally"]["n_queued"].values()))
    assert v == (96, "23.867", "2.120", "25.987", "2.013", 0) and not r["flag"]["flag_to_max"]
    lo, hi = r["flag"]["total"] + 3 * H.run_h(L.STEPS) + 1, r["flag"]["total"] + 3 * H.run_h(L.STEPS) + 2
    assert (f"{lo:.1f}", f"{hi:.1f}") == ("27.9", "28.9")
    assert re.search(r"96 runs measured, 23\.867 h, \+ 2\.120 h OTHER = 25\.987 h; queued 0 of 90 registered runs; "
                     r"2\.013 h of headroom under 28 h", CAPF) and "would make 27.9 to 28.9 h" in CAPF


RAN = ("stage1_select", "stage1_s003B", "stage1_s003C", "stage1_seeds", "stage2_select", "stage2_s006x", "stage2_seeds",
       "stage2_s006_seeds", "stage2_base_diag", "stage2_replay")   # every SCREENS plan that ran (to 2026-10-10 23:51:19)


def test_every_queue_run_is_registered():
    """models/registry.jsonl (2026-10-10; stage2_replay's 3 runs 2026-10-11): one saved line per run of every plan that
    ran (its final checkpoint, in its run folder), the 33 stables of the 11 S003 trunks, the 18 runs of the last four
    plans (BASE 101/102 aside, scored in stage 1) marked not read, and the 16 smokes not_saved."""
    import json
    runs = {ln.split()[1] for p in RAN for ln in open(os.path.join(L.HERE, "plans", p + ".txt")) if ln.startswith("train ")}
    reg = [json.loads(x) for x in open(os.path.join(L.ROOT, "models", "registry.jsonl")) if x.strip()]
    sc = [e for e in reg if e["experiment"] == "SCREENS"]
    fin = {e["locations"][0].rsplit("/", 1)[1]: e for e in sc if e["kind"] == "from_scratch" and e["status"] != "not_saved"}
    assert set(fin) == runs and len(runs) == 99 and all(e["files"][0]["path"].startswith("final_") for e in fin.values())
    st = [e for e in sc if e["kind"] == "checkpoint"]
    trunks = {r for r in runs if r.endswith("_trunk")}
    assert len(trunks) == 11 and len(st) == 33 and {e["parent"] for e in st} == {fin[t]["id"] for t in trunks}
    late = {ln.split()[1] for p in RAN[-4:] for ln in open(os.path.join(L.HERE, "plans", p + ".txt"))
            if ln.startswith("train ")} - {"base_s101", "base_s102"}
    assert len(late) == 18 and {r for r, e in fin.items() if e["result"].startswith("not read")} == late
    assert sum(1 for e in sc if e["status"] == "not_saved" and e["role"] == "smoke") == 16 == len(sc) - 132
